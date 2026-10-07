"""Frozen predictor: native images and description strings, never target labels."""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
EXP, ROOT = HERE.parent, HERE.parent.parent
sys.path.insert(0, str(ROOT / "MM-PoreFormer/scripts"))
import run_fusion_order_ablation as order
sys.path.insert(0, str(EXP / "native_detail_context_control_20260930"))
from model_no_context import NativeNoContext
from architecture import sample_native_grid
sys.path.insert(0, str(EXP / "body_interface_decoder_20260930"))
from model import BodyInterfaceDecoder
from representation import describe
sys.path.insert(0, str(EXP / "body_pointwise_control_20260930"))
from pointwise_model import PointwiseBodyDecoder

METHODS = ("global", "native_detail", "body_mean", "pointwise", "separate")


def low_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (1024, 1024):
        raise ValueError("Expected native 1024x1024 uint8 grayscale image")
    return np.asarray(Image.fromarray(image).resize((512, 512), Image.Resampling.BILINEAR))


def assign_bodies(original, components, assigned):
    """Map each predicted oil body to one subtype, retaining support exactly."""
    assert original.dtype == np.uint8 and original.shape == components.shape
    assigned = np.asarray(assigned)
    assert assigned.shape == (int(components.max()),)
    assert assigned.size == 0 or (assigned.min() >= 3 and assigned.max() <= 7)
    assert np.array_equal(components > 0, original >= 3)
    lookup = np.zeros(len(assigned) + 1, np.uint8)
    lookup[1:] = assigned
    result = original.copy()
    selected = components > 0
    result[selected] = lookup[components[selected]]
    assert np.array_equal(result >= 3, original >= 3)
    assert np.array_equal(result[~selected], original[~selected])
    return result


class FrozenPipeline:
    def __init__(self, seed, files):
        order.base.set_seed(seed)
        self.tokenizer = order.base.CharTokenizer.load(files["tokenizer"])
        self.global_model = order.build_model("fusion_before_ppm", self.tokenizer.vocab_size)
        saved = torch.load(files["global_checkpoint"], map_location="cpu", weights_only=False)
        self.global_model.load_state_dict(saved["model_state_dict"], strict=True)
        self.global_epoch = saved["epoch"]
        self.native_model = NativeNoContext()
        saved = torch.load(files["native_checkpoint"], map_location="cpu", weights_only=False)
        self.native_model.load_state_dict(saved["model"], strict=True)
        self.native_epoch = saved["epoch"]
        self.heads = {"separate": BodyInterfaceDecoder("separate"), "pointwise": PointwiseBodyDecoder()}
        self.head_epochs = {}
        for name, model in self.heads.items():
            saved = torch.load(files[f"{name}_checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"], strict=True)
            self.head_epochs[name] = saved["epoch"]
            model.cuda().eval().requires_grad_(False)
        for model in (self.global_model, self.native_model):
            model.cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)
        self.captured = {}
        self.hook = self.native_model.up_fine.register_forward_hook(lambda m, a, out: self.captured.update(feature=out))

    def close(self):
        self.hook.remove()
        self.captured.clear()
        self.global_model.cpu(); self.native_model.cpu()
        for model in self.heads.values():
            model.cpu()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict_batch(self, images, descriptions, diagnostics=False):
        if not 1 <= len(images) <= 4 or len(images) != len(descriptions):
            raise ValueError("Use 1--4 aligned images and descriptions")
        lows = [low_image(im) for im in images]
        tensors = [torch.from_numpy(low.copy()).float().unsqueeze(0).repeat(3, 1, 1) / 255 for low in lows]
        tokens = [self.tokenizer.encode(text) for text in descriptions]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.global_model(torch.stack(tensors).cuda().to(memory_format=torch.channels_last),
                torch.stack([t[0] for t in tokens]).cuda(), torch.stack([t[1] for t in tokens]).cuda())
        assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
        # Preserve the archived cache boundary exactly, including BF16 bits.
        coarse_bits = logits.cpu().view(torch.uint16).numpy().copy()
        del logits
        outputs = []
        for index, raw in enumerate(images):
            image = raw.astype(np.float32) / 255
            coarse = torch.from_numpy(coarse_bits[index].copy()).view(torch.bfloat16).cuda()[None]
            up = sample_native_grid(coarse, torch.zeros((1, 2), device="cuda"), 1024)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                refined = self.native_model(torch.from_numpy(image).cuda()[None, None], up, None)
            probabilities = refined.float().softmax(1)[0].cpu().numpy()
            feature = self.captured.pop("feature").float()[0].cpu().numpy()
            descriptors, components, native_pred = describe(image, probabilities, feature)
            masks = {"global": up.argmax(1)[0].cpu().numpy().astype(np.uint8), "native_detail": native_pred}
            mean = descriptors["summary"][:, 3:8].argmax(1) + 3
            masks["body_mean"] = assign_bodies(native_pred, components, mean)
            for name, model in self.heads.items():
                choices = []
                for start in range(0, len(mean), 512):
                    end = start + 512
                    args = [torch.from_numpy(np.array(descriptors[k][start:end], dtype=dtype)).cuda()
                            for k, dtype in (("points", np.float32), ("valid", bool), ("summary", np.float32))]
                    choices.append(model(*args).argmax(1).cpu().numpy() + 3)
                assigned = np.concatenate(choices) if choices else np.empty(0, np.int64)
                masks[name] = assign_bodies(native_pred, components, assigned)
            canonical = {name: value[1::2, 1::2].copy() for name, value in masks.items()}
            canonical["global"] = coarse.argmax(1)[0].cpu().numpy().astype(np.uint8)
            result = {"native": masks, "canonical512": canonical, "bodies": len(mean)}
            if diagnostics:
                result["diagnostics"] = {"coarse_bits": coarse_bits[index], "low": lows[index], **descriptors}
            outputs.append(result)
        return outputs
