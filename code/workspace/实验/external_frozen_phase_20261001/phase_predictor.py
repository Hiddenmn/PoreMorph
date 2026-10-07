"""Fixed global/native branches; no target, filename or sample-ID input."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

from external_inputs import ROOT, EXP, nearest512
sys.path.insert(0, str(EXP / 'body_inference_locked_20260930'))
import locked_common as locked
from inference import order, NativeNoContext

METHODS = ('global', 'native_detail')
RULES = ('collapse8', 'marginal4')
GRIDS = ('canonical512', 'native')


def project_native(coarse, height, width):
    yy = (torch.arange(height, device=coarse.device, dtype=torch.float32)+.5)*(2./height)-1
    xx = (torch.arange(width, device=coarse.device, dtype=torch.float32)+.5)*(2./width)-1
    grid = torch.stack((xx[None, :].expand(height, -1), yy[:, None].expand(-1, width)), -1)[None]
    return F.grid_sample(coarse.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)


def decisions(logits):
    assert logits.ndim == 4 and logits.shape[:2] == (1, 8) and torch.isfinite(logits).all()
    fine = logits.argmax(1)[0]
    collapse = fine.clamp_max(3)
    scores = logits.float()
    marginal = torch.cat((scores[:, :3], torch.logsumexp(scores[:, 3:], 1, keepdim=True)), 1).argmax(1)[0]
    return {key: value.cpu().numpy().astype(np.uint8)
            for key, value in (('collapse8', collapse), ('marginal4', marginal))}


class PhasePredictor:
    def __init__(self, seed):
        files = locked.paths(seed)
        order.base.set_seed(seed)
        self.tokenizer = order.base.CharTokenizer.load(files['tokenizer'])
        self.global_model = order.build_model('fusion_before_ppm', self.tokenizer.vocab_size)
        saved = torch.load(files['global_checkpoint'], map_location='cpu', weights_only=False)
        self.global_model.load_state_dict(saved['model_state_dict'], strict=True)
        self.epochs = {'global': saved['epoch']}
        self.native_model = NativeNoContext()
        saved = torch.load(files['native_checkpoint'], map_location='cpu', weights_only=False)
        self.native_model.load_state_dict(saved['model'], strict=True)
        self.epochs['native_detail'] = saved['epoch']
        for model in (self.global_model, self.native_model):
            model.cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, native_image, global_image, description, diagnostics=False):
        for image in (native_image, global_image):
            assert image.dtype == np.float32 and image.ndim == 2
            assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
        assert global_image.shape == (512, 512) and min(native_image.shape) >= 4
        ids, attention = self.tokenizer.encode(description)
        image = torch.from_numpy(global_image.copy())[None, None].repeat(1, 3, 1, 1).cuda().to(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            coarse = self.global_model(image, ids[None].cuda(), attention[None].cuda())
        assert coarse.dtype == torch.bfloat16
        # Preserve the formal global-output precision boundary, without a new uint8 image conversion.
        bits = coarse.cpu().view(torch.uint16).numpy().copy()
        coarse = torch.from_numpy(bits.copy()).view(torch.bfloat16).cuda()
        up = project_native(coarse, *native_image.shape)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            refined = self.native_model(torch.from_numpy(native_image.copy()).cuda()[None, None], up, None)
        glob_low, glob_native, detail_native = decisions(coarse), decisions(up), decisions(refined)
        result = {grid: {rule: {} for rule in RULES} for grid in GRIDS}
        for rule in RULES:
            result['canonical512'][rule] = {'global': glob_low[rule], 'native_detail': nearest512(detail_native[rule])}
            result['native'][rule] = {'global': glob_native[rule], 'native_detail': detail_native[rule]}
        if diagnostics:
            result['diagnostics'] = {'coarse_bits': bits[0], 'fine_native': refined.argmax(1)[0].cpu().numpy().astype(np.uint8)}
        return result

    def close(self):
        self.global_model.cpu(); self.native_model.cpu()
        torch.cuda.empty_cache()
