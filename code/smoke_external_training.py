"""Bounded CUDA save/resume check on synthetic inputs; never a research run."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import reproduce as r
import external_runtime as e
import train_external as t


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            a = item.detach().cpu().contiguous()
            digest.update(str((a.dtype, tuple(a.shape))).encode())
            digest.update(a.numpy().tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                digest.update(repr(key).encode())
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


class SyntheticData:
    def __init__(self, catalog, dataset, d4):
        self.dataset, self.split, self.d4 = dataset, "synthetic", d4
        self.description = catalog["descriptions"][dataset]
        rng = np.random.default_rng(41006)
        self.images = rng.random((2, 96, 112), dtype=np.float32)
        self.low = np.stack([e.preprocessing().low_image(x) for x in self.images])
        self.labels = ((np.indices((2, 96, 112)).sum(0) // 7) % 8).astype(np.uint8)
        self.entry = {"class_weights": np.ones(8, np.float32).tolist()}

    def __len__(self):
        return 2

    def get(self, index, transform=None):
        values = (self.images[index], self.low[index], self.labels[index])
        return tuple(x.copy() if transform is None else self.d4(x, transform) for x in values)


def prediction(engine, model, data):
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        image, low, _ = engine.adapter.to_device(data.get(0))
        result = model(image, low, *engine.original.tokens(model, data))
        return [x.detach().float().cpu() for x in result]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-checkpoints", action="store_true")
    args = parser.parse_args()
    if not args.trusted_checkpoints:
        raise ValueError("Explicit trust of local initialization checkpoints is required")
    catalog = t.load_catalog()
    output = t.destination(args.output, args.resource_root, catalog)
    if output.exists():
        raise FileExistsError("Use a fresh synthetic-check folder")
    t.stop_check(args.resource_root)
    initial = t.verify_resources(catalog, args.resource_root, "wf1", 42, include_caches=False)
    t.require_cuda()
    free, _ = torch.cuda.mem_get_info()
    if free < 4 * 2**30:
        raise RuntimeError("Less than 4 GiB free VRAM; do not disturb other work")
    output.mkdir(parents=True, exist_ok=False)
    engine = t.build_engine(catalog, 42, initial, args.resource_root, output)
    data = SyntheticData(catalog, "wf1", engine.c.d4)
    report = {"status": "running", "scope": "synthetic engineering check only",
              "research_pixels_read": False, "research_training_started": False,
              "test_pixels_read": False, "native_shape": [96, 112], "global_shape": [512, 512],
              "seed": 42, "arms": {}, "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json")}
    writer = t.atomic_writer(output)
    writer(output / "report.json", report)
    try:
        for arm in t.ARMS:
            t.stop_check(args.resource_root, output)
            model = engine.adapter.AdaptationModel(arm).cuda().to(memory_format=torch.channels_last)
            optimizer = engine.original.optimizer_for(model)
            initial_hash = engine.c.state_hash(model.state_dict())
            frozen_hash = engine.c.state_hash(model.frozen_state())
            rows_before = model.global_model.text_encoder.token_embed.new_weight.detach().cpu().clone()
            torch.cuda.reset_peak_memory_stats()
            plan = [(0, 0), (1, 1), (0, 4), (1, 8)]
            first = engine.original.train_epoch(model, data, optimizer, 1, plan=plan)
            assert first["images"] == 4 and first["updates"] == 1
            first_state = engine.c.state_hash(model.state_dict())
            assert first_state != initial_hash
            assert not torch.equal(rows_before, model.global_model.text_encoder.token_embed.new_weight.detach().cpu())
            before = prediction(engine, model, data)
            first_optimizer = tree_hash(optimizer.state_dict())
            checkpoint = output / f"synthetic_{arm}.pth"
            saved_hash = engine.original.save_checkpoint(checkpoint, {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "epoch": 1, "synthetic_only": True})
            engine.original.close_model(model, optimizer)
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            # New instances exercise the same model/AdamW restoration as run_epoch.
            model = engine.adapter.AdaptationModel(arm).cuda().to(memory_format=torch.channels_last)
            optimizer = engine.original.optimizer_for(model)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            del saved
            assert engine.c.state_hash(model.state_dict()) == first_state
            assert tree_hash(optimizer.state_dict()) == first_optimizer
            after = prediction(engine, model, data)
            for old, new in zip(before, after):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            second = engine.original.train_epoch(model, data, optimizer, 2, plan=plan)
            assert second["updates"] == 1 and engine.c.state_hash(model.state_dict()) != first_state
            assert engine.c.state_hash(model.frozen_state()) == frozen_hash
            report["arms"][arm] = {"first_update": first, "resumed_update": second,
                "model_and_optimizer_restore_exact": True, "restored_logits_exact": True,
                "frozen_state_unchanged": True, "new_character_rows_updated": True,
                "synthetic_checkpoint_sha256": saved_hash, "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20}
            engine.original.close_model(model, optimizer)
            del model, optimizer, before, after
            gc.collect()
            torch.cuda.empty_cache()
            writer(output / "report.json", report)
        report.update(status="PASS", completed_utc=t.utc(), synthetic_updates=4)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        writer(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
