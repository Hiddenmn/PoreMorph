"""Synthetic CUDA update at the native training batch/crop size; no research pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import reproduce as r
import main_native as n
from train_external_body import binary_writer


def exact(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = r.new_output(args.output)
    c = n.load_catalog()
    n.setup(42)
    if torch.cuda.mem_get_info()[0] < 4 * 2**30:
        raise RuntimeError("Leave existing jobs unchanged; need at least 4 GiB free for this check")
    model = n.build_model(c["cases"]["42"]).cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    weights = torch.tensor(c["cases"]["42"]["config"]["class_weights"], device="cuda")
    batch = {"image": torch.randint(256, (4, 384, 384), dtype=torch.uint8),
        "mask": torch.randint(8, (4, 384, 384), dtype=torch.uint8),
        "low": torch.randint(256, (4, 512, 512), dtype=torch.uint8),
        "coarse": torch.randn(4, 8, 512, 512).bfloat16(),
        "context": torch.zeros(4, 32, 128, 128),
        "origin": torch.tensor([[0, 0], [640, 640], [128, 64], [64, 128]]),
        "aug": torch.tensor([[0, 0, 0], [1, 0, 1], [0, 1, 2], [1, 1, 3]])}
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    loss = n.train_epoch(model, [batch], optimizer, scheduler, weights)
    if not any(not exact(before[k], value) for k, value in model.state_dict().items()):
        raise ValueError("Synthetic optimizer did not update the model")
    for prefix in ("fine", "middle", "deep", "up_middle", "up_fine", "output"):
        if not any(x.grad is not None and bool(x.grad.abs().sum() > 0)
                   for name, x in model.named_parameters() if name.startswith(prefix)):
            raise ValueError(f"Missing synthetic gradient: {prefix}")
    model.eval()
    image, coarse, context, _ = n.ops.local_batch(batch, "native")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(image, coarse, context).cpu()
    output.mkdir(parents=True, exist_ok=False)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    binary_writer(output, lambda handle, value: torch.save(value, handle))(output / "synthetic_only.pth", payload)
    saved = torch.load(output / "synthetic_only.pth", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for x in model.parameters():
            x.zero_()
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    if not all(exact(state, saved[key]) for key, state in (("model", model.state_dict()),
            ("optimizer", optimizer.state_dict()), ("scheduler", scheduler.state_dict()))):
        raise ValueError("Synthetic checkpoint restoration differs")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(image, coarse, context).cpu()
    if not torch.equal(expected, actual):
        raise ValueError("Restored synthetic native logits differ")
    report = {"status": "PASS", "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "synthetic_updates": 1, "synthetic_batch_size": 4, "crop": 384, "loss_grid": [320, 320],
        "parameters": 118744, "loss": loss, "checkpoint_state_exact": True, "restored_logits_exact": True,
        "all_six_parameter_groups_receive_gradients": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "research_pixels_read": False, "research_training_started": False, "test_pixels_read": False,
        "full_training_reproduced": False, "resume_supported": False}
    r.write_json(output / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
