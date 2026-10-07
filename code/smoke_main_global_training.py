"""One synthetic 512-grid CUDA update; not a formal batch-16 reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import reproduce as r
import train_main_global as m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = r.new_output(args.output)
    c = m.load_catalog()
    m.configure(c, 42)
    device = m.base.get_device()
    free, _ = torch.cuda.mem_get_info()
    if free < 4 * 1024 ** 3:
        raise RuntimeError("Leave running jobs unchanged; synthetic smoke needs 4 GiB free")
    m.base.set_seed(42)
    torch.set_num_threads(2)
    tok = m.tokenizer_for(c)
    model = m.order.build_model("fusion_before_ppm", tok.vocab_size).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    scaler = torch.amp.GradScaler("cuda") if m.base.autocast_dtype() == torch.float16 else None
    weights = torch.tensor(c["seeds"]["42"]["weights"], device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights)
    dice = m.base.WeightedDiceLoss(8, weights).to(device)
    ids, mask = tok.encode("Oil rock water")
    batch = {"image": torch.rand(2, 3, 512, 512), "mask": torch.randint(8, (2, 512, 512)),
             "input_ids": ids.repeat(2, 1), "attention_mask": mask.repeat(2, 1)}
    before = model.final_conv.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    train = m.base.train_one_epoch(model, [batch], optimizer, scheduler, scaler,
                                  device, ce, dice, 1, "SYNTHETIC main Global")
    if torch.equal(before, model.final_conv.weight) or not torch.isfinite(torch.tensor(train["loss"])):
        raise ValueError("Synthetic update did not change finite model parameters")
    model.eval()
    with torch.no_grad():
        logits_before = m.base.forward_loss(model, batch, device, ce, dice)[3].detach().cpu()
    output.mkdir(parents=True, exist_ok=False)
    save = m.binary_writer(output, lambda handle, value: torch.save(value, handle))
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": scheduler.state_dict(),
               "scaler_state_dict": None if scaler is None else scaler.state_dict()}
    save(output / "synthetic_only.pth", payload)
    saved = torch.load(output / "synthetic_only.pth", map_location="cpu", weights_only=False)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    scheduler.load_state_dict(saved["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(saved["scaler_state_dict"])
    if any(not torch.equal(v.detach().cpu(), saved["model_state_dict"][k]) for k, v in model.state_dict().items()):
        raise ValueError("Synthetic checkpoint state differs")
    with torch.no_grad():
        logits_after = m.base.forward_loss(model, batch, device, ce, dice)[3].detach().cpu()
    if not torch.equal(logits_before, logits_after):
        raise ValueError("Synthetic restored evaluation logits differ")
    report = {"status": "PASS", "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "synthetic_updates": 1, "synthetic_batch_size": 2, "grid": [512, 512],
        "parameters": sum(p.numel() for p in model.parameters()), "train": train,
        "checkpoint_state_exact": True, "restored_logits_exact": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "amp_dtype": str(m.base.autocast_dtype()), "device": torch.cuda.get_device_name(),
        "formal_batch16_memory_fit_tested": False, "persistent_worker_resume_tested": False,
        "research_pixels_read": False, "research_training_started": False, "test_pixels_read": False}
    r.write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
