"""Two synthetic CUDA updates per seed; never uses research descriptors."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import reproduce as r
import main_body as b
from test_main_body import synthetic_data
from train_external_body import binary_writer


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def original_update(model, data, opt, sched, weights, seed):
    # Literal epoch computation from replicate_body.py, evaluated independently.
    order = np.random.default_rng(seed*1000+1).permutation(data.indices)
    generator = torch.Generator(device="cuda").manual_seed(seed*10000+1)
    model.train(); losses = []
    for start in range(0, len(order), 512):
        idx = order[start:start+512]
        p, v, g = data.input_batch(idx)
        target = torch.from_numpy(np.array(data.arrays["native_counts"][idx], copy=True)).cuda().float()
        opt.zero_grad(set_to_none=True)
        loss = b.model_ops.body_loss(model(b.model_ops.augment(p, generator), v, g), target, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); losses.append(float(loss.detach()))
    sched.step()
    return float(np.mean(losses))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    c = b.load_catalog()
    out = r.new_output(args.output); out.mkdir(parents=True, exist_ok=False)
    save = binary_writer(out, lambda h, x: torch.save(x, h))
    cases = []
    for seed in (42, 2024, 2025):
        b.setup(seed)
        data = synthetic_data(520, "cuda")
        initial_arrays = {k: x.tobytes() for k, x in data.arrays.items()}
        case = c["cases"][str(seed)]
        model = b.build_model(case).cuda()
        reference = copy.deepcopy(model)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        ropt = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
        rsched = torch.optim.lr_scheduler.CosineAnnealingLR(ropt, T_max=12)
        weights = torch.tensor(case["config"]["class_weights"], device="cuda")
        torch.cuda.reset_peak_memory_stats()
        loss = b.train_epoch(model, data, opt, sched, weights, seed, 1)
        original_loss = original_update(reference, data, ropt, rsched, weights, seed)
        assert loss == original_loss and equal(model.state_dict(), reference.state_dict())
        assert equal(opt.state_dict(), ropt.state_dict()) and equal(sched.state_dict(), rsched.state_dict())
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
        payload = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}
        path = out / f"synthetic_seed{seed}.pth"
        save(path, payload)
        # AdamW step counters are CPU tensors in the original non-capturable
        # optimizer. Preserve that storage on load; moments follow parameters.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        restored = b.build_model(case).cuda(); restored.load_state_dict(saved["model"])
        new_opt = torch.optim.AdamW(restored.parameters(), lr=.001, weight_decay=.0001)
        new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=12)
        new_opt.load_state_dict(saved["optimizer"]); new_sched.load_state_dict(saved["scheduler"])
        assert equal(payload, {"model": restored.state_dict(), "optimizer": new_opt.state_dict(), "scheduler": new_sched.state_dict()})
        model.eval(); restored.eval()
        with torch.inference_mode():
            assert torch.equal(model(*data.input_batch(data.indices[:8])), restored(*data.input_batch(data.indices[:8])))
        assert initial_arrays == {k: x.tobytes() for k, x in data.arrays.items()}
        cases.append({"seed": seed, "parameters": 25573, "loss": loss,
            "original_epoch_computation_exact": True, "all_parameters_receive_finite_gradients": True,
            "checkpoint_state_exact": True, "restored_logits_exact": True,
            "descriptor_arrays_unchanged": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        del model, reference, restored, opt, ropt, sched, rsched, new_opt, new_sched, payload, saved
        torch.cuda.empty_cache()
    report = {"status": "PASS", "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "cases": cases, "synthetic_updates": 6, "reference_synthetic_updates": 6, "body_batch_sizes": [512, 8],
        "research_pixels_read": False, "research_training_started": False, "test_pixels_read": False,
        "full_training_reproduced": False, "resume_supported": False}
    r.write_json(out / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
