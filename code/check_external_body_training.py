"""Bounded engineering checks; never calls the formal body-training run loop."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import reproduce as r
import external_runtime as e
import train_external as t
import train_external_body as b


class SyntheticBodies:
    def __init__(self):
        generator = torch.Generator().manual_seed(6100642)
        self.indices = np.arange(520)
        self.points = torch.randn(520, 48, 31, generator=generator) * .1
        self.valid = torch.rand(520, 48, generator=generator) > .2
        self.valid[:, 0] = True
        self.summary = torch.rand(520, 13, generator=generator) + .1
        self.summary[:, :8] /= self.summary[:, :8].sum(1, keepdim=True)
        self.counts = torch.randint(1, 20, (520, 8), generator=generator).float()

    def inputs(self, indices, device="cuda"):
        return tuple(value[indices].to(device) for value in (self.points, self.valid, self.summary))

    def targets(self, indices, device="cuda"):
        return self.counts[indices].to(device)


def synthetic(catalog, resource_root, output):
    case = catalog["cases"]["original_bentheimer_seed42"]
    engine = b.build_engine(case, resource_root, output)
    b.require_cuda()
    data = SyntheticBodies()
    model = engine.c.source_head("separate").cuda()
    before = copy.deepcopy(model.state_dict())
    config = case["config"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    one = engine.runner.train_epoch(model, data, optimizer, case["dataset"], "separate", 1)
    scheduler.step()
    if all(torch.equal(before[k], value) for k, value in model.state_dict().items()):
        raise ValueError("No parameter updated")
    path = output / "synthetic_epoch_001.pth"
    engine.c.save_checkpoint(path, {"purpose": "synthetic only; never a research checkpoint",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": scheduler.state_dict()})
    restored = engine.c.source_head("separate").cuda()
    other = torch.optim.AdamW(restored.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    other_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(other, T_max=12)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_state_dict(saved["model"], strict=True)
    other.load_state_dict(saved["optimizer"])
    other_schedule.load_state_dict(saved["schedule"])
    torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), other.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        x = data.inputs(np.arange(7))
        torch.testing.assert_close(model(*x), restored(*x), rtol=0, atol=0)
    two = engine.runner.train_epoch(model, data, optimizer, case["dataset"], "separate", 2)
    scheduler.step()
    resumed_two = engine.runner.train_epoch(restored, data, other, case["dataset"], "separate", 2)
    other_schedule.step()
    torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), other.state_dict(), rtol=0, atol=0)
    if scheduler.state_dict() != other_schedule.state_dict() or two["loss"] != resumed_two["loss"]:
        raise ValueError("Resumed update differs")
    for report in (one, two, resumed_two):
        if report["bodies"] != 520 or report["updates"] != 2:
            raise ValueError("Full/partial synthetic batch was not exercised")
    initial = e.checked_resource(resource_root, case["initial_head"])
    result = {"status": "PASS", "purpose": "synthetic CUDA body-head engineering check",
              "synthetic_updates": 6, "body_batch_sizes": [512, 8], "precision": "FP32",
              "model_optimizer_scheduler_restore_exact": True, "restored_logits_exact": True,
              "next_epoch_matches_uninterrupted": True, "initial_checkpoint_sha256": r.sha256(initial),
              "research_pixels_read": False, "research_training_started": False, "test_pixels_read": False,
              "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json")}
    model.cpu(); restored.cpu()
    del model, restored, optimizer, other, scheduler, other_schedule, x
    torch.cuda.empty_cache()
    return result


def validation(catalog, resource_root, output):
    b.require_cuda()
    results = []
    for name, case in catalog["cases"].items():
        t.stop_check(output, resource_root)
        _, resources = b.verify_resources(case, resource_root)
        path = e.checked_resource(resource_root, case["archived_head"])
        selected = r.read_json(e.checked_resource(resource_root, case["archived_selection"]))
        run = r.read_json(e.checked_resource(resource_root, case["archived_run"]))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved["identity"] != run["identity"] or saved["epoch"] != selected["selected_epoch"]
                or saved["dataset"] != case["dataset"] or saved["arm"] != "separate"):
            raise ValueError("Wrong archived body-checkpoint role")
        engine = b.build_engine(case, resource_root, output)
        model = engine.core.BodyInterfaceDecoder("separate").cuda()
        model.load_state_dict(saved["model"], strict=True)
        model.eval().requires_grad_(False)
        data = engine.data.Bodies(r.local_path(resource_root, case["cache_folder"]), "val")
        measured = engine.data.evaluate(model, data)
        if measured != selected["validation"]:
            raise ValueError(f"Archived validation metrics differ: {name}")
        dense_check = None
        if name in ("original_bentheimer_seed42", "original_wf1_seed42"):
            # Fixed domain/seed choices, not selected from the replay results.
            head_folder = r.local_path(resource_root, case["archived_selection"]["path"]).parent
            dense_check = engine.runner.verify_assignments(head_folder, selected, data, case["dataset"])
        results.append({"case": name, "selected_epoch": selected["selected_epoch"],
            "validation_images": len(data.records), "validation_bodies": len(data.indices),
            "all_metrics_and_confusion_matrices_exact": True, "dense_replay": dense_check, **resources})
        print(f"BODY_VALIDATION_EXACT {name} epoch={selected['selected_epoch']}", flush=True)
        model.cpu()
        del model, saved, data, engine
        torch.cuda.empty_cache()
    return {"status": "PASS", "purpose": "All recorded VAL descriptors, not TEST or retraining",
            "cases": len(results), "results": results, "canonical_and_native_exact": True,
            "training_started": False, "test_pixels_read": False, "paper_metrics_changed": False,
            "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("synthetic", "validation"))
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-checkpoints", action="store_true")
    args = parser.parse_args()
    if not args.trusted_checkpoints:
        raise ValueError("Explicit trust of local pickle checkpoints is required")
    catalog = b.load_catalog()
    output = b.output_path(args.output, args.resource_root, catalog)
    t.stop_check(output, args.resource_root)
    output.mkdir(parents=True, exist_ok=False)
    with t.output_lock(output):
        result = globals()[args.mode](catalog, args.resource_root, output)
        r.write_json(output / "report.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
