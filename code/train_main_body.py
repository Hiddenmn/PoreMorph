"""Fresh main separate-head training on the recorded frozen TRAIN/VAL descriptors.

This command never updates Global/native models, loads TEST, or overwrites a run.
Native-to-descriptor cache generation is not part of this entry point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

import reproduce as r
import main_body as b
import train_main_native as native_training
import train_main_global as g
import train_external as t
from train_external_body import binary_writer

choose = native_training.choose  # Identical first maximum, epochs 1..12, native8 VAL.


def run_epochs(model, optimizer, scheduler, train, val, weights, output, identity, stop_roots):
    output = Path(output)
    write = t.atomic_writer(output)
    save = binary_writer(output, lambda handle, value: torch.save(value, handle))
    state = {"identity": identity, "status": "running", "started": t.utc(),
        "completed_epochs": 0, "test_pixels_read": False, "upstream_updated": False,
        "resume_supported": False}
    history, best = [], -1.
    write(output / "run_manifest.json", state)
    try:
        for epoch in range(1, 13):
            g.stop_check(*stop_roots)
            print(f"MAIN_BODY_EPOCH_START {epoch}/12", flush=True)
            loss = b.train_epoch(model, train, optimizer, scheduler, weights, identity["seed"], epoch)
            validation = b.ops.evaluate(model, val)
            row = {"epoch": epoch, "loss": loss, "validation": validation,
                   "lr_after_epoch": optimizer.param_groups[0]["lr"]}
            chosen = choose([*history, row])
            payload = {"model": model.state_dict(), "epoch": epoch, "validation": validation,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "history": [*history, row], "best_epoch": chosen["epoch"], "arm": "separate",
                "seed": identity["seed"], "identity": identity, "completed_epoch": True}
            save(output / "last_completed_epoch.pth", payload)
            if validation["native"]["mIoU"] > best:
                best = validation["native"]["mIoU"]
                save(output / "best.pth", payload)
            history.append(row)
            write(output / "history.json", history)
            state.update(completed_epochs=epoch, best_epoch=chosen["epoch"], best_native_mIoU=best, updated=t.utc())
            write(output / "run_manifest.json", state)
            print(f"MAIN_BODY_EPOCH_DONE {epoch}/12 native_mIoU={best:.8f} (best)", flush=True)
            g.stop_check(*stop_roots)
        saved = torch.load(output / "best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        replay = b.ops.evaluate(model, val)
        if replay != choose(history)["validation"] or replay != saved["validation"]:
            raise ValueError("Reloaded body validation differs from selected epoch")
        write(output / "selected_validation.json", {"epoch": saved["epoch"], "validation": replay,
            "checkpoint": "best.pth", "sha256": r.sha256(output / "best.pth"),
            "selection": identity["selection"], "test_pixels_read": False})
        state.update(status="completed", selected_replay_exact=True)
    except BaseException as exc:
        state.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error=f"{type(exc).__name__}: {exc}",
                     detail="Fresh-run entry: incomplete epoch discarded; completed checkpoints retained.")
        raise
    finally:
        state["updated"] = t.utc()
        write(output / "run_manifest.json", state)
    return state


def run(c, args):
    root = args.resource_root.resolve()
    prepared = getattr(args, "prepared_cache", None)
    output = b.output_path(args.output, root, c, prepared)
    stops = (output, output.parent, root, root / "实验", *([prepared] if prepared else []))
    g.stop_check(*stops)
    binding = b.bind(c, root, args.seed, prepared)
    case, folder = binding["case"], binding["folder"]
    b.setup(args.seed)
    model = b.build_model(case).cuda()
    train, val = b.Bodies(folder, "train", "cuda"), b.Bodies(folder, "val", "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    weights = torch.tensor(case["config"]["class_weights"], device="cuda")
    identity = {"stage": "main separate body head", "seed": args.seed, "config": case["config"],
        "selection": c["selection"], "selected_native": case["upstream"],
        "cache_manifest_sha256": case["resources"]["manifest"]["sha256"],
        "initial_sha256": case["initial_sha256"],
        "body_catalog_sha256": r.sha256(r.PACKAGE / "main_body_training.json"),
        "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"), "fresh_run": True,
        "python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name()}
    output.mkdir(parents=True, exist_ok=False)
    with t.output_lock(output):
        return run_epochs(model, optimizer, scheduler, train, val, weights, output, identity, stops)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 2024, 2025), required=True)
    parser.add_argument("--prepared-cache", type=Path, help="Complete new native-to-body descriptor preparation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    c = b.load_catalog()
    if args.action == "run":
        if args.output is None or args.report is not None:
            parser.error("run requires --output only")
        result = run(c, args)
    else:
        if args.report is None or args.output is not None:
            parser.error("check requires --report only")
        report = b.output_path(args.report, args.resource_root, c, args.prepared_cache)
        binding = b.bind(c, args.resource_root, args.seed, args.prepared_cache)
        b.build_model(binding["case"])
        result = {"status": "PASS", "seed": args.seed, "images": len(binding["records"]),
            "bodies": binding["case"]["bodies_by_split"], "cache_hashes_exact": True,
            "initial_state_exact": True, "gpu_initialized": torch.cuda.is_initialized(),
            "research_training_started": False, "test_pixels_read": False,
            "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json")}
        r.write_json(report, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
