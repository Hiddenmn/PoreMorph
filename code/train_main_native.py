"""Fresh-run main-domain native-detail training from a fixed Global cache.

Only TRAIN crops update the 118744-parameter NativeNoContext network. The
Global network is not constructed by this trainer; its BF16 logits are fixed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

import reproduce as r
import train_main_global as g
import train_external as t
from train_external_body import binary_writer
import main_native as n


def choose(history):
    if (not history or [x["epoch"] for x in history] != list(range(1, len(history) + 1))
            or len(history) > 12):
        raise ValueError("Native selection requires completed epochs starting at 1")
    scores = [float(x["validation"]["native"]["mIoU"]) for x in history]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError("Invalid native validation score")
    return history[max(range(len(scores)), key=scores.__getitem__)]


def run_epochs(model, optimizer, scheduler, train, val, sampler, weights, output, identity, stop_roots):
    output = Path(output)
    write = t.atomic_writer(output)
    save = binary_writer(output, lambda handle, value: torch.save(value, handle))
    state = {"identity": identity, "status": "running", "started": t.utc(),
             "completed_epochs": 0, "test_pixels_read": False, "resume_supported": False}
    history, best = [], -1.
    write(output / "run_manifest.json", state)
    try:
        for epoch in range(1, 13):
            g.stop_check(*stop_roots)
            sampler.epoch = epoch
            print(f"NATIVE_EPOCH_START {epoch}/12", flush=True)
            loss = n.train_epoch(model, train, optimizer, scheduler, weights)
            validation = n.ops.evaluate(model, val, "native")
            row = {"epoch": epoch, "loss": loss, "validation": validation,
                   "lr_after_epoch": optimizer.param_groups[0]["lr"]}
            chosen = choose([*history, row])
            payload = {"model": model.state_dict(), "epoch": epoch, "validation": validation,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "history": [*history, row], "best_epoch": chosen["epoch"],
                "identity": identity, "completed_epoch": True}
            save(output / "last_completed_epoch.pth", payload)
            if validation["native"]["mIoU"] > best:
                best = validation["native"]["mIoU"]
                save(output / "best.pth", payload)
            history.append(row)
            write(output / "history.json", history)
            state.update(completed_epochs=epoch, best_epoch=chosen["epoch"], best_native_mIoU=best, updated=t.utc())
            write(output / "run_manifest.json", state)
            print(f"NATIVE_EPOCH_DONE {epoch}/12 native_mIoU={best:.8f} (best)", flush=True)
            g.stop_check(*stop_roots)
        # Read only this run's own checkpoint, never an untrusted downloaded file.
        saved = torch.load(output / "best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        replay = n.ops.evaluate(model, val, "native")
        if replay != choose(history)["validation"] or saved["validation"] != replay:
            raise ValueError("Reloaded native validation differs from selected epoch")
        write(output / "selected_validation.json", {"epoch": saved["epoch"], "validation": replay,
            "checkpoint": "best.pth", "sha256": r.sha256(output / "best.pth"),
            "selection": identity["selection"], "test_pixels_read": False})
        state.update(status="completed", selected_replay_exact=True)
    except BaseException as exc:
        state.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error=f"{type(exc).__name__}: {exc}",
                     detail="Incomplete epoch discarded; completed checkpoints retained. Fresh-run entry only.")
        raise
    finally:
        state["updated"] = t.utc()
        write(output / "run_manifest.json", state)
    return state


def run(c, args):
    root = args.resource_root.resolve()
    output = n.output_path(args.output, root, c, args.prepared_cache)
    stops = (output, output.parent, root, root / "实验")
    g.stop_check(*stops)
    binding = n.bind(c, root, args.seed, args.prepared_cache)
    n.setup(args.seed)
    case = c["cases"][str(args.seed)]
    train, val, sampler = n.loaders(binding)
    model = n.build_model(case).cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    weights = torch.tensor(case["config"]["class_weights"], device="cuda")
    identity = {"stage": "main native-detail", "seed": args.seed, "config": case["config"],
        "selection": c["selection"], "selected_global": binding["selected_global"],
        "coarse_sha256": binding["coarse_sha256"], "full_initial_sha256": model.full_initial_sha256,
        "native_catalog_sha256": r.sha256(r.PACKAGE / "main_native_training.json"),
        "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "fresh_run": True}
    output.mkdir(parents=True, exist_ok=False)
    with t.output_lock(output):
        return run_epochs(model, optimizer, scheduler, train, val, sampler, weights, output, identity, stops)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 2024, 2025), required=True)
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    c = n.load_catalog()
    if args.action == "run":
        if args.output is None or args.report is not None:
            parser.error("run requires --output only")
        result = run(c, args)
    else:
        if args.report is None or args.output is not None:
            parser.error("check requires --report only")
        report = r.new_output(args.report)
        binding = n.bind(c, args.resource_root, args.seed, args.prepared_cache)
        result = {"status": "PASS", "seed": args.seed, "images": binding["n"],
            "train": 2364, "val": 676, "cache_hashes_exact": True,
            "coarse_sha256": binding["coarse_sha256"], "selected_global": binding["selected_global"],
            "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
            "test_pixels_read": False, "research_training_started": False,
            "gpu_initialized": torch.cuda.is_initialized()}
        r.write_json(report, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
