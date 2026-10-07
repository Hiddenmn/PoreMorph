"""Cache frozen selected Global logits for main TRAIN/VAL native-detail training.

Use an archived selected model or a completed portable main Global run. Inputs
are existing native pixel caches. No raw-cache creation or optimizer is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

import reproduce as r
import external_runtime as e
import train_main_global as g
import train_external as t
import main_native as n


def selection_contract(meta, selected, history, catalog, global_catalog, seed):
    identity = meta["identity"]
    compatible = [*catalog["compatible_global_contents_sha256"], r.sha256(r.PACKAGE / "contents.json")]
    if (meta["status"] != "completed" or meta["test_pixels_read"]
            or identity["seed"] != seed or identity["config"] != global_catalog["config"]
            or identity["selection"] != global_catalog["selection"]
            or identity["main_catalog_sha256"] != r.sha256(r.PACKAGE / "main_training.json")
            or identity["source_contents_sha256"] not in compatible
            or not identity["fresh_run"] or meta["completed_epochs"] != len(history)
            or not 1 <= len(history) <= 60
            or [x["epoch"] for x in history] != list(range(1, len(history) + 1))):
        raise ValueError("A completed, matching portable Global run is required")
    scores = [float(x["val"]["mIoU"]) for x in history]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError("Nonfinite/out-of-range Global validation score")
    best, best_epoch, stale = -1., -1, 0
    for i, score in enumerate(scores, 1):
        best, best_epoch, stale, _ = g.advance_selection(i, score, best, best_epoch, stale)
        if stale >= 10 and i != len(scores):
            raise ValueError("History continued past the recorded early-stopping rule")
    stopped_by = "early_stopping" if stale >= 10 else "epoch_limit"
    if (len(scores) != 60 and stale < 10 or meta["stopped_by"] != stopped_by
            or selected["epoch"] != best_epoch or meta["best_epoch"] != best_epoch
            or selected["val_mIoU"] != best or meta["best_val_miou"] != best
            or selected["selection"] != global_catalog["selection"]
            or selected["checkpoint"] != "best_model.pth"):
        raise ValueError("Global selection is not the first maximum from a completed run")
    return best_epoch


def selected_global(c, root, seed, global_run=None):
    case = c["cases"][str(seed)]
    global_c = g.load_catalog()
    if global_run is None:
        path = e.checked_resource(root, case["global_checkpoint"])
        token = e.checked_resource(root, case["tokenizer"])
        info = {"kind": "archived-selected", "seed": seed, "checkpoint_sha256": case["global_checkpoint"]["sha256"]}
        return path, token, info, None
    folder = Path(global_run).resolve()
    g.stop_check(folder)
    meta = r.read_json(folder / "run_manifest.json")
    history = r.read_json(folder / "history.json")
    selected = r.read_json(folder / "selected_validation.json")
    epoch = selection_contract(meta, selected, history, c, global_c, seed)
    path = e.checked_resource(folder, {"path": selected["checkpoint"], "sha256": selected["sha256"]})
    token = folder / "tokenizer.json"
    if r.read_json(token) != global_c["tokenizer"]:
        raise ValueError("New Global tokenizer differs from the fixed TRAIN vocabulary")
    info = {"kind": "new-portable-selected", "seed": seed, "checkpoint_sha256": selected["sha256"],
        "epoch": epoch, "source_contents_sha256": meta["identity"]["source_contents_sha256"],
        "run_manifest_sha256": r.sha256(folder / "run_manifest.json"),
        "selected_validation_sha256": r.sha256(folder / "selected_validation.json")}
    return path, token, info, {"identity": meta["identity"], "history": history, "selected": selected}


def load_teacher(c, seed, path, token_path, info, portable):
    if r.sha256(path) != info["checkpoint_sha256"]:
        raise ValueError("Selected Global checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tok = g.base.CharTokenizer.load(token_path)
    if {"max_len": tok.max_len, "idx2token": tok.idx2token} != g.load_catalog()["tokenizer"]:
        raise ValueError("Wrong Global tokenizer")
    if payload["experiment"]["model_type"] != "fusion_before_ppm":
        raise ValueError("Wrong Global model role")
    if portable is not None:
        selected = portable["selected"]
        if (not payload["completed_epoch"] or payload["portable_identity"] != portable["identity"]
                or payload["epoch"] != selected["epoch"] or payload["best_epoch"] != selected["epoch"]
                or payload["best_val_miou"] != selected["val_mIoU"]
                or payload["history"] != portable["history"][:selected["epoch"]]):
            raise ValueError("Selected Global tensor payload differs from its completed run")
    else:
        history = payload["history"]
        best_index = int(np.argmax(history["val_mIoU"]))
        if (payload["checkpoint_type"] != "best"
                or payload["epoch"] != payload["best_epoch"]
                or payload["epoch"] != history["epoch"][best_index]
                or payload["best_val_miou"] != history["val_mIoU"][best_index]):
            raise ValueError("Archived Global selection differs")
    teacher = g.order.build_model("fusion_before_ppm", tok.vocab_size)
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    if not all(torch.isfinite(x).all() for x in teacher.state_dict().values()):
        raise ValueError("Nonfinite Global parameters")
    return teacher.cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False), tok


def run(args):
    if not args.trusted_checkpoints:
        raise ValueError("Explicit trust of local pickle checkpoints is required")
    c = n.load_catalog()
    root = args.resource_root.resolve()
    output = n.output_path(args.output, root, c, args.global_run)
    g.stop_check(root, root / "实验", output.parent)
    path, token, info, portable = selected_global(c, root, args.seed, args.global_run)
    for item in c["shared"].values():
        e.checked_resource(root, item)
    rows = c["records"]
    if args.smoke:
        rows = [x for x in rows if x["split"] == "train"][:4] + [x for x in rows if x["split"] == "val"][:4]
    byte_count = len(rows) * 8 * 512**2 * 2
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < byte_count + 2 * 2**30:
        raise RuntimeError("Insufficient space for a separate coarse cache and reserve")
    n.setup(args.seed)
    teacher, tok = load_teacher(c, args.seed, path, token, info, portable)
    output.mkdir(parents=True, exist_ok=False)
    meta = {"status": "preparing", "seed": args.seed, "smoke": args.smoke, "n": len(rows),
        "records": rows, "completed_records": 0, "selected_global": info, "test_pixels_read": False,
        "native_catalog_sha256": r.sha256(r.PACKAGE / "main_native_training.json"),
        "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"), "batch_size": 4,
        "research_training_started": False, "created": t.utc()}
    write = t.atomic_writer(output)
    with t.output_lock(output):
        write(output / "native_cache.json", meta)
        coarse = np.memmap(output / "coarse.dat", mode="w+", dtype=np.uint16, shape=(len(rows), 8, 512, 512))
        images = np.memmap(r.local_path(root, c["shared"]["low"]["path"]), mode="r", dtype=np.uint8, shape=(3040, 512, 512))
        try:
            for start in range(0, len(rows), 4):
                g.stop_check(root, root / "实验", output, output.parent, *([args.global_run] if args.global_run else []))
                batch = rows[start:start+4]
                local = torch.stack([torch.from_numpy(np.array(images[x["shared_index"]], copy=True)).float()
                                     .unsqueeze(0).repeat(3, 1, 1)/255 for x in batch])
                tokens = [tok.encode(x["text"]) for x in batch]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = teacher(local.cuda().to(memory_format=torch.channels_last),
                        torch.stack([x[0] for x in tokens]).cuda(), torch.stack([x[1] for x in tokens]).cuda())
                if logits.dtype != torch.bfloat16 or not torch.isfinite(logits).all():
                    raise ValueError("Unexpected frozen Global logits")
                encoded = logits.cpu().view(torch.uint16).numpy()
                if not torch.equal(torch.from_numpy(encoded).view(torch.bfloat16).float(), logits.float().cpu()):
                    raise ValueError("BF16 cache round-trip failed")
                coarse[start:start+len(batch)] = encoded
                if start % 100 == 0 or start + len(batch) == len(rows):
                    coarse.flush()
                    meta.update(completed_records=start+len(batch), updated=t.utc())
                    write(output / "native_cache.json", meta)
                    print(f"NATIVE_COARSE {start+len(batch)}/{len(rows)}", flush=True)
            coarse.flush()
            meta.update(status="completed", completed_records=len(rows),
                        coarse={"path": "coarse.dat", "sha256": r.sha256(output / "coarse.dat")})
        except BaseException as exc:
            meta.update(status="stopped" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            meta["updated"] = t.utc()
            write(output / "native_cache.json", meta)
            del teacher, coarse, images
            torch.cuda.empty_cache()
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 2024, 2025), required=True)
    parser.add_argument("--global-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trusted-checkpoints", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "seed", "n", "smoke", "coarse")}))


if __name__ == "__main__":
    main()
