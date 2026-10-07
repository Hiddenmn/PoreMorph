"""Portable, fresh-run main-domain Global training (no TEST loader).

The archived model, augmentation, batch loss, optimizer step and evaluation
definitions are reused. Resource routing and the outer run loop are portable.
This entry does not train the native refiner or the oil-body classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

import reproduce as r
import train_external as t
from train_external_body import binary_writer

sys.dont_write_bytecode = True
sys.path.insert(0, str(r.WORKSPACE / "MM-PoreFormer/scripts"))
order = importlib.import_module("run_fusion_order_ablation")
base = order.base
for mod in (order, base):
    Path(mod.__file__).resolve().relative_to(r.WORKSPACE.resolve())


def validate_catalog(c):
    if (c["version"] != "main-global-training-v1" or c["test_loader"]
            or c["test_pixels_read"] or set(c["seeds"]) != {"42", "2024", "2025"}):
        raise ValueError("Unexpected main-training scope")
    cfg = c["config"]
    for k, v in dict(target_size=[512, 512], batch_size=16, num_epochs=60,
                     patience=10, num_classes=8, max_text_len=192, num_workers=8).items():
        if cfg[k] != v:
            raise ValueError(f"Changed main protocol: {k}")
    rows = c["records"]
    if len(rows) != 3040 or len({x["id"] for x in rows}) != 3040:
        raise ValueError("Duplicate or incomplete TRAIN/VAL rows")
    if len({x["cache_index"] for x in rows}) != 3040:
        raise ValueError("Overlapping cache rows")
    for split, expected_ids in (("train", set(range(591))), ("val", set(range(591, 760)))):
        for stage in ("NPI-006", "NPI-012", "Oil", "WI"):
            ids = {x["slice_id"] for x in rows if x["split"] == split and x["sample"] == stage}
            if ids != expected_ids:
                raise ValueError("The fixed contiguous main split changed")
    for x in rows:
        if (x["split"] not in ("train", "val") or x["id"] != f"{x['sample']}_{x['slice_id']:03d}"
                or not 0 <= x["cache_index"] < 3380 or len(x["counts"]) != 8
                or min(x["counts"]) < 0 or sum(x["counts"]) != 512 * 512):
            raise ValueError("Invalid main cache record")
    train = {x["id"] for x in rows if x["split"] == "train"}
    for seed, info in c["seeds"].items():
        expected = sorted(train, key=lambda key: (key.rsplit("_", 1)[0], int(key.rsplit("_", 1)[1])))
        random.Random(int(seed)).shuffle(expected)
        sampled = expected.copy()
        random.Random(int(seed)).shuffle(sampled)
        if info["train_ids"] != expected or info["weight_sample_ids"] != sampled[:600]:
            raise ValueError("Training/shuffled weight-sampling order changed")
    if c["cache"]["shape"] != [3380, 512, 512]:
        raise ValueError("Wrong cache geometry")
    return c


def load_catalog():
    r.verify_contents()
    c = validate_catalog(r.read_json(r.PACKAGE / "main_training.json"))
    for source in c["sources"]:
        if r.sha256(r.local_path(r.WORKSPACE, source["path"])) != source["sha256"]:
            raise ValueError("Archived computational source changed")
    return c


def configure(c, seed):
    if str(seed) not in c["seeds"]:
        raise ValueError("Choose seed 42, 2024 or 2025")
    for key, value in c["config"].items():
        setattr(base.Config, key, tuple(value) if key == "target_size" else value)
    base.Config.seed = seed


def stop_check(*roots):
    t.stop_check(*roots)
    for root in roots:
        marker = Path(root) / "STOP_TRAINING.txt"
        if marker.exists():
            raise InterruptedError(f"Stop marker retained: {marker}")


def cache_binding(c, resource_root):
    cache = c["cache"]
    path = r.local_path(resource_root, cache["metadata"]["path"])
    if r.sha256(path) != cache["metadata"]["sha256"]:
        raise ValueError("Shared cache metadata changed")
    meta = r.read_json(path)
    for row in c["records"]:
        key = meta["record_keys"][row["cache_index"]].split("|")
        if (key[0], int(key[1])) != (row["sample"], row["slice_id"]):
            raise ValueError("Wrong cache row identity")
    files = {}
    for key, item in cache["arrays"].items():
        files[key] = str(r.local_path(resource_root, item["path"]))
        if Path(files[key]).stat().st_size != item["bytes"]:
            raise ValueError("Shared cache file size changed")
    return {"files": files, "shape": tuple(cache["shape"])}


class MainDataset(base.MultimodalRemainingOilDataset):
    """Only routing differs: explicit, read-only memmaps survive Windows spawn."""
    def __init__(self, records, tokenizer, binding, augment=False):
        if any(x["split"] not in ("train", "val") for x in records):
            raise ValueError("No TEST records accepted")
        super().__init__(records, tokenizer, (512, 512), augment)
        self.binding = binding
        self.use_cache = True

    def _ensure_cache_open(self):
        if self._images_mm is None or self._masks_mm is None:
            self._images_mm, self._masks_mm = (
                np.memmap(self.binding["files"][k], mode="r", dtype=np.uint8,
                          shape=self.binding["shape"]) for k in ("image", "mask"))
            self._cache_meta = {"shape": self.binding["shape"]}


def records_for(c, seed, split):
    if str(seed) not in c["seeds"] or split not in ("train", "val"):
        raise ValueError("Only the recorded TRAIN/VAL splits and seeds are accepted")
    by_id = {x["id"]: x for x in c["records"]}
    if split == "train":
        return [by_id[key] for key in c["seeds"][str(seed)]["train_ids"]]
    return sorted((x for x in c["records"] if x["split"] == "val"),
                  key=lambda x: (x["sample"], x["slice_id"]))


def tokenizer_for(c):
    tok = base.CharTokenizer([x["text"] for x in c["records"] if x["split"] == "train"], max_len=192)
    if {"max_len": tok.max_len, "idx2token": tok.idx2token} != c["tokenizer"]:
        raise ValueError("TRAIN-only vocabulary differs from the original tokenizer")
    return tok


def check_resources(c, root, seed):
    """Check row bytes/counts, then independently use the original weight code."""
    configure(c, seed)
    stop_check(root, Path(root) / "实验")
    binding = cache_binding(c, root)
    data = MainDataset(c["records"], tokenizer_for(c), binding)
    data._ensure_cache_open()
    for row in c["records"]:
        stop_check(root, Path(root) / "实验")
        for key, mm in (("image", data._images_mm), ("mask", data._masks_mm)):
            a = np.asarray(mm[row["cache_index"]])
            if hashlib.sha256(a.tobytes()).hexdigest() != row[f"{key}_sha256"]:
                raise ValueError(f"Main {key} bytes changed: {row['id']}")
            if key == "mask" and np.bincount(a.ravel(), minlength=8).tolist() != row["counts"]:
                raise ValueError(f"Main counts changed: {row['id']}")
    # The original helper is used, but its global cache root is explicit here.
    # Spawned datasets above never depend on this parent-process configuration.
    base.Config.cache_dir = Path(binding["files"]["image"]).parent
    weights = base.compute_auto_class_weights(records_for(c, seed, "train"), 600)
    expected = torch.tensor(c["seeds"][str(seed)]["weights"], dtype=torch.float32)
    if not torch.equal(weights, expected):
        raise ValueError("Original TRAIN-only class weights differ")
    return binding, weights, {"images": 3040, "train": 2364, "val": 676,
        "cache_row_bytes_exact": True, "class_weights_exact": True,
        "weights": weights.tolist(), "test_pixels_read": False}


def advance_selection(epoch, score, best, best_epoch, stale):
    if not 1 <= epoch <= 60 or not math.isfinite(score):
        raise ValueError("Only finite validation scores from completed epochs 1..60")
    improved = score > best
    return (score, epoch, 0, True) if improved else (best, best_epoch, stale + 1, False)


def run_epochs(model, optimizer, scheduler, scaler, tokenizer, train_loader, val_loader,
               weights, device, output, identity, stop_roots):
    """No mid-epoch state is ever saved as a completed epoch."""
    write = t.atomic_writer(output)
    save = binary_writer(output, lambda handle, value: torch.save(value, handle))
    ce = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    dice = base.WeightedDiceLoss(8, weights.to(device)).to(device)
    best, best_epoch, stale, completed = -1., -1, 0, 0
    history = []
    state = {"identity": identity, "status": "running", "started": t.utc(),
             "completed_epochs": 0, "test_pixels_read": False, "resume_supported": False}
    write(Path(output) / "run_manifest.json", state)
    try:
        for epoch in range(1, 61):
            stop_check(*stop_roots)
            train = base.train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                device, ce, dice, epoch, "main_global")
            val = base.evaluate_model(model, val_loader, device, ce, dice, "main_global VAL")
            best, best_epoch, stale, improved = advance_selection(epoch, float(val["mIoU"]), best, best_epoch, stale)
            row = {"epoch": epoch, "train": train, "val": val,
                   "lr_after_epoch": optimizer.param_groups[0]["lr"]}
            payload = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": None if scaler is None else scaler.state_dict(),
                "best_val_miou": best, "best_epoch": best_epoch, "patience_counter": stale,
                "history": [*history, row], "vocab_size": tokenizer.vocab_size, "max_text_len": 192,
                "class_names": base.Config.class_names,
                "experiment": {"name": f"seed_{base.Config.seed}_fusion_before_ppm", "model_type": "fusion_before_ppm"},
                "portable_identity": identity, "completed_epoch": True, "saved_time": t.utc()}
            save(Path(output) / "last_completed_epoch.pth", payload)
            if improved:
                save(Path(output) / "best_model.pth", payload)
            history.append(row)
            completed = epoch
            write(Path(output) / "history.json", history)
            state.update(completed_epochs=completed, best_epoch=best_epoch, best_val_miou=best, updated=t.utc())
            write(Path(output) / "run_manifest.json", state)
            stop_check(*stop_roots)
            if stale >= 10:
                break
        selected = {"epoch": best_epoch, "val_mIoU": best, "selection": identity["selection"],
                    "checkpoint": "best_model.pth", "sha256": r.sha256(Path(output) / "best_model.pth")}
        write(Path(output) / "selected_validation.json", selected)
        state.update(status="completed", stopped_by="early_stopping" if stale >= 10 else "epoch_limit")
    except BaseException as exc:
        state.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error=f"{type(exc).__name__}: {exc}", completed_epochs=completed,
                     detail="Incomplete epoch discarded; last completed checkpoint retained")
        raise
    finally:
        state["updated"] = t.utc()
        write(Path(output) / "run_manifest.json", state)
    return state


def output_path(path, root, c):
    resources = {"meta": c["cache"]["metadata"], **c["cache"]["arrays"]}
    target = t.destination(path, root, {"entries": {"main": {"resources": resources}}})
    if target.exists():
        raise FileExistsError("Fresh runs only: choose a new output directory")
    return target


def run(c, args):
    output = output_path(args.output, args.resource_root, c)
    stop_check(args.resource_root, args.resource_root / "实验", output.parent)
    binding, weights, checked = check_resources(c, args.resource_root, args.seed)
    device = base.get_device()  # Requires CUDA; preserves TF32/BF16 policy.
    base.set_seed(args.seed)
    tok = tokenizer_for(c)
    train_data = MainDataset(records_for(c, args.seed, "train"), tok, binding, augment=True)
    val_data = MainDataset(records_for(c, args.seed, "val"), tok, binding)
    train_loader = base.build_loader(train_data, 16, shuffle=True, drop_last=True)
    val_loader = base.build_loader(val_data, 16, shuffle=False, drop_last=False)
    model = order.build_model("fusion_before_ppm", tok.vocab_size).to(device)
    model = model.to(memory_format=torch.channels_last)
    if sum(x.numel() for x in model.parameters()) != 9023884:
        raise ValueError("Global architecture differs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    scaler = torch.amp.GradScaler("cuda") if base.autocast_dtype() == torch.float16 else None
    identity = {"seed": args.seed, "config": c["config"], "selection": c["selection"],
        "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "main_catalog_sha256": r.sha256(r.PACKAGE / "main_training.json"),
        "resource_root": str(args.resource_root.resolve()), "checks": checked,
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "amp_dtype": str(base.autocast_dtype()),
        "stage": "Global scratch training only", "fresh_run": True}
    output.mkdir(parents=True, exist_ok=False)
    with t.output_lock(output):
        tok.save(output / "tokenizer.json")
        return run_epochs(model, optimizer, scheduler, scaler, tok, train_loader, val_loader,
            weights, device, output, identity, (output, output.parent, args.resource_root, args.resource_root / "实验"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 2024, 2025), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    c = load_catalog()
    if args.action == "run":
        if args.output is None or args.report is not None:
            parser.error("run requires --output only")
        result = run(c, args)
    else:
        if args.report is None or args.output is not None:
            parser.error("check requires --report only")
        report = r.new_output(args.report)
        _, _, details = check_resources(c, args.resource_root, args.seed)
        result = {"status": "PASS", "seed": args.seed, **details,
            "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
            "gpu_initialized": torch.cuda.is_initialized(), "research_training_started": False}
        r.write_json(report, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
