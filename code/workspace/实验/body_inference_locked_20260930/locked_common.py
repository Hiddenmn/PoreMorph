"""Immutable input references and small CPU-only evaluation helpers."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP, ROOT = HERE.parent, HERE.parent.parent
BODY = EXP / "body_interface_decoder_20260930"
POINT = EXP / "body_pointwise_control_20260930"
REPEAT = EXP / "body_decoder_replication_20260930"
SHARED = EXP / "native_detail_decoder_20260930/cache"
CANONICAL = EXP / "multimodal_remaining_oil_outputs_fast_ppm_before_fusion"
SPLITS = CANONICAL / "results/dataset_splits"
SEEDS = (42, 2024, 2025)
GRIDS = ("native", "canonical512")
METHODS = ("global", "native_detail", "body_mean", "pointwise", "separate")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stop_check():
    for folder in (HERE, ROOT, BODY, POINT, REPEAT):
        if (folder / "STOP_AFTER_EPOCH.txt").exists():
            raise InterruptedError(f"User stop marker: {folder}; never remove automatically")


def records():
    meta = read(CANONICAL / "preprocessed_cache_512/cache_meta.json")
    lookup = {}
    for i, name in enumerate(meta["record_keys"]):
        stage, z, *_ = name.split("|")
        key = (stage, int(z))
        assert key not in lookup
        lookup[key] = i
    result, used = {}, set()
    for split, expected in (("train", 2364), ("val", 676), ("test", 340)):
        with (SPLITS / f"{split}_records.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected
        for row in rows:
            row["slice_id"] = int(row["slice_id"])
            key = (row["sample"], row["slice_id"])
            assert key not in used
            used.add(key)
            row["cache_index"] = lookup[key]
            row["split"] = split
        assert all(sum(r["sample"] == stage for r in rows) == expected // 4 for stage in ("Oil", "WI", "NPI-006", "NPI-012"))
        result[split] = rows
    assert used == set(lookup)
    return result


def resolve(row, kind):
    original = Path(row[f"{kind}_path"])
    path = EXP / original.parent.name / original.name
    if path.is_file():
        return path
    suffix = f"filtered{row['slice_id']}" + (".tif" if kind == "image" else "_semantic_8bit_palette.tif")
    candidates = list(path.parent.glob("*" + suffix))
    if len(candidates) != 1:
        raise ValueError(f"No unique raw {kind} path: {path}")
    return candidates[0]


def paths(seed):
    global_dir = EXP / f"multimodal_remaining_oil_outputs_fusion_order_ablation/models/seed_{seed}_fusion_before_ppm"
    native_home = EXP / ({42: "native_detail_context_control_20260930", 2024: "native_detail_replication_20260930", 2025: "native_detail_replication_seed2025_20260930"}[seed])
    native_run = native_home / f"outputs/native_no_context_seed{seed}"
    body_run = BODY / "outputs/pair12_seed42" if seed == 42 else REPEAT / f"outputs/pair12_seed{seed}"
    point_run = POINT / "outputs/control12_seed42" if seed == 42 else body_run
    return {"global_checkpoint": global_dir / "best_model.pth", "tokenizer": global_dir / "tokenizer.json",
            "native_checkpoint": native_run / "best.pth", "native_manifest": native_run / "run_manifest.json",
            "native_selected": native_run / "selected_validation.json",
            "separate_checkpoint": body_run / "separate_best.pth", "body_manifest": body_run / "run_manifest.json",
            "pointwise_checkpoint": point_run / ("best.pth" if seed == 42 else "pointwise_best.pth"),
            "pointwise_manifest": point_run / "run_manifest.json",
            "separate_selected": body_run / "separate_selected_validation.json",
            "pointwise_selected": point_run / ("selected_validation.json" if seed == 42 else "pointwise_selected_validation.json"),
            "body_mean_selected": body_run / "body_mean_validation.json"}


def body_cache(seed):
    return BODY / "cache" if seed == 42 else REPEAT / f"cache_seed{seed}"


def coarse_cache(seed):
    if seed == 42:
        return SHARED
    home = EXP / ("native_detail_replication_20260930" if seed == 2024 else "native_detail_replication_seed2025_20260930")
    return home / f"coarse_cache_seed{seed}"


def identity():
    old = read(REPEAT / "run_manifest.json")
    assert old["status"] == "completed" and old["engineering_screen_passed"]
    sources = dict(old["identity"]["sources"])
    for path, expected in sources.items():
        assert sha(path) == expected, path
    for name in ("inference.py", "locked_common.py", "evaluate_locked.py", "test_inference.py", "PROTOCOL.md"):
        sources[str(HERE / name)] = sha(HERE / name)
    bundles = {}
    for seed in SEEDS:
        files = paths(seed)
        n, b, p = (read(files[k]) for k in ("native_manifest", "body_manifest", "pointwise_manifest"))
        assert n["status"] == b["status"] == p["status"] == "completed"
        assert n["selected_validation_verified"] and n["best_epoch"] == 11
        assert sha(files["native_checkpoint"]) == n["checkpoint_sha256"]
        assert sha(files["separate_checkpoint"]) == b["arms"]["separate"]["checkpoint_sha256"]
        info = p if seed == 42 else p["arms"]["pointwise"]
        assert sha(files["pointwise_checkpoint"]) == info["checkpoint_sha256"]
        assert b["arms"]["separate"]["best_epoch"] == 12
        assert info["best_epoch"] == (10 if seed == 2025 else 12)
        cached = read(coarse_cache(seed) / "manifest.json")["identity"]
        assert sha(files["global_checkpoint"]) == cached["checkpoint"]
        assert sha(files["tokenizer"]) == cached["tokenizer"]
        bundles[str(seed)] = {str(path): sha(path) for path in files.values()}
        for path in (body_cache(seed) / "manifest.json", body_cache(seed) / "records.json", coarse_cache(seed) / "manifest.json", coarse_cache(seed) / "records.json"):
            bundles[str(seed)][str(path)] = sha(path)
    return {"sources": sources, "bundles": bundles, "replication_manifest": sha(REPEAT / "run_manifest.json"),
            "splits": {split: sha(SPLITS / f"{split}_records.csv") for split in ("train", "val", "test")},
            "canonical_meta": sha(CANONICAL / "preprocessed_cache_512/cache_meta.json"),
            "shared_manifest": sha(SHARED / "manifest.json"), "shared_records": sha(SHARED / "records.json")}


def settings():
    return {"seeds": list(SEEDS), "methods": list(METHODS), "global_batch": 4, "native_batch": 1,
            "head_batch": 512, "global_native_precision": "BF16", "head_precision": "FP32",
            "descriptor_quantization": "points float16; summaries float32", "grids": list(GRIDS),
            "primary_grid": "canonical512", "test_count": 340, "val_count": 676, "gpu_cap": .35,
            "protocol": "frozen_body_inference_existing_holdout_v1", "optimization": None}


def histogram(prediction, target):
    assert prediction.shape == target.shape and prediction.min() >= 0 and prediction.max() <= 7
    assert target.min() >= 0 and target.max() <= 7
    return np.bincount((target.astype(np.int64) * 8 + prediction).ravel(), minlength=64).reshape(8, 8)


def metrics(h):
    h = np.asarray(h, dtype=np.int64)
    d, actual, predicted = np.diag(h).astype(np.float64), h.sum(1), h.sum(0)
    iou = d / (actual + predicted - d + 1e-8)
    dice = 2 * d / (actual + predicted + 1e-8)
    return {"mIoU": float(iou.mean()), "mDice": float(dice.mean()), "oil_macro_IoU": float(iou[3:].mean()),
            "confusion_matrix": h.tolist(), "class_IoU": iou.tolist(), "class_Dice": dice.tolist(),
            "gt_pixels": actual.tolist(), "pred_pixels": predicted.tolist()}


def allow_test(reports):
    assert set(reports) == {str(s) for s in SEEDS}
    assert all(r["status"] == "completed" and r["split"] == "val" and r["images"] == 676 and
               r["all_parity_exact"] is True and r["test_pixels_read"] is False for r in reports.values())
    return True
