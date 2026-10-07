"""Rebuild the exact external TRAIN/VAL caches from original images and fine8 masks.

CPU only; no model, optimizer, TEST loader, label editing or fitted normalization.
Inputs remain read-only. A separate output resource tree mirrors the catalog's
relative cache paths and can be used with train_external.py --cache-root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

import reproduce as r
import external_runtime as e
import train_external as t

ROLES = ("image", "low", "mask")


def fs(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def geometries():
    external = r.read_json(r.PACKAGE / "external_checkpoints.json")
    e.verify_sources(external)
    fields = ("raw_grid_yx", "native_grid_yx", "crop_yx")
    out = {}
    for case in external["cases"].values():
        current = {key: case[key] for key in fields}
        if case["dataset"] in out and out[case["dataset"]] != current:
            raise ValueError("Different preprocessing geometry across recorded cases")
        out[case["dataset"]] = current
    return out


def entries_for(catalog, dataset):
    if dataset not in ("all", "bentheimer", "wf1"):
        raise ValueError("Only the recorded external TRAIN/VAL datasets are allowed")
    return {key: value for key, value in catalog["entries"].items()
            if dataset == "all" or value["dataset"] == dataset}


def destination(path, resource_root, catalog):
    output = t.destination(path, resource_root, catalog)
    # Raw images and accepted labels are protected as well as existing caches.
    for entry in catalog["entries"].values():
        for row in entry["records"]:
            for role in ("raw", "label"):
                item = r.local_path(resource_root, Path(row[role]["path"]).parts[0])
                if output == item or output.is_relative_to(item) or item.is_relative_to(output):
                    raise ValueError("Output overlaps a raw-image or accepted-label tree")
    return output


def read_sample(row, geometry, resource_root):
    image = e.checked_resource(resource_root, row["raw"])
    label = e.checked_resource(resource_root, row["label"])
    native, low = e.prepare_image(image, geometry)
    with Image.open(fs(label)) as stream:
        mask = np.array(stream)
    if (mask.dtype != np.uint8 or mask.ndim != 2 or list(mask.shape) != geometry["raw_grid_yx"]
            or mask.size == 0 or mask.max() > 7):
        raise ValueError(f"Invalid original fine8 label grid: {row['label_id']}")
    if geometry["crop_yx"] is not None:
        (y0, y1), (x0, x1) = geometry["crop_yx"]
        mask = mask[y0:y1, x0:x1]
    if mask.shape != native.shape or list(mask.shape) != geometry["native_grid_yx"]:
        raise ValueError("Image and label native grids differ")
    return {"image": native, "low": low, "mask": mask}


def layout(entry, role):
    return (np.dtype("uint8") if role == "mask" else np.dtype("float32"),
            (entry["count"], 512, 512) if role == "low" else tuple(entry["shape"]))


def validate_array(path, entry, role, mode="r"):
    array = np.load(fs(path), mmap_mode=mode, allow_pickle=False)
    dtype, shape = layout(entry, role)
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f"Cache shape/type mismatch: {path}")
    return array


def close_arrays(arrays):
    for array in arrays.values():
        if array.flags.writeable:
            array.flush()
        array._mmap.close()


def verify_group(entry, root):
    for role, resource in entry["resources"].items():
        path = e.checked_resource(root, resource)
        array = validate_array(path, entry, role)
        array._mmap.close()
        if path.stat().st_size != resource["bytes"]:
            raise ValueError("Unexpected cache length")


def prepare_group(key, entry, geometry, resource_root, output, meta, writer):
    if entry["split"] not in ("train", "val"):
        raise ValueError("No TEST preparation route")
    state = meta["entries"].setdefault(key, {"completed_rows": 0, "status": "preparing"})
    if state["status"] == "completed":
        verify_group(entry, output)
        return
    count = entry["count"]
    if not 0 <= state["completed_rows"] <= count:
        raise ValueError("Invalid resume row")
    targets = {role: r.local_path(output, entry["resources"][role]["path"]) for role in ROLES}
    partial = {role: path.with_name(path.name + ".partial") for role, path in targets.items()}
    missing_bytes = sum(entry["resources"][role]["bytes"] for role in ROLES
                        if not targets[role].exists() and not partial[role].exists())
    if shutil.disk_usage(output).free < missing_bytes + 512 * 2**20:
        raise OSError("Insufficient disk space for the new cache and safety margin")
    arrays = {}
    try:
        for role in ROLES:
            target, temp = targets[role], partial[role]
            if target.exists():
                if state["completed_rows"] != count or temp.exists():
                    raise ValueError("Unexpected existing final cache")
                e.checked_resource(output, entry["resources"][role])
                arrays[role] = validate_array(target, entry, role)
            elif temp.exists():
                arrays[role] = validate_array(temp, entry, role, "r+")
            else:
                if state["completed_rows"] != 0:
                    raise ValueError("A partially prepared cache is missing")
                fs(temp.parent).mkdir(parents=True, exist_ok=True)
                dtype, shape = layout(entry, role)
                arrays[role] = np.lib.format.open_memmap(fs(temp), mode="w+", dtype=dtype, shape=shape)
        writer(output / "preparation.json", meta)
        for index in range(state["completed_rows"], count):
            t.stop_check(resource_root, output)
            sample = read_sample(entry["records"][index], geometry, resource_root)
            for role in ROLES:
                arrays[role][index] = sample[role]
            if (index + 1) % 25 == 0 or index + 1 == count:
                for array in arrays.values():
                    array.flush()
                state["completed_rows"] = index + 1
                writer(output / "preparation.json", meta)
                print(f"RAW_CACHE {key} {index + 1}/{count}", flush=True)
        counts = np.zeros(8, np.int64)
        for mask in arrays["mask"]:
            counts += np.bincount(mask.ravel(), minlength=8)
        if counts.tolist() != entry["class_pixel_counts"]:
            raise ValueError(f"Class pixel counts differ: {key}")
        # The exact original TRAIN-only weight function, with no fitting on VAL.
        weights = e.definitions("external_eightclass_adaptation_20261003/bridge.py",
                                 ("fine_weights",), {"np": np}).fine_weights
        if entry["split"] == "train":
            np.testing.assert_array_equal(weights(counts), np.asarray(entry["class_weights"], np.float32))
        elif entry["class_weights"] is not None:
            raise ValueError("VAL must not supply training class weights")
    finally:
        close_arrays(arrays)
    for role in ROLES:
        path = targets[role] if targets[role].exists() else partial[role]
        if r.sha256(path) != entry["resources"][role]["sha256"]:
            raise ValueError(f"Rebuilt cache differs from the archived bytes: {key}/{role}")
        if path == partial[role]:
            # No existing final path is replaced. Only this run's verified partial is moved.
            if targets[role].exists():
                raise FileExistsError(targets[role])
            path.rename(targets[role])
    state.update(status="completed", cache_files_exact=3, class_pixel_counts_exact=True,
                 training_weights_exact=entry["split"] == "train", completed_utc=t.utc())
    writer(output / "preparation.json", meta)


def identity(catalog, geometry, dataset):
    encoded = json.dumps({"catalog": catalog, "geometry": geometry}, sort_keys=True, ensure_ascii=False).encode()
    return {"entry_point": "external-raw-cache-v1", "dataset": dataset,
            "protocol_sha256": hashlib.sha256(encoded).hexdigest(),
            "package_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
            "python": sys.version, "numpy": np.__version__, "pillow": Image.__version__}


def build(catalog, geometry, dataset, resource_root, output, resume=False):
    output = destination(output, resource_root, catalog)
    t.stop_check(resource_root, output)
    ident = identity(catalog, geometry, dataset)
    entries = entries_for(catalog, dataset)
    if output.exists():
        if not resume:
            raise FileExistsError("Use a new output root, or explicitly resume the same preparation")
        meta = r.read_json(output / "preparation.json")
        if meta["identity"] != ident:
            raise ValueError("Preparation source, inputs, geometry or environment changed")
    else:
        if resume:
            raise FileNotFoundError("No preparation to resume")
        output.mkdir(parents=True, exist_ok=False)
        meta = {"identity": ident, "status": "preparing", "created_utc": t.utc(), "entries": {}}
    with t.output_lock(output):
        writer = t.atomic_writer(output)
        if (output / "preparation.json").exists():
            meta = r.read_json(output / "preparation.json")
            if meta["identity"] != ident:
                raise ValueError("Preparation identity changed before lock acquisition")
        writer(output / "preparation.json", meta)
        try:
            for key, entry in entries.items():
                t.stop_check(resource_root, output)
                prepare_group(key, entry, geometry[entry["dataset"]], resource_root, output, meta, writer)
            meta.update(status="completed", completed_utc=t.utc())
        except BaseException as error:
            meta.update(status="paused" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error=repr(error))
            raise
        finally:
            writer(output / "preparation.json", meta)
        report = {"status": "PASS", "dataset": dataset,
            "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
            "cache_files_exact": 3 * len(entries), "images": sum(x["count"] for x in entries.values()),
            "splits": list(entries), "class_counts_and_training_weights_exact": True,
            "all_rebuilt_cache_bytes_exact": True, "cache_root": str(output),
            "gpu_initialized": torch.cuda.is_initialized(), "training_started": False,
            "test_pixels_read": False, "paper_metrics_changed": False}
        writer(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "bentheimer", "wf1"), default="all")
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = build(t.load_catalog(), geometries(), args.dataset, args.resource_root, args.output, args.resume)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
