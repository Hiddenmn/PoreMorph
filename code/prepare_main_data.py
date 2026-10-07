"""Original main TRAIN/VAL TIFFs -> exact Global/native pixel resource tree.

No TEST input, fitted normalization, model or optimizer. Historical small binding
records are copied verbatim only after pixel verification; preparation.json is
the new run record. No old cache, checkpoint or manuscript is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

import reproduce as r
import train_main_global as g
import main_native as n
import external_runtime as e
import train_external as t
from prepare_external_data import fs
from train_external_body import binary_writer

ROLES = ("native_image", "native_mask", "native_low", "global_image", "global_mask")


def load_catalog():
    gc, nc = g.load_catalog(), n.load_catalog()
    c = r.read_json(r.PACKAGE/"main_raw_preparation.json")
    if (c["version"] != "main-raw-preparation-v1" or c["test_pixels_read"] or c["dtype"] != "uint8"
            or c["geometry"] != {"native": 1024, "global": 512}
            or c["images_by_split"] != {"train": 2364, "val": 676}
            or len(c["records"]) != 3040 or set(c["arrays"]) != set(ROLES)):
        raise ValueError("Unexpected raw main TRAIN/VAL scope")
    for path, digest in c["sources"].items():
        e.checked_resource(r.WORKSPACE, {"path": path, "sha256": digest})
    old_rows = r.read_json(r.WORKSPACE/nc["shared"]["records"]["path"])
    for i, (row, old, gr) in enumerate(zip(c["records"], old_rows, gc["records"], strict=True)):
        if (any(row[k] != gr[k] for k in ("id", "sample", "slice_id", "split", "cache_index", "counts"))
                or row["shared_index"] != i or row["split"] not in ("train", "val")
                or row["slice_id"] >= 760 or set(row["pixels"]) != set(ROLES)):
            raise ValueError("Raw/native/Global split or row identity changed")
        for key in ("image", "mask"):
            if row["raw"][key]["sha256"] != old[key+"_sha256"]:
                raise ValueError("Raw file identity differs from the original native-cache record")
            if row["pixels"]["global_"+key] != gr[key+"_sha256"]:
                raise ValueError("Canonical pixel identity changed")
    for key in ("image", "mask", "low"):
        a = c["arrays"]["native_"+key]
        side = 512 if key == "low" else 1024
        if a != dict(nc["shared"][key], shape=[3040, side, side]):
            raise ValueError("Wrong native target resource")
    for key in ("image", "mask"):
        if c["arrays"]["global_"+key] != dict(gc["cache"]["arrays"][key], shape=[3380, 512, 512]):
            raise ValueError("Wrong canonical target resource")
    expected = sorted(set(range(3380))-{row["cache_index"] for row in c["records"]})
    if c["unpopulated_global_indices"] != expected or len(expected) != 340:
        raise ValueError("Unexpected non-training cache slots")
    references = [gc["cache"]["metadata"], nc["shared"]["manifest"], nc["shared"]["records"]]
    if c["historical_bindings"] != references:
        raise ValueError("Historical binding records differ")
    for reference in references:
        e.checked_resource(r.WORKSPACE, reference)
    return c


def read_sample(row, c, resource_root):
    if row["split"] not in ("train", "val"):
        raise ValueError("No TEST input is accepted")
    ipath = e.checked_resource(resource_root, row["raw"]["image"])
    mpath = e.checked_resource(resource_root, row["raw"]["mask"])
    with Image.open(fs(ipath)) as handle:
        image = np.array(handle.convert("L"), dtype=np.uint8)
    with Image.open(fs(mpath)) as handle:
        mask = np.array(handle)  # Preserve palette indices, not palette colours.
    native, low_side = c["geometry"]["native"], c["geometry"]["global"]
    if (image.shape != (native, native) or mask.shape != image.shape
            or mask.dtype != np.uint8 or mask.min() < 0 or mask.max() > 7):
        raise ValueError("Expected the recorded native grey image and eight-class palette mask")
    low = np.array(Image.fromarray(image).resize((low_side, low_side), Image.Resampling.BILINEAR))
    small_mask = np.array(Image.fromarray(mask).resize((low_side, low_side), Image.Resampling.NEAREST))
    if native != 2*low_side or not np.array_equal(small_mask, mask[1::2, 1::2]):
        raise ValueError("Canonical nearest-neighbour pixel centres changed")
    return {"native_image": image, "native_mask": mask, "native_low": low,
            "global_image": low, "global_mask": small_mask}


def row_index(row, role):
    return row["cache_index"] if role.startswith("global_") else row["shared_index"]


def check_row(row, values):
    for role in ROLES:
        a = values[role]
        if a.dtype != np.uint8 or hashlib.sha256(a.tobytes()).hexdigest() != row["pixels"][role]:
            raise ValueError(f"Decoded pixel/cache mismatch: {row['id']} {role}")
    if np.bincount(values["global_mask"].ravel(), minlength=8).tolist() != row["counts"]:
        raise ValueError("Canonical class counts differ")


def destination(output, resource_root, c):
    resources = dict(c["arrays"])
    resources.update({str(i)+role: item for i, row in enumerate(c["records"]) for role, item in row["raw"].items()})
    return t.destination(output, resource_root, {"entries": {"main": {"resources": resources}}})


def identity(c):
    return {"version": "main-raw-cache-v1",
        "catalog_sha256": hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "implementation_sha256": r.sha256(__file__), "python": sys.version,
        "numpy": np.__version__, "pillow": Image.__version__}


def close_arrays(arrays):
    for array in arrays.values():
        if array.flags.writeable:
            array.flush()
        array._mmap.close()
    arrays.clear()


def verify_arrays(c, paths):
    arrays = {}
    try:
        for role, resource in c["arrays"].items():
            path = paths[role]
            if path.stat().st_size != resource["bytes"]:
                raise ValueError("Wrong prepared array length")
            if "sha256" in resource and r.sha256(path) != resource["sha256"]:
                raise ValueError(f"Native pixel bytes differ: {role}")
            arrays[role] = np.memmap(fs(path), mode="r", dtype=np.uint8, shape=tuple(resource["shape"]))
        for row in c["records"]:
            check_row(row, {role: np.asarray(arrays[role][row_index(row, role)]) for role in ROLES})
        for role in ("global_image", "global_mask"):
            for index in c["unpopulated_global_indices"]:
                if np.any(arrays[role][index]):
                    raise ValueError("Non-training Global slot is not an empty placeholder")
    finally:
        close_arrays(arrays)


def copy_bindings(c, output):
    # Exact legacy identity records let the unchanged TRAIN/VAL readers bind the
    # regenerated pixels. They are references, not a new GPU/cache experiment.
    save = binary_writer(output, lambda handle, content: handle.write(content))
    for item in c["historical_bindings"]:
        source = e.checked_resource(r.WORKSPACE, item)
        target = r.local_path(output, item["path"])
        fs(target.parent).mkdir(parents=True, exist_ok=True)
        if target.exists():
            e.checked_resource(output, item)
        else:
            save(target, fs(source).read_bytes())


def prepare(c, resource_root, output, resume=False, reader=read_sample):
    if any(row["split"] not in ("train", "val") for row in c["records"]):
        raise ValueError("Only TRAIN/VAL pixel resources may be prepared")
    root = Path(resource_root).resolve()
    out = destination(output, root, c)
    g.stop_check(root, root/"实验", out, out.parent)
    ident = identity(c)
    if out.exists():
        if not resume:
            raise FileExistsError("Choose a fresh output or explicitly resume the same preparation")
        meta = r.read_json(out/"preparation.json")
        if meta["identity"] != ident:
            raise ValueError("Cannot resume different sources, geometry or environment")
    else:
        if resume:
            raise FileNotFoundError("No preparation to resume")
        out.mkdir(parents=True, exist_ok=False)
        meta = {"identity": ident, "status": "preparing", "completed_rows": 0, "created": t.utc(),
            "source_contents_sha256": r.sha256(r.PACKAGE/"contents.json"), "test_pixels_read": False,
            "research_training_started": False, "checkpoint_loaded": False,
            "historical_bindings_role": "Exact reference records only; preparation.json describes this new run",
            "unpopulated_global_policy": c["unpopulated_global_policy"]}
    targets = {key: r.local_path(out, value["path"]) for key, value in c["arrays"].items()}
    partial = {key: path.with_name(path.name+".partial") for key, path in targets.items()}
    write = t.atomic_writer(out)
    stops = (root, root/"实验", out, out.parent)
    with t.output_lock(out):
        if (out/"preparation.json").exists():
            meta = r.read_json(out/"preparation.json")
            if meta["identity"] != ident:
                raise ValueError("Preparation changed before lock acquisition")
        g.stop_check(*stops)
        if meta["status"] == "completed":
            verify_arrays(c, targets)
            for item in c["historical_bindings"]:
                e.checked_resource(out, item)
            return r.read_json(out/"report.json")
        if not 0 <= meta["completed_rows"] <= len(c["records"]):
            raise ValueError("Invalid completed row count")
        meta.update(status="preparing", updated=t.utc())
        write(out/"preparation.json", meta)
        arrays = {}
        try:
            required = sum(x["bytes"] for key, x in c["arrays"].items() if not partial[key].exists() and not targets[key].exists())
            if shutil.disk_usage(out).free < required+512*2**20:
                raise OSError("Insufficient space for the new pixel cache")
            for key, item in c["arrays"].items():
                target, temp = targets[key], partial[key]
                if target.exists():
                    if meta["completed_rows"] != len(c["records"]) or temp.exists():
                        raise ValueError("Unexpected existing final array")
                    path, mode = target, "r"
                elif temp.exists():
                    path, mode = temp, "r+"
                else:
                    if meta["completed_rows"]:
                        raise ValueError("A partial array is missing")
                    fs(temp.parent).mkdir(parents=True, exist_ok=True)
                    path, mode = temp, "w+"
                if mode != "w+" and path.stat().st_size != item["bytes"]:
                    raise ValueError("Truncated pixel cache")
                arrays[key] = np.memmap(fs(path), mode=mode, dtype=np.uint8, shape=tuple(item["shape"]))
                if mode == "w+":
                    arrays[key][:] = 0  # Only placeholders, not loaded TEST data.
            for row in c["records"][:meta["completed_rows"]]:
                g.stop_check(*stops)
                check_row(row, {role: np.asarray(arrays[role][row_index(row, role)]) for role in ROLES})
            for i in range(meta["completed_rows"], len(c["records"])):
                g.stop_check(*stops)
                row = c["records"][i]
                values = reader(row, c, root)
                check_row(row, values)
                for role in ROLES:
                    arrays[role][row_index(row, role)] = values[role]
                if (i+1) % 25 == 0 or i+1 == len(c["records"]):
                    for array in arrays.values():
                        array.flush()
                    meta.update(completed_rows=i+1, updated=t.utc())
                    write(out/"preparation.json", meta)
                    if (i+1) % 100 == 0 or i+1 == len(c["records"]):
                        print(f"MAIN_RAW_CACHE {i+1}/{len(c['records'])}", flush=True)
            close_arrays(arrays)
            verify_arrays(c, {key: targets[key] if targets[key].exists() else partial[key] for key in ROLES})
            for role in ROLES:
                g.stop_check(*stops)
                if partial[role].exists():
                    if targets[role].exists():
                        raise FileExistsError(targets[role])
                    fs(partial[role]).rename(fs(targets[role]))
            copy_bindings(c, out)
            report = {"status": "PASS", "images": len(c["records"]), "images_by_split": c["images_by_split"],
                "native_files_byte_identical": 3, "all_five_pixel_rows_exact": True,
                "global_unpopulated_rows_zero": len(c["unpopulated_global_indices"]),
                "canonical_class_counts_exact": True, "gpu_initialized": torch.cuda.is_initialized(),
                "checkpoint_loaded": False, "research_training_started": False, "test_pixels_read": False,
                "paper_metrics_changed": False, "source_contents_sha256": r.sha256(r.PACKAGE/"contents.json"),
                "prepared_files": {key: {"path": item["path"], "sha256": r.sha256(targets[key])}
                                   for key, item in c["arrays"].items()},
                "scope": "TRAIN/VAL pixels only; historical binding records are unchanged references; no logits or weights generated"}
            write(out/"report.json", report)
            meta.update(status="completed", completed_rows=len(c["records"]), completed=t.utc())
        except BaseException as exc:
            meta.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            close_arrays(arrays)
            meta["updated"] = t.utc()
            write(out/"preparation.json", meta)
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--resource-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    report = prepare(load_catalog(), args.resource_root, args.output, args.resume)
    print(json.dumps({k: report[k] for k in ("status", "images", "native_files_byte_identical", "all_five_pixel_rows_exact")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
