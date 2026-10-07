"""Frozen main native network -> original oil-body descriptors, TRAIN/VAL only.

Accept an archived selection or a completed new portable native run. Preparation
can resume explicitly from identical inputs; it never starts optimizer updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import ndimage
import torch

import reproduce as r
import external_runtime as e
import main_native as n
import main_body as b
import train_main_native as nt
import train_main_global as g
import train_external as t
from train_external_body import binary_writer

representation = e.definitions("body_interface_decoder_20260930/representation.py",
    ("sample_group", "describe", "targets"),
    {"np": np, "ndimage": ndimage, "POINTS": 16, "TYPES": 3, "FEATURES": 31, "SUMMARY": 13})
GRIDS = ("native", "canonical512")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def implementation():
    names = ("reproduce.py", "external_runtime.py", "main_native.py", "train_main_native.py",
             "main_body.py", "prepare_main_bodies.py")
    files = {name: r.sha256(r.PACKAGE / name) for name in names}
    for rel in b.load_catalog()["sources"]:
        files["实验/" + rel] = r.sha256(r.EXPERIMENTS / rel)
    return digest_json(files)


def validate_metrics(value, images=676):
    if set(value) != set(GRIDS):
        raise ValueError("Both validation grids are required")
    for grid, side in (("native", 1024), ("canonical512", 512)):
        h = np.asarray(value[grid]["confusion_matrix"])
        if (h.shape != (8, 8) or not np.issubdtype(h.dtype, np.integer) or (h < 0).any()
                or int(h.sum()) != images * side**2 or n.ops.metrics(h) != value[grid]):
            raise ValueError("Invalid native validation metrics/counts")


def selection_contract(meta, selected, history, bc, nc, seed, binding):
    ident = meta["identity"]
    compatible = [*bc["compatible_native_contents_sha256"], r.sha256(r.PACKAGE / "contents.json")]
    if (meta["status"] != "completed" or not meta["selected_replay_exact"] or meta["test_pixels_read"]
            or meta["completed_epochs"] != 12 or len(history) != 12
            or ident["stage"] != "main native-detail" or ident["seed"] != seed
            or ident["config"] != nc["cases"][str(seed)]["config"] or ident["selection"] != nc["selection"]
            or ident["native_catalog_sha256"] != r.sha256(r.PACKAGE / "main_native_training.json")
            or ident["source_contents_sha256"] not in compatible or not ident["fresh_run"]
            or ident["selected_global"] != binding["selected_global"]
            or ident["coarse_sha256"] != binding["coarse_sha256"]
            or ident["full_initial_sha256"] != nc["cases"][str(seed)]["full_initial_sha256"]):
        raise ValueError("A completed native run with exactly matching coarse inputs is required")
    chosen = nt.choose(history)
    if (selected["epoch"] != chosen["epoch"] or meta["best_epoch"] != chosen["epoch"]
            or selected["validation"] != chosen["validation"]
            or meta["best_native_mIoU"] != chosen["validation"]["native"]["mIoU"]
            or selected["selection"] != nc["selection"] or selected["test_pixels_read"]
            or selected["checkpoint"] != "best.pth"):
        raise ValueError("Native selection must be the first maximum VAL checkpoint")
    validate_metrics(selected["validation"])
    return chosen["epoch"]


def selected_native(bc, nc, root, seed, binding, native_run=None):
    case = nc["cases"][str(seed)]
    if native_run is None:
        if (binding["coarse_sha256"] != case["coarse"]["sha256"]
                or binding["selected_global"]["checkpoint_sha256"] != case["global_checkpoint"]["sha256"]):
            raise ValueError("Archived native weights require their original coarse logits")
        path = e.checked_resource(root, case["native_checkpoint"])
        info = {"kind": "archived-selected", "seed": seed,
            "checkpoint_sha256": case["native_checkpoint"]["sha256"],
            "epoch": case["archived_selected_epoch"], "validation": case["archived_validation"]}
        return path, info, None
    folder = Path(native_run).resolve()
    g.stop_check(folder)
    meta = r.read_json(folder / "run_manifest.json")
    selected = r.read_json(folder / "selected_validation.json")
    history = r.read_json(folder / "history.json")
    epoch = selection_contract(meta, selected, history, bc, nc, seed, binding)
    path = e.checked_resource(folder, {"path": "best.pth", "sha256": selected["sha256"]})
    info = {"kind": "new-portable-selected", "seed": seed, "checkpoint_sha256": selected["sha256"],
        "epoch": epoch, "validation": selected["validation"],
        "run_manifest_sha256": r.sha256(folder / "run_manifest.json"),
        "selected_validation_sha256": r.sha256(folder / "selected_validation.json"),
        "source_contents_sha256": meta["identity"]["source_contents_sha256"]}
    return path, info, {"identity": meta["identity"], "history": history, "selected": selected}


def load_native(nc, seed, path, info, portable):
    if r.sha256(path) != info["checkpoint_sha256"]:
        raise ValueError("Selected native checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["epoch"] != info["epoch"] or saved["validation"] != info["validation"]:
        raise ValueError("Native tensor payload and selection disagree")
    if portable is not None:
        epoch = info["epoch"]
        if (not saved["completed_epoch"] or saved["identity"] != portable["identity"]
                or saved["history"] != portable["history"][:epoch] or saved["best_epoch"] != epoch):
            raise ValueError("Incomplete or mismatched portable native tensor payload")
    model = n.build_model(nc["cases"][str(seed)])
    model.load_state_dict(saved["model"], strict=True)
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError("Nonfinite native weights")
    return model.cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)


def row_values(image, probabilities, fine, truth, row, shared_index):
    data, labels, prediction = representation.describe(image, probabilities, fine)
    counts, outside = representation.targets(labels, prediction, truth)
    values = {**data, "row": np.array(row, dtype=np.int64), "shared_index": np.array(shared_index, dtype=np.int64)}
    for grid, p, y in (("native", prediction, truth),
                       ("canonical512", prediction[1::2, 1::2], truth[1::2, 1::2])):
        values[grid + "_counts"] = counts[grid]
        values[grid + "_outside"] = outside[grid]
        values[grid + "_frozen"] = np.bincount((y.astype(np.int64)*8+p).ravel(), minlength=64).reshape(8, 8)
    return values


class NativeRows:
    def __init__(self, binding, model):
        self.binding, self.model, self.box = binding, model, {}
        self.arrays = {k: np.memmap(binding["files"][k], mode="r", dtype=np.uint8,
                        shape=(binding["shared_n"], 1024, 1024)) for k in ("image", "mask")}
        self.coarse = np.memmap(binding["files"]["coarse"], mode="r", dtype=np.uint16,
                                shape=(binding["n"], 8, 512, 512))
        self.hook = model.up_fine.register_forward_hook(lambda m, a, out: self.box.update(feature=out))

    @torch.inference_mode()
    def __call__(self, row):
        shared = self.binding["records"][row]["shared_index"]
        im = np.array(self.arrays["image"][shared], copy=True).astype(np.float32)/255
        image = torch.from_numpy(im).cuda()[None, None]
        low = torch.from_numpy(np.array(self.coarse[row], copy=True)).view(torch.bfloat16).cuda()[None]
        up = n.arch.sample_native_grid(low, torch.zeros((1, 2), device="cuda"), 1024)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(image, up, None)
        probabilities = logits.float().softmax(1)[0].cpu().numpy()
        fine = self.box.pop("feature").float()[0].cpu().numpy()
        # Label targets are read only after all forward inputs have been computed.
        truth = np.array(self.arrays["mask"][shared], copy=True)
        return row_values(im, probabilities, fine, truth, row, shared)

    def close(self):
        self.hook.remove()
        self.arrays.clear(); self.coarse = None; self.box.clear()


def selected_rows(records, smoke):
    if any(x["split"] not in ("train", "val") for x in records):
        raise ValueError("No TEST row can enter descriptor preparation")
    if not smoke:
        return list(range(len(records)))
    # Fixed by image identity/order, not prediction quality; one per stage/split.
    seen, rows = set(), []
    for i, record in enumerate(records):
        group = (record["sample"], record["split"])
        if group not in seen:
            seen.add(group); rows.append(i)
    return rows


def cache_identity(bc, nc, binding, info, smoke):
    return {"version": "prepared-main-bodies-v1", "seed": binding["seed"], "smoke": smoke,
        "native": info, "selected_global": binding["selected_global"], "coarse_sha256": binding["coarse_sha256"],
        "shared_pixels": {k: nc["shared"][k]["sha256"] for k in ("image", "mask")},
        "records_sha256": digest_json(bc["records"]), "implementation": implementation(),
        "native_catalog_sha256": r.sha256(r.PACKAGE / "main_native_training.json"),
        "body_protocol_sha256": digest_json(bc["cases"][str(binding["seed"])]["config"]),
        "support": "probability argmax; unfiltered 4-connected native oil support",
        "batch_size": 1, "test_pixels_read": False}


def verify_prepared_files(folder, meta):
    for key in ("records", "outside", "frozen_validation"):
        e.checked_resource(folder, {"path": key + ".json", "sha256": meta[key + "_sha256"]})
    for key in b.LAYOUT:
        e.checked_resource(folder, {"path": key + ".npy", "sha256": meta["array_sha256"][key]})


def prepare_cache(output, records, chosen, identity, producer, expected_validation, stop_roots, resume=False):
    output = Path(output)
    write = t.atomic_writer(output)
    save = binary_writer(output, lambda handle, values: np.savez(handle, **values))
    if output.exists():
        if not resume:
            raise FileExistsError("Use a fresh preparation or explicit identical --resume")
        meta = r.read_json(output / "manifest.json")
        if meta["identity"] != identity or meta["selected_rows"] != chosen:
            raise ValueError("Cannot resume changed descriptors, model, source or inputs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "shards").mkdir()
        meta = {"identity": identity, "status": "preparing", "seed": identity["seed"],
            "smoke": identity["smoke"], "selected_rows": chosen, "shards": {}, "completed_images": 0,
            "test_pixels_read": False, "research_training_started": False, "trainable": False,
            "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"), "created": t.utc()}
    with t.output_lock(output):
        # Another writer may have finished between the first read and this lock.
        if (output / "manifest.json").exists():
            meta = r.read_json(output / "manifest.json")
            if meta["identity"] != identity or meta["selected_rows"] != chosen:
                raise ValueError("Preparation identity changed before acquiring the lock")
        g.stop_check(*stop_roots)
        if meta["status"] == "completed":
            verify_prepared_files(output, meta)
            return meta
        meta.update(status="preparing", trainable=False)
        write(output / "manifest.json", meta)
        started = time.monotonic()
        try:
            entries = []
            for ordinal, row in enumerate(chosen):
                g.stop_check(*stop_roots)
                path = output / "shards" / f"{row:04d}.npz"
                old_hash = meta["shards"].get(str(row))
                if old_hash is not None:
                    if r.sha256(path) != old_hash:
                        raise ValueError("Committed descriptor shard changed")
                else:
                    values = producer(row)
                    if int(values["row"]) != row:
                        raise ValueError("Wrong prepared image row")
                    save(path, values)
                    meta["shards"][str(row)] = r.sha256(path)
                    meta.update(completed_images=ordinal+1, updated=t.utc())
                    write(output / "manifest.json", meta)
                with np.load(path, allow_pickle=False) as shard:
                    if int(shard["row"]) != row or int(shard["shared_index"]) != records[row]["shared_index"]:
                        raise ValueError("Prepared shard and pixel row differ")
                    count = len(shard["points"])
                    for key, (dtype, tail) in b.LAYOUT.items():
                        if shard[key].dtype != dtype or shard[key].shape != (count, *tail):
                            raise ValueError("Wrong shard array type or shape")
                entries.append({"row": row, "shared_index": records[row]["shared_index"],
                    **{k: records[row][k] for k in ("sample", "slice_id", "split")},
                    "bodies": count, "shard_sha256": meta["shards"][str(row)]})
                if (ordinal+1) % 100 == 0 or ordinal+1 == len(chosen):
                    print(f"MAIN_BODY_CACHE {ordinal+1}/{len(chosen)} elapsed={time.monotonic()-started:.1f}s", flush=True)
            g.stop_check(*stop_roots)
            count = sum(x["bodies"] for x in entries)
            arrays = {key: np.lib.format.open_memmap(output / (key + ".npy"), mode="w+", dtype=dtype,
                      shape=(count, *tail)) for key, (dtype, tail) in b.LAYOUT.items()}
            outside = {split: {grid: np.zeros((8, 8), np.int64) for grid in GRIDS} for split in ("train", "val")}
            frozen = {split: {grid: np.zeros((8, 8), np.int64) for grid in GRIDS} for split in ("train", "val")}
            cursor = 0
            try:
                for entry in entries:
                    g.stop_check(*stop_roots)
                    end = cursor + entry["bodies"]
                    entry.update(start=cursor, end=end)
                    path = output / "shards" / f"{entry['row']:04d}.npz"
                    if r.sha256(path) != entry["shard_sha256"]:
                        raise ValueError("Shard changed before merge")
                    with np.load(path, allow_pickle=False) as shard:
                        for key in arrays:
                            arrays[key][cursor:end] = shard[key]
                        for grid in GRIDS:
                            outside[entry["split"]][grid] += shard[grid + "_outside"]
                            frozen[entry["split"]][grid] += shard[grid + "_frozen"]
                    cursor = end
                for array in arrays.values():
                    array.flush()
                del array
            finally:
                arrays.clear()
            measured = {grid: n.ops.metrics(h) for grid, h in frozen["val"].items()}
            if not identity["smoke"] and measured != expected_validation:
                raise ValueError("Full frozen native validation changed; do not train a head")
            write(output / "records.json", entries)
            write(output / "outside.json", {split: {grid: h.tolist() for grid, h in grids.items()} for split, grids in outside.items()})
            write(output / "frozen_validation.json", measured)
            meta.update(status="completed", completed_images=len(chosen), n=count,
                trainable=not identity["smoke"], frozen_validation_replayed=not identity["smoke"],
                images_by_split={split: sum(x["split"] == split for x in entries) for split in ("train", "val")},
                bodies_by_split={split: sum(x["bodies"] for x in entries if x["split"] == split) for split in ("train", "val")},
                array_sha256={key: r.sha256(output / (key + ".npy")) for key in b.LAYOUT},
                **{key + "_sha256": r.sha256(output / (key + ".json")) for key in ("records", "outside", "frozen_validation")})
        except BaseException as exc:
            meta.update(status="stopped" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
                        trainable=False, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            meta["updated"] = t.utc()
            write(output / "manifest.json", meta)
    return meta


def prepared_contract(meta, c, seed):
    identity = meta["identity"]
    if (meta["status"] != "completed" or meta["smoke"] or not meta["trainable"]
            or not meta["frozen_validation_replayed"] or meta["test_pixels_read"] or meta["research_training_started"]
            or meta["seed"] != seed or meta["completed_images"] != 3040
            or meta["selected_rows"] != list(range(3040)) or meta["images_by_split"] != {"train": 2364, "val": 676}
            or identity["version"] != "prepared-main-bodies-v1" or identity["seed"] != seed or identity["smoke"]
            or identity["test_pixels_read"] or identity["batch_size"] != 1
            or identity["support"] != "probability argmax; unfiltered 4-connected native oil support"
            or identity["implementation"] != implementation() or identity["records_sha256"] != digest_json(c["records"])
            or identity["native_catalog_sha256"] != r.sha256(r.PACKAGE / "main_native_training.json")
            or identity["body_protocol_sha256"] != digest_json(c["cases"][str(seed)]["config"])):
        raise ValueError("Complete matching TRAIN/VAL body preparation required; smoke caches cannot train")
    info = identity["native"]
    nc = n.load_catalog()
    native_case = nc["cases"][str(seed)]
    if identity["shared_pixels"] != {k: nc["shared"][k]["sha256"] for k in ("image", "mask")}:
        raise ValueError("Prepared native image/mask resources differ")
    if info["seed"] != seed or not 1 <= info["epoch"] <= 12:
        raise ValueError("Prepared native model/seed mismatch")
    if info["kind"] == "archived-selected":
        if (info["checkpoint_sha256"] != c["cases"][str(seed)]["upstream"]["sha256"]
                or info["epoch"] != native_case["archived_selected_epoch"]
                or info["validation"] != native_case["archived_validation"]
                or identity["coarse_sha256"] != native_case["coarse"]["sha256"]
                or identity["selected_global"] != {"kind": "archived-selected", "seed": seed,
                    "checkpoint_sha256": native_case["global_checkpoint"]["sha256"]}):
            raise ValueError("Wrong archived native model")
    elif info["kind"] == "new-portable-selected":
        allowed = [*c["compatible_native_contents_sha256"], r.sha256(r.PACKAGE / "contents.json")]
        if (any(len(info[k]) != 64 for k in ("checkpoint_sha256", "run_manifest_sha256", "selected_validation_sha256"))
                or info["source_contents_sha256"] not in allowed or len(identity["coarse_sha256"]) != 64
                or identity["selected_global"]["seed"] != seed):
            raise ValueError("Incomplete portable native model identity")
    else:
        raise ValueError("Unknown native model role")
    validate_metrics(info["validation"])


def bind_prepared(c, folder, seed):
    folder = Path(folder).resolve()
    g.stop_check(folder)
    meta = r.read_json(folder / "manifest.json")
    prepared_contract(meta, c, seed)
    verify_prepared_files(folder, meta)
    template = c["cases"][str(seed)]
    case = {key: template[key] for key in ("seed", "mode", "initial_sha256", "config")}
    case.update({key: meta[key] for key in ("n", "images_by_split", "bodies_by_split")})
    resources = {"manifest": {"path": "manifest.json", "sha256": r.sha256(folder / "manifest.json")}}
    for key, name in (("records", "records"), ("outside", "outside"), ("frozen", "frozen_validation")):
        resources[key] = {"path": name + ".json", "sha256": meta[name + "_sha256"]}
    resources.update({key: {"path": key + ".npy", "sha256": meta["array_sha256"][key]} for key in b.LAYOUT})
    case.update(resources=resources, cache_folder=".", upstream=meta["identity"]["native"],
                preparation_identity=meta["identity"])
    rows = r.read_json(folder / "records.json")
    b.validate_records(case, rows, c["records"])
    b.validate_arrays(folder, case, rows)
    if r.read_json(folder / "frozen_validation.json") != meta["identity"]["native"]["validation"]:
        raise ValueError("Prepared validation differs from selected upstream")
    return {"case": case, "folder": folder, "records": rows}


def run(args):
    if not args.trusted_checkpoints:
        raise ValueError("Explicit trust of local pickle checkpoints required")
    bc, nc = b.load_catalog(), n.load_catalog()
    root = args.resource_root.resolve()
    out = n.output_path(args.output, root, nc, args.prepared_cache, allow_existing=args.resume)
    if args.native_run:
        run_folder = args.native_run.resolve()
        if out.is_relative_to(run_folder) or run_folder.is_relative_to(out):
            raise ValueError("Preparation output overlaps native-run inputs")
    stops = (root, root / "实验", out, out.parent, *([args.native_run] if args.native_run else []),
             *([args.prepared_cache] if args.prepared_cache else []))
    g.stop_check(*stops)
    binding = n.bind(nc, root, args.seed, args.prepared_cache)
    path, info, portable = selected_native(bc, nc, root, args.seed, binding, args.native_run)
    chosen = selected_rows(binding["records"], args.smoke)
    ident = cache_identity(bc, nc, binding, info, args.smoke)
    ancestor = out.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    # Reserve two uncompressed descriptor copies (shards + merged arrays), with
    # a 2x margin against that seed's original body count and an extra 2 GiB.
    estimate = 2 * 2 * bc["cases"][str(args.seed)]["n"] * 3144 * len(chosen) / 3040
    if shutil.disk_usage(ancestor).free < estimate + 2 * 2**30:
        raise RuntimeError("Insufficient space for separate descriptor shards and merged arrays")
    n.setup(args.seed)
    model = load_native(nc, args.seed, path, info, portable)
    producer = NativeRows(binding, model)
    try:
        return prepare_cache(out, binding["records"], chosen, ident, producer, info["validation"], stops, args.resume)
    finally:
        producer.close()
        del producer, model
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resource-root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(42, 2024, 2025), required=True)
    p.add_argument("--native-run", type=Path)
    p.add_argument("--prepared-cache", type=Path, help="Matching newly prepared Global coarse cache, if used")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--trusted-checkpoints", action="store_true")
    args = p.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "seed", "n", "smoke", "trainable", "completed_images")}))


if __name__ == "__main__":
    main()
