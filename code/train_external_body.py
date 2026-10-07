"""Portable separate oil-body-head adaptation from recorded TRAIN/VAL caches.

The original FP32 trainer, loss, augmentation, validation, selection and resume
definitions are reused unchanged. This does not build descriptors, unfreeze the
upstream model, load TEST pixels or replace any historical run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import uuid

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import reproduce as r
import external_runtime as e
import train_external as t

sys.dont_write_bytecode = True
GRIDS = ("canonical512", "native")
LAYOUT = {"points": (np.float16, (48, 31)), "valid": (np.uint8, (48,)),
          "summary": (np.float32, (13,)), "native_counts": (np.uint32, (8,)),
          "canonical512_counts": (np.uint32, (8,))}


def load_catalog():
    r.verify_contents()
    catalog = r.read_json(r.PACKAGE / "external_body_training.json")
    e.verify_sources(catalog)
    external = r.read_json(r.PACKAGE / "external_checkpoints.json")
    if (catalog["version"] != "external-body-training-v1" or catalog["test_loader"]
            or not catalog["upstream_frozen"] or set(catalog["cases"]) != set(external["cases"])):
        raise ValueError("Unexpected body-training catalog")
    required = {"epochs": 12, "batch_size": 512, "lr": .001, "weight_decay": .0001,
                "grad_clip": 1., "precision": "FP32", "primary_grid": "canonical512"}
    for name, case in catalog["cases"].items():
        source = external["cases"][name]
        if any(case[key] != source[key] for key in ("seed", "dataset", "attempt", "descriptor_support")):
            raise ValueError("Body case differs from frozen external case")
        if case["config"]["seed"] != case["seed"] or any(case["config"][k] != v for k, v in required.items()):
            raise ValueError("Body training protocol changed")
        if case["selected_upstream"] != source["weights"]["native_detail"]:
            raise ValueError("Body cache is linked to a different upstream")
        if case["archived_head"] != source["weights"]["body"]:
            raise ValueError("Wrong archived head")
    return catalog


def choose(catalog, name):
    if name not in catalog["cases"]:
        raise ValueError(f"Choose an explicit recorded case: {list(catalog['cases'])}")
    return catalog["cases"][name]


def output_path(path, resource_root, catalog):
    # Use the same protected-tree and Windows-length contract as upstream.
    entries = {}
    for name, case in catalog["cases"].items():
        entries[name] = {"resources": {**case["resources"], "mask": case["validation_mask"],
                                      "initial": case["initial_head"]}}
    return t.destination(path, resource_root, {"entries": entries})


def cache_records(case, resource_root):
    folder = r.local_path(resource_root, case["cache_folder"])
    records = r.read_json(e.checked_resource(resource_root, case["resources"]["records"]))
    cursor = 0
    for split in ("train", "val"):
        group = [item for item in records if item["split"] == split]
        if len(group) != case["images_by_split"][split] or [x["row"] for x in group] != list(range(len(group))):
            raise ValueError("Unexpected TRAIN/VAL body records")
        if len({x["label_id"] for x in group}) != len(group):
            raise ValueError("Duplicate body image IDs")
        if sum(x["bodies"] for x in group) != case["bodies_by_split"][split]:
            raise ValueError("Body count changed")
    if {x["label_id"] for x in records if x["split"] == "train"} & {x["label_id"] for x in records if x["split"] == "val"}:
        raise ValueError("Overlapping TRAIN/VAL IDs")
    for item in records:
        if item["split"] not in ("train", "val") or item["bodies"] < 0:
            raise ValueError("Only TRAIN/VAL caches are accepted")
        if item["start"] != cursor or item["end"] != cursor + item["bodies"]:
            raise ValueError("Non-contiguous body indices")
        cursor = item["end"]
        r.local_path(folder / "shards", item["file"])
    if cursor != case["n"] or cursor <= 0:
        raise ValueError("Incomplete body cache")
    return folder, records


def verify_resources(case, resource_root, descriptor_root=None, cache_root=None):
    descriptor_root = descriptor_root or resource_root
    cache_root = cache_root or resource_root
    t.stop_check(resource_root, descriptor_root, cache_root)
    for item in case["resources"].values():
        t.stop_check(resource_root, descriptor_root, cache_root)
        e.checked_resource(descriptor_root, item)
    initial = e.checked_resource(resource_root, case["initial_head"])
    mask = e.checked_resource(cache_root, case["validation_mask"])
    folder, records = cache_records(case, descriptor_root)
    meta = r.read_json(folder / "manifest.json")
    if (meta["status"] != "completed" or meta["smoke"] or meta["test_pixels_read"]
            or meta["identity"]["config"] != case["config"] or meta["n"] != case["n"]):
        raise ValueError("Cache metadata differs from recorded case")
    if "preparation_identity" in case and meta["identity"] != case["preparation_identity"]:
        raise ValueError("Descriptor cache belongs to a different upstream preparation")
    for name, (dtype, tail) in LAYOUT.items():
        array = np.load(r.local_path(descriptor_root, case["resources"][name]["path"]), mmap_mode="r", allow_pickle=False)
        if array.dtype != dtype or array.shape != (case["n"], *tail) or array.flags.writeable:
            raise ValueError(f"Body cache shape, type or mutability differs: {name}")
    # The final selected-checkpoint verification replays only VAL shards.
    val_shards = 0
    for item in records:
        if item["split"] == "val":
            t.stop_check(resource_root)
            if r.sha256(r.local_path(folder / "shards", item["file"])) != item["sha256"]:
                raise ValueError("Validation body shard changed")
            val_shards += 1
    truth = np.load(mask, mmap_mode="r", allow_pickle=False)
    shape = (case["images_by_split"]["val"], *((980, 980) if case["dataset"] == "bentheimer" else (1033, 1030)))
    if truth.shape != shape or truth.dtype != np.uint8 or truth.flags.writeable:
        raise ValueError("Native validation mask grid changed")
    return initial, {"arrays_verified": len(LAYOUT), "validation_shards_verified": val_shards}


def mechanics(case):
    c = types.SimpleNamespace(CONFIG=case["config"], GRIDS=GRIDS, LAYOUT=LAYOUT,
                              DATASETS=(case["dataset"],), ARMS=("separate",))
    c.u = e.definitions("external_eightclass_adaptation_20261003/bridge.py",
                        ("metrics", "histogram", "fine_weights"), {"np": np})
    weights = c.u.fine_weights(case["training_class_counts"])
    np.testing.assert_array_equal(weights, np.asarray(case["training_weights"], np.float32))
    def body_weights(dataset):
        if dataset != case["dataset"]:
            raise ValueError("Wrong dataset")
        return weights[3:].copy()
    c.weights = body_weights
    c.__dict__.update(e.definitions("body_interface_decoder_20260930/model.py",
        ("augment", "body_loss"), {"torch": torch}).__dict__)
    c.aggregate = e.definitions("body_interface_decoder_20260930/representation.py",
                                ("aggregate",), {"np": np}).aggregate
    c.__dict__.update(e.definitions("external_fine8_multiseed_body_recovery_20261005/run_heads.py",
        ("order_for", "stream"), {"c": c, "np": np, "torch": torch}).__dict__)
    c.__dict__.update(e.definitions("external_body_adaptation_20261003/body_common.py",
        ("primary", "selected_epoch"), {"primary": lambda result: result["canonical512"]["mIoU"]}).__dict__)
    spatial = e.definitions("external_body_adaptation_20261003/spatial.py",
        ("nearest512_ids", "recount", "assert_fixed_support"), {"np": np, "Image": Image})
    return c, spatial


def binary_writer(root, save):
    root = Path(root).resolve()
    def write(path, value):
        path = Path(path).resolve()
        path.relative_to(root)
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with tmp.open("xb") as handle:
            save(handle, value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    return write


def build_engine(case, resource_root, output, descriptor_root=None, cache_root=None):
    descriptor_root = descriptor_root or resource_root
    cache_root = cache_root or resource_root
    c, spatial = mechanics(case)
    core, _ = r.pipeline_modules()
    c.HERE, c.read, c.sha, c.utc = Path(output), r.read_json, r.sha256, t.utc
    c.write = t.atomic_writer(output)
    c.stop_check = lambda: t.stop_check(output, resource_root, descriptor_root, cache_root)
    c.save_checkpoint = binary_writer(output, lambda handle, value: torch.save(value, handle))
    c.save_npz = binary_writer(output, lambda handle, value: np.savez_compressed(handle, **value))
    c.assign_bodies = core.assign_bodies
    def source_head(arm):
        if arm != "separate":
            raise ValueError("This entry point trains the complete separate head only")
        path = e.checked_resource(resource_root, case["initial_head"])
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = core.BodyInterfaceDecoder("separate")
        model.load_state_dict(saved["model"], strict=True)
        if sum(p.numel() for p in model.parameters()) != 25573 or not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("Invalid source head")
        return model
    c.source_head = source_head
    def cache_folder(dataset):
        if dataset != case["dataset"]:
            raise ValueError("Wrong body-cache dataset")
        return r.local_path(descriptor_root, case["cache_folder"])
    def validation_masks(dataset, split):
        if dataset != case["dataset"] or split != "val":
            raise ValueError("Dense replay accepts this case's VAL masks only")
        return types.SimpleNamespace(masks=np.load(r.local_path(cache_root, case["validation_mask"]["path"]), mmap_mode="r", allow_pickle=False))
    c.CachedFine = validation_masks
    preparation = e.definitions("external_body_adaptation_20261003/prepare_bodies.py",
        ("Bodies", "evaluate"), {"c": c, "s": spatial, "Path": Path, "np": np, "torch": torch})
    preparation.cache_folder = cache_folder
    runner = e.definitions("external_body_adaptation_20261003/run_body.py",
        ("train_epoch", "completed_checkpoint", "verify_assignments", "train_arm"),
        {"c": c, "p": preparation, "s": spatial, "np": np, "torch": torch, "time": time, "os": os})
    return types.SimpleNamespace(c=c, spatial=spatial, data=preparation, runner=runner, core=core)


def run_identity(case, case_name):
    identity = {"entry_point": "portable-external-body-v1", "case": case_name,
            "config": case["config"], "dataset": case["dataset"], "seed": case["seed"],
            "catalog_sha256": r.sha256(r.PACKAGE / "external_body_training.json"),
            "package_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
            "environment": {"python": sys.version, "torch": str(torch.__version__),
                            "numpy": np.__version__, "cuda": torch.version.cuda}}
    if "preparation_identity" in case:
        identity["prepared_case_sha256"] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
        identity["selected_upstream"] = case["selected_upstream"]
    return identity


def prepared_case(folder, catalog):
    """Accept a complete newly generated cache, never a bounded smoke cache."""
    folder = Path(folder).resolve()
    payload = r.read_json(folder / "body_case.json")
    if (payload["version"] != "prepared-external-bodies-v1" or not payload["trainable"]
            or payload["test_pixels_read"]):
        raise ValueError("Only complete TRAIN/VAL descriptor preparations can train a head")
    case = payload["case"]
    template = choose(catalog, case["template_case"])
    constants = ("dataset", "seed", "descriptor_support", "config", "initial_head",
                 "training_class_counts", "training_weights", "validation_mask")
    if any(case[k] != template[k] for k in constants) or case["attempt"] != "new-descriptor-preparation":
        raise ValueError("Prepared case changed the matched body-training protocol")
    if any(k.startswith("archived_") for k in case):
        raise ValueError("A new preparation must not inherit archived head results")
    identity = case["preparation_identity"]
    if (identity["entry_point"] != "portable-external-body-preparation-v1" or identity["smoke"]
            or identity["template_case"] != case["template_case"] or identity["dataset"] != case["dataset"]
            or identity["seed"] != case["seed"] or identity["config"] != case["config"]
            or identity["descriptor_support"] != case["descriptor_support"]
            or identity["selected_upstream"][case["dataset"]] != case["selected_upstream"]
            or identity["package_contents_sha256"] != r.sha256(r.PACKAGE / "contents.json")):
        raise ValueError("Prepared cache identity differs from this source and case")
    meta = r.read_json(e.checked_resource(folder, case["resources"]["manifest"]))
    if (meta["status"] != "completed" or meta["smoke"] or meta["test_pixels_read"]
            or meta["identity"] != identity or not meta["dense_sparse_exact"]
            or not meta["selected_validation_pixel_replay_exact"]):
        raise ValueError("Descriptor preparation is incomplete or mismatched")
    for key in ("n", "images_by_split", "bodies_by_split"):
        if case[key] != meta[key]:
            raise ValueError("Prepared cache counts changed")
    for split in ("train", "val"):
        if case["images_by_split"][split] != t.EXPECTED[case["dataset"] + "_" + split][0]:
            raise ValueError("Prepared cache does not contain every TRAIN/VAL image")
    for key in LAYOUT:
        if case["resources"][key]["sha256"] != meta["array_sha256"][key]:
            raise ValueError("Prepared array belongs to a different preparation")
    for key in ("records", "matrices"):
        if case["resources"][key]["sha256"] != meta[key + "_sha256"]:
            raise ValueError("Prepared bookkeeping changed")
    _, records = cache_records(case, folder)
    upstream = t.load_catalog()
    for split in ("train", "val"):
        expected = upstream["entries"][case["dataset"] + "_" + split]["records"]
        if [x["label_id"] for x in records if x["split"] == split] != [x["label_id"] for x in expected]:
            raise ValueError("Prepared image IDs differ from the upstream TRAIN/VAL split")
    return case


def requested_case(args, catalog):
    prepared = getattr(args, "prepared_cache", None)
    if prepared is None:
        return choose(catalog, args.case), args.case, None
    case = prepared_case(prepared, catalog)
    return case, "prepared/" + case["template_case"], Path(prepared).resolve()


def requested_output(path, args, catalog, descriptor_root):
    output = output_path(path, args.resource_root, catalog)
    cache_root = getattr(args, "cache_root", None)
    if cache_root is not None:
        t.destination(output, cache_root, t.load_catalog())
    if descriptor_root is not None:
        root = Path(descriptor_root).resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Head-training output must be separate from descriptor inputs")
    return output


def check(args):
    catalog = load_catalog()
    case, name, descriptor_root = requested_case(args, catalog)
    _, checked = verify_resources(case, args.resource_root, descriptor_root, getattr(args, "cache_root", None))
    mechanics(case)
    return {"status": "PASS", "case": name, **checked,
            "gpu_initialized": torch.cuda.is_initialized(), "training_started": False,
            "test_pixels_read": False, "scope": "Read-only TRAIN/VAL body-cache and initialization check"}


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("The unchanged body-head trainer requires CUDA")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)


def run(args):
    if not args.trusted_checkpoints:
        raise ValueError("Explicit trust of local pickle checkpoints is required")
    catalog = load_catalog()
    case, name, descriptor_root = requested_case(args, catalog)
    cache_root = getattr(args, "cache_root", None)
    output = requested_output(args.output, args, catalog, descriptor_root)
    t.stop_check(output, args.resource_root)
    identity = run_identity(case, name)
    t.start_output(output, identity, args.resume)
    with t.output_lock(output):
        writer = t.atomic_writer(output)
        meta_path = output / "portable_run.json"
        if meta_path.exists():
            meta = r.read_json(meta_path)
            if not args.resume or meta["identity"] != identity:
                raise ValueError("Another run created or changed this output")
        else:
            meta = {"identity": identity, "status": "prepared", "created_utc": t.utc()}
            writer(meta_path, meta)
        folder = output / "outputs" / case["dataset"] / "separate"
        if meta["status"] == "completed":
            head = r.read_json(folder / "run_manifest.json")
            chosen = r.read_json(folder / "selected_validation.json")
            if (head["status"] != "completed" or head["identity"] != identity
                    or r.sha256(folder / "selected_validation.json") != head["selected_sha256"]
                    or r.sha256(folder / "epochs" / chosen["checkpoint"]["file"]) != chosen["checkpoint"]["sha256"]):
                raise ValueError("Completed output has changed")
            return {"status": "ALREADY_COMPLETE", "output": str(output)}
        try:
            verify_resources(case, args.resource_root, descriptor_root, cache_root)
            require_cuda()
            engine = build_engine(case, args.resource_root, output, descriptor_root, cache_root)
            meta.update(status="running", pid=os.getpid(), started_utc=t.utc())
            writer(meta_path, meta)
            selected = engine.runner.train_arm(case["dataset"], "separate", identity, args.resume)
            meta.update(status="completed", completed_utc=t.utc(), selected_epoch=selected["selected_epoch"],
                        test_pixels_read=False, upstream_updated=False)
        except BaseException as error:
            meta.update(status="paused" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error=repr(error), traceback=traceback.format_exc(), stopped_utc=t.utc())
            raise
        finally:
            writer(meta_path, meta)
    return {"status": meta["status"], "case": name, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        command = sub.add_parser(name)
        choice = command.add_mutually_exclusive_group(required=True)
        choice.add_argument("--case")
        choice.add_argument("--prepared-cache", type=Path, help="New completed descriptor-preparation folder")
        command.add_argument("--resource-root", type=Path, required=True)
        command.add_argument("--cache-root", type=Path, help="Separate upstream raw-to-cache root for VAL masks")
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--trusted-checkpoints", action="store_true")
        else:
            command.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "check" and args.report is not None:
        catalog = load_catalog()
        _, _, descriptor_root = requested_case(args, catalog)
        requested_output(args.report, args, catalog, descriptor_root)
        r.new_output(args.report)
    result = check(args) if args.command == "check" else run(args)
    if args.command == "check" and args.report is not None:
        r.write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
