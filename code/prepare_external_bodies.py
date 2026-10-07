"""Prepare TRAIN/VAL oil-body descriptors from a selected frozen upstream.

Original descriptor, inference and dense/sparse bookkeeping definitions are
executed unchanged. No optimizer or TEST loader is provided by this command.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import os
import sys
import time
import traceback
import types

import numpy as np
from PIL import Image
import torch

import reproduce as r
import external_runtime as e
import train_external as t
import train_external_body as b

sys.dont_write_bytecode = True
CONSTANTS = ("dataset", "seed", "descriptor_support", "config", "initial_head",
             "training_class_counts", "training_weights", "validation_mask")


def selection_contract(meta, selected, case, entries):
    """Selection is the first maximum of all 21 VAL epochs, including epoch 0."""
    if (meta["status"] != "completed" or meta["dataset"] != case["dataset"]
            or meta["arm"] != "native_detail" or meta["last_epoch"] != 20):
        raise ValueError("A completed, matching native-detail upstream is required")
    history = meta["history"]
    if [x["epoch"] for x in history] != list(range(21)):
        raise ValueError("Complete epoch 0..20 validation history is required")
    scores = [x["validation"]["metrics"]["canonical512"]["eightclass"]["mIoU"] for x in history]
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite validation score")
    chosen = int(np.argmax(scores))
    if chosen != meta["best_epoch"] or chosen != selected["selected_epoch"]:
        raise ValueError("Upstream was not selected by the first validation maximum")
    if (selected["split"] != "val" or selected["label_space"] != "fine8"
            or selected["external_test_pixels_read"] or not selected["saved_pixel_replay_exact"]
            or selected["metrics"] != history[chosen]["validation"]["metrics"]
            or selected["frozen_sha256"] != meta["frozen_sha256"]):
        raise ValueError("Selected validation metadata differs from the completed run")
    records = selected["saved_predictions"]
    if selected["images"] != len(entries) or len(records) != len(entries):
        raise ValueError("Incomplete selected validation images")
    for index, (record, entry) in enumerate(zip(records, entries)):
        if record["index"] != index or record["label_id"] != entry["label_id"]:
            raise ValueError("Selected validation image order differs")
    return chosen


def selected_upstream(case, catalog, resource_root, upstream_run=None):
    if upstream_run is None:
        checkpoint = e.checked_resource(resource_root, case["selected_upstream"])
        folder = checkpoint.parent.parent
        expected = case["selected_upstream"]
        identity = None
        kind = "archived-selected"
    else:
        upstream_run = Path(upstream_run).resolve()
        portable = r.read_json(upstream_run / "portable_run.json")
        identity = portable["identity"]
        expected_identity = t.run_identity(catalog, case["seed"], case["dataset"])
        # A moved run may have a different environment now, but not different
        # source, training protocol, seed, data catalog or initialization.
        keys = tuple(k for k in expected_identity if k != "environment")
        if (portable["status"] != "completed" or portable.get("test_pixels_read", True)
                or any(identity.get(k) != expected_identity[k] for k in keys)):
            raise ValueError("Not a completed matching portable upstream run")
        folder = r.local_path(upstream_run, f"outputs/{case['dataset']}_seed{case['seed']}/native_detail")
        kind = "new-portable-selected"
    meta = r.read_json(folder / "run_manifest.json")
    selected_path = folder / "selected_validation.json"
    if r.sha256(selected_path) != meta["selected_sha256"]:
        raise ValueError("Selected validation file changed")
    selected = r.read_json(selected_path)
    epoch = selection_contract(meta, selected, case, catalog["entries"][case["dataset"] + "_val"]["records"])
    if identity is not None:
        if meta["identity"] != identity or meta["best_checkpoint"] != selected["checkpoint"]:
            raise ValueError("Selected checkpoint is not from this portable run")
        expected = {**selected["checkpoint"], "path": "epochs/" + selected["checkpoint"]["file"]}
        checkpoint = e.checked_resource(folder, expected)
    if selected["checkpoint"]["sha256"] != expected["sha256"] or expected["epoch"] != epoch:
        raise ValueError("Wrong selected upstream checkpoint")
    predictions = r.local_path(folder, selected["predictions_directory"])
    for row in selected["saved_predictions"]:
        r.local_path(predictions, row["file"])
    return {"folder": str(folder), "checkpoint": str(checkpoint), "sha256": expected["sha256"],
            "epoch": epoch, "frozen_sha256": meta["frozen_sha256"], "kind": kind,
            "selected_validation_sha256": meta["selected_sha256"], "run_identity": identity}


def independent_output(output, *inputs):
    output = Path(output).resolve()
    for value in inputs:
        if value is None:
            continue
        root = Path(value).resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("New output overlaps an input or an upstream run")
    return output


def build_engine(case, catalog, resource_root, cache_root, output, initial, upstream):
    original = t.build_engine(catalog, case["seed"], initial, resource_root, output, cache_root=cache_root)
    c, _ = b.mechanics(case)
    core, _ = r.pipeline_modules()
    c.HERE, c.read, c.sha, c.utc = Path(output), r.read_json, r.sha256, t.utc
    c.write = t.atomic_writer(output)
    c.save_npz = b.binary_writer(output, lambda handle, value: np.savez_compressed(handle, **value))
    c.stop_check = lambda: t.stop_check(output, resource_root, cache_root, Path(upstream["folder"]))
    c.u.legacy = original.adapter
    c.assign_bodies = core.assign_bodies
    def cached(dataset, split):
        data = t.Cached(dataset, split, catalog, cache_root, original.c.d4)
        data.images, data.masks = data.image, data.mask
        return data
    c.CachedFine = cached
    def load_checkpoint(item, identity):
        path = Path(item["checkpoint"])
        if r.sha256(path) != item["sha256"]:
            raise ValueError("Upstream weights changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved["epoch"] != item["epoch"] or saved["dataset"] != case["dataset"]
                or saved["arm"] != "native_detail" or saved["frozen_sha256"] != item["frozen_sha256"]
                or original.c.state_hash(original.adapter.frozen_from_state(saved["model"])) != item["frozen_sha256"]
                or (item["run_identity"] is not None and saved["identity"] != item["run_identity"])):
            raise ValueError("Checkpoint role, selected epoch or frozen initialization differs")
        if not all(torch.isfinite(x).all() for x in saved["model"].values()):
            raise ValueError("Non-finite upstream weights")
        return saved
    c.load_upstream_checkpoint = load_checkpoint
    if case["descriptor_support"] == "probability_argmax":
        c.describe = core.describe
        source = "external_body_extension20_20261004/prepare_bodies.py"
    elif case["descriptor_support"] == "logit_argmax":
        c.describe = r.module("portable_body_logit_representation", "external_fine8_multiseed_body_recovery_20261005/representation_authoritative.py").describe
        source = "external_fine8_multiseed_body_recovery_20261005/prepare_bodies.py"
    else:
        raise ValueError("Unknown descriptor support convention")
    spatial = e.definitions("external_body_adaptation_20261003/spatial.py",
        ("nearest512_ids", "targets", "verify_dense", "assert_fixed_support", "recount"),
        {"np": np, "Image": Image, "c": c})
    preparation = e.definitions(source,
        ("digest_identity", "cache_folder", "validate_shard", "prepare", "verify_cache"),
        {"np": np, "torch": torch, "Path": Path, "time": time, "hashlib": hashlib, "json": json, "c": c, "s": spatial})
    return types.SimpleNamespace(c=c, preparation=preparation, spatial=spatial)


def export_case(case, name, folder, output, identity, smoke):
    meta = r.read_json(folder / "manifest.json")
    result = {key: copy.deepcopy(case[key]) for key in CONSTANTS}
    result.update(attempt="new-descriptor-preparation", template_case=name,
                  cache_folder=folder.relative_to(output).as_posix(),
                  n=meta["n"], images_by_split=meta["images_by_split"], bodies_by_split=meta["bodies_by_split"])
    result["resources"] = {}
    for key in ("manifest", "records", "matrices", *b.LAYOUT):
        path = folder / (key + (".npy" if key in b.LAYOUT else ".json"))
        result["resources"][key] = {"path": path.relative_to(output).as_posix(), "sha256": r.sha256(path)}
    result["selected_upstream"] = identity["selected_upstream"][case["dataset"]]
    result["preparation_identity"] = identity
    payload = {"version": "prepared-external-bodies-v1", "trainable": not smoke,
               "test_pixels_read": False, "case": result}
    t.atomic_writer(output)(output / "body_case.json", payload)
    return payload


def run(args):
    if not args.trusted_checkpoints:
        raise ValueError("Explicit trust of local pickle checkpoints is required")
    bodies, catalog = b.load_catalog(), t.load_catalog()
    case = b.choose(bodies, args.case)
    resource_root = Path(args.resource_root).resolve()
    cache_root = Path(args.cache_root or resource_root).resolve()
    output = b.output_path(args.output, resource_root, bodies)
    t.destination(output, cache_root, catalog)
    independent_output(output, args.upstream_run)
    t.stop_check(output, resource_root, cache_root)
    upstream = selected_upstream(case, catalog, resource_root, args.upstream_run)
    independent_output(output, upstream["folder"])
    identity = {"entry_point": "portable-external-body-preparation-v1", "template_case": args.case,
        "dataset": case["dataset"], "seed": case["seed"], "config": case["config"],
        "descriptor_support": case["descriptor_support"], "smoke": args.smoke,
        "selected_upstream": {case["dataset"]: upstream},
        "package_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                        "cuda": torch.version.cuda}}
    t.start_output(output, identity, args.resume)
    with t.output_lock(output):
        path = output / "portable_run.json"
        if path.exists():
            meta = r.read_json(path)
            if meta["identity"] != identity or not args.resume:
                raise ValueError("Another preparation changed this output")
        else:
            meta = {"identity": identity, "created_utc": t.utc(), "status": "prepared"}
        write = t.atomic_writer(output)
        write(path, meta)
        try:
            initial = t.verify_resources(catalog, resource_root, case["dataset"], case["seed"], cache_root=cache_root)
            engine = build_engine(case, catalog, resource_root, cache_root, output, initial, upstream)
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Original upstream descriptor preparation requires CUDA/BF16")
            torch.set_num_threads(4)
            torch.cuda.set_per_process_memory_fraction(.35)
            meta.update(status="running", pid=os.getpid(), started_utc=t.utc())
            write(path, meta)
            prepared = engine.preparation.prepare(case["dataset"], identity, smoke=args.smoke)
            folder = engine.preparation.cache_folder(case["dataset"], args.smoke)
            export_case(case, args.case, folder, output, identity, args.smoke)
            meta.update(status="completed", completed_utc=t.utc(), images_by_split=prepared["images_by_split"],
                trainable=not args.smoke, training_started=False, test_pixels_read=False)
        except BaseException as error:
            meta.update(status="paused" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error=repr(error), traceback=traceback.format_exc(), stopped_utc=t.utc())
            raise
        finally:
            write(path, meta)
    return {"status": meta["status"], "output": str(output), "trainable": not args.smoke,
            "images_by_split": meta["images_by_split"], "training_started": False, "test_pixels_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Recorded dataset/seed/support-protocol template")
    parser.add_argument("--upstream-run", type=Path, help="Completed new portable upstream run; omit to reconstruct the archived selected case")
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Only first/last TRAIN/VAL images; output cannot train a head")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trusted-checkpoints", action="store_true")
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
