"""Portable fixed-protocol external upstream adaptation, separate from inference.

No experiment driver's top-level code is imported. Its original computational
definitions are bound to explicit read-only resources and a new output folder.
The body-recognition head is NOT trained by this entry point.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
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
from torch import nn
from torch.nn import functional as F

import reproduce as r
import external_runtime as e

sys.dont_write_bytecode = True
ARMS = ("global", "native_detail")
EXPECTED = {"bentheimer_train": (630, 980, 980), "bentheimer_val": (90, 980, 980),
            "wf1_train": (1600, 1033, 1030), "wf1_val": (300, 1033, 1030)}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_catalog():
    r.verify_contents()
    catalog = r.read_json(r.PACKAGE / "external_training.json")
    e.verify_sources(catalog)
    if catalog["version"] != "external-upstream-training-v1" or set(catalog["entries"]) != set(EXPECTED):
        raise ValueError("Unexpected training catalog")
    if catalog["arms"] != list(ARMS) or catalog["seeds"] != [42, 2024, 2025] or catalog["test_loader"]:
        raise ValueError("Unexpected training scope")
    for key, shape in EXPECTED.items():
        entry = catalog["entries"][key]
        if tuple(entry["shape"]) != shape or entry["count"] != shape[0] or len(entry["records"]) != shape[0]:
            raise ValueError(f"Invalid cache catalog: {key}")
        if key != f"{entry['dataset']}_{entry['split']}" or entry["split"] not in ("train", "val"):
            raise ValueError("Only TRAIN and VAL may be loaded")
        ids = [row["label_id"] for row in entry["records"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sample IDs")
    for name in ("bentheimer", "wf1"):
        a, b = (catalog["entries"][f"{name}_{split}"]["records"] for split in ("train", "val"))
        if {row["label_id"] for row in a} & {row["label_id"] for row in b}:
            raise ValueError("Overlapping training and validation sample IDs")
        if {row["raw"]["sha256"] for row in a} & {row["raw"]["sha256"] for row in b}:
            raise ValueError("Overlapping training and validation image hashes")
    return catalog


def config_for(catalog, seed):
    if seed not in catalog["seeds"]:
        raise ValueError("Use one of the three recorded seeds")
    return {**catalog["config_without_seed"], "seed": seed}


def stop_check(*roots):
    for root in roots:
        for name in ("STOP.txt", "STOP_AFTER_EPOCH.txt"):
            if (Path(root) / name).exists():
                raise InterruptedError(f"Stop marker retained: {Path(root) / name}")


def destination(path, resource_root, catalog):
    path, root = Path(path).resolve(), Path(resource_root).resolve()
    if os.name == "nt" and len(str(path)) > 140:
        raise ValueError("Choose a shorter output root (at most 140 characters on Windows) for the archived checkpoint writers")
    if path == root or root.is_relative_to(path):
        raise ValueError("The output cannot be the resource root or its ancestor")
    # Protect every existing top-level input tree, not just the selected files.
    protected = [r.PACKAGE.resolve()]
    for entry in catalog["entries"].values():
        for resource in entry["resources"].values():
            protected.append(r.local_path(root, Path(resource["path"]).parts[0]))
    for bundle in r.read_json(r.PACKAGE / "main_checkpoints.json").values():
        for resource in bundle.values():
            protected.append(r.local_path(root, Path(resource["path"]).parts[0]))
    if any(path == item or path.is_relative_to(item) or item.is_relative_to(path) for item in protected):
        raise ValueError("The output overlaps package or input trees")
    return path


def atomic_writer(root):
    root = Path(root).resolve()

    def write(path, value):
        path = Path(path).resolve()
        path.relative_to(root)
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        def fs(p):
            return Path("\\\\?\\" + str(p)) if os.name == "nt" and not str(p).startswith("\\\\?\\") else p
        with fs(tmp).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(fs(tmp), fs(path))
    return write


@contextlib.contextmanager
def output_lock(folder):
    """OS-held lock is released on crashes; a stale text file is not a lock."""
    with (Path(folder) / "run.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def mechanics(catalog, seed):
    config = config_for(catalog, seed)
    c = types.SimpleNamespace(SEED=seed, CONFIG=config, ARMS=ARMS,
                              GRIDS=("canonical512", "native"), RULES=("eightclass",))
    common = e.definitions("external_clean_prior_adaptation_20261001/common.py",
        ("state_hash", "epoch_plan", "d4", "window_size"),
        {"np": np, "hashlib": hashlib, "SEED": seed, "CONFIG": config})
    c.__dict__.update(common.__dict__)
    c.lr_at = e.definitions("external_fine8_multiseed_20261004/run_upstream.py",
                            ("upstream_lr",), {"np": np}).upstream_lr
    c.weighted_dice_from_logp = e.definitions("architecture_upgrade_20260928/mixed_label_loss.py",
        ("weighted_dice_from_logp",), {"torch": torch, "F": F}).weighted_dice_from_logp
    c.data = e.definitions("external_frozen_phase_20261001/external_inputs.py",
                          ("nearest512",), {"np": np, "Image": Image})
    b = e.definitions("external_eightclass_adaptation_20261003/bridge.py",
        ("fine_weights", "fine_loss", "histogram", "metrics"), {"np": np, "torch": torch, "F": F, "c": c})
    b.c = c
    return c, b


class Cached:
    """Only previously hash-checked TRAIN/VAL memmaps; no test path exists."""
    def __init__(self, dataset, split, catalog, resource_root, d4):
        if split not in ("train", "val"):
            raise ValueError("Only train/val splits are accepted")
        self.entry = catalog["entries"][f"{dataset}_{split}"]
        self.dataset, self.split, self.d4 = dataset, split, d4
        self.description = catalog["descriptions"][dataset]
        arrays = {key: np.load(r.local_path(resource_root, entry["path"]), mmap_mode="r", allow_pickle=False)
                  for key, entry in self.entry["resources"].items()}
        self.image, self.low, self.mask = (arrays[key] for key in ("image", "low", "mask"))
        shape = tuple(self.entry["shape"])
        if self.image.shape != shape or self.mask.shape != shape or self.low.shape != (shape[0], 512, 512):
            raise ValueError("Cache dimensions differ from the recorded protocol")
        if self.image.dtype != np.float32 or self.low.dtype != np.float32 or self.mask.dtype != np.uint8:
            raise ValueError("Cache dtype differs from the recorded protocol")

    def __len__(self):
        return self.entry["count"]

    def get(self, index, transform=None):
        arrays = tuple(x[index] for x in (self.image, self.low, self.mask))
        return tuple(x.copy() if transform is None else self.d4(x, transform) for x in arrays)


def verify_resources(catalog, resource_root, dataset, seed, include_caches=True, cache_root=None):
    if dataset not in ("bentheimer", "wf1"):
        raise ValueError("Unknown dataset")
    config_for(catalog, seed)
    main = r.read_json(r.PACKAGE / "main_checkpoints.json")[str(seed)]
    initial = {key: e.checked_resource(resource_root, main[key])
               for key in ("tokenizer", "global_checkpoint", "native_checkpoint")}
    if include_caches:
        for split in ("train", "val"):
            for entry in catalog["entries"][f"{dataset}_{split}"]["resources"].values():
                stop_check(resource_root, cache_root or resource_root)
                e.checked_resource(cache_root or resource_root, entry)
    return initial


def build_engine(catalog, seed, initial, resource_root, output, cache_root=None):
    c, b = mechanics(catalog, seed)
    core, _ = r.pipeline_modules()
    prior = r.module("pack_training_prior", "external_clean_prior_adaptation_20261001/prior_schema.py")
    if prior.PROMPTS != catalog["descriptions"]:
        raise ValueError("Prior descriptions changed")
    c.HERE, c.utc, c.sha, c.read = Path(output), utc, r.sha256, r.read_json
    c.write = atomic_writer(output)
    c.stop_check = lambda: stop_check(output, resource_root, cache_root or resource_root)
    c.order, c.NativeNoContext = core.order, core.NativeNoContext
    c.locked = types.SimpleNamespace(paths=lambda requested: initial if requested == seed else None)
    c.project_native = e.definitions("external_frozen_phase_20261001/phase_predictor.py",
                                    ("project_native",), {"torch": torch, "F": F}).project_native
    adapter = e.definitions("external_clean_prior_adaptation_20261001/model_adapter.py",
        ("AdaptationModel", "frozen_from_state", "to_device"), {"torch": torch, "nn": nn, "F": F, "c": c,
            "extend_tokenizer": prior.extend_tokenizer, "AppendOnlyEmbedding": prior.AppendOnlyEmbedding})
    namespace = {"torch": torch, "np": np, "gc": gc, "time": time, "os": os, "hashlib": hashlib, "json": json,
        "c": c, "AdaptationModel": adapter.AdaptationModel, "to_device": adapter.to_device,
        "frozen_from_state": adapter.frozen_from_state, "phase_loss": b.fine_loss,
        "Cached": lambda dataset, split: Cached(dataset, split, catalog, cache_root or resource_root, c.d4),
        "arm_folder": lambda dataset, arm: c.HERE / "outputs" / f"{dataset}_seed{seed}" / arm}
    names = ("selection", "tokens", "close_model", "train_epoch", "optimizer_for", "cpu_tree", "save_checkpoint",
             "register_checkpoint", "run_epoch", "finish_arm")
    original = e.definitions("external_clean_prior_adaptation_20261001/run_adaptation.py", names, namespace)
    b.legacy = types.SimpleNamespace(tokens=original.tokens, to_device=adapter.to_device)
    b.stop_check, b.sha = c.stop_check, r.sha256
    evaluation = e.definitions("external_eightclass_adaptation_20261003/run_eightclass.py",
                              ("counts", "primary", "evaluate"), {"b": b, "torch": torch, "np": np, "time": time})
    namespace.update(primary=evaluation.primary, evaluate=evaluation.evaluate)
    return types.SimpleNamespace(c=c, b=b, adapter=adapter, original=original, evaluate=evaluation.evaluate)


def run_identity(catalog, seed, dataset):
    return {"entry_point": "portable-external-upstream-v1", "dataset": dataset, "seed": seed,
        "config": config_for(catalog, seed), "catalog_sha256": r.sha256(r.PACKAGE / "external_training.json"),
        "main_catalog_sha256": r.sha256(r.PACKAGE / "main_checkpoints.json"),
        "package_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                        "pillow": Image.__version__, "cuda": torch.version.cuda}}


def start_output(output, identity, resume):
    if output.exists():
        if not resume:
            raise FileExistsError("Use a new output folder, or explicitly resume this same portable run")
        metadata = r.read_json(output / "portable_run.json")
        if metadata["identity"] != identity:
            raise ValueError("Resume requires the same source, inputs, protocol and environment")
    else:
        if resume:
            raise FileNotFoundError("Cannot resume a missing portable run")
        output.mkdir(parents=True, exist_ok=False)


def require_cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This unchanged training protocol requires CUDA/BF16")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.45)


def run(args):
    if not args.trusted_checkpoints:
        raise ValueError("Explicit trust of local pickle checkpoints is required")
    catalog = load_catalog()
    output = destination(args.output, args.resource_root, catalog)
    cache_root = getattr(args, "cache_root", None)
    if cache_root is not None:
        output = destination(output, cache_root, catalog)
    stop_check(output, args.resource_root, cache_root or args.resource_root)
    identity = run_identity(catalog, args.seed, args.dataset)
    start_output(output, identity, args.resume)
    with output_lock(output):
        writer = atomic_writer(output)
        meta_path = output / "portable_run.json"
        if meta_path.exists():
            meta = r.read_json(meta_path)
            if not args.resume or meta["identity"] != identity:
                raise ValueError("Another run created or changed this output")
        else:
            meta = {"identity": identity, "created_utc": utc(), "status": "prepared", "completed_arms": {}}
            writer(meta_path, meta)
        # All original input files are read-only, including cache arrays.
        initial = verify_resources(catalog, args.resource_root, args.dataset, args.seed, cache_root=cache_root)
        engine = build_engine(catalog, args.seed, initial, args.resource_root, output, cache_root=cache_root)
        if meta["status"] == "completed":
            for arm in ARMS:
                folder = output / "outputs" / f"{args.dataset}_seed{args.seed}" / arm
                arm_meta = r.read_json(folder / "run_manifest.json")
                if arm_meta["identity"] != identity or arm_meta["status"] != "completed":
                    raise ValueError("Completed arm metadata is missing or different")
                if r.sha256(folder / "selected_validation.json") != arm_meta["selected_sha256"]:
                    raise ValueError("Completed validation output changed")
                entry = arm_meta["best_checkpoint"]
                if r.sha256(folder / "epochs" / entry["file"]) != entry["sha256"]:
                    raise ValueError("Completed selected checkpoint changed")
            return {"status": "ALREADY_COMPLETE", "output": str(output)}
        require_cuda()
        meta.update(status="running", pid=os.getpid(), started_utc=utc())
        writer(meta_path, meta)
        try:
            for epoch in range(engine.c.CONFIG["epochs"] + 1):
                for arm in ARMS:
                    engine.c.stop_check()
                    engine.original.run_epoch(args.dataset, arm, epoch, identity, args.resume)
            for arm in ARMS:
                engine.c.stop_check()
                result = engine.original.finish_arm(args.dataset, arm, identity)
                meta["completed_arms"][arm] = {"selected_epoch": result["selected_epoch"],
                                               "metrics": result["metrics"]}
            meta.update(status="completed", completed_utc=utc(), test_pixels_read=False,
                        body_head_adaptation_complete=False)
        except BaseException as error:
            meta.update(status="paused" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        error=repr(error), traceback=traceback.format_exc(), stopped_utc=utc())
            raise
        finally:
            writer(meta_path, meta)
    return {"status": meta["status"], "scope": "upstream adaptation only", "output": str(output)}


def check(args):
    catalog = load_catalog()
    cache_root = getattr(args, "cache_root", None) or args.resource_root
    stop_check(args.resource_root, cache_root)
    verify_resources(catalog, args.resource_root, args.dataset, args.seed, cache_root=cache_root)
    c, b = mechanics(catalog, args.seed)
    for split in ("train", "val"):
        data = Cached(args.dataset, split, catalog, cache_root, c.d4)
        if split == "train":
            np.testing.assert_array_equal(b.fine_weights(data.entry["class_pixel_counts"]),
                                          np.asarray(data.entry["class_weights"], dtype=np.float32))
        elif data.entry["class_weights"] is not None:
            raise ValueError("Validation labels must not supply training weights")
    return {"status": "PASS", "dataset": args.dataset, "seed": args.seed,
            "cache_files_verified": 6, "gpu_initialized": torch.cuda.is_initialized(),
            "training_started": False, "test_pixels_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", choices=("bentheimer", "wf1"), required=True)
        command.add_argument("--seed", type=int, choices=(42, 2024, 2025), required=True)
        command.add_argument("--resource-root", type=Path, required=True)
        command.add_argument("--cache-root", type=Path, help="Separate byte-identical raw-to-cache output root")
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--trusted-checkpoints", action="store_true")
        else:
            command.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "check" and args.report is not None:
        r.new_output(args.report)
    result = check(args) if args.command == "check" else run(args)
    if args.command == "check" and args.report is not None:
        r.write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
