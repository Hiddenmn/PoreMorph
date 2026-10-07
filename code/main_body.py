"""Portable main separate-head mechanics; original model/loss/evaluator reused."""
from __future__ import annotations

import hashlib
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import reproduce as r
import external_runtime as e
import main_native as n
import train_main_global as g
import train_external as t
from train_external_body import LAYOUT

model_ops = e.definitions("body_interface_decoder_20260930/model.py",
    ("ContactBlock", "BodyInterfaceDecoder", "augment", "body_loss"),
    {"torch": torch, "nn": nn, "F": F, "POINTS": 16, "TYPES": 3, "FEATURES": 31, "SUMMARY": 13})
aggregate = e.definitions("body_interface_decoder_20260930/representation.py", ("aggregate",), {"np": np}).aggregate
ops = e.definitions("body_interface_decoder_20260930/run_pair.py",
    ("state_digest", "CachedBodies", "evaluate"), {"torch": torch, "np": np,
        "hashlib": hashlib, "s": types.SimpleNamespace(read=r.read_json, metrics=n.ops.metrics),
        "LAYOUT": LAYOUT, "aggregate": aggregate})
random_ops = e.definitions("body_decoder_replication_20260930/replicate_body.py",
    ("order_for", "stream"), {"np": np, "torch": torch})


def load_catalog():
    r.verify_contents()
    c = r.read_json(r.PACKAGE / "main_body_training.json")
    e.verify_sources(c)
    nc = n.load_catalog()
    checkpoints = r.read_json(r.PACKAGE / "main_checkpoints.json")
    if (c["version"] != "main-body-training-v1" or c["test_loader"] or c["test_pixels_read"]
            or not c["upstream_frozen"] or not c["fresh_runs_only"]
            or set(c["cases"]) != {"42", "2024", "2025"}
            or c["records"] != [{k: x[k] for k in ("sample", "slice_id", "split")} for x in nc["records"]]):
        raise ValueError("Wrong main body training catalog")
    for seed, case in c["cases"].items():
        required = {"seed": int(seed), "epochs": 12, "batch_size": 512, "lr": .001,
            "weight_decay": .0001, "grad_clip": 1., "precision": "FP32", "parameters": 25573,
            "class_weights": nc["cases"][seed]["config"]["class_weights"][3:],
            "epoch_zero_eligible": False, "scheduler": "CosineAnnealingLR(T_max=12)",
            "order_seed": "seed * 1000 + epoch", "augmentation_seed": "seed * 10000 + epoch"}
        if (case["seed"] != int(seed) or case["mode"] != "separate" or case["config"] != required
                or case["upstream"] != nc["cases"][seed]["native_checkpoint"]
                or case["archived_checkpoint"] != checkpoints[seed]["separate_checkpoint"]):
            raise ValueError("Main body architecture, seed or protocol changed")
    return c


def build_model(case):
    # Both the seed42 constructor and initialise_pair() construct this head first.
    # Keeping this fork CPU-only makes identity checks independent of CUDA state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case["seed"])
        model = model_ops.BodyInterfaceDecoder("separate")
    if (ops.state_digest(model.state_dict()) != case["initial_sha256"]
            or sum(x.numel() for x in model.parameters()) != 25573):
        raise ValueError("Main body initialization changed")
    return model


def validate_records(case, records, expected):
    if (len(records) != 3040 or [{k: x[k] for k in ("sample", "slice_id", "split")} for x in records] != expected
            or len({(x["sample"], x["slice_id"]) for x in records}) != 3040):
        raise ValueError("Main body image IDs or TRAIN/VAL order differ")
    cursor = 0
    for row, item in enumerate(records):
        if (item["row"] != row or item["split"] not in ("train", "val")
                or item.get("shared_index", row) != row or item["bodies"] < 0
                or item["start"] != cursor or item["end"] != cursor + item["bodies"]):
            raise ValueError("Noncontiguous or misaligned body records")
        cursor = item["end"]
    if cursor != case["n"]:
        raise ValueError("Body total differs")
    for split in ("train", "val"):
        rows = [x for x in records if x["split"] == split]
        if (len(rows) != case["images_by_split"][split]
                or sum(x["bodies"] for x in rows) != case["bodies_by_split"][split]):
            raise ValueError("Body split population differs")


def bind(c, root, seed, prepared=None):
    root = Path(root).resolve()
    g.stop_check(root, root / "实验")
    if prepared is not None:
        from prepare_main_bodies import bind_prepared
        return bind_prepared(c, prepared, seed)
    case = c["cases"][str(seed)]
    for item in case["resources"].values():
        e.checked_resource(root, item)
    folder = r.local_path(root, case["cache_folder"])
    meta, rows = r.read_json(folder / "manifest.json"), r.read_json(folder / "records.json")
    if (meta["status"] != "completed" or meta["smoke"] or meta["test_pixels_read"]
            or not meta["frozen_validation_replayed"] or meta["completed_images"] != 3040
            or meta["selected_rows"] != list(range(3040)) or meta["n"] != case["n"]
            or meta["images_by_split"] != case["images_by_split"] or meta["bodies_by_split"] != case["bodies_by_split"]):
        raise ValueError("Incomplete or incompatible main body cache")
    upstream = meta["identity"] if seed == 42 else meta["identity"]["seeds"][str(seed)]
    if upstream["native_checkpoint"] != case["upstream"]["sha256"]:
        raise ValueError("Descriptor/native model mismatch")
    validate_records(case, rows, c["records"])
    if (meta["records_sha256"] != case["resources"]["records"]["sha256"]
            or meta["outside_sha256"] != case["resources"]["outside"]["sha256"]
            or any(meta["array_sha256"][k] != case["resources"][k]["sha256"] for k in LAYOUT)
            or r.read_json(folder / "frozen_validation.json") != case["frozen_validation"]):
        raise ValueError("Body cache metadata disagrees with resources")
    validate_arrays(folder, case, rows)
    return {"case": case, "folder": folder, "records": rows}


def validate_arrays(folder, case, rows):
    arrays = {}
    for key, (dtype, tail) in LAYOUT.items():
        a = np.load(folder / (key + ".npy"), mmap_mode="r", allow_pickle=False)
        if a.dtype != dtype or a.shape != (case["n"], *tail) or a.flags.writeable:
            raise ValueError(f"Wrong body array: {key}")
        arrays[key] = a
    # Pixel-count targets stay outside forward(). Check each split's accounting.
    outside = r.read_json(folder / "outside.json")
    for split in ("train", "val"):
        group = [x for x in rows if x["split"] == split]
        for grid, side in (("native", 1024), ("canonical512", 512)):
            matrix = np.asarray(outside[split][grid], dtype=np.int64)
            total = sum(int(arrays[grid + "_counts"][x["start"]:x["end"]].sum(dtype=np.uint64)) for x in group)
            if (matrix.shape != (8, 8) or (matrix < 0).any() or matrix[:, 3:].any()
                    or total + int(matrix.sum()) != len(group) * side * side):
                raise ValueError("Body and fixed nonoil pixels do not cover the grid")


class Bodies(ops.CachedBodies):
    def __init__(self, folder, split, device="cpu"):
        if split not in ("train", "val"):
            raise ValueError("This trainer has no TEST loader")
        super().__init__(Path(folder), split)
        self.device, self.split = device, split

    def input_batch(self, index):
        return tuple(torch.from_numpy(np.array(self.arrays[k][index], copy=True)).to(self.device, dtype=dtype)
                     for k, dtype in (("points", torch.float32), ("valid", torch.bool), ("summary", torch.float32)))


def setup(seed):
    if not torch.cuda.is_available():
        raise RuntimeError("The unchanged main body trainer requires CUDA")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    g.base.set_seed(seed)


def train_epoch(model, data, optimizer, scheduler, weights, seed, epoch):
    if data.split != "train":
        raise ValueError("Only TRAIN bodies can update the model")
    order = random_ops.order_for(data.indices, seed, epoch)
    generator = random_ops.stream(seed, epoch, data.device)
    model.train()
    losses = []
    for start in range(0, len(order), 512):
        index = order[start:start + 512]
        points, valid, summary = data.input_batch(index)
        target = torch.from_numpy(np.array(data.arrays["native_counts"][index], copy=True)).to(data.device).float()
        optimizer.zero_grad(set_to_none=True)
        loss = model_ops.body_loss(model(model_ops.augment(points, generator), valid, summary), target, weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite main body loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("Empty TRAIN body cache")
    scheduler.step()
    return float(np.mean(losses))


def output_path(path, root, c, prepared=None):
    entries = {seed: {"resources": case["resources"]} for seed, case in c["cases"].items()}
    path = r.new_output(t.destination(path, root, {"entries": entries}))
    if prepared is not None:
        source = Path(prepared).resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Head output overlaps its prepared descriptor inputs")
    return path
