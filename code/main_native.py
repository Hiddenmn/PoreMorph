"""Shared portable native-detail mechanics; original tensor operations retained."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler, DataLoader

import reproduce as r
import external_runtime as e
import train_main_global as g
import train_external as t

sys.dont_write_bytecode = True
arch = e.definitions("native_detail_decoder_20260930/architecture.py",
    ("PixelNorm", "Block", "sample_native_grid", "NativeDetailDecoder"), {"torch": torch, "nn": nn, "F": F})
model_ops = e.definitions("native_detail_context_control_20260930/model_no_context.py",
    ("tensor_digest", "NativeNoContext"), {"torch": torch, "F": F, "hashlib": hashlib,
                                          "NativeDetailDecoder": arch.NativeDetailDecoder})
ops = e.definitions("native_detail_decoder_20260930/run_study.py",
    ("metrics", "add_counts", "local_batch", "evaluate", "loss_function"),
    {"torch": torch, "np": np, "F": F, "p": types.SimpleNamespace(base=g.base),
     "sample_native_grid": arch.sample_native_grid})
item_ops = e.definitions("native_detail_replication_20260930/data.py", ("__getitem__",),
                         {"np": np, "torch": torch}, owner="ReplicationDataset")
sampler_ops = e.definitions("native_detail_replication_20260930/data.py", ("SeedSampler",),
                            {"torch": torch, "Sampler": Sampler})


def load_catalog():
    r.verify_contents()
    c = r.read_json(r.PACKAGE / "main_native_training.json")
    e.verify_sources(c)
    if (c["version"] != "main-native-training-v1" or c["test_loader"] or c["test_pixels_read"]
            or set(c["cases"]) != {"42", "2024", "2025"} or c["n"] != 3040
            or len(c["records"]) != 3040 or len({x["id"] for x in c["records"]}) != 3040
            or [x["shared_index"] for x in c["records"]] != list(range(3040))):
        raise ValueError("Invalid native TRAIN/VAL catalog")
    global_c = g.load_catalog()
    expected = global_c["records"]
    for a, b in zip(c["records"], expected):
        if any(a[key] != b[key] for key in ("id", "sample", "slice_id", "text", "split")):
            raise ValueError("Native row order or priors differ from Global TRAIN/VAL")
    for seed, case in c["cases"].items():
        required = {"seed": int(seed), "frozen_global_seed": int(seed), "epochs": 12,
            "crop": 384, "halo": 32, "batch_size": 4, "lr": .001, "weight_decay": .0001,
            "train_count": 2364, "val_count": 676, "trainable_parameters": 118744,
            "class_weights": global_c["seeds"]["42"]["weights"], "test_pixels_read": False}
        if any(case["config"][key] != value for key, value in required.items()):
            raise ValueError("Native configuration differs from the recorded protocol")
    return c


def setup(seed):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Original native computation requires CUDA/BF16")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    # Original native drivers ran in a fresh process, without get_device()'s
    # Global-training TF32/benchmark changes. Keep that separate numerical path.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    g.base.set_seed(seed)


def build_model(case):
    g.base.set_seed(case["seed"])
    model = model_ops.NativeNoContext()
    if (model.full_initial_sha256 != case["full_initial_sha256"]
            or sum(x.numel() for x in model.parameters()) != 118744):
        raise ValueError("Native shared initialization changed")
    return model


def prepared_contract(meta, c, seed):
    if (meta["status"] != "completed" or meta["smoke"] or meta["n"] != 3040
            or meta["seed"] != seed or meta["test_pixels_read"] or meta["research_training_started"]
            or meta["records"] != c["records"] or meta["batch_size"] != 4
            or meta["native_catalog_sha256"] != r.sha256(r.PACKAGE / "main_native_training.json")
            or meta["completed_records"] != 3040):
        raise ValueError("A completed, full matching TRAIN/VAL native cache is required")
    teacher = meta["selected_global"]
    if teacher["seed"] != seed:
        raise ValueError("Native cache and Global seeds differ")
    if teacher["kind"] == "archived-selected":
        if teacher["checkpoint_sha256"] != c["cases"][str(seed)]["global_checkpoint"]["sha256"]:
            raise ValueError("Wrong archived Global teacher")
    elif teacher["kind"] == "new-portable-selected":
        allowed = [*c["compatible_global_contents_sha256"], r.sha256(r.PACKAGE / "contents.json")]
        if (teacher["source_contents_sha256"] not in allowed or not 1 <= teacher["epoch"] <= 60
                or any(len(teacher[k]) != 64 for k in ("checkpoint_sha256", "run_manifest_sha256", "selected_validation_sha256"))):
            raise ValueError("Unrecognized portable Global teacher")
    else:
        raise ValueError("Unknown Global teacher role")


def bind(c, root, seed, prepared=None, verify=True):
    if str(seed) not in c["cases"]:
        raise ValueError("Unknown native seed")
    root = Path(root).resolve()
    g.stop_check(root, root / "实验")
    if verify:
        for resource in c["shared"].values():
            g.stop_check(root, root / "实验")
            e.checked_resource(root, resource)
    files = {key: str(r.local_path(root, c["shared"][key]["path"])) for key in ("image", "mask", "low")}
    case = c["cases"][str(seed)]
    if prepared is None:
        if verify:
            for key in ("coarse", "coarse_manifest", "coarse_records"):
                e.checked_resource(root, case[key])
        files["coarse"] = str(r.local_path(root, case["coarse"]["path"]))
        teacher = {"kind": "archived-selected", "seed": seed,
                   "checkpoint_sha256": case["global_checkpoint"]["sha256"]}
        coarse_hash = case["coarse"]["sha256"]
    else:
        prepared = Path(prepared).resolve()
        g.stop_check(prepared)
        meta = r.read_json(prepared / "native_cache.json")
        prepared_contract(meta, c, seed)
        file = e.checked_resource(prepared, meta["coarse"]) if verify else r.local_path(prepared, meta["coarse"]["path"])
        files["coarse"], coarse_hash, teacher = str(file), meta["coarse"]["sha256"], meta["selected_global"]
    for key, side in (("image", 1024), ("mask", 1024), ("low", 512)):
        if Path(files[key]).stat().st_size != 3040 * side ** 2:
            raise ValueError("Wrong native pixel-cache size")
    if Path(files["coarse"]).stat().st_size != 3040 * 8 * 512 ** 2 * 2:
        raise ValueError("Wrong native logit-cache size")
    return {"n": 3040, "shared_n": 3040, "records": c["records"], "files": files,
            "seed": seed, "selected_global": teacher, "coarse_sha256": coarse_hash}


class NativeDataset(Dataset):
    def __init__(self, binding, split, training):
        if split not in ("train", "val") or any(x["split"] not in ("train", "val") for x in binding["records"]):
            raise ValueError("No TEST route")
        if training and split != "train":
            raise ValueError("Only TRAIN may supply optimizer batches")
        self.binding, self.records = binding, binding["records"]
        self.indices = [i for i, x in enumerate(self.records) if x["split"] == split]
        self.training, self.seed, self.arrays = training, binding["seed"], None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = None
        return state

    def __getitem__(self, requested):
        if self.arrays is None:
            self.arrays = {k: np.memmap(self.binding["files"][k], mode="r", dtype=np.uint8,
                                      shape=(self.binding["shared_n"], side, side))
                           for k, side in (("image", 1024), ("mask", 1024), ("low", 512))}
            self.arrays["coarse"] = np.memmap(self.binding["files"]["coarse"], mode="r", dtype=np.uint16,
                shape=(self.binding["n"], 8, 512, 512))
        # The archived per-seed crop/augmentation expression is unchanged.
        # Its context is zero and ignored by NativeNoContext, for every seed.
        return item_ops.__getitem__(self, requested)


def loaders(binding):
    train = NativeDataset(binding, "train", True)
    sampler = sampler_ops.SeedSampler(train, seed=binding["seed"])
    train_loader = DataLoader(train, batch_size=4, sampler=sampler, num_workers=2,
                              persistent_workers=True, prefetch_factor=2, pin_memory=True)
    val_loader = DataLoader(NativeDataset(binding, "val", False), batch_size=1, num_workers=0)
    return train_loader, val_loader, sampler


def output_path(path, root, c, prepared=None, allow_existing=False):
    entries = {"shared": {"resources": c["shared"]}}
    for seed, case in c["cases"].items():
        entries[seed] = {"resources": {"coarse": case["coarse"]}}
    path = t.destination(path, root, {"entries": entries})
    if prepared is not None:
        source = Path(prepared).resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Output overlaps its prepared input")
    if path.exists() and not allow_existing:
        raise FileExistsError("Use a new output or explicitly resume the same run")
    return path


def train_epoch(model, loader, optimizer, scheduler, weights):
    model.train()
    total = 0.
    for batch in loader:
        image, coarse, context, target = ops.local_batch(batch, "native")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ops.loss_function(model(image, coarse, context), target, weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach())
    scheduler.step()
    return total / len(loader)
