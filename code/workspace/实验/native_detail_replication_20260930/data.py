"""Seeded crops and read-only shared pixels, with a seed-specific coarse cache."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "native_detail_decoder_20260930"
CONTROL = HERE.parent / "native_detail_context_control_20260930"
sys.path.insert(0, str(CONTROL))
from model_no_context import NativeNoContext
import run_study as s


class ReplicationDataset(Dataset):
    def __init__(self, coarse_cache, split, training, seed=2024):
        self.folder = Path(coarse_cache)
        self.meta = s.read(self.folder / "manifest.json")
        if self.meta["status"] != "completed":
            raise ValueError("Incomplete coarse cache")
        self.records = s.read(self.folder / "records.json")
        self.shared = Path(self.meta["shared_pixel_cache"])
        self.shared_n = s.read(self.shared / "manifest.json")["n"]
        self.indices = [i for i, r in enumerate(self.records) if r["split"] == split]
        self.seed, self.training, self.arrays = seed, training, None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = None
        return state

    def __getitem__(self, requested):
        index, epoch = requested if isinstance(requested, tuple) else (requested, 0)
        row = self.indices[index]
        shared_row = self.records[row]["shared_index"]
        if self.arrays is None:
            self.arrays = {k: np.memmap(self.shared / f"{k}.dat", mode="r", dtype=np.uint8,
                                      shape=(self.shared_n, side, side))
                           for k, side in (("image", 1024), ("mask", 1024), ("low", 512))}
            self.arrays["coarse"] = np.memmap(self.folder / "coarse.dat", mode="r", dtype=np.uint16,
                                             shape=(self.meta["n"], 8, 512, 512))
        rng = np.random.default_rng(self.seed + 1000003 * epoch + 97 * index)
        if self.training:
            y, x = (4 * rng.integers(0, (1024-384)//4+1, size=2)).tolist()
            side, aug = 384, rng.integers(0, [2, 2, 4]).tolist()
        else:
            y = x = 0
            side, aug = 1024, [0, 0, 0]
        return {"image": torch.from_numpy(np.array(self.arrays["image"][shared_row, y:y+side, x:x+side], copy=True)),
                "mask": torch.from_numpy(np.array(self.arrays["mask"][shared_row, y:y+side, x:x+side], copy=True)),
                "low": torch.from_numpy(np.array(self.arrays["low"][shared_row], copy=True)),
                "coarse": torch.from_numpy(np.array(self.arrays["coarse"][row], copy=True)).view(torch.bfloat16),
                "context": torch.zeros(32, 128, 128),  # Never read by NativeNoContext.
                "origin": torch.tensor([y, x]), "aug": torch.tensor(aug)}


class SeedSampler(Sampler):
    def __init__(self, data, seed=2024):
        self.data, self.seed, self.epoch = data, seed, 1

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        order = torch.randperm(len(self.data), generator=generator)
        return iter((int(i), self.epoch) for i in order)
