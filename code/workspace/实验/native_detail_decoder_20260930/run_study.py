"""One-shot native/degraded paired decoder experiment; never loads test pixels."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from architecture import NativeDetailDecoder, sample_native_grid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "architecture_upgrade_20260928"))
import prototype as p

BASELINE = HERE.parent / "multimodal_remaining_oil_outputs_fusion_order_ablation"
CHECKPOINT = BASELINE / "models/seed_42_fusion_before_ppm/best_model.pth"
TOKENIZER = CHECKPOINT.parent / "tokenizer.json"
LAYOUT = {"image": (np.uint8, (1024, 1024)), "mask": (np.uint8, (1024, 1024)),
          "low": (np.uint8, (512, 512)), "coarse": (np.uint16, (8, 512, 512)),
          "context": (np.float32, (32, 128, 128))}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_hashes():
    paths = [HERE / name for name in ("run_study.py", "architecture.py", "PROTOCOL.md")]
    paths += [Path(p.__file__), ROOT / "MM-PoreFormer/scripts/run_fusion_order_ablation.py",
              ROOT / "MM-PoreFormer/scripts/run_seed_statistics.py"]
    return {str(path): digest(path) for path in paths}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def setup():
    p.configure(42)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(.35)


def metrics(hist):
    h = np.asarray(hist, dtype=np.int64)
    diagonal = np.diag(h).astype(np.float64)
    gt, pred = h.sum(1), h.sum(0)
    iou = diagonal / (gt + pred - diagonal + 1e-8)
    dice = 2 * diagonal / (gt + pred + 1e-8)
    return {"mIoU": float(iou.mean()), "mDice": float(dice.mean()),
            "oil_macro_IoU": float(iou[3:].mean()), "confusion_matrix": h.tolist(),
            "per_class": {str(i): {"name": p.base.Config.class_names[i], "iou": float(iou[i]),
                                  "dice": float(dice[i]), "gt_pixels": int(gt[i]),
                                  "pred_pixels": int(pred[i])} for i in range(8)}}


def add_counts(hist, prediction, target):
    pr = prediction.detach().cpu().numpy().astype(np.int64)
    gt = target.detach().cpu().numpy().astype(np.int64)
    if gt.shape != pr.shape or gt.min() < 0 or gt.max() > 7:
        raise ValueError("Invalid evaluation label shape or range")
    hist += np.bincount((gt * 8 + pr).ravel(), minlength=64).reshape(8, 8)


def resolve(row, kind):
    original = Path(row[f"{kind}_path"])
    path = ROOT / "实验" / original.parent.name / original.name
    if path.is_file():
        return path
    suffix = f"filtered{row['slice_id']}" + (".tif" if kind == "image" else "_semantic_8bit_palette.tif")
    candidates = list(path.parent.glob("*" + suffix))
    if len(candidates) != 1:
        raise ValueError(f"No unique same-stage axial-ID match: {path}")
    return candidates[0]


def cache_arrays(folder, n, mode="r"):
    return {key: np.memmap(folder / f"{key}.dat", dtype=dtype, mode=mode, shape=(n, *shape))
            for key, (dtype, shape) in LAYOUT.items()}


def cache_identity():
    return {"checkpoint": digest(CHECKPOINT), "tokenizer": digest(TOKENIZER),
            "canonical_metadata": digest(p.CACHE / "cache_meta.json"),
            "splits": {s: digest(p.SPLITS / f"{s}_records.csv") for s in ("train", "val")},
            "sources": source_hashes()}


def prepare_cache(folder, smoke=False):
    identity = cache_identity()
    if folder.exists():
        meta = read(folder / "manifest.json")
        if meta["status"] != "completed" or meta["identity"] != identity:
            raise RuntimeError("Existing cache is incomplete or has changed provenance; preserve and inspect it")
        if meta["smoke"] != smoke:
            raise ValueError("Cache scope differs")
        for key, (dtype, shape) in LAYOUT.items():
            expected = meta["n"] * int(np.prod(shape)) * np.dtype(dtype).itemsize
            if (folder / f"{key}.dat").stat().st_size != expected:
                raise ValueError(f"Truncated cache: {key}")
        return meta
    folder.mkdir(parents=True, exist_ok=False)
    saved, canonical_meta = p.load_records()
    rows = []
    for split in ("train", "val"):
        selected = saved[split][:4 if split == "train" else 2] if smoke else saved[split]
        rows += [{**r, "split": split} for r in selected]
    n = len(rows)
    meta = {"status": "preparing", "pid": os.getpid(), "created_utc": utc(), "identity": identity,
            "n": n, "smoke": smoke, "train_count": sum(r["split"] == "train" for r in rows),
            "val_count": sum(r["split"] == "val" for r in rows), "test_pixels_read": False,
            "precision": "BF16 logits stored as uint16 bit patterns; pooled context stored in float32; exact round-trip checked"}
    write(folder / "manifest.json", meta)
    shape = (canonical_meta["num_records"], 512, 512)
    original_images = np.memmap(p.CACHE / "images_uint8.dat", dtype=np.uint8, mode="r", shape=shape)
    original_masks = np.memmap(p.CACHE / "masks_uint8.dat", dtype=np.uint8, mode="r", shape=shape)
    arrays = cache_arrays(folder, n, "w+")
    tokenizer = p.base.CharTokenizer.load(TOKENIZER)
    teacher = p.order.build_model("fusion_before_ppm", tokenizer.vocab_size)
    reference = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if reference["epoch"] != 58:
        raise ValueError("Unexpected global checkpoint epoch")
    teacher.load_state_dict(reference["model_state_dict"], strict=True)
    del reference
    teacher = teacher.cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)
    captured = {}
    hook = teacher.dec1.register_forward_hook(lambda module, args, output: captured.update(state=output))
    records = []
    try:
        for start in range(0, n, 4):
            batch = rows[start:start + 4]
            image_list, ids_list, masks_list = [], [], []
            for offset, row in enumerate(batch):
                index = start + offset
                ipath, mpath = resolve(row, "image"), resolve(row, "mask")
                with Image.open(ipath) as im:
                    raw_image = np.array(im.convert("L"), dtype=np.uint8)
                with Image.open(mpath) as im:
                    raw_mask = np.array(im)
                if raw_image.shape != (1024, 1024) or raw_mask.shape != (1024, 1024):
                    raise ValueError(f"Unexpected native shape: {ipath}, {mpath}")
                if raw_mask.min() < 0 or raw_mask.max() > 7:
                    raise ValueError(f"Invalid raw label IDs: {mpath}")
                low = np.array(Image.fromarray(raw_image).resize((512, 512), Image.Resampling.BILINEAR))
                mask512 = np.array(Image.fromarray(raw_mask).resize((512, 512), Image.Resampling.NEAREST))
                ci = row["cache_index"]
                if not np.array_equal(low, original_images[ci]) or not np.array_equal(mask512, original_masks[ci]):
                    raise ValueError(f"Raw/cache mismatch: {row['sample']} {row['slice_id']}")
                if not np.array_equal(mask512, raw_mask[1::2, 1::2]):
                    raise ValueError("Native/canonical pixel-center mismatch")
                arrays["image"][index], arrays["mask"][index], arrays["low"][index] = raw_image, raw_mask, low
                image_list.append(torch.from_numpy(low.copy()).float().unsqueeze(0).repeat(3, 1, 1) / 255)
                ids, mask = tokenizer.encode(row["text"])
                ids_list.append(ids)
                masks_list.append(mask)
                records.append({"sample": row["sample"], "slice_id": row["slice_id"], "split": row["split"],
                                "cache_index": ci, "image_path": str(ipath), "mask_path": str(mpath),
                                "image_sha256": digest(ipath), "mask_sha256": digest(mpath)})
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = teacher(torch.stack(image_list).cuda().to(memory_format=torch.channels_last),
                                 torch.stack(ids_list).cuda(), torch.stack(masks_list).cuda())
                context = F.avg_pool2d(captured.pop("state"), 4)
            for key, tensor in (("coarse", logits), ("context", context)):
                if key == "coarse":
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError("Expected BF16 frozen logits")
                    encoded = tensor.cpu().view(torch.uint16).numpy()
                    restored = torch.from_numpy(encoded).view(torch.bfloat16).float()
                else:
                    encoded = tensor.float().cpu().numpy()
                    restored = torch.from_numpy(encoded)
                if not torch.equal(restored, tensor.float().cpu()) or not torch.isfinite(restored).all():
                    raise ValueError(f"Cache precision loss: {key}")
                arrays[key][start:start + len(batch)] = encoded
            del logits, context, encoded
            if start % 100 == 0 or start + len(batch) == n:
                meta.update(completed_records=start + len(batch), updated_utc=utc())
                write(folder / "manifest.json", meta)
                print(f"CACHE {start + len(batch)}/{n}", flush=True)
        for arr in arrays.values():
            arr.flush()
        write(folder / "records.json", records)
        meta.update(status="completed", updated_utc=utc(), records_sha256=digest(folder / "records.json"),
                    frozen_parameter_gradients=0, native_to_canonical_match_all=True)
        write(folder / "manifest.json", meta)
    except BaseException as exc:
        meta.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), updated_utc=utc())
        write(folder / "manifest.json", meta)
        raise
    finally:
        hook.remove()
        del teacher, arrays
        torch.cuda.empty_cache()
    return meta


class CachedDataset(Dataset):
    def __init__(self, folder, split, training):
        self.folder = Path(folder)
        self.meta = read(self.folder / "manifest.json")
        self.records = read(self.folder / "records.json")
        self.indices = [i for i, r in enumerate(self.records) if r["split"] == split]
        self.training = training
        self.arrays = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = None
        return state

    def __getitem__(self, requested):
        if isinstance(requested, tuple):
            index, epoch = requested
        else:
            index, epoch = requested, 0
        row = self.indices[index]
        if self.arrays is None:
            self.arrays = cache_arrays(self.folder, self.meta["n"])
        randomizer = np.random.default_rng(42 + 1000003 * epoch + 97 * index)
        if self.training:
            y, x = (4 * randomizer.integers(0, (1024 - 384) // 4 + 1, size=2)).tolist()
            side = 384
            aug = randomizer.integers(0, [2, 2, 4]).tolist()
        else:
            y = x = 0
            side = 1024
            aug = [0, 0, 0]
        result = {key: torch.from_numpy(np.array(self.arrays[key][row], copy=True))
                  for key in ("low", "coarse", "context")}
        result["coarse"] = result["coarse"].view(torch.bfloat16)
        result.update(image=torch.from_numpy(np.array(self.arrays["image"][row, y:y+side, x:x+side], copy=True)),
                      mask=torch.from_numpy(np.array(self.arrays["mask"][row, y:y+side, x:x+side], copy=True)),
                      origin=torch.tensor([y, x]), aug=torch.tensor(aug))
        return result


class EpochSampler(Sampler):
    def __init__(self, data):
        self.data = data
        self.epoch = 1

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        order = torch.randperm(len(self.data), generator=torch.Generator().manual_seed(42000 + self.epoch))
        return iter((int(i), self.epoch) for i in order)


def local_batch(batch, arm):
    origins = batch["origin"].cuda()
    side = batch["image"].shape[-1]
    coarse = sample_native_grid(batch["coarse"].cuda(), origins, side)
    context = sample_native_grid(batch["context"].cuda(), origins, side, stride=4)
    if arm == "native":
        image = batch["image"].cuda().float().unsqueeze(1) / 255
    else:
        image = sample_native_grid(batch["low"].cuda().float().unsqueeze(1) / 255, origins, side)
    target = batch["mask"].cuda().long()
    values = [image, coarse, context, target]
    for i, (flip_y, flip_x, k) in enumerate(batch["aug"].tolist()):
        for item in values:
            sample = item[i]
            if flip_y:
                sample = sample.flip(-2)
            if flip_x:
                sample = sample.flip(-1)
            if k:
                sample = sample.rot90(k, (-2, -1))
            item[i] = sample.clone()
    return image, coarse, context, target


@torch.inference_mode()
def evaluate(model, loader, arm, frozen=False):
    if model is not None:
        model.eval()
    hist_native, hist512 = np.zeros((8, 8), np.int64), np.zeros((8, 8), np.int64)
    for index, batch in enumerate(loader):
        image, coarse, context, target = local_batch(batch, arm)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = coarse if frozen else model(image, coarse, context)
        add_counts(hist_native, logits.argmax(1), target)
        # The canonical reference is the original frozen 512 prediction, while
        # the new decoder's native prediction is sampled on canonical label centers.
        pred512 = batch["coarse"].cuda().argmax(1) if frozen else logits[:, :, 1::2, 1::2].argmax(1)
        add_counts(hist512, pred512, target[:, 1::2, 1::2])
        if index % 150 == 0:
            print(f"VAL {'frozen' if frozen else arm} {index+1}/{len(loader)}", flush=True)
    assert hist_native.sum() == len(loader.dataset) * 1024**2
    assert hist512.sum() == len(loader.dataset) * 512**2
    return {"native": metrics(hist_native), "canonical512": metrics(hist512)}


def loss_function(logits, target, weights):
    # Exclude the 32-pixel crop border, whose context differs from full-image inference.
    logits, target = logits[:, :, 32:-32, 32:-32].float(), target[:, 32:-32, 32:-32]
    return .5 * F.cross_entropy(logits, target, weight=weights) + .5 * p.base.WeightedDiceLoss(8, weights)(logits, target)


def smoke(folder):
    setup()
    meta = prepare_cache(folder, smoke=True)
    data = CachedDataset(folder, "train", True)
    loader = DataLoader(data, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    model = NativeDetailDecoder().cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    weights = torch.tensor(read(HERE.parent / "wall_geometry_refinement_20260929/geometry/outputs/legacy60_scratch_seed42/run_manifest.json")["class_weights"], device="cuda")
    observations = []
    torch.cuda.reset_peak_memory_stats()
    for arm in ("degraded", "native"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        image, coarse, context, target = local_batch(batch, arm)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(image, coarse, context)
            loss = loss_function(logits, target, weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite smoke loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        for prefix in ("fine", "middle", "deep", "context_projection", "context_gate", "up_middle", "up_fine", "output"):
            if not any(t.grad is not None and bool(t.grad.abs().sum() > 0) for n, t in model.named_parameters() if n.startswith(prefix)):
                raise ValueError(f"No gradient: {prefix}")
        optimizer.step()
        observations.append({"arm": arm, "loss": float(loss.detach()), "gradient_norm": float(norm)})
    val = DataLoader(CachedDataset(folder, "val", False), batch_size=1, num_workers=0)
    result = evaluate(model, val, "native")
    report = {"status": "passed", "steps": observations, "validation_images": meta["val_count"],
              "native_forward_shape": [1, 8, 1024, 1024], "parameters": sum(t.numel() for t in model.parameters()),
              "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
              "finite_validation": bool(np.isfinite(result["native"]["mIoU"])), "weights_reused_in_formal": False,
              "sources": source_hashes()}
    write(HERE / "smoke_report.json", report)
    print("SMOKE_PASSED", json.dumps(report), flush=True)


def train_pair(folder, cache):
    setup()
    check = read(HERE / "smoke_report.json")
    if check["status"] != "passed" or check["sources"] != source_hashes():
        raise RuntimeError("Current source has not passed the real-data smoke")
    meta = prepare_cache(cache)
    weights_source = HERE.parent / "wall_geometry_refinement_20260929/geometry/outputs/legacy60_scratch_seed42/run_manifest.json"
    class_weights = read(weights_source)["class_weights"]
    config = {"seed": 42, "epochs": 12, "crop": 384, "halo": 32, "batch_size": 4,
              "lr": .001, "weight_decay": .0001, "gpu_allocator_fraction": .35,
              "class_weights": class_weights, "class_weights_source_sha256": digest(weights_source),
              "selection": "maximum native validation eight-class mIoU; first tie",
              "train_count": meta["train_count"], "val_count": meta["val_count"], "test_pixels_read": False}
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=False)
        manifest = {"status": "initializing", "created_utc": utc(), "sources": source_hashes(),
                    "config": config, "cache_identity": meta["identity"], "arms": {}}
        snapshot = folder / "source_snapshot"
        snapshot.mkdir()
        for source in source_hashes():
            shutil.copy2(source, snapshot / Path(source).name)
        write(folder / "run_manifest.json", manifest)
    else:
        manifest = read(folder / "run_manifest.json")
        if manifest["sources"] != source_hashes() or manifest["config"] != config or manifest["cache_identity"] != meta["identity"]:
            raise ValueError("Run provenance/configuration changed")
        if manifest["status"] == "completed":
            raise ValueError("Pair is already complete")
    if (folder / "STOP_AFTER_EPOCH.txt").exists():
        raise RuntimeError("Explicit stop marker is present")
    manifest.update(status="running", pid=os.getpid(), updated_utc=utc())
    write(folder / "run_manifest.json", manifest)
    train_data = CachedDataset(cache, "train", True)
    sampler = EpochSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=4, sampler=sampler, num_workers=2,
                              persistent_workers=True, prefetch_factor=2, pin_memory=True)
    val_loader = DataLoader(CachedDataset(cache, "val", False), batch_size=1, num_workers=0)
    weights = torch.tensor(class_weights, device="cuda")
    try:
        baseline_path = folder / "frozen_validation.json"
        if not baseline_path.exists():
            write(baseline_path, evaluate(None, val_loader, "degraded", frozen=True))
        for arm in ("degraded", "native"):
            if manifest["arms"].get(arm, {}).get("status") == "completed":
                continue
            p.base.set_seed(42)
            model = NativeDetailDecoder().cuda().to(memory_format=torch.channels_last)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
            history, best, best_epoch = [], -1., None
            initial_hash = hashlib.sha256(b"".join(t.detach().cpu().contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            last_path = folder / f"{arm}_last.pth"
            if last_path.exists():
                saved = torch.load(last_path, map_location="cpu", weights_only=False)
                if saved["sources"] != source_hashes() or saved["config"] != config:
                    raise ValueError("Checkpoint source/config mismatch")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                history, best, best_epoch = saved["history"], saved["best"], saved["best_epoch"]
                random.setstate(saved["rng"]["python"])
                np.random.set_state(saved["rng"]["numpy"])
                torch.set_rng_state(saved["rng"]["torch"])
                torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
                del saved
            manifest["arms"][arm] = {"status": "training", "initial_sha256": initial_hash,
                                     "parameters": sum(t.numel() for t in model.parameters())}
            for epoch in range(len(history)+1, 13):
                if (folder / "STOP_AFTER_EPOCH.txt").exists():
                    manifest.update(status="paused", updated_utc=utc())
                    write(folder / "run_manifest.json", manifest)
                    return
                start = time.monotonic()
                sampler.epoch = epoch
                model.train()
                manifest.update(current_arm=arm, current_epoch=epoch, phase="train", updated_utc=utc())
                write(folder / "run_manifest.json", manifest)
                total = 0.
                print(f"EPOCH_START arm={arm} epoch={epoch}/12", flush=True)
                for batch in train_loader:
                    image, coarse, context, target = local_batch(batch, arm)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(image, coarse, context)
                        loss = loss_function(logits, target, weights)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach())
                scheduler.step()
                manifest.update(phase="validation", updated_utc=utc())
                write(folder / "run_manifest.json", manifest)
                score = evaluate(model, val_loader, arm)
                row = {"epoch": epoch, "loss": total/len(train_loader), "seconds": time.monotonic()-start, "validation": score}
                history.append(row)
                if score["native"]["mIoU"] > best:
                    best, best_epoch = score["native"]["mIoU"], epoch
                    save(folder / f"{arm}_best.pth", {"model": model.state_dict(), "epoch": epoch, "validation": score})
                save(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scheduler": scheduler.state_dict(), "history": history, "best": best,
                                 "best_epoch": best_epoch, "sources": source_hashes(), "config": config,
                                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                                         "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}})
                write(folder / f"{arm}_history.json", history)
                manifest["arms"][arm].update(completed_epochs=epoch, best_epoch=best_epoch, best_native_mIoU=best)
                manifest.update(phase="epoch_complete", updated_utc=utc())
                write(folder / "run_manifest.json", manifest)
                print(f"EPOCH_DONE arm={arm} epoch={epoch} native_mIoU={score['native']['mIoU']:.8f} canonical_mIoU={score['canonical512']['mIoU']:.8f} seconds={row['seconds']:.1f}", flush=True)
            chosen = torch.load(folder / f"{arm}_best.pth", map_location="cpu", weights_only=False)
            model.load_state_dict(chosen["model"])
            verified = evaluate(model, val_loader, arm)
            if verified != chosen["validation"]:
                raise ValueError("Reloaded chosen validation does not match")
            write(folder / f"{arm}_selected_validation.json", verified)
            manifest["arms"][arm].update(status="completed", selected_validation_verified=True,
                                         checkpoint_sha256=digest(folder / f"{arm}_best.pth"))
            write(folder / "run_manifest.json", manifest)
            del model, optimizer, scheduler, chosen
            torch.cuda.empty_cache()
        if manifest["arms"]["native"]["initial_sha256"] != manifest["arms"]["degraded"]["initial_sha256"]:
            raise ValueError("Paired initialization mismatch")
        frozen, degraded, native = (read(folder / name) for name in ("frozen_validation.json", "degraded_selected_validation.json", "native_selected_validation.json"))
        gates = {"native_vs_frozen_0.005": native["native"]["mIoU"] >= frozen["native"]["mIoU"] + .005,
                 "native_vs_degraded_0.0025": native["native"]["mIoU"] >= degraded["native"]["mIoU"] + .0025,
                 "canonical_no_material_regression": native["canonical512"]["mIoU"] >= frozen["canonical512"]["mIoU"] - .001,
                 "oil_macro_no_regression": native["native"]["oil_macro_IoU"] >= frozen["native"]["oil_macro_IoU"]}
        for key in ("3", "4"):
            gates[f"class_{key}_no_regression"] = native["native"]["per_class"][key]["iou"] >= frozen["native"]["per_class"][key]["iou"]
        write(folder / "summary.json", {"frozen": frozen, "degraded": degraded, "native": native,
              "gates": gates, "passed": all(gates.values()), "single_seed_development": True, "test_pixels_read": False})
        manifest.update(status="completed", phase="complete", updated_utc=utc(), screening=gates)
        write(folder / "run_manifest.json", manifest)
        print("PAIR_COMPLETED", json.dumps(gates), flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), updated_utc=utc())
        write(folder / "run_manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke(HERE / "smoke_cache_v2")
    else:
        train_pair(HERE / "outputs/paired12_seed42", HERE / "cache")


if __name__ == "__main__":
    main()
