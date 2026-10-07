"""Seed-specific frozen descriptors and matched body-head repetitions."""
from __future__ import annotations
import argparse
import contextlib
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
PARENT = EXP / "body_interface_decoder_20260930"
CONTROL = EXP / "body_pointwise_control_20260930"
sys.path.insert(0, str(PARENT))
import run_pair as b
sys.path.insert(0, str(CONTROL))
from pointwise_model import PointwiseBodyDecoder, initialise
sys.path.insert(0, str(HERE))
s = b.s
SEEDS = (2024, 2025)
ARMS = ("separate", "pointwise")
PRIOR = PARENT / "outputs/pair12_seed42"
PRIOR_CONTROL = CONTROL / "outputs/control12_seed42"
NATIVE = {2024: EXP / "native_detail_replication_20260930",
          2025: EXP / "native_detail_replication_seed2025_20260930"}
SHARED = EXP / "native_detail_decoder_20260930/cache"


def check_stop():
    for folder in (HERE, ROOT, PARENT, CONTROL):
        if (folder / "STOP_AFTER_EPOCH.txt").exists():
            raise InterruptedError(f"User stop marker: {folder}; never remove automatically")


@contextlib.contextmanager
def lock():
    import msvcrt
    with (HERE / "run.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"\0"); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def map_rows(records, shared):
    seen = set()
    for row in records:
        index = row["shared_index"]
        assert isinstance(index, int) and 0 <= index < len(shared) and index not in seen
        seen.add(index)
        assert row["split"] in ("train", "val")
        assert all(row[k] == shared[index][k] for k in
                   ("sample", "slice_id", "split", "image_sha256", "mask_sha256"))
    assert seen == set(range(len(shared)))
    return [row["shared_index"] for row in records]


def reference(seed):
    home = NATIVE[seed]
    coarse = home / f"coarse_cache_seed{seed}"
    run = home / f"outputs/native_no_context_seed{seed}"
    meta, cm = s.read(run / "run_manifest.json"), s.read(coarse / "manifest.json")
    assert meta["status"] == cm["status"] == "completed"
    assert meta["selected_validation_verified"] and meta["config"]["seed"] == seed
    assert meta["identity"] == cm["identity"] and cm["identity"]["global_seed"] == seed
    assert cm["n"] == cm["completed_records"] == 3040 and not cm["smoke"]
    assert not cm["test_pixels_read"] and Path(cm["shared_pixel_cache"]) == SHARED
    assert s.digest(coarse / "records.json") == cm["records_sha256"]
    assert s.digest(SHARED / "manifest.json") == cm["identity"]["shared_manifest"]
    assert s.digest(SHARED / "records.json") == cm["identity"]["shared_records"]
    for path, expected in meta["identity"]["sources"].items():
        assert s.digest(path) == expected, path
    global_path = EXP / f"multimodal_remaining_oil_outputs_fusion_order_ablation/models/seed_{seed}_fusion_before_ppm/best_model.pth"
    assert s.digest(global_path) == cm["identity"]["checkpoint"]
    assert s.digest(global_path.parent / "tokenizer.json") == cm["identity"]["tokenizer"]
    assert s.digest(run / "best.pth") == meta["checkpoint_sha256"]
    old = torch.load(run / "best.pth", map_location="cpu", weights_only=False)
    assert old["epoch"] == meta["best_epoch"] == 11
    assert old["validation"] == s.read(run / "selected_validation.json")
    history = s.read(run / "history.json")
    assert len(history) == 12
    assert max(history, key=lambda r: r["validation"]["native"]["mIoU"])["epoch"] == old["epoch"]
    records = s.read(coarse / "records.json")
    mapping = map_rows(records, s.read(SHARED / "records.json"))
    assert {split: sum(r["split"] == split for r in records) for split in ("train", "val")} == {"train": 2364, "val": 676}
    assert (coarse / "coarse.dat").stat().st_size == 3040 * 8 * 512 * 512 * 2
    return run, coarse, meta, old, records, mapping


def identity():
    prior, control = s.read(PRIOR / "run_manifest.json"), s.read(PRIOR_CONTROL / "run_manifest.json")
    assert prior["status"] == control["status"] == "completed" and prior["passed"] is False
    assert prior["identity"] == b.identity()
    sources = {**prior["identity"]["sources"], **control["identity"]["sources"]}
    for path, expected in sources.items():
        assert s.digest(path) == expected, path
    for name in ("replicate_body.py", "test_replication.py", "PROTOCOL.md"):
        sources[str(HERE / name)] = s.digest(HERE / name)
    inputs = {}
    for seed in SEEDS:
        print(f"VERIFY_INPUTS seed={seed}", flush=True)
        run, coarse, meta, old, records, mapping = reference(seed)
        assert meta["config"]["class_weights"][3:] == prior["config"]["class_weights"]
        sources.update(meta["identity"]["sources"])
        inputs[str(seed)] = {
            "native_checkpoint": s.digest(run / "best.pth"),
            "native_manifest": s.digest(run / "run_manifest.json"),
            "selected_validation": s.digest(run / "selected_validation.json"),
            "coarse_manifest": s.digest(coarse / "manifest.json"),
            "coarse_records": s.digest(coarse / "records.json"),
            "coarse_array": s.digest(coarse / "coarse.dat"),
        }
    pixels = {}
    for key in ("image", "mask"):
        path = SHARED / f"{key}.dat"
        assert path.stat().st_size == 3040 * 1024 * 1024
        pixels[key] = s.digest(path)
    return {"sources": sources, "seeds": inputs, "shared_pixels": pixels,
            "shared_manifest": s.digest(SHARED / "manifest.json"),
            "shared_records": s.digest(SHARED / "records.json"),
            "seed42_pair_manifest": s.digest(PRIOR / "run_manifest.json"),
            "seed42_control_manifest": s.digest(PRIOR_CONTROL / "run_manifest.json")}


def settings():
    return {"seeds": list(SEEDS), "arms": list(ARMS), "epochs": 12, "batch_size": 512,
            "lr": .001, "weight_decay": .0001, "gpu_cap": .35,
            "class_weights": s.read(PRIOR / "run_manifest.json")["config"]["class_weights"],
            "selection": "first maximum native validation mIoU, both grids at same checkpoint",
            "protocol": "body_separate_pointwise_repeat12_seeds2024_2025_v1", "test_pixels_read": False}


def initialise_pair(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        attention = b.BodyInterfaceDecoder("separate")
        pointwise = PointwiseBodyDecoder()
        old, current = attention.state_dict(), pointwise.state_dict()
        shared = {k: value for k, value in old.items() if k in current}
        for k, value in shared.items():
            assert value.shape == current[k].shape
            current[k].copy_(value)
        pointwise.load_state_dict(current, strict=True)
    assert sum(v.numel() for v in shared.values()) == 17125
    return {"separate": attention, "pointwise": pointwise}


def order_for(indices, seed, epoch):
    return np.random.default_rng(seed * 1000 + epoch).permutation(indices)


def stream(seed, epoch, device="cuda"):
    return torch.Generator(device=device).manual_seed(seed * 10000 + epoch)


@torch.inference_mode()
def prepare(seed, folder, ids, smoke=False):
    run, coarse_path, ref, old, records, mapping = reference(seed)
    selected = list(range(len(records)))
    if smoke:
        selected = [i for i, r in enumerate(records) if r["split"] == "train"][:2] + \
                   [i for i, r in enumerate(records) if r["split"] == "val"][:2]
    if folder.exists():
        meta = s.read(folder / "manifest.json")
        assert meta["identity"] == ids and meta["seed"] == seed and meta["smoke"] == smoke
        assert meta["selected_rows"] == selected
        if meta["status"] == "completed":
            verify_cache(folder)
            return meta
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "shards").mkdir()
        meta = {"status": "preparing", "identity": ids, "seed": seed, "smoke": smoke,
                "selected_rows": selected, "created_utc": s.utc(), "completed_images": 0,
                "test_pixels_read": False}
        s.write(folder / "manifest.json", meta)
    model = b.NativeNoContext().eval().requires_grad_(False)
    model.load_state_dict(old["model"], strict=True)
    model = model.cuda().to(memory_format=torch.channels_last)
    box = {}
    hook = model.up_fine.register_forward_hook(lambda m, a, out: box.update(feature=out))
    arrays = {k: np.memmap(SHARED / f"{k}.dat", mode="r", dtype=np.uint8, shape=(3040, 1024, 1024)) for k in ("image", "mask")}
    coarse = np.memmap(coarse_path / "coarse.dat", mode="r", dtype=np.uint16, shape=(3040, 8, 512, 512))
    entries, started = [], time.monotonic()
    try:
        for ordinal, row in enumerate(selected):
            check_stop()
            record = records[row]
            path = folder / "shards" / f"{row:04d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as shard:
                    assert int(shard["row"]) == row and int(shard["shared_index"]) == mapping[row]
                    n = len(shard["points"])
            else:
                im = np.array(arrays["image"][mapping[row]], copy=True).astype(np.float32) / 255
                image = torch.from_numpy(im).cuda()[None, None]
                low = torch.from_numpy(np.array(coarse[row], copy=True)).view(torch.bfloat16).cuda()[None]
                up = s.sample_native_grid(low, torch.zeros((1, 2), device="cuda"), 1024)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(image, up, None)
                probabilities = logits.float().softmax(1)[0].cpu().numpy()
                fine = box.pop("feature").float()[0].cpu().numpy()
                descriptors, labels, prediction = b.describe(im, probabilities, fine)
                gt = np.array(arrays["mask"][mapping[row]], copy=True)
                counts, outside = b.targets(labels, prediction, gt)
                values = {**descriptors, "row": np.array(row), "shared_index": np.array(mapping[row])}
                for grid, p, t in (("native", prediction, gt), ("canonical512", prediction[1::2, 1::2], gt[1::2, 1::2])):
                    values[f"{grid}_counts"] = counts[grid]
                    values[f"{grid}_outside"] = outside[grid]
                    values[f"{grid}_frozen"] = np.bincount((t.astype(np.int64) * 8 + p).ravel(), minlength=64).reshape(8, 8)
                b.save_npz(path, values)
                n = len(descriptors["points"])
            entries.append({"row": row, "shared_index": mapping[row], "sample": record["sample"],
                            "slice_id": record["slice_id"], "split": record["split"], "bodies": n,
                            "shard_sha256": s.digest(path)})
            if (ordinal + 1) % 100 == 0 or ordinal + 1 == len(selected):
                meta.update(completed_images=ordinal + 1, updated_utc=s.utc(), elapsed_seconds=time.monotonic() - started)
                s.write(folder / "manifest.json", meta)
                print(f"CACHE seed={seed} {ordinal+1}/{len(selected)} seconds={meta['elapsed_seconds']:.1f}", flush=True)
    finally:
        hook.remove()
        del model, arrays, coarse
        torch.cuda.empty_cache()
    check_stop()
    n = sum(e["bodies"] for e in entries)
    total = {k: np.lib.format.open_memmap(folder / f"{k}.npy", mode="w+", dtype=dtype, shape=(n, *shape)) for k, (dtype, shape) in b.LAYOUT.items()}
    outside = {split: {g: np.zeros((8, 8), np.int64) for g in ("native", "canonical512")} for split in ("train", "val")}
    frozen = {split: {g: np.zeros((8, 8), np.int64) for g in ("native", "canonical512")} for split in ("train", "val")}
    start = 0
    for entry in entries:
        check_stop()
        end = start + entry["bodies"]
        entry.update(start=start, end=end)
        path = folder / "shards" / f"{entry['row']:04d}.npz"
        assert s.digest(path) == entry["shard_sha256"]
        with np.load(path, allow_pickle=False) as shard:
            for key in total:
                total[key][start:end] = shard[key]
            for g in ("native", "canonical512"):
                outside[entry["split"]][g] += shard[f"{g}_outside"]
                frozen[entry["split"]][g] += shard[f"{g}_frozen"]
        start = end
    for array in total.values():
        array.flush()
    del total
    measured = {g: s.metrics(h) for g, h in frozen["val"].items()}
    if not smoke:
        assert measured == old["validation"], "Frozen validation replay differs: do not train"
    s.write(folder / "records.json", entries)
    s.write(folder / "outside.json", {split: {g: h.tolist() for g, h in groups.items()} for split, groups in outside.items()})
    s.write(folder / "frozen_validation.json", measured)
    meta.update(status="completed", n=n, updated_utc=s.utc(), frozen_validation_replayed=not smoke,
                records_sha256=s.digest(folder / "records.json"), outside_sha256=s.digest(folder / "outside.json"),
                array_sha256={k: s.digest(folder / f"{k}.npy") for k in b.LAYOUT},
                images_by_split={split: sum(e["split"] == split for e in entries) for split in ("train", "val")},
                bodies_by_split={split: sum(e["bodies"] for e in entries if e["split"] == split) for split in ("train", "val")})
    s.write(folder / "manifest.json", meta)
    return meta


def verify_cache(folder):
    meta = s.read(folder / "manifest.json")
    assert meta["status"] == "completed"
    assert s.digest(folder / "records.json") == meta["records_sha256"]
    assert s.digest(folder / "outside.json") == meta["outside_sha256"]
    for key, (dtype, shape) in b.LAYOUT.items():
        a = np.load(folder / f"{key}.npy", mmap_mode="r")
        assert a.dtype == dtype and a.shape == (meta["n"], *shape)
        assert s.digest(folder / f"{key}.npy") == meta["array_sha256"][key]
    return meta


@torch.inference_mode()
def cpu_replay(path, data, frozen):
    """Independent class-by-class accumulation, not the GPU evaluator."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = b.BodyInterfaceDecoder("separate") if saved["arm"] == "separate" else PointwiseBodyDecoder()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    matrices = {g: np.array(h, dtype=np.int64) for g, h in data.outside.items()}
    for start in range(0, len(data.indices), 512):
        idx = data.indices[start:start + 512]
        args = [torch.from_numpy(np.array(data.arrays[k][idx], dtype=dtype)) for k, dtype in
                (("points", np.float32), ("valid", bool), ("summary", np.float32))]
        assigned = model(*args).argmax(1).numpy() + 3
        for g, h in matrices.items():
            counts = np.asarray(data.arrays[f"{g}_counts"][idx], dtype=np.int64)
            for c in range(3, 8):
                h[:, c] += counts[assigned == c].sum(0)
    result = {}
    for g, h in matrices.items():
        expected = np.array(saved["validation"][g]["confusion_matrix"])
        base = np.array(frozen[g]["confusion_matrix"])
        assert np.array_equal(h[:, :3], base[:, :3])
        assert np.array_equal(h[:, 3:].sum(1), base[:, 3:].sum(1))
        result[g] = {"confusion_exact": bool(np.array_equal(h, expected)),
                     "absolute_difference": int(np.abs(h - expected).sum()), "metrics": s.metrics(h)}
    return {"checkpoint_sha256": s.digest(path), "selected_epoch": saved["epoch"],
            "device": "cpu", "grids": result, "exact": all(v["confusion_exact"] for v in result.values())}


def smoke():
    check_stop()
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_replication.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful(), "CPU contracts failed"
    ids, config = identity(), settings()
    s.setup()
    reports = {}
    for seed in SEEDS:
        folder = HERE / f"smoke_cache_seed{seed}"
        torch.cuda.reset_peak_memory_stats()
        prepare(seed, folder, ids, smoke=True)
        inference_peak = torch.cuda.max_memory_allocated() / 2**20
        data = b.CachedBodies(folder, "train")
        idx = np.resize(data.indices, 512)
        p, v, g = data.input_batch(idx)
        labels = torch.from_numpy(np.array(data.arrays["native_counts"][idx], copy=True)).cuda().float()
        weights = torch.tensor(config["class_weights"], device="cuda")
        reports[str(seed)] = {"native_inference_peak_MiB": inference_peak, "arms": {}}
        for arm, model in initialise_pair(seed).items():
            initial = b.state_digest(model.state_dict())
            model = model.cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            torch.cuda.reset_peak_memory_stats()
            loss = b.body_loss(model(b.augment(p, stream(seed, 1)), v, g), labels, weights)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            reports[str(seed)]["arms"][arm] = {"initial_sha256": initial, "loss": float(loss.detach()),
                "gradient_norm": float(norm), "peak_MiB": torch.cuda.max_memory_allocated() / 2**20,
                "parameters": sum(x.numel() for x in model.parameters())}
            del model, optimizer, loss
        torch.cuda.empty_cache()
    s.write(HERE / "smoke_report.json", {"status": "passed", "identity": ids, "config": config,
            "cpu_tests": result.testsRun, "reports": reports, "smoke_weights_reused": False})
    print("SMOKE_PASSED", reports, flush=True)


def train_seed(seed, ids, config, smoke_report, root_manifest):
    folder, run = HERE / f"cache_seed{seed}", HERE / f"outputs/pair12_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "run_manifest.json"
    if manifest_path.exists():
        meta = s.read(manifest_path)
        assert meta["identity"] == ids and meta["config"] == config and meta["seed"] == seed
        if meta["status"] == "completed":
            return
    else:
        meta = {"status": "preparing", "identity": ids, "config": config, "seed": seed,
                "arms": {}, "created_utc": s.utc()}
    try:
        meta.update(status="running", phase="cache", pid=os.getpid(), updated_utc=s.utc())
        s.write(manifest_path, meta)
        cache = prepare(seed, folder, ids)
        assert cache["images_by_split"] == {"train": 2364, "val": 676} and cache["frozen_validation_replayed"]
        verify_cache(folder)
        meta["cache_manifest_sha256"] = s.digest(folder / "manifest.json")
        train, val = b.CachedBodies(folder, "train"), b.CachedBodies(folder, "val")
        frozen, mean = s.read(folder / "frozen_validation.json"), b.evaluate(None, val)
        s.write(run / "frozen_validation.json", frozen)
        s.write(run / "body_mean_validation.json", mean)
        weights = torch.tensor(config["class_weights"], device="cuda")
        for arm in ARMS:
            check_stop()
            if meta["arms"].get(arm, {}).get("status") == "completed":
                assert s.digest(run / f"{arm}_best.pth") == meta["arms"][arm]["checkpoint_sha256"]
                continue
            model = initialise_pair(seed)[arm].cuda()
            initial = b.state_digest(model.state_dict())
            assert initial == smoke_report["reports"][str(seed)]["arms"][arm]["initial_sha256"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
            history, best, best_epoch = [], -1., None
            last = run / f"{arm}_last.pth"
            if last.exists():
                saved = torch.load(last, map_location="cpu", weights_only=False)
                assert saved["identity"] == ids and saved["config"] == config
                assert saved["arm"] == arm and saved["seed"] == seed
                assert saved["cache_manifest_sha256"] == meta["cache_manifest_sha256"]
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                schedule.load_state_dict(saved["schedule"])
                history, best, best_epoch = saved["history"], saved["best"], saved["best_epoch"]
            info = {"status": "running", "initial_sha256": initial,
                    "parameters": sum(p.numel() for p in model.parameters())}
            meta["arms"][arm] = info
            for epoch in range(len(history) + 1, 13):
                check_stop()
                started = time.monotonic()
                meta.update(current_arm=arm, current_epoch=epoch, phase="train", updated_utc=s.utc())
                s.write(manifest_path, meta)
                order, generator = order_for(train.indices, seed, epoch), stream(seed, epoch)
                model.train(); losses = []
                print(f"EPOCH_START seed={seed} {arm} {epoch}/12 bodies={len(order)}", flush=True)
                for start in range(0, len(order), 512):
                    idx = order[start:start + 512]
                    p, v, g = train.input_batch(idx)
                    target = torch.from_numpy(np.array(train.arrays["native_counts"][idx], copy=True)).cuda().float()
                    optimizer.zero_grad(set_to_none=True)
                    loss = b.body_loss(model(b.augment(p, generator), v, g), target, weights)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); losses.append(float(loss.detach()))
                    if start % (512 * 400) == 0:
                        print(f"TRAIN seed={seed} {arm} {epoch} bodies={start+len(idx)}/{len(order)}", flush=True)
                schedule.step()
                meta.update(phase="validation", updated_utc=s.utc()); s.write(manifest_path, meta)
                measured = b.evaluate(model, val)
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.monotonic() - started, "validation": measured})
                common = {"model": model.state_dict(), "identity": ids, "config": config, "seed": seed,
                          "arm": arm, "cache_manifest_sha256": meta["cache_manifest_sha256"]}
                if measured["native"]["mIoU"] > best:
                    best, best_epoch = measured["native"]["mIoU"], epoch
                    s.save(run / f"{arm}_best.pth", {**common, "epoch": epoch, "validation": measured})
                s.save(last, {**common, "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                              "history": history, "best": best, "best_epoch": best_epoch})
                s.write(run / f"{arm}_history.json", history)
                info.update(completed_epochs=epoch, best_epoch=best_epoch, best_native_mIoU=best)
                meta.update(phase="epoch_complete", updated_utc=s.utc()); s.write(manifest_path, meta)
                print(f"EPOCH_DONE seed={seed} {arm} {epoch}/12 native={measured['native']['mIoU']:.8f} canonical={measured['canonical512']['mIoU']:.8f} seconds={history[-1]['seconds']:.1f}", flush=True)
            selected = torch.load(run / f"{arm}_best.pth", map_location="cpu", weights_only=False)
            assert selected["epoch"] == max(history, key=lambda r: r["validation"]["native"]["mIoU"])["epoch"]
            model.load_state_dict(selected["model"])
            assert b.evaluate(model, val) == selected["validation"]
            del model, optimizer, schedule
            torch.cuda.empty_cache()
            meta.update(phase="cpu_replay", updated_utc=s.utc()); s.write(manifest_path, meta)
            replay = cpu_replay(run / f"{arm}_best.pth", val, frozen)
            s.write(run / f"{arm}_cpu_replay.json", replay)
            assert replay["exact"], "CPU/GPU prediction mismatch; retain and investigate"
            s.write(run / f"{arm}_selected_validation.json", selected["validation"])
            info.update(status="completed", checkpoint_sha256=replay["checkpoint_sha256"],
                        selected_validation_verified=True, independent_cpu_replay_exact=True)
            s.write(manifest_path, meta)
        chosen = {arm: s.read(run / f"{arm}_selected_validation.json") for arm in ARMS}
        s.write(run / "summary.json", {"seed": seed, "frozen": frozen, "body_mean": mean,
                                      "selected": chosen, "test_pixels_read": False})
        meta.update(status="completed", phase="complete", updated_utc=s.utc())
        s.write(manifest_path, meta)
        print(f"SEED_COMPLETED {seed}", flush=True)
    except BaseException as exc:
        meta.update(status="paused" if isinstance(exc, InterruptedError) else "failed", error=repr(exc),
                    traceback=traceback.format_exc(), updated_utc=s.utc())
        s.write(manifest_path, meta)
        raise


def finish():
    first = s.read(PRIOR / "summary.json")
    rows = {"42": {"frozen": first["frozen"], "body_mean": first["body_mean"],
                   "selected": {"separate": first["selected"]["separate"],
                                "pointwise": s.read(PRIOR_CONTROL / "selected_validation.json")}}}
    for seed in SEEDS:
        rows[str(seed)] = s.read(HERE / f"outputs/pair12_seed{seed}/summary.json")
    summary = {"seeds": rows, "aggregates": {}, "test_pixels_read": False,
               "note": "Three paired development seeds, reused same-core validation. Not an independent test, significance test or novelty proof. Coupled-attention screen stays failed."}
    getters = {"mIoU": lambda v: v["mIoU"], "oil_macro_IoU": lambda v: v["oil_macro_IoU"]}
    getters.update({f"class_{i}_IoU": lambda v, i=str(i): v["per_class"][i]["iou"] for i in range(8)})
    for grid in ("native", "canonical512"):
        summary["aggregates"][grid] = {}
        for name, getter in getters.items():
            values = {arm: np.array([getter(r["selected"][arm][grid]) for r in rows.values()]) for arm in ARMS}
            delta = values["separate"] - values["pointwise"]
            summary["aggregates"][grid][name] = {**{arm: {"values": val.tolist(), "mean": float(val.mean()), "sample_sd": float(val.std(ddof=1))} for arm, val in values.items()},
                "paired_delta": delta.tolist(), "delta_mean": float(delta.mean()), "delta_sample_sd": float(delta.std(ddof=1))}
    a = summary["aggregates"]
    gates = {"all_seeds_positive_both_grid_mIoU": all(d > 0 for g in a.values() for d in g["mIoU"]["paired_delta"]),
             "native_mean_mIoU_at_least_0.0025": a["native"]["mIoU"]["delta_mean"] >= .0025,
             "native_mean_oil_macro_at_least_0.004": a["native"]["oil_macro_IoU"]["delta_mean"] >= .004,
             "native_mean_film_nonnegative": a["native"]["class_3_IoU"]["delta_mean"] >= 0,
             "native_mean_corner_nonnegative": a["native"]["class_4_IoU"]["delta_mean"] >= 0,
             "all_separate_over_frozen_at_least_0.005": all(r["selected"]["separate"]["native"]["mIoU"] >= r["frozen"]["native"]["mIoU"] + .005 for r in rows.values())}
    summary.update(gates=gates, engineering_screen_passed=all(gates.values()))
    s.write(HERE / "summary.json", summary)
    lines = ["# Paired body-decoder replication", "", summary["note"], ""]
    for grid in a:
        lines += [f"## {grid}", "", "| Seed | Frozen mIoU (%) | Pointwise (%) | Separate (%) | Difference (pp) |", "|---|---:|---:|---:|---:|"]
        for seed, r in rows.items():
            x, y = (r["selected"][arm][grid]["mIoU"] for arm in ARMS)
            lines.append(f"| {seed} | {100*r['frozen'][grid]['mIoU']:.4f} | {100*y:.4f} | {100*x:.4f} | {100*(x-y):+.4f} |")
        lines += ["", "| Metric | Separate mean +/- sample SD (%) | Pointwise mean +/- sample SD (%) | Mean paired difference (pp) |", "|---|---:|---:|---:|"]
        for name, value in a[grid].items():
            x, y = value["separate"], value["pointwise"]
            lines.append(f"| {name} | {100*x['mean']:.4f} +/- {100*x['sample_sd']:.4f} | {100*y['mean']:.4f} +/- {100*y['sample_sd']:.4f} | {100*value['delta_mean']:+.4f} |")
        lines.append("")
    lines.append(f"Prespecified engineering screen: {summary['engineering_screen_passed']}. No automatic promotion or further experiments.")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run():
    with lock():
        check_stop()
        s.setup()
        ids, config = identity(), settings()
        smoke_report = s.read(HERE / "smoke_report.json")
        assert smoke_report["status"] == "passed" and smoke_report["identity"] == ids and smoke_report["config"] == config
        path = HERE / "run_manifest.json"
        if path.exists():
            meta = s.read(path)
            assert meta["identity"] == ids and meta["config"] == config
            if meta["status"] == "completed":
                print("ALREADY_COMPLETED", flush=True); return
        else:
            snapshot = HERE / "source_snapshot"; snapshot.mkdir(exist_ok=False)
            for i, (source, expected) in enumerate(sorted(ids["sources"].items())):
                assert s.digest(source) == expected
                shutil.copy2(source, snapshot / f"{i:02d}_{Path(source).name}")
            meta = {"created_utc": s.utc(), "identity": ids, "config": config, "command": sys.argv,
                    "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda}}
        try:
            for seed in SEEDS:
                check_stop()
                meta.update(status="running", current_seed=seed, pid=os.getpid(), updated_utc=s.utc())
                s.write(path, meta)
                train_seed(seed, ids, config, smoke_report, meta)
            assert identity() == ids, "Sources/inputs changed during formal run"
            summary = finish()
            meta.update(status="completed", updated_utc=s.utc(), engineering_screen_passed=summary["engineering_screen_passed"])
            s.write(path, meta)
            print("REPLICATION_COMPLETED", summary["gates"], flush=True)
        except BaseException as exc:
            meta.update(status="paused" if isinstance(exc, InterruptedError) else "failed", error=repr(exc),
                        traceback=traceback.format_exc(), updated_utc=s.utc())
            s.write(path, meta)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    args = parser.parse_args()
    smoke() if args.mode == "smoke" else run()
