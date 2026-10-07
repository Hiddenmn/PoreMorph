"""One new global/decoder seed, without modifying the completed experiments."""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import HERE, ORIGINAL, CONTROL, ReplicationDataset, SeedSampler, NativeNoContext, s
from verify_results import verify as verify_pair, check_metrics

GLOBAL_DIR = s.BASELINE / "models/seed_2025_fusion_before_ppm"
CHECKPOINT = GLOBAL_DIR / "best_model.pth"
TOKENIZER = GLOBAL_DIR / "tokenizer.json"
OLD_RUN = CONTROL / "outputs/native_no_context_seed42"
RUN = HERE / "outputs/native_no_context_seed2025"
CACHE = HERE / "coarse_cache_seed2025"
SHARED = ORIGINAL / "cache"


def sources():
    values = s.source_hashes()
    for path in [HERE / n for n in ("run_repeat.py", "data.py", "test_repeat.py", "collect_results.py", "PROTOCOL.md")]:
        values[str(path)] = s.digest(path)
    for path in (CONTROL / "model_no_context.py", ORIGINAL / "verify_results.py"):
        values[str(path)] = s.digest(path)
    return values


def identity():
    return {"sources": sources(), "global_seed": 2025, "checkpoint": s.digest(CHECKPOINT),
            "tokenizer": s.digest(TOKENIZER), "shared_manifest": s.digest(SHARED / "manifest.json"),
            "shared_records": s.digest(SHARED / "records.json"),
            "control_manifest": s.digest(OLD_RUN / "run_manifest.json"),
            "splits": {split: s.digest(s.p.SPLITS / f"{split}_records.csv") for split in ("train", "val", "test")}}


def verify_control():
    verify_pair()
    manifest = s.read(OLD_RUN / "run_manifest.json")
    if manifest["status"] != "completed":
        raise ValueError("Original control incomplete")
    for path, hash_value in manifest["sources"].items():
        if s.digest(path) != hash_value:
            raise ValueError(f"Changed original source: {path}")
    history = s.read(OLD_RUN / "history.json")
    assert [r["epoch"] for r in history] == list(range(1, 13))
    for row in history:
        check_metrics(row["validation"], 676)
    chosen = max(history, key=lambda r: r["validation"]["native"]["mIoU"])
    best = torch.load(OLD_RUN / "best.pth", map_location="cpu", weights_only=False)
    selected = s.read(OLD_RUN / "selected_validation.json")
    assert best["validation"] == chosen["validation"] == selected
    assert best["epoch"] == chosen["epoch"] == manifest["best_epoch"]
    assert s.digest(OLD_RUN / "best.pth") == manifest["checkpoint_sha256"]
    return manifest


def setup():
    s.setup()
    s.p.configure(2025)
    torch.set_num_threads(4)


def check_split_and_rows():
    splits, _ = s.p.load_records()  # Test CSV IDs only, not test image/label pixels.
    if s.p.base.Config.split_mode != "block_per_sample":
        raise ValueError("Archived global model did not use this split mode")
    s.p.base.Config.seed = 2025
    rebuilt = s.p.base.split_records_by_sample(sum((splits[k] for k in ("train", "val", "test")), []))
    for split, rows in zip(("train", "val", "test"), rebuilt):
        assert {(r["sample"], r["slice_id"]) for r in rows} == {(r["sample"], r["slice_id"]) for r in splits[split]}
    shared_records = s.read(SHARED / "records.json")
    lookup = {(r["sample"], r["slice_id"]): r for key in ("train", "val") for r in splits[key]}
    records = []
    for index, record in enumerate(shared_records):
        source = lookup[(record["sample"], record["slice_id"])]
        assert source["cache_index"] == record["cache_index"]
        records.append(dict(record, shared_index=index, text=source["text"]))
    assert len(records) == 3040 and sum(r["split"] == "train" for r in records) == 2364
    return records


def prepare_cache(folder, smoke=False):
    current_identity = identity()
    records = check_split_and_rows()
    if smoke:
        records = ([r for r in records if r["split"] == "train"][:4]
                   + [r for r in records if r["split"] == "val"][:2])
    n = len(records)
    byte_count = n * 8 * 512 * 512 * 2
    if folder.exists():
        meta = s.read(folder / "manifest.json")
        if meta["identity"] != current_identity or meta["smoke"] != smoke:
            raise ValueError("Existing coarse cache provenance differs; preserve it")
        if s.read(folder / "records.json") != records or (folder / "coarse.dat").stat().st_size != byte_count:
            raise ValueError("Coarse cache records or file size differs")
        if meta["status"] == "completed":
            assert meta["records_sha256"] == s.digest(folder / "records.json")
            return meta
        if meta["status"] not in ("preparing", "paused", "failed"):
            raise ValueError("Unknown partial cache status")
    else:
        if shutil.disk_usage(HERE).free < byte_count + 5*2**30:
            raise RuntimeError("Not enough disk space for a separate cache with reserve")
        folder.mkdir(parents=True, exist_ok=False)
        meta = {"status": "preparing", "identity": current_identity, "smoke": smoke, "n": n,
                "shared_pixel_cache": str(SHARED), "completed_records": 0, "test_pixels_read": False,
                "train_count": sum(r["split"] == "train" for r in records),
                "val_count": sum(r["split"] == "val" for r in records), "created_utc": s.utc()}
        s.write(folder / "records.json", records)
        with (folder / "coarse.dat").open("wb") as stream:
            stream.truncate(byte_count)
        s.write(folder / "manifest.json", meta)
    coarse = np.memmap(folder / "coarse.dat", dtype=np.uint16, mode="r+", shape=(n, 8, 512, 512))
    images = np.memmap(SHARED / "low.dat", dtype=np.uint8, mode="r", shape=(3040, 512, 512))
    tokenizer = s.p.base.CharTokenizer.load(TOKENIZER)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    history = checkpoint["history"]
    selected_index = int(np.argmax(history["val_mIoU"]))
    assert checkpoint["checkpoint_type"] == "best"
    assert checkpoint["epoch"] == checkpoint["best_epoch"] == history["epoch"][selected_index] == 59
    assert checkpoint["best_val_miou"] == history["val_mIoU"][selected_index]
    teacher = s.p.order.build_model("fusion_before_ppm", tokenizer.vocab_size)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    teacher = teacher.cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)
    meta.update(status="preparing", pid=os.getpid(), updated_utc=s.utc(), frozen_best_epoch=59)
    s.write(folder / "manifest.json", meta)
    try:
        for start in range(meta["completed_records"], n, 4):
            if (RUN / "STOP_AFTER_EPOCH.txt").exists():
                meta.update(status="paused", updated_utc=s.utc())
                s.write(folder / "manifest.json", meta)
                raise InterruptedError("Explicit pause marker during cache preparation")
            batch = records[start:start+4]
            local = torch.stack([torch.from_numpy(np.array(images[r["shared_index"]], copy=True)).float()
                                 .unsqueeze(0).repeat(3, 1, 1)/255 for r in batch])
            tokens = [tokenizer.encode(r["text"]) for r in batch]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = teacher(local.cuda().to(memory_format=torch.channels_last),
                                 torch.stack([t[0] for t in tokens]).cuda(), torch.stack([t[1] for t in tokens]).cuda())
            if logits.dtype != torch.bfloat16 or not torch.isfinite(logits).all():
                raise ValueError("Unexpected coarse logits")
            encoded = logits.cpu().view(torch.uint16).numpy()
            if not torch.equal(torch.from_numpy(encoded).view(torch.bfloat16).float(), logits.float().cpu()):
                raise ValueError("Coarse cache precision loss")
            coarse[start:start+len(batch)] = encoded
            if start % 100 == 0 or start+len(batch) == n:
                coarse.flush()
                meta.update(completed_records=start+len(batch), updated_utc=s.utc())
                s.write(folder / "manifest.json", meta)
                print(f"COARSE_CACHE {start+len(batch)}/{n}", flush=True)
        coarse.flush()
        meta.update(status="completed", completed_records=n, updated_utc=s.utc(),
                    records_sha256=s.digest(folder / "records.json"))
        s.write(folder / "manifest.json", meta)
        return meta
    except BaseException as exc:
        if not isinstance(exc, InterruptedError):
            meta.update(status="failed", error=repr(exc), updated_utc=s.utc())
            s.write(folder / "manifest.json", meta)
        raise
    finally:
        del teacher, coarse, images
        torch.cuda.empty_cache()


def build():
    s.p.base.set_seed(2025)
    model = NativeNoContext().cuda().to(memory_format=torch.channels_last)
    assert sum(p.numel() for p in model.parameters()) == 118744
    return model


def base_config():
    return dict(s.read(OLD_RUN / "run_manifest.json")["config"], seed=2025, frozen_global_seed=2025,
                protocol="native_detail_repeat12_seed2025", arm="native_no_context")


@contextlib.contextmanager
def locked():
    import msvcrt
    with (HERE / "run.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def smoke():
    setup()
    verify_control()
    folder = HERE / "smoke_cache_seed2025"
    meta = prepare_cache(folder, smoke=True)
    model = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    data = ReplicationDataset(folder, "train", True)
    batch = next(iter(DataLoader(data, batch_size=4, num_workers=0)))
    image, coarse, context, target = s.local_batch(batch, "native")
    weights = torch.tensor(base_config()["class_weights"], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = s.loss_function(model(image, coarse, context), target, weights)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite smoke loss")
    loss.backward()
    for prefix in ("fine", "middle", "deep", "up_middle", "up_fine", "output"):
        assert any(t.grad is not None and bool(t.grad.abs().sum()>0) for n,t in model.named_parameters() if n.startswith(prefix))
    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    result = s.evaluate(model, DataLoader(ReplicationDataset(folder, "val", False), batch_size=1), "native")
    check_metrics(result, meta["val_count"])
    report = {"status": "passed", "identity": identity(), "config": base_config(), "loss": float(loss.detach()),
              "parameters": 118744, "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
              "smoke_weights_reused": False, "validation_images": meta["val_count"]}
    s.write(HERE / "smoke_report.json", report)
    print("SMOKE_PASSED", json.dumps({k:report[k] for k in ("loss", "parameters", "peak_allocated_MiB")}), flush=True)


def finish(frozen, selected):
    gates = {"native_improvement_at_least_0.005": selected["native"]["mIoU"] >= frozen["native"]["mIoU"]+.005,
             "canonical_improvement_at_least_0.005": selected["canonical512"]["mIoU"] >= frozen["canonical512"]["mIoU"]+.005,
             "oil_macro_no_regression": selected["native"]["oil_macro_IoU"] >= frozen["native"]["oil_macro_IoU"]}
    for i in ("3", "4"):
        gates[f"class_{i}_no_regression"] = selected["native"]["per_class"][i]["iou"] >= frozen["native"]["per_class"][i]["iou"]
    result = {"global_seed": 2025, "decoder_seed": 2025, "frozen": frozen, "native_no_context": selected,
              "gates": gates, "passed": all(gates.values()), "test_pixels_read": False,
              "main_model_promoted": False, "scope": "same-split validation; combined native-input and decoder-training change"}
    s.write(RUN / "summary.json", result)
    lines = ["# 原生细节简化模型：seed 2025", "", "全局模型与细节解码器均为 seed 2025；只在固定验证集上评价。", ""]
    for grid in ("canonical512", "native"):
        lines += [f"## {grid}", "", "| 指标 | 冻结全局模型 | 加入原生细节解码器 | 差值（百分点） |", "|---|---:|---:|---:|"]
        specs = [("mIoU",lambda m:m["mIoU"]),("油相宏平均 IoU",lambda m:m["oil_macro_IoU"])]
        specs += [(selected[grid]["per_class"][str(i)]["name"],lambda m,i=i:m["per_class"][str(i)]["iou"]) for i in range(8)]
        for label, get in specs:
            a,b = get(frozen[grid])*100,get(selected[grid])*100
            lines.append(f"| {label} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
        lines.append("")
    lines += [f"筛选：{'通过' if result['passed'] else '未通过'}。", "",
              "增益同时包含原生输入、新增解码器与额外训练，不能全部归因于架构。",
              "第三随机种子不是独立岩心/外部泛化证据，且验证集已反复使用；未提升为论文主模型。", ""]
    (RUN / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def train():
    setup()
    verify_control()
    ids, config = identity(), base_config()
    smoke_report = s.read(HERE / "smoke_report.json")
    if smoke_report["status"] != "passed" or smoke_report["identity"] != ids or smoke_report["config"] != config:
        raise RuntimeError("Smoke no longer matches current sources")
    with locked():
        if RUN.exists():
            manifest = s.read(RUN / "run_manifest.json")
            if manifest["identity"] != ids or manifest["config"] != config:
                raise RuntimeError("Run provenance changed; refusing resume")
            if manifest["status"] == "completed":
                raise RuntimeError("Run already complete")
        else:
            RUN.mkdir(parents=True, exist_ok=False)
            manifest = {"status": "initializing", "created_utc": s.utc(), "identity": ids, "config": config,
                        "command": [sys.executable,*sys.argv], "environment": {"python":sys.version,"torch":torch.__version__,
                        "numpy":np.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name()}}
            snapshot = RUN / "source_snapshot"
            snapshot.mkdir()
            for i,path in enumerate(sources()):
                shutil.copy2(path,snapshot/f"{i:02d}_{Path(path).name}")
            s.write(RUN / "run_manifest.json",manifest)
        if (RUN / "STOP_AFTER_EPOCH.txt").exists():
            raise RuntimeError("Explicit user pause marker is present")
        try:
            manifest.update(status="running",phase="cache",pid=os.getpid(),updated_utc=s.utc())
            s.write(RUN / "run_manifest.json",manifest)
            prepare_cache(CACHE)
            train_data = ReplicationDataset(CACHE,"train",True)
            sampler = SeedSampler(train_data)
            loader = DataLoader(train_data,batch_size=4,sampler=sampler,num_workers=2,persistent_workers=True,
                                prefetch_factor=2,pin_memory=True)
            val = DataLoader(ReplicationDataset(CACHE,"val",False),batch_size=1,num_workers=0)
            if not (RUN / "frozen_validation.json").exists():
                s.write(RUN / "frozen_validation.json",s.evaluate(None,val,"native",frozen=True))
            frozen = s.read(RUN / "frozen_validation.json")
            check_metrics(frozen,676)
            # Minor BF16 batch-size differences are expected; a large discrepancy
            # against the archived seed-2025 validation indicates wrong data/model.
            if abs(frozen["canonical512"]["mIoU"] - .760878729563444) > .0002:
                raise ValueError("Frozen reference differs materially from archived validation")
            model = build()
            manifest["full_initial_sha256"] = model.full_initial_sha256
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12)
            history,best,best_epoch = [],-1.,None
            if (RUN / "last.pth").exists():
                saved = torch.load(RUN / "last.pth",map_location="cpu",weights_only=False)
                if saved["identity"] != ids or saved["config"] != config:
                    raise ValueError("Checkpoint provenance mismatch")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                history,best,best_epoch = saved["history"],saved["best"],saved["best_epoch"]
                random.setstate(saved["rng"]["python"])
                np.random.set_state(saved["rng"]["numpy"])
                torch.set_rng_state(saved["rng"]["torch"])
                torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
                del saved
            weights = torch.tensor(config["class_weights"],device="cuda")
            for epoch in range(len(history)+1,13):
                if (RUN / "STOP_AFTER_EPOCH.txt").exists():
                    raise InterruptedError("Pause requested at epoch boundary")
                start = time.monotonic()
                sampler.epoch = epoch
                model.train()
                manifest.update(current_epoch=epoch,phase="train",updated_utc=s.utc())
                s.write(RUN / "run_manifest.json",manifest)
                print(f"EPOCH_START seed2025 {epoch}/12",flush=True)
                total = 0.
                for batch in loader:
                    image,coarse,context,target = s.local_batch(batch,"native")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        loss = s.loss_function(model(image,coarse,context),target,weights)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach())
                scheduler.step()
                manifest.update(phase="validation",updated_utc=s.utc())
                s.write(RUN / "run_manifest.json",manifest)
                score = s.evaluate(model,val,"native")
                check_metrics(score,676)
                row = {"epoch":epoch,"loss":total/len(loader),"seconds":time.monotonic()-start,"validation":score}
                history.append(row)
                if score["native"]["mIoU"] > best:
                    best,best_epoch = score["native"]["mIoU"],epoch
                    s.save(RUN / "best.pth",{"model":model.state_dict(),"epoch":epoch,"validation":score})
                s.save(RUN / "last.pth",{"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                       "scheduler":scheduler.state_dict(),"history":history,"best":best,"best_epoch":best_epoch,
                       "identity":ids,"config":config,"rng":{"python":random.getstate(),"numpy":np.random.get_state(),
                       "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}})
                s.write(RUN / "history.json",history)
                manifest.update(completed_epochs=epoch,best_epoch=best_epoch,best_native_mIoU=best,
                                phase="epoch_complete",updated_utc=s.utc())
                s.write(RUN / "run_manifest.json",manifest)
                print(f"EPOCH_DONE {epoch}/12 native_mIoU={score['native']['mIoU']:.8f} canonical_mIoU={score['canonical512']['mIoU']:.8f} seconds={row['seconds']:.1f}",flush=True)
            chosen = torch.load(RUN / "best.pth",map_location="cpu",weights_only=False)
            model.load_state_dict(chosen["model"])
            selected = s.evaluate(model,val,"native")
            if selected != chosen["validation"]:
                raise ValueError("Reloaded best validation mismatch")
            s.write(RUN / "selected_validation.json",selected)
            result = finish(frozen,selected)
            manifest.update(status="completed",phase="complete",updated_utc=s.utc(),selected_validation_verified=True,
                            checkpoint_sha256=s.digest(RUN / "best.pth"),screening=result["gates"])
            s.write(RUN / "run_manifest.json",manifest)
            print("REPLICATION_COMPLETED",json.dumps(result["gates"]),flush=True)
            from collect_results import collect
            collect()
        except BaseException as exc:
            manifest.update(status="paused" if isinstance(exc,InterruptedError) else "failed",
                            error=repr(exc),traceback=traceback.format_exc(),updated_utc=s.utc())
            s.write(RUN / "run_manifest.json",manifest)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("smoke","run"))
    args = parser.parse_args()
    smoke() if args.mode == "smoke" else train()
