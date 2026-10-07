"""One fixed native-input ablation; imports the unchanged matched run utilities."""
from __future__ import annotations
import argparse
import contextlib
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

from model_no_context import NativeNoContext, ORIGINAL, tensor_digest
import run_study as s
from verify_results import verify

HERE = Path(__file__).resolve().parent
RUN = HERE / "outputs/native_no_context_seed42"
REFERENCE = ORIGINAL / "outputs/paired12_seed42"
CACHE = ORIGINAL / "cache"


def sources():
    result = s.source_hashes()
    paths = [HERE / n for n in ("run_control.py", "model_no_context.py", "test_control.py", "PROTOCOL.md")]
    paths += [ORIGINAL / "verify_results.py"]
    result.update({str(p): s.digest(p) for p in paths})
    return result


def evidence():
    if not (CACHE / "manifest.json").is_file():
        raise RuntimeError("Completed cache required; this control must not rebuild it")
    report = verify()
    if not report["passed"]:
        raise RuntimeError("Original pair did not pass screening")
    reference = s.read(REFERENCE / "run_manifest.json")
    config = dict(reference["config"], arm="native_no_context", protocol="native_detail_paired12_seed42",
                  frozen_global_seed=42, trainable_parameters=118744,
                  no_context_rule="remove gate and projection; retain coarse logits and native input")
    links = {str(REFERENCE / n): s.digest(REFERENCE / n) for n in
             ("run_manifest.json", "summary.json", "native_best.pth", "native_selected_validation.json")}
    links[str(CACHE / "manifest.json")] = s.digest(CACHE / "manifest.json")
    return reference, config, links


def build(reference):
    s.p.base.set_seed(42)
    model = NativeNoContext()
    if model.full_initial_sha256 != reference["arms"]["native"]["initial_sha256"]:
        raise ValueError("Shared initialization does not match the completed full arm")
    if sum(t.numel() for t in model.parameters()) != 118744:
        raise ValueError("Unexpected parameter count")
    return model.cuda().to(memory_format=torch.channels_last)


@contextlib.contextmanager
def run_lock():
    # Windows byte-range locks are released by the OS on process exit; the
    # persistent sentinel is harmless and never needs deletion for recovery.
    import msvcrt
    with (HERE / "formal_run.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another control process holds the run lock") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def smoke():
    s.setup()
    reference, config, links = evidence()
    folder = ORIGINAL / "smoke_cache_v2"
    if not (folder / "manifest.json").is_file():
        raise RuntimeError("Existing smoke cache required")
    meta = s.prepare_cache(folder, smoke=True)
    loader = DataLoader(s.CachedDataset(folder, "train", True), batch_size=4, num_workers=0)
    batch = next(iter(loader))
    model = build(reference)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    image, coarse, context, target = s.local_batch(batch, "native")
    weights = torch.tensor(config["class_weights"], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(image, coarse, context)
        loss = s.loss_function(result, target, weights)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite smoke loss")
    loss.backward()
    for prefix in ("fine", "middle", "deep", "up_middle", "up_fine", "output"):
        if not any(t.grad is not None and bool(t.grad.abs().sum() > 0)
                   for n, t in model.named_parameters() if n.startswith(prefix)):
            raise ValueError(f"No gradient in {prefix}")
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    val = DataLoader(s.CachedDataset(folder, "val", False), batch_size=1, num_workers=0)
    value = s.evaluate(model, val, "native")
    report = {"status": "passed", "sources": sources(), "config": config, "reference_evidence": links,
              "loss": float(loss.detach()), "gradient_norm": float(norm), "validation_images": meta["val_count"],
              "native_forward_shape": [1, 8, 1024, 1024], "parameters": 118744,
              "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
              "finite_validation": bool(np.isfinite(value["native"]["mIoU"])), "smoke_weights_reused": False}
    if not report["finite_validation"]:
        raise ValueError("Nonfinite smoke validation")
    s.write(HERE / "smoke_report.json", report)
    print("SMOKE_PASSED", json.dumps({k: report[k] for k in ("loss", "parameters", "peak_allocated_MiB")}), flush=True)


def finish(selected):
    full = s.read(REFERENCE / "native_selected_validation.json")
    f, c = full["native"], selected["native"]
    gates = {"full_minus_control_native_mIoU_at_least_0.0025": f["mIoU"] - c["mIoU"] >= .0025,
             "full_oil_macro_no_regression": f["oil_macro_IoU"] >= c["oil_macro_IoU"]}
    for key in ("3", "4"):
        gates[f"full_class_{key}_no_regression"] = f["per_class"][key]["iou"] >= c["per_class"][key]["iou"]
    result = {"full_native": full, "no_context": selected, "gates": gates,
              "feature_path_screen_passed": all(gates.values()), "single_seed_development": True,
              "test_pixels_read": False, "full_params": 127064, "control_params": 118744,
              "main_model_promoted": False}
    s.write(RUN / "summary.json", result)
    lines = ["# 原生细节解码器：去全局特征路径对照", "", "seed 42；各 12 轮；均按原生验证 mIoU 选择检查点。单位：百分比，差值：百分点。", ""]
    for grid, title in (("canonical512", "统一 512 网格"), ("native", "原生 1024 网格")):
        lines += [f"## {title}", "", "| 指标 | 全结构 | 去特征路径 | 全结构−对照 |", "|---|---:|---:|---:|"]
        specs = [("整体 mIoU", lambda m: m["mIoU"]), ("整体 mDice", lambda m: m["mDice"]),
                 ("五类油相宏平均 IoU", lambda m: m["oil_macro_IoU"])]
        specs += [(full[grid]["per_class"][str(i)]["name"], lambda m, i=i: m["per_class"][str(i)]["iou"]) for i in range(8)]
        for label, get in specs:
            a, b = get(full[grid]) * 100, get(selected[grid]) * 100
            lines.append(f"| {label} | {a:.4f} | {b:.4f} | {a-b:+.4f} |")
        lines.append("")
    lines += ["## 结论边界", "", f"预先设定的特征路径筛选：{'通过' if result['feature_path_screen_passed'] else '未通过'}。",
              "两组都保留原生图像与多模态粗预测；这不是图像单模态对照。",
              "全结构 127064、对照 118744 个可训练参数，属于结构消融而非等参数机制证明。",
              "单种子、反复使用的岩心内验证集；不能据此宣称独立泛化、方法新颖性或直接替换论文主模型。",
              "若筛选未过，不继续宣称该特征路径必需，保留原生细节路线及更简单结构的候选资格。", ""]
    (RUN / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def train():
    s.setup()
    reference, config, links = evidence()
    smoke_result = s.read(HERE / "smoke_report.json")
    current_sources = sources()
    if (smoke_result["status"] != "passed" or smoke_result["sources"] != current_sources
            or smoke_result["config"] != config or smoke_result["reference_evidence"] != links):
        raise RuntimeError("Current sources have not passed smoke validation")
    with run_lock():
        if RUN.exists():
            manifest = s.read(RUN / "run_manifest.json")
            if (manifest["sources"] != current_sources or manifest["config"] != config
                    or manifest["reference_evidence"] != links):
                raise RuntimeError("Source/config/reference identity changed; refusing resume")
            if manifest["status"] == "completed":
                raise RuntimeError("This control is already completed")
        else:
            RUN.mkdir(parents=True, exist_ok=False)
            manifest = {"status": "initializing", "created_utc": s.utc(), "sources": current_sources,
                        "config": config, "cache_identity": reference["cache_identity"], "reference_evidence": links,
                        "command": [sys.executable, *sys.argv],
                        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}}
            snapshot = RUN / "source_snapshot"
            snapshot.mkdir()
            for number, path in enumerate(current_sources):
                shutil.copy2(path, snapshot / f"{number:02d}_{Path(path).name}")
            s.write(RUN / "run_manifest.json", manifest)
        if (RUN / "STOP_AFTER_EPOCH.txt").exists():
            raise RuntimeError("Explicit pause marker is present")
        try:
            model = build(reference)
            manifest.update(status="running", pid=os.getpid(), updated_utc=s.utc(),
                            full_initial_sha256=model.full_initial_sha256,
                            shared_initial_sha256=tensor_digest(model.state_dict()))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
            history, best, best_epoch = [], -1., None
            if (RUN / "last.pth").exists():
                last = torch.load(RUN / "last.pth", map_location="cpu", weights_only=False)
                if last["sources"] != current_sources or last["config"] != config or last["reference_evidence"] != links:
                    raise ValueError("Checkpoint provenance changed")
                model.load_state_dict(last["model"])
                optimizer.load_state_dict(last["optimizer"])
                scheduler.load_state_dict(last["scheduler"])
                history, best, best_epoch = last["history"], last["best"], last["best_epoch"]
                random.setstate(last["rng"]["python"])
                np.random.set_state(last["rng"]["numpy"])
                torch.set_rng_state(last["rng"]["torch"])
                torch.cuda.set_rng_state_all(last["rng"]["cuda"])
                del last
            s.write(RUN / "run_manifest.json", manifest)
            dataset = s.CachedDataset(CACHE, "train", True)
            sampler = s.EpochSampler(dataset)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2,
                                persistent_workers=True, prefetch_factor=2, pin_memory=True)
            val = DataLoader(s.CachedDataset(CACHE, "val", False), batch_size=1, num_workers=0)
            weights = torch.tensor(config["class_weights"], device="cuda")
            for epoch in range(len(history)+1, 13):
                if (RUN / "STOP_AFTER_EPOCH.txt").exists():
                    manifest.update(status="paused", phase="epoch_boundary", updated_utc=s.utc())
                    s.write(RUN / "run_manifest.json", manifest)
                    return
                start = time.monotonic()
                sampler.epoch = epoch
                model.train()
                manifest.update(current_epoch=epoch, phase="train", updated_utc=s.utc())
                s.write(RUN / "run_manifest.json", manifest)
                print(f"EPOCH_START native_no_context {epoch}/12", flush=True)
                total = 0.
                for batch in loader:
                    image, coarse, context, target = s.local_batch(batch, "native")
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(image, coarse, context)
                        loss = s.loss_function(logits, target, weights)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach())
                scheduler.step()
                manifest.update(phase="validation", updated_utc=s.utc())
                s.write(RUN / "run_manifest.json", manifest)
                score = s.evaluate(model, val, "native")
                row = {"epoch": epoch, "loss": total/len(loader), "seconds": time.monotonic()-start, "validation": score}
                history.append(row)
                if score["native"]["mIoU"] > best:
                    best, best_epoch = score["native"]["mIoU"], epoch
                    s.save(RUN / "best.pth", {"model": model.state_dict(), "epoch": epoch, "validation": score})
                s.save(RUN / "last.pth", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "history": history, "best": best, "best_epoch": best_epoch,
                       "sources": current_sources, "config": config, "reference_evidence": links,
                       "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}})
                s.write(RUN / "history.json", history)
                manifest.update(completed_epochs=epoch, best_epoch=best_epoch, best_native_mIoU=best,
                                phase="epoch_complete", updated_utc=s.utc())
                s.write(RUN / "run_manifest.json", manifest)
                print(f"EPOCH_DONE {epoch}/12 native_mIoU={score['native']['mIoU']:.8f} canonical_mIoU={score['canonical512']['mIoU']:.8f} seconds={row['seconds']:.1f}", flush=True)
            chosen = torch.load(RUN / "best.pth", map_location="cpu", weights_only=False)
            model.load_state_dict(chosen["model"])
            selected = s.evaluate(model, val, "native")
            if selected != chosen["validation"]:
                raise ValueError("Reloaded best checkpoint validation differs")
            s.write(RUN / "selected_validation.json", selected)
            result = finish(selected)
            manifest.update(status="completed", phase="complete", updated_utc=s.utc(),
                            selected_validation_verified=True, checkpoint_sha256=s.digest(RUN / "best.pth"),
                            screening=result["gates"])
            s.write(RUN / "run_manifest.json", manifest)
            print("CONTROL_COMPLETED", json.dumps(result["gates"]), flush=True)
        except BaseException as exc:
            manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), updated_utc=s.utc())
            s.write(RUN / "run_manifest.json", manifest)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "run"))
    args = parser.parse_args()
    smoke() if args.mode == "smoke" else train()
