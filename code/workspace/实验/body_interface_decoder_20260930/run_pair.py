"""One whole-body/interface pair; immutable frozen images, no test loader."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from model import BodyInterfaceDecoder, augment, body_loss, POINTS, TYPES, FEATURES, SUMMARY
from representation import describe, targets, aggregate

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent / "native_detail_context_control_20260930"))
from model_no_context import NativeNoContext
sys.path.insert(0,str(HERE.parent / "native_detail_decoder_20260930"))
import run_study as s

REFERENCE = HERE.parent / "native_detail_context_control_20260930/outputs/native_no_context_seed42"
PIXELS = s.HERE / "cache"
CACHE = HERE / "cache"
RUN = HERE / "outputs/pair12_seed42"
STOP = HERE / "STOP_AFTER_EPOCH.txt"
ARMS = ("separate","coupled")
LAYOUT = {"points":(np.float16,(TYPES*POINTS,FEATURES)),"valid":(np.uint8,(TYPES*POINTS,)),
          "summary":(np.float32,(SUMMARY,)),"native_counts":(np.uint32,(8,)),
          "canonical512_counts":(np.uint32,(8,))}


def sources():
    files = [HERE/n for n in ("model.py","representation.py","run_pair.py","test_model.py","PROTOCOL.md")]
    files += [Path(sys.modules["model_no_context"].__file__),Path(sys.modules["architecture"].__file__)]
    files += [Path(p) for p in s.source_hashes()]
    return {str(p):s.digest(p) for p in set(files)}


def identity():
    return {"sources":sources(),"native_checkpoint":s.digest(REFERENCE/"best.pth"),
            "native_manifest":s.digest(REFERENCE/"run_manifest.json"),
            "pixels_manifest":s.digest(PIXELS/"manifest.json"),"pixels_records":s.digest(PIXELS/"records.json")}


def settings():
    return {"seed":42,"epochs":12,"body_batch_size":512,"lr":.001,"weight_decay":.0001,
            "loss":"0.5 conditional weighted CE + 0.5 weighted oil Dice from pixel counts",
            "class_weights":s.read(REFERENCE/"run_manifest.json")["config"]["class_weights"][3:],
            "selection":"first maximum native1024 validation eight-class mIoU",
            "point_groups":["body","oil-rock","oil-water"],"points_per_group":POINTS,
            "feature_width":FEATURES,"hidden":32,"heads":4,"layers":2,"gpu_cap":.35,
            "test_pixels_read":False,"protocol":"body_interface_pair12_seed42_v1"}


def verify_reference():
    s.prepare_cache(PIXELS)
    ref = s.read(REFERENCE/"run_manifest.json")
    assert ref["status"] == "completed" and ref["selected_validation_verified"]
    assert s.digest(REFERENCE/"best.pth") == ref["checkpoint_sha256"]
    for p,h in ref["sources"].items():
        assert s.digest(p) == h,p
    old = torch.load(REFERENCE/"best.pth",map_location="cpu",weights_only=False)
    assert old["epoch"] == ref["best_epoch"] == 11
    assert old["validation"] == s.read(REFERENCE/"selected_validation.json")
    return old


def state_digest(state):
    return hashlib.sha256(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in state.values())).hexdigest()


def build(mode):
    s.p.base.set_seed(42)
    return BodyInterfaceDecoder(mode).cuda()


def check_stop():
    if STOP.exists():
        raise InterruptedError("User pause marker; never remove automatically")


@contextlib.contextmanager
def lock():
    import msvcrt
    with (HERE/"run.lock").open("a+b") as handle:
        handle.seek(0,os.SEEK_END)
        if handle.tell()==0:
            handle.write(b"\0"); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def save_npz(path, values):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez(stream,**values)
    temp.replace(path)


@torch.inference_mode()
def prepare(folder, smoke=False):
    ids = identity()
    old = verify_reference()
    records = s.read(PIXELS/"records.json")
    if smoke:
        chosen = [i for i,r in enumerate(records) if r["split"]=="train"][:2] + \
                 [i for i,r in enumerate(records) if r["split"]=="val"][:2]
    else:
        chosen = list(range(len(records)))
    assert all(records[i]["split"] in ("train","val") for i in chosen)
    if folder.exists():
        meta = s.read(folder/"manifest.json")
        assert meta["identity"] == ids and meta["smoke"] == smoke
        if meta["status"] == "completed":
            for name,(dtype,shape) in LAYOUT.items():
                arr = np.load(folder/f"{name}.npy",mmap_mode="r")
                assert arr.shape == (meta["n"],*shape) and arr.dtype == dtype
            assert s.digest(folder/"records.json") == meta["records_sha256"]
            return meta
    else:
        folder.mkdir(parents=True,exist_ok=False)
        (folder/"shards").mkdir()
        meta = {"status":"preparing","identity":ids,"smoke":smoke,"selected_rows":chosen,
                "created_utc":s.utc(),"completed_images":0,"test_pixels_read":False}
        s.write(folder/"manifest.json",meta)
    assert meta["selected_rows"] == chosen
    model = NativeNoContext().eval().requires_grad_(False)
    model.load_state_dict(old["model"],strict=True)
    model = model.cuda().to(memory_format=torch.channels_last)
    feature_box = {}
    hook = model.up_fine.register_forward_hook(lambda m,a,o:feature_box.update(feature=o))
    pixel_meta = s.read(PIXELS/"manifest.json")
    arrays = {k:np.memmap(PIXELS/f"{k}.dat",mode="r",dtype=dtype,shape=(pixel_meta["n"],*shape))
              for k,(dtype,shape) in s.LAYOUT.items() if k in ("image","mask","coarse")}
    entries = []
    started = time.monotonic()
    try:
        for ordinal,row in enumerate(chosen):
            check_stop()
            record = records[row]
            path = folder/"shards"/f"{row:04d}.npz"
            if path.exists():
                # Each shard is atomically complete. Only identical locked
                # sources/checkpoint/cache may resume the next missing image.
                with np.load(path,allow_pickle=False) as saved:
                    assert int(saved["row"]) == row
                    n = len(saved["points"])
            else:
                im = np.array(arrays["image"][row],copy=True).astype(np.float32)/255
                image = torch.from_numpy(im).cuda()[None,None]
                coarse = torch.from_numpy(np.array(arrays["coarse"][row],copy=True)).view(torch.bfloat16).cuda()[None]
                coarse = s.sample_native_grid(coarse,torch.zeros((1,2),device="cuda"),1024)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    logits = model(image,coarse,None)
                probs = logits.float().softmax(1)[0].cpu().numpy()
                fine = feature_box.pop("feature").float()[0].cpu().numpy()
                data,labels,pred = describe(im,probs,fine)
                gt = np.array(arrays["mask"][row],copy=True)
                counts,outside = targets(labels,pred,gt)
                frozen = {grid:np.bincount((t.astype(np.int64)*8+p).ravel(),minlength=64).reshape(8,8)
                          for grid,p,t in (("native",pred,gt),("canonical512",pred[1::2,1::2],gt[1::2,1::2]))}
                values = {**data,"row":np.array(row,dtype=np.int64)}
                for grid in counts:
                    values[f"{grid}_counts"] = counts[grid]
                    values[f"{grid}_outside"] = outside[grid]
                    values[f"{grid}_frozen"] = frozen[grid]
                n = len(data["points"])
                save_npz(path,values)
            entries.append({"row":row,"sample":record["sample"],"slice_id":record["slice_id"],
                            "split":record["split"],"bodies":n,"shard_sha256":s.digest(path)})
            if (ordinal+1)%100==0 or ordinal+1==len(chosen):
                meta.update(completed_images=ordinal+1,updated_utc=s.utc(),elapsed_seconds=time.monotonic()-started)
                s.write(folder/"manifest.json",meta)
                print(f"CACHE {ordinal+1}/{len(chosen)} images; seconds={time.monotonic()-started:.1f}",flush=True)
    finally:
        hook.remove()
        del model,arrays
        torch.cuda.empty_cache()
    check_stop()
    n = sum(e["bodies"] for e in entries)
    total_arrays = {k:np.lib.format.open_memmap(folder/f"{k}.npy",mode="w+",dtype=dtype,shape=(n,*shape))
                    for k,(dtype,shape) in LAYOUT.items()}
    outside = {split:{g:np.zeros((8,8),np.int64) for g in ("native","canonical512")} for split in ("train","val")}
    frozen = {split:{g:np.zeros((8,8),np.int64) for g in ("native","canonical512")} for split in ("train","val")}
    start = 0
    for entry in entries:
        check_stop()
        end = start+entry["bodies"]
        entry.update(start=start,end=end)
        path = folder/"shards"/f"{entry['row']:04d}.npz"
        assert s.digest(path) == entry["shard_sha256"]
        with np.load(path,allow_pickle=False) as saved:
            for key in total_arrays:
                total_arrays[key][start:end] = saved[key]
            for grid in ("native","canonical512"):
                outside[entry["split"]][grid] += saved[f"{grid}_outside"]
                frozen[entry["split"]][grid] += saved[f"{grid}_frozen"]
        start = end
    for arr in total_arrays.values():
        arr.flush()
    del total_arrays
    original = {g:s.metrics(h) for g,h in frozen["val"].items()}
    if not smoke:
        assert original == old["validation"],"Frozen validation replay drift; do not train"
    s.write(folder/"records.json",entries)
    s.write(folder/"outside.json",{split:{g:h.tolist() for g,h in values.items()} for split,values in outside.items()})
    s.write(folder/"frozen_validation.json",original)
    meta.update(status="completed",n=n,records_sha256=s.digest(folder/"records.json"),
                outside_sha256=s.digest(folder/"outside.json"),updated_utc=s.utc(),
                array_sha256={key:s.digest(folder/f"{key}.npy") for key in LAYOUT},
                frozen_validation_replayed=not smoke,
                images_by_split={split:sum(e["split"]==split for e in entries) for split in ("train","val")},
                bodies_by_split={split:sum(e["bodies"] for e in entries if e["split"]==split) for split in ("train","val")})
    s.write(folder/"manifest.json",meta)
    return meta


class CachedBodies:
    def __init__(self,folder,split):
        self.folder = folder
        self.meta = s.read(folder/"manifest.json")
        assert self.meta["status"] == "completed"
        entries = s.read(folder/"records.json")
        self.indices = np.concatenate([np.arange(e["start"],e["end"],dtype=np.int64) for e in entries if e["split"]==split])
        self.arrays = {k:np.load(folder/f"{k}.npy",mmap_mode="r") for k in LAYOUT}
        self.outside = s.read(folder/"outside.json")[split]
        self.image_count = self.meta["images_by_split"][split]

    def input_batch(self,index):
        return (torch.from_numpy(np.array(self.arrays["points"][index],copy=True)).to("cuda",dtype=torch.float32),
                torch.from_numpy(np.array(self.arrays["valid"][index],copy=True)).to("cuda",dtype=torch.bool),
                torch.from_numpy(np.array(self.arrays["summary"][index],copy=True)).to("cuda",dtype=torch.float32))


@torch.inference_mode()
def evaluate(model,data):
    if model is not None:
        model.eval()
    matrices = {g:np.array(h,dtype=np.int64) for g,h in data.outside.items()}
    for start in range(0,len(data.indices),512):
        idx = data.indices[start:start+512]
        if model is None:
            assigned = np.asarray(data.arrays["summary"][idx,3:8]).argmax(1)
        else:
            assigned = model(*data.input_batch(idx)).argmax(1).cpu().numpy()
        for grid in matrices:
            matrices[grid] = aggregate(data.arrays[f"{grid}_counts"][idx],assigned,matrices[grid])
    for grid,h in matrices.items():
        size = 1024 if grid=="native" else 512
        assert int(h.sum()) == data.image_count*size*size
    return {g:s.metrics(h) for g,h in matrices.items()}


def tests():
    suite = unittest.defaultTestLoader.discover(str(HERE),pattern="test_model.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("CPU contracts failed")
    return result.testsRun


def smoke():
    count = tests()
    s.setup()
    folder = HERE/"smoke_cache"
    prepare(folder,smoke=True)
    data,val = CachedBodies(folder,"train"),CachedBodies(folder,"val")
    weights = torch.tensor(settings()["class_weights"],device="cuda")
    # Repeating smoke bodies exercises the actual 512-body memory batch; these
    # weights are discarded and never become formal initialisation.
    idx = np.resize(data.indices,512)
    p,v,g = data.input_batch(idx)
    labels = torch.from_numpy(np.array(data.arrays["native_counts"][idx],copy=True)).cuda().float()
    reports,hashes = [],[]
    for mode in ARMS:
        model = build(mode).train()
        hashes.append(state_digest(model.state_dict()))
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        augmented = augment(p,torch.Generator(device="cuda").manual_seed(42001))
        loss = body_loss(model(augmented,v,g),labels,weights)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        score = evaluate(model,val)
        reports.append({"arm":mode,"trainable_parameters":sum(x.numel() for x in model.parameters()),
                        "batch512_loss":float(loss.detach()),"gradient_norm":float(norm),
                        "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
                        "smoke_validation_mIoU":score["native"]["mIoU"],"seconds":time.monotonic()-started})
        del model,optimizer,loss
        torch.cuda.empty_cache()
    assert len(set(hashes))==1
    s.write(HERE/"smoke_report.json",{"status":"passed","identity":identity(),"config":settings(),
            "cpu_tests":count,"initial_sha256":hashes[0],"reports":reports,"smoke_weights_reused":False})
    print("SMOKE_PASSED",json.dumps(reports),flush=True)


def finish(frozen,mean,chosen):
    a,b = chosen["separate"],chosen["coupled"]
    gates = {"native_coupled_over_separate_0.0025":b["native"]["mIoU"]>=a["native"]["mIoU"]+.0025,
             "native_oil_macro_over_separate_0.004":b["native"]["oil_macro_IoU"]>=a["native"]["oil_macro_IoU"]+.004,
             "native_over_body_mean_0.0025":b["native"]["mIoU"]>=mean["native"]["mIoU"]+.0025,
             "native_over_frozen_0.005":b["native"]["mIoU"]>=frozen["native"]["mIoU"]+.005,
             "canonical_over_frozen_0.005":b["canonical512"]["mIoU"]>=frozen["canonical512"]["mIoU"]+.005,
             "native_oil_macro_no_frozen_regression":b["native"]["oil_macro_IoU"]>=frozen["native"]["oil_macro_IoU"]}
    for c in ("3","4"):
        gates[f"class{c}_native_over_separate_0.005"] = b["native"]["per_class"][c]["iou"]>=a["native"]["per_class"][c]["iou"]+.005
        gates[f"class{c}_native_no_frozen_regression"] = b["native"]["per_class"][c]["iou"]>=frozen["native"]["per_class"][c]["iou"]
    summary = {"frozen":frozen,"body_mean":mean,"selected":chosen,"gates":gates,"passed":all(gates.values()),
               "test_pixels_read":False,"note":"Exploratory seed42 same-core validation, not an independent test or novelty proof"}
    s.write(RUN/"summary.json",summary)
    lines = ["# Whole-body/interface decoder: seed42",""]
    for grid in ("canonical512","native"):
        lines += [f"## {grid}","","| Metric | Frozen | Body mean | Separate | Coupled | Coupled - separate (pp) |",
                  "|---|---:|---:|---:|---:|---:|"]
        columns = [frozen[grid],mean[grid],a[grid],b[grid]]
        for name,get in [("mIoU",lambda m:m["mIoU"]),("Oil macro",lambda m:m["oil_macro_IoU"])]+[(str(c),lambda m,c=str(c):m["per_class"][c]["iou"]) for c in range(8)]:
            vals = [get(x)*100 for x in columns]
            lines.append(f"| {name} | "+" | ".join(f"{x:.4f}" for x in vals)+f" | {vals[3]-vals[2]:+.4f} |")
        lines.append("")
    lines += [f"Prespecified screen: {'passed' if summary['passed'] else 'failed'}.","",summary["note"],
              "The binary oil support and nonoil labels are fixed for all body decoders.",
              "No automatic additional seeds, test evaluation, manuscript promotion or release."]
    (RUN/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def run():
    s.setup()
    with lock():
        check_stop()
        ids,config = identity(),settings()
        smoke_report = s.read(HERE/"smoke_report.json")
        assert smoke_report["status"]=="passed" and smoke_report["identity"]==ids and smoke_report["config"]==config
        if RUN.exists():
            manifest = s.read(RUN/"run_manifest.json")
            assert manifest["identity"]==ids and manifest["config"]==config
            if manifest["status"]=="completed":
                print("ALREADY_COMPLETED",flush=True); return
        else:
            RUN.mkdir(parents=True,exist_ok=False)
            snapshot = RUN/"source_snapshot"; snapshot.mkdir()
            for index,(path,h) in enumerate(sorted(ids["sources"].items())):
                shutil.copy2(path,snapshot/f"{index:02d}_{Path(path).name}")
            manifest = {"identity":ids,"config":config,"arms":{},"created_utc":s.utc(),
                        "environment":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"cuda":torch.version.cuda},
                        "command":sys.argv,"status":"preparing_cache","phase":"cache","pid":os.getpid()}
            s.write(RUN/"run_manifest.json",manifest)
        try:
            manifest.update(status="running",phase="cache",pid=os.getpid(),updated_utc=s.utc())
            s.write(RUN/"run_manifest.json",manifest)
            cache_meta = prepare(CACHE)
            assert cache_meta["images_by_split"]=={"train":2364,"val":676}
            # Stream hashes once at the formal training boundary, not per epoch.
            for key,h in cache_meta["array_sha256"].items():
                assert s.digest(CACHE/f"{key}.npy")==h
            assert s.digest(CACHE/"outside.json")==cache_meta["outside_sha256"]
            manifest["cache_manifest_sha256"] = s.digest(CACHE/"manifest.json")
            train,val = CachedBodies(CACHE,"train"),CachedBodies(CACHE,"val")
            frozen = s.read(CACHE/"frozen_validation.json")
            mean = evaluate(None,val)
            s.write(RUN/"frozen_validation.json",frozen)
            s.write(RUN/"body_mean_validation.json",mean)
            weights = torch.tensor(config["class_weights"],device="cuda")
            for arm in ARMS:
                check_stop()
                if manifest["arms"].get(arm,{}).get("status")=="completed":
                    continue
                model = build(arm)
                initial = state_digest(model.state_dict())
                assert initial==smoke_report["initial_sha256"]
                optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12)
                history,best,best_epoch = [],-1.,None
                last = RUN/f"{arm}_last.pth"
                if last.exists():
                    saved = torch.load(last,map_location="cpu",weights_only=False)
                    assert saved["identity"]==ids and saved["config"]==config and saved["arm"]==arm
                    assert saved["cache_manifest_sha256"]==manifest["cache_manifest_sha256"]
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    schedule.load_state_dict(saved["schedule"])
                    history,best,best_epoch = saved["history"],saved["best"],saved["best_epoch"]
                info = {"status":"running","initial_sha256":initial,"parameters":sum(p.numel() for p in model.parameters())}
                manifest["arms"][arm] = info
                for epoch in range(len(history)+1,13):
                    check_stop()
                    started = time.monotonic()
                    manifest.update(current_arm=arm,current_epoch=epoch,phase="train",updated_utc=s.utc())
                    s.write(RUN/"run_manifest.json",manifest)
                    order = np.random.default_rng(42000+epoch).permutation(train.indices)
                    generator = torch.Generator(device="cuda").manual_seed(420000+epoch)
                    model.train(); losses = []
                    print(f"EPOCH_START {arm} {epoch}/12 bodies={len(order)}",flush=True)
                    for start in range(0,len(order),512):
                        idx = order[start:start+512]
                        p,v,g = train.input_batch(idx)
                        target = torch.from_numpy(np.array(train.arrays["native_counts"][idx],copy=True)).cuda().float()
                        optimizer.zero_grad(set_to_none=True)
                        loss = body_loss(model(augment(p,generator),v,g),target,weights)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step(); losses.append(float(loss.detach()))
                        if start%(512*400)==0:
                            print(f"TRAIN {arm} {epoch} bodies={start+len(idx)}/{len(order)}",flush=True)
                    schedule.step()
                    manifest.update(phase="validation",updated_utc=s.utc()); s.write(RUN/"run_manifest.json",manifest)
                    measured = evaluate(model,val)
                    history.append({"epoch":epoch,"loss":float(np.mean(losses)),"seconds":time.monotonic()-started,"validation":measured})
                    common = {"model":model.state_dict(),"identity":ids,"config":config,"arm":arm,
                              "cache_manifest_sha256":manifest["cache_manifest_sha256"]}
                    if measured["native"]["mIoU"]>best:
                        best,best_epoch = measured["native"]["mIoU"],epoch
                        s.save(RUN/f"{arm}_best.pth",{**common,"epoch":epoch,"validation":measured})
                    s.save(last,{**common,"optimizer":optimizer.state_dict(),"schedule":schedule.state_dict(),
                                 "history":history,"best":best,"best_epoch":best_epoch})
                    s.write(RUN/f"{arm}_history.json",history)
                    info.update(completed_epochs=epoch,best_epoch=best_epoch,best_native_mIoU=best)
                    manifest.update(phase="epoch_complete",updated_utc=s.utc()); s.write(RUN/"run_manifest.json",manifest)
                    print(f"EPOCH_DONE {arm} {epoch}/12 native={measured['native']['mIoU']:.8f} canonical={measured['canonical512']['mIoU']:.8f} seconds={history[-1]['seconds']:.1f}",flush=True)
                selected = torch.load(RUN/f"{arm}_best.pth",map_location="cpu",weights_only=False)
                model.load_state_dict(selected["model"])
                replay = evaluate(model,val)
                assert replay==selected["validation"],"Selected checkpoint replay drift"
                s.write(RUN/f"{arm}_selected_validation.json",replay)
                info.update(status="completed",selected_validation_verified=True,checkpoint_sha256=s.digest(RUN/f"{arm}_best.pth"))
                s.write(RUN/"run_manifest.json",manifest)
                del model,optimizer,schedule
                torch.cuda.empty_cache()
            assert manifest["arms"]["separate"]["initial_sha256"]==manifest["arms"]["coupled"]["initial_sha256"]
            chosen = {arm:s.read(RUN/f"{arm}_selected_validation.json") for arm in ARMS}
            summary = finish(frozen,mean,chosen)
            manifest.update(status="completed",phase="complete",updated_utc=s.utc(),passed=summary["passed"],gates=summary["gates"])
            s.write(RUN/"run_manifest.json",manifest)
            print("PAIR_COMPLETED",json.dumps(summary["gates"]),flush=True)
        except BaseException as exc:
            manifest.update(status="paused" if isinstance(exc,InterruptedError) else "failed",error=repr(exc),
                            traceback=traceback.format_exc(),updated_utc=s.utc())
            s.write(RUN/"run_manifest.json",manifest)
            raise


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("smoke","run"))
    args = parser.parse_args()
    smoke() if args.mode=="smoke" else run()
