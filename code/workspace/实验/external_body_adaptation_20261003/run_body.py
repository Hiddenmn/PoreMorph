"""Explicit staged entry point; never polls or starts before stage1 verification."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import unittest

import numpy as np
import torch

import body_common as c
import prepare_bodies as p
import spatial as s


def contracts():
    c.setup('cpu')
    spec = importlib.util.spec_from_file_location('external_body_contracts', c.HERE / 'test_body.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():
        raise AssertionError('CPU contracts failed')
    c.write(c.HERE / 'preparation_checks.json', dict(status='PASS', cpu_tests=result.testsRun,
            sources={str(c.HERE / n): c.sha(c.HERE / n) for n in c.SOURCES}, checked_utc=c.utc(),
            scope='CPU synthetic geometry/loss/state contracts and original source-head loading only',
            stage2_gpu_run_started=False, real_frozen_cache_verified=False, test_pixels_read=False))
    print(f'BODY_CPU_CONTRACTS_PASS {result.testsRun}; no stage2 GPU run', flush=True)


def train_epoch(model, data, optimizer, dataset, arm, epoch):
    model.train(); order = c.order_for(data.indices, epoch); generator = c.stream(epoch)
    weights = torch.tensor(c.weights(dataset), device='cuda')
    losses = []; start = time.monotonic()
    for offset in range(0, len(order), c.CONFIG['batch_size']):
        c.stop_check(); index = order[offset:offset + c.CONFIG['batch_size']]
        points, valid, summary = data.inputs(index)
        target = data.targets(index)
        optimizer.zero_grad(set_to_none=True)
        loss = c.body_loss(model(c.augment(points, generator), valid, summary), target, weights)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite body-head loss')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.CONFIG['grad_clip'])
        assert torch.isfinite(norm)
        optimizer.step(); losses.append(float(loss.detach()))
    assert losses, 'No TRAIN bodies'
    return dict(loss=float(np.mean(losses)), bodies=len(order), updates=len(losses),
                seconds=time.monotonic() - start, lr=optimizer.param_groups[0]['lr'])


def smoke():
    contracts(); ids = c.identity()
    from data8 import verify as verify_pixels
    verify_pixels(); c.setup()
    reports = {}
    for dataset in c.DATASETS:
        meta = p.prepare(dataset, ids, smoke=True)
        data = p.Bodies(p.cache_folder(dataset, True), 'train')
        assert len(data.indices) > 0
        index = np.resize(data.indices, 512)
        for arm in c.ARMS:
            model = c.source_head(arm).cuda()
            initial = c.u.c.state_hash(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=c.CONFIG['lr'], weight_decay=c.CONFIG['weight_decay'])
            points, valid, summary = data.inputs(index)
            target = data.targets(index)
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            loss = c.body_loss(model(c.augment(points, c.stream(1)), valid, summary), target,
                               torch.tensor(c.weights(dataset), device='cuda'))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            assert c.u.c.state_hash(model.state_dict()) != initial
            reports[dataset + '_' + arm] = dict(parameters=sum(x.numel() for x in model.parameters()),
                    loss=float(loss.detach()), peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                    source_state_sha256=initial, cache_images=meta['images_by_split'], weights_discarded=True)
            model.cpu(); del model, optimizer, loss, points, valid, summary, target
            torch.cuda.empty_cache()
    assert c.identity() == ids
    c.write(c.HERE / 'smoke_report.json', dict(status='PASS', identity=ids, reports=reports,
            completed_utc=c.utc(), test_pixels_read=False))
    print('ALL_FOUR_EXTERNAL_BODY_SMOKES_PASS', flush=True)


def completed_checkpoint(path, ids, cache_sha, dataset, arm):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['identity'] == ids and saved['cache_sha256'] == cache_sha
    assert saved['dataset'] == dataset and saved['arm'] == arm
    assert [r['epoch'] for r in saved['history']] == list(range(saved['epoch'] + 1))
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    return saved


def verify_assignments(folder, selected, data, dataset):
    path = folder / selected['assignments_file']
    assert c.sha(path) == selected['assignments_sha256']
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['indices'], data.indices)
        assigned = np.array(z['assigned'])
    assert assigned.shape == data.indices.shape and (assigned.size == 0 or assigned.max() < 5)
    dense_total = {g: np.zeros((8, 8), np.int64) for g in c.GRIDS}
    pixels = c.CachedFine(dataset, 'val'); cursor = 0
    for record in data.records:
        c.stop_check()
        n = record['bodies']; chosen = assigned[cursor:cursor + n]; cursor += n
        shard = data.folder / 'shards' / record['file']
        assert c.sha(shard) == record['sha256']
        with np.load(shard, allow_pickle=False) as z:
            truth = np.asarray(pixels.masks[record['row']])
            dense = c.assign_bodies(z['prediction'], z['components'], chosen.astype(np.int64) + 3)
            dense_total['native'] += c.u.histogram(dense, truth)
            dense_total['canonical512'] += c.u.histogram(s.nearest512_ids(dense), s.nearest512_ids(truth))
    assert cursor == len(assigned)
    for grid in c.GRIDS:
        actual = s.recount(assigned, data.arrays[grid + '_counts'][data.indices], data.outside[grid])
        np.testing.assert_array_equal(actual, dense_total[grid])
        assert c.u.metrics(actual) == selected['validation'][grid]
    s.assert_fixed_support(dense_total, data.frozen)
    return dict(images=len(data.records), bodies=len(assigned), dense_sparse_exact=True,
                fixed_oil_and_nonoil_exact=True)


def train_arm(dataset, arm, ids, resume):
    data_dir = p.cache_folder(dataset); cache_sha = c.sha(data_dir / 'manifest.json')
    train, val = p.Bodies(data_dir, 'train'), p.Bodies(data_dir, 'val')
    folder = c.HERE / f'outputs/{dataset}/{arm}'
    if not folder.exists():
        folder.mkdir(parents=True); (folder / 'epochs').mkdir()
        meta = dict(status='prepared', identity=ids, cache_sha256=cache_sha,
                    dataset=dataset, arm=arm, history=[], checkpoints=[])
        c.write(folder / 'run_manifest.json', meta)
    else:
        meta = c.read(folder / 'run_manifest.json')
        assert meta['identity'] == ids and meta['cache_sha256'] == cache_sha
        assert resume, 'Existing head output requires explicit --resume'
    for entry in meta['checkpoints']:
        assert c.sha(folder / 'epochs' / entry['file']) == entry['sha256']
    history = meta['history']
    # Recover only an atomically completed next epoch; never choose among reruns.
    next_path = folder / 'epochs' / f'epoch_{len(history):03d}.pth'
    if next_path.exists():
        assert resume
        saved = completed_checkpoint(next_path, ids, cache_sha, dataset, arm)
        assert saved['epoch'] == len(history) and saved['history'][:-1] == history
        history = saved['history']
        meta['checkpoints'].append(dict(epoch=saved['epoch'], file=next_path.name, sha256=c.sha(next_path)))
        meta.update(history=history, status='running')
        c.write(folder / 'run_manifest.json', meta)
    model = c.source_head(arm).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.CONFIG['lr'], weight_decay=c.CONFIG['weight_decay'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    if history:
        previous = folder / 'epochs' / meta['checkpoints'][-1]['file']
        saved = completed_checkpoint(previous, ids, cache_sha, dataset, arm)
        assert saved['history'] == history
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); schedule.load_state_dict(saved['schedule'])
        del saved
    try:
        for epoch in range(len(history), 13):
            c.stop_check(); start = time.monotonic()
            c.write(c.HERE / 'progress.json', dict(stage='head_train' if epoch else 'head_epoch0',
                    dataset=dataset, arm=arm, epoch=epoch, pid=os.getpid(), updated_utc=c.utc()))
            training = train_epoch(model, train, optimizer, dataset, arm, epoch) if epoch else None
            if epoch:
                schedule.step()
            measured = p.evaluate(model, val)
            history = history + [dict(epoch=epoch, training=training, validation=measured,
                                      seconds=time.monotonic() - start)]
            destination = folder / 'epochs' / f'epoch_{epoch:03d}.pth'
            assert not destination.exists()
            c.save_checkpoint(destination, dict(identity=ids, cache_sha256=cache_sha,
                    dataset=dataset, arm=arm, epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), schedule=schedule.state_dict(), history=history))
            meta['checkpoints'].append(dict(epoch=epoch, file=destination.name, sha256=c.sha(destination)))
            meta.update(status='running', history=history, best_epoch=c.selected_epoch(history), updated_utc=c.utc())
            c.write(folder / 'history.json', history); c.write(folder / 'run_manifest.json', meta)
            print(f'BODY_EPOCH {dataset} {arm} epoch={epoch} mIoU={c.primary(measured):.8f}', flush=True)
        best = c.selected_epoch(history)
        checkpoint = meta['checkpoints'][best]
        saved = completed_checkpoint(folder / 'epochs' / checkpoint['file'], ids, cache_sha, dataset, arm)
        model.load_state_dict(saved['model'], strict=True)
        replay = p.evaluate(model, val, save_to=folder / 'selected_assignments.npz')
        assert replay == history[best]['validation']
        selected = dict(selected_epoch=best, checkpoint=checkpoint, validation=replay,
                        assignments_file='selected_assignments.npz',
                        assignments_sha256=c.sha(folder / 'selected_assignments.npz'))
        selected['verification'] = verify_assignments(folder, selected, val, dataset)
        c.write(folder / 'selected_validation.json', selected)
        meta.update(status='completed', best_epoch=best, completed_utc=c.utc(),
                    selected_sha256=c.sha(folder / 'selected_validation.json'))
        c.write(folder / 'run_manifest.json', meta)
        return selected
    finally:
        model.cpu(); del model, optimizer, schedule
        torch.cuda.empty_cache()


def run(resume):
    ids = c.identity()
    smoke_report = c.read(c.HERE / 'smoke_report.json')
    assert smoke_report['status'] == 'PASS' and smoke_report['identity'] == ids
    path = c.HERE / 'run_manifest.json'
    if path.exists():
        assert resume, 'Existing experiment requires --resume'
        meta = c.read(path); assert meta['identity'] == ids
        if meta['status'] == 'completed':
            raise RuntimeError('Already completed: use verify, not another training run')
    else:
        assert not resume
        snapshot = c.HERE / 'source_snapshot'; snapshot.mkdir(exist_ok=False)
        for name in c.SOURCES:
            shutil.copy2(c.HERE / name, snapshot / name)
        meta = dict(status='prepared', identity=ids, completed_arms={}, command=sys.argv,
                    created_utc=c.utc(), test_pixels_read=False,
                    environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__))
        c.write(path, meta)
    try:
        from data8 import verify as verify_pixels
        verify_pixels(); c.setup()
        meta.update(status='running', pid=os.getpid(), updated_utc=c.utc()); c.write(path, meta)
        results = dict(stage='external_body_adaptation', seed=42, datasets={}, test_pixels_read=False,
                       main_model_promoted=False, scope='Target-domain fine-tuning; validation only')
        for dataset in c.DATASETS:
            p.prepare(dataset, ids)
            val = p.Bodies(p.cache_folder(dataset), 'val')
            entry = dict(native_detail={g: c.u.metrics(np.asarray(h)) for g, h in val.frozen.items()},
                         body_mean=p.evaluate(None, val), heads={})
            for arm in c.ARMS:
                key = dataset + '_' + arm
                if key in meta['completed_arms']:
                    folder = c.HERE / f'outputs/{dataset}/{arm}'
                    assert c.sha(folder / 'selected_validation.json') == meta['completed_arms'][key]
                    chosen = c.read(folder / 'selected_validation.json')
                    verify_assignments(folder, chosen, val, dataset)
                else:
                    chosen = train_arm(dataset, arm, ids, resume)
                    meta['completed_arms'][key] = c.sha(c.HERE / f'outputs/{dataset}/{arm}/selected_validation.json')
                    c.write(path, meta)
                entry['heads'][arm] = chosen
            entry['separate_minus_pointwise_pp'] = {
                g: {metric: 100 * (entry['heads']['separate']['validation'][g][metric] -
                                   entry['heads']['pointwise']['validation'][g][metric])
                    for metric in ('mIoU', 'oil_macro_IoU')} for g in c.GRIDS}
            results['datasets'][dataset] = entry
        assert c.identity() == ids
        c.write(c.HERE / 'summary.json', results)
        meta.update(status='completed', completed_utc=c.utc(), summary_sha256=c.sha(c.HERE / 'summary.json'))
        c.write(path, meta)
        verify()
    except BaseException as error:
        meta.update(status='stopped' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
                    error=str(error), traceback=traceback.format_exc(), updated_utc=c.utc())
        c.write(path, meta)
        raise


def verify():
    ids = c.identity(); meta = c.read(c.HERE / 'run_manifest.json')
    assert meta['status'] == 'completed' and meta['identity'] == ids
    assert c.sha(c.HERE / 'summary.json') == meta['summary_sha256']
    reports = []
    for dataset in c.DATASETS:
        p.verify_cache(p.cache_folder(dataset), ids)
        data = p.Bodies(p.cache_folder(dataset), 'val')
        for arm in c.ARMS:
            folder = c.HERE / f'outputs/{dataset}/{arm}'; run_meta = c.read(folder / 'run_manifest.json')
            assert run_meta['status'] == 'completed' and run_meta['identity'] == ids
            assert len(run_meta['history']) == len(run_meta['checkpoints']) == 13
            assert run_meta['best_epoch'] == c.selected_epoch(run_meta['history'])
            for record in run_meta['checkpoints']:
                assert c.sha(folder / 'epochs' / record['file']) == record['sha256']
            assert c.sha(folder / 'selected_validation.json') == run_meta['selected_sha256']
            selected = c.read(folder / 'selected_validation.json')
            assert selected['selected_epoch'] == run_meta['best_epoch']
            reports.append(dict(dataset=dataset, arm=arm, **verify_assignments(folder, selected, data, dataset)))
    c.write(c.HERE / 'verification.json', dict(status='PASS', reports=reports, verified_utc=c.utc(),
            test_pixels_read=False, source_identity=ids))
    print('EXTERNAL_BODY_COMPLETED_VERIFIED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('contracts', 'smoke', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert not args.resume or args.stage == 'run'
    with c.lock():
        run(args.resume) if args.stage == 'run' else globals()[args.stage]()
