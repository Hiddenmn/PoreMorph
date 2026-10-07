"""Paired fine8 external upstream replication for locked seeds 2024/2025.

Imports the completed seed42 trainer/evaluator without editing its files. The
single-stage 0..20 loop reproduces its 12-epoch cosine plus eight terminal-LR
continuation, preserving AdamW state at every epoch.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import os
from pathlib import Path
import statistics
import sys
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'external_eightclass_adaptation_20261003'
EXTENDED = HERE.parent / 'external_eightclass_extension20_recovery_20261004'
SEEDS = (2024, 2025)
ALL_SEEDS = (42, 2024, 2025)
sys.path.insert(0, str(PILOT))
import bridge as b
import run_eightclass as fine
from data8 import CachedFine, verify as verify_fine_cache

fine.install()
c = b.c
legacy = b.legacy
BASE_CONFIG = copy.deepcopy(c.CONFIG)
SOURCES = ('run_upstream.py', 'PROTOCOL.md')


def case(seed):
    if seed not in SEEDS:
        raise ValueError('Only seeds 2024/2025 are authorized here')
    return HERE / 'cases' / f'seed{seed}'


def stop_check():
    folders = (c.ROOT, HERE, c.HERE)
    for folder in folders:
        for name in ('STOP.txt', 'STOP_AFTER_EPOCH.txt'):
            if (folder / name).exists():
                raise InterruptedError(f'User stop marker: {folder / name}')


def upstream_lr(epoch):
    if not 1 <= epoch <= 20:
        raise ValueError(epoch)
    if epoch >= 13:
        return 5e-6
    return 5e-6 + .5 * (5e-5 - 5e-6) * (1 + np.cos(np.pi * (epoch - 1) / 11))


def job_identity():
    parent = c.read(PILOT / 'run_manifest.json')
    extended = c.read(EXTENDED / 'run_manifest.json')
    assert parent['status'] == extended['status'] == 'completed'
    assert c.read(EXTENDED / 'verification.json')['status'] == 'PASS'
    sources = dict(parent['identity']['sources'])
    inputs = dict(parent['identity']['inputs'])
    for paths in (sources, inputs):
        for path, digest in paths.items():
            assert c.sha(path) == digest, path
    for name in SOURCES:
        sources[str(HERE / name)] = c.sha(HERE / name)
    for path in (PILOT / 'run_manifest.json', PILOT / 'verification.json',
                 EXTENDED / 'run_manifest.json', EXTENDED / 'verification.json'):
        inputs[str(path)] = c.sha(path)
    for seed in SEEDS:
        files = c.locked.paths(seed)
        for key in ('global_checkpoint', 'tokenizer', 'native_checkpoint',
                    'native_manifest', 'native_selected'):
            path = files[key]
            inputs[str(path)] = c.sha(path)
        native = c.read(files['native_manifest'])
        assert native['status'] == 'completed'
        assert native['checkpoint_sha256'] == inputs[str(files['native_checkpoint'])]
    return dict(sources=sources, inputs=inputs,
                cache_manifest=c.sha(PILOT / 'cache/manifest.json'),
                label_fingerprint=c.read(PILOT / 'label_freeze.json')['dataset_fingerprint'],
                seeds=list(SEEDS), schedule='cosine 1..12 then constant 5e-6 13..20')


def identity():
    job = job_identity()
    return dict(sources=job['sources'], inputs=job['inputs'],
                cache_manifest=job['cache_manifest'],
                label_fingerprint=job['label_fingerprint'], config=copy.deepcopy(c.CONFIG))


def arm_folder(dataset, arm):
    assert dataset in c.DATASETS and arm in c.ARMS
    return c.HERE / f'outputs/{dataset}_seed{c.SEED}/{arm}'


def summary(reports):
    result = dict(status='completed', stage='upstream_adaptation_only',
                  seed=c.SEED, label_space='fine8', test_pixels_read=False,
                  datasets={}, replication_screen_passed=None)
    lines = [f'# Fine8 external upstream seed {c.SEED}', '',
             'TRAIN adaptation and VAL selection only; no body head or TEST.', '',
             '| Domain | Global | Native | Native−Global (pp) | selected G/N |',
             '|---|---:|---:|---:|---|']
    for domain in c.DATASETS:
        g, n = reports[f'{domain}_global'], reports[f'{domain}_native_detail']
        result['datasets'][domain] = dict(global_model=g, native_detail=n)
        a, z = fine.primary(g), fine.primary(n)
        lines.append(f'| {domain} | {a:.8f} | {z:.8f} | {100*(z-a):+.4f} | '
                     f'{g["selected_epoch"]}/{n["selected_epoch"]} |')
    c.write(c.HERE / 'summary.json', result)
    (c.HERE / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def configure(seed):
    destination = case(seed)
    c.HERE = destination
    # data8.py resolves its frozen label and cache identities via b.HERE.
    # Keep that input root pinned to the completed seed42 fine8 directory;
    # only c.HERE is a writable per-seed output destination.
    assert b.HERE == PILOT
    c.SEED = seed
    c.CONFIG = {**BASE_CONFIG, 'protocol': 'external_fine8_multiseed_upstream_v1',
                'seed': seed, 'epochs': 20,
                'schedule': 'epochs1..12 original cosine; epochs13..20 constant 5e-6'}
    c.identity = identity
    c.stop_check = stop_check
    b.stop_check = stop_check
    c.lr_at = upstream_lr
    legacy.arm_folder = arm_folder
    legacy.summary = summary
    return destination


@contextlib.contextmanager
def job_lock():
    import msvcrt
    with (HERE / 'run.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'\0'); stream.flush()
        stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def prepare():
    assert not torch.cuda.is_initialized()
    stop_check()
    job = job_identity()
    verify_fine_cache()
    assert abs(upstream_lr(1) - 5e-5) < 1e-14
    assert abs(upstream_lr(12) - 5e-6) < 1e-14
    assert all(upstream_lr(epoch) == 5e-6 for epoch in range(13, 21))
    for seed in SEEDS:
        path = configure(seed)
        path.mkdir(parents=True, exist_ok=True)
        assert not (path / 'run_manifest.json').exists()
        files = c.locked.paths(seed)
        assert all(files[key].is_file() for key in ('global_checkpoint', 'native_checkpoint', 'tokenizer'))
        c.write(path / 'preparation_report.json', dict(status='PASS', seed=seed,
            identity=identity(), test_pixels_read=False, created_utc=c.utc()))
    c.write(HERE / 'preparation_report.json', dict(status='PASS', identity=job,
        seeds=list(SEEDS), test_pixels_read=False, created_utc=c.utc()))
    print('FINE8_MULTISEED_PREPARED', flush=True)


def smoke():
    job = job_identity()
    assert c.read(HERE / 'preparation_report.json')['identity'] == job
    verify_fine_cache()
    c.setup()
    reports = {}
    for seed in SEEDS:
        configure(seed)
        assert not (c.HERE / 'run_manifest.json').exists()
        assert c.read(c.HERE / 'preparation_report.json')['identity'] == identity()
        entries = {}
        for domain in c.DATASETS:
            data = CachedFine(domain, 'train')
            common_global = None
            for arm in c.ARMS:
                stop_check()
                model = legacy.AdaptationModel(arm).cuda().to(memory_format=torch.channels_last)
                assert legacy.legacy_id_gpu_parity(model, data)
                optimizer = legacy.optimizer_for(model)
                initial = c.state_hash(model.global_model.state_dict())
                if common_global is None:
                    common_global = initial
                assert initial == common_global
                frozen = c.state_hash(model.frozen_state())
                torch.cuda.reset_peak_memory_stats()
                result = legacy.train_epoch(model, data, optimizer, 1,
                    plan=[(0, 0), (len(data)-1, 1), (0, 4), (len(data)-1, 8)])
                assert result['images'] == 4 and result['updates'] == 1
                assert c.state_hash(model.global_model.state_dict()) != initial
                assert c.state_hash(model.frozen_state()) == frozen
                entries[f'{domain}_{arm}'] = dict(update=result,
                    peak_MiB=torch.cuda.max_memory_allocated() / 2**20,
                    smoke_weights_discarded=True)
                legacy.close_model(model, optimizer)
        report = dict(status='passed', seed=seed, identity=identity(),
                      reports=entries, test_pixels_read=False, completed_utc=c.utc())
        c.write(c.HERE / 'smoke_report.json', report)
        reports[str(seed)] = c.sha(c.HERE / 'smoke_report.json')
    c.write(HERE / 'smoke_report.json', dict(status='PASS', identity=job,
        seed_reports=reports, test_pixels_read=False, completed_utc=c.utc()))
    print('FINE8_MULTISEED_SMOKE_PASS', flush=True)


def verify_case(seed):
    configure(seed)
    root = c.read(c.HERE / 'run_manifest.json')
    assert root['status'] == 'completed' and root['identity'] == identity()
    for domain in c.DATASETS:
        for arm in c.ARMS:
            folder = arm_folder(domain, arm)
            meta = c.read(folder / 'run_manifest.json')
            selected = c.read(folder / 'selected_validation.json')
            assert meta['status'] == 'completed' and meta['identity'] == root['identity']
            assert [r['epoch'] for r in meta['history']] == list(range(21))
            assert len(meta['epochs']) == 21
            assert meta['best_epoch'] == legacy.selection(meta['history']) == selected['selected_epoch']
            assert selected['saved_pixel_replay_exact'] and selected['split'] == 'val'
            assert c.sha(folder / 'selected_validation.json') == meta['selected_sha256']
            for entry in meta['epochs']:
                assert c.sha(folder / 'epochs' / entry['file']) == entry['sha256']
            for row in selected['saved_predictions']:
                assert c.sha(folder / selected['predictions_directory'] / row['file']) == row['sha256']
    return c.sha(c.HERE / 'summary.json')


def run(resume):
    stop_check()
    job = job_identity()
    smoke_report = c.read(HERE / 'smoke_report.json')
    assert smoke_report['status'] == 'PASS' and smoke_report['identity'] == job
    target = HERE / 'run_manifest.json'
    if target.exists():
        assert resume
        meta = c.read(target)
        assert meta['identity'] == job and meta['status'] != 'completed'
    else:
        assert not resume
        meta = dict(status='prepared', identity=job, completed_seeds={},
                    command=sys.argv, created_utc=c.utc(), test_pixels_read=False)
    meta.update(status='running', pid=os.getpid(), updated_utc=c.utc())
    c.write(target, meta)
    try:
        for seed in SEEDS:
            configure(seed)
            stop_check()
            assert c.sha(c.HERE / 'smoke_report.json') == smoke_report['seed_reports'][str(seed)]
            meta.update(current_seed=seed, updated_utc=c.utc())
            c.write(target, meta)
            existing = (c.HERE / 'run_manifest.json').exists()
            if str(seed) not in meta['completed_seeds']:
                legacy.run(resume=existing)
                meta['completed_seeds'][str(seed)] = verify_case(seed)
                c.write(target, meta)
                print(f'FINE8_SEED_COMPLETED {seed}', flush=True)
            else:
                assert verify_case(seed) == meta['completed_seeds'][str(seed)]
        meta.update(status='completed', completed_utc=c.utc(), current_seed=None)
        print('FINE8_UPSTREAM_ALL_SEEDS_COMPLETED', flush=True)
    except BaseException as error:
        meta.update(status='paused' if isinstance(error, InterruptedError) else 'failed',
                    error=repr(error), traceback=traceback.format_exc(), stopped_utc=c.utc())
        raise
    finally:
        c.write(target, meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert not args.resume or args.stage == 'run'
    with job_lock():
        if args.stage == 'prepare':
            prepare()
        elif args.stage == 'smoke':
            smoke()
        elif args.stage == 'verify':
            for item in SEEDS:
                print(item, verify_case(item))
        else:
            run(args.resume)
