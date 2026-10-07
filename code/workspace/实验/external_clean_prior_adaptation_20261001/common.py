"""Shared identities and constants for the fixed external adaptation pilot."""
from __future__ import annotations
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP, ROOT = HERE.parent, HERE.parent.parent
FROZEN = EXP / 'external_frozen_phase_20261001'
PREVIOUS = EXP / 'external_supervised_adaptation_20261001'
sys.path.insert(0, str(FROZEN))
import external_inputs as data
from phase_predictor import locked, order, NativeNoContext, project_native, decisions
sys.path.insert(0, str(EXP / 'architecture_upgrade_20260928'))
from mixed_label_loss import phase_marginal_logp, weighted_dice_from_logp

SEED = 42
DATASETS = ('bentheimer', 'wf1')
ARMS = ('global', 'native_detail')
GRIDS = ('canonical512', 'native')
RULES = ('collapse8', 'marginal4')
CONFIG = {'protocol': 'native_phase_clean_prior_adaptation_v1', 'seed': SEED, 'epochs': 12,
          'microbatch': 1, 'accumulate': 4, 'lr': 5e-5, 'min_lr': 5e-6,
          'weight_decay': 1e-4, 'grad_clip': 1.0, 'gpu_memory_fraction': .45,
          'primary_grid': 'canonical512', 'primary_rule': 'collapse8', 'precision': 'BF16',
          'freeze_existing_text_encoder': True, 'train_new_character_rows_only': True,
          'prior_schema': 'external_physical_conditions_v1_20261001',
          'new_row_seed': 4201001, 'new_row_noise_std': 0.01,
          'freeze_bn_running_statistics': True,
          'selection': 'first maximum validation mIoU, including epoch0',
          'external_test_pixels_read': False}
SOURCES = ('common.py', 'prepare_data.py', 'model_adapter.py', 'run_adaptation.py', 'test_contracts.py', 'prior_schema.py',
           'check_preparation.py', 'PROTOCOL.md', 'PRIOR_CHANGES.md')

read, sha = data.read, data.sha


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stop_check():
    locked.stop_check()
    for folder in (HERE, ROOT, FROZEN):
        for name in ('STOP_AFTER_EPOCH.txt', 'STOP.txt'):
            if (folder / name).exists():
                raise InterruptedError(f'User stop marker {folder / name}; never remove automatically')


@contextlib.contextmanager
def lock():
    import msvcrt
    with (HERE / 'run.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b'\0'); stream.flush()
        stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def identity(include_cache=True):
    prior = read(FROZEN / 'run_manifest.json')
    assert prior['status'] == 'completed'
    sources = dict(prior['identity']['sources'])
    inputs = dict(prior['identity']['inputs'])
    for entries in (sources, inputs):
        for path, expected in entries.items():
            assert sha(path) == expected, path
    for name in SOURCES:
        sources[str(HERE / name)] = sha(HERE / name)
    inputs[str(FROZEN / 'run_manifest.json')] = sha(FROZEN / 'run_manifest.json')
    for dataset in DATASETS:
        report = FROZEN / f'outputs/{dataset}_seed42/report.json'
        inputs[str(report)] = sha(report)
    result = {'sources': sources, 'inputs': inputs, 'config': CONFIG}
    if include_cache:
        result['cache_manifest'] = sha(PREVIOUS / 'cache/manifest.json')
    return result


def setup():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'])


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def lr_at(epoch):
    assert 1 <= epoch <= CONFIG['epochs']
    return CONFIG['min_lr'] + .5*(CONFIG['lr']-CONFIG['min_lr'])*(1+np.cos(np.pi*(epoch-1)/(CONFIG['epochs']-1)))


def weights_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    assert counts.shape == (4,) and (counts >= 0).all() and counts.sum() > 0
    weights = 1/np.sqrt(counts/counts.sum()+1e-8)
    return np.clip(weights/weights.mean(), .3, 6).astype(np.float32)


def epoch_plan(count, epoch):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, epoch, 129]))
    indices = rng.permutation(count)
    transforms = rng.integers(0, 16, size=count)
    return [(int(i), int(t)) for i, t in zip(indices, transforms)]


def d4(array, code):
    result = array
    if code & 1:
        result = np.flip(result, 0)
    if code & 2:
        result = np.flip(result, 1)
    return np.ascontiguousarray(np.rot90(result, code >> 2))


def window_size(step, count):
    start = (step//CONFIG['accumulate'])*CONFIG['accumulate']
    return min(CONFIG['accumulate'], count-start)
