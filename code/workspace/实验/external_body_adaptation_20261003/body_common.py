"""Read-only reuse of the accepted stage-1 inputs and original body heads."""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP, ROOT = HERE.parent, HERE.parent.parent
UPSTREAM = EXP / 'external_eightclass_adaptation_20261003'
sys.path.insert(0, str(UPSTREAM))
import bridge as u
from data8 import CachedFine
from model import BodyInterfaceDecoder, augment, body_loss
from pointwise_model import PointwiseBodyDecoder
from representation import describe, aggregate
from inference import assign_bodies

read, write, sha, utc = u.read, u.write, u.sha, u.c.utc
DATASETS = ('bentheimer', 'wf1')
ARMS = ('separate', 'pointwise')
GRIDS = ('canonical512', 'native')
CONFIG = dict(protocol='external_fine8_body_stage2_seed42_v1', seed=42, epochs=12,
              batch_size=512, lr=.001, weight_decay=.0001, grad_clip=1.,
              gpu_memory_fraction=.35, precision='FP32', primary_grid='canonical512',
              selection='first maximum eight-class validation mIoU, including epoch0',
              initialization='each original main-domain seed42 head; no external head reruns',
              upstream='completed stage1 native_detail selection, frozen',
              model_test_pixels_read=False)
LAYOUT = {'points': (np.float16, (48, 31)), 'valid': (np.uint8, (48,)),
          'summary': (np.float32, (13,)), 'native_counts': (np.uint32, (8,)),
          'canonical512_counts': (np.uint32, (8,))}
SOURCES = ('body_common.py', 'spatial.py', 'prepare_bodies.py', 'run_body.py',
           'test_body.py', 'PROTOCOL.md')


def stop_check():
    for folder in (HERE, ROOT, UPSTREAM):
        for name in ('STOP.txt', 'STOP_AFTER_EPOCH.txt'):
            if (folder / name).exists():
                raise InterruptedError(f'User pause marker: {folder / name}')


@contextlib.contextmanager
def lock():
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


def setup(device='cuda'):
    torch.set_num_threads(4)
    if device == 'cuda':
        assert torch.cuda.is_available(), 'Use the crazyDog CUDA environment'
        torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'])


def source_head(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    files = u.c.locked.paths(42)
    checkpoint = files[arm + '_checkpoint']
    parent = read(UPSTREAM / 'run_manifest.json')['identity']['inputs']
    assert sha(checkpoint) == parent[str(checkpoint)], checkpoint
    model = BodyInterfaceDecoder('separate') if arm == 'separate' else PointwiseBodyDecoder()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == (25573 if arm == 'separate' else 25509)
    return model


def weights(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    data = read(UPSTREAM / 'cache/manifest.json')['entries'][dataset + '_train']
    expected = u.fine_weights(data['class_pixel_counts'])
    np.testing.assert_array_equal(expected, np.asarray(data['class_weights'], np.float32))
    return expected[3:].copy()


def require_completed_upstream():
    """One immediate check, never a polling worker or automatic queue."""
    stop_check()
    meta = read(UPSTREAM / 'run_manifest.json')
    if meta['status'] != 'completed':
        raise RuntimeError('Upstream four-arm training is not complete; do not start stage2')
    verified = read(UPSTREAM / 'verification.json')
    assert verified['status'] == 'PASS' and verified['histories'] == 52
    assert verified['saved_prediction_records'] == 780
    assert meta['identity'] == u.identity()
    inputs = {}
    for dataset in DATASETS:
        folder = UPSTREAM / f'outputs/{dataset}_seed42/native_detail'
        run = read(folder / 'run_manifest.json')
        assert run['status'] == 'completed' and run['identity'] == meta['identity']
        selected = read(folder / 'selected_validation.json')
        assert selected['saved_pixel_replay_exact']
        assert sha(folder / 'selected_validation.json') == run['selected_sha256']
        assert run['best_epoch'] == selected['selected_epoch']
        assert run['best_epoch'] == max(range(13), key=lambda i: run['history'][i]['validation']['metrics']['canonical512']['eightclass']['mIoU'])
        checkpoint = folder / 'epochs' / selected['checkpoint']['file']
        assert sha(checkpoint) == selected['checkpoint']['sha256']
        inputs[dataset] = dict(folder=str(folder), checkpoint=str(checkpoint),
                               checkpoint_sha256=sha(checkpoint), epoch=selected['selected_epoch'],
                               selected_sha256=sha(folder / 'selected_validation.json'))
    return meta, inputs


def identity():
    upstream, selected = require_completed_upstream()
    return dict(sources={str(HERE / n): sha(HERE / n) for n in SOURCES}, config=CONFIG,
                upstream_identity=upstream['identity'], selected_upstream=selected,
                upstream_manifest_sha256=sha(UPSTREAM / 'run_manifest.json'),
                upstream_verification_sha256=sha(UPSTREAM / 'verification.json'))


def order_for(indices, epoch):
    return np.random.default_rng(42 * 1000 + epoch).permutation(indices)


def stream(epoch, device='cuda'):
    return torch.Generator(device=device).manual_seed(42 * 10000 + epoch)


def save_npz(path, values):
    path = Path(path); tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as out:
        np.savez_compressed(out, **values)
    tmp.replace(path)


def save_checkpoint(path, values):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(values, tmp); tmp.replace(path)


def primary(result):
    return result['canonical512']['mIoU']


def selected_epoch(history):
    if not history:
        raise ValueError('Empty history')
    return max(range(len(history)), key=lambda i: primary(history[i]['validation']))
