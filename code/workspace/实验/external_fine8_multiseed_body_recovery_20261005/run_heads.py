"""Recovery of seed-matched external heads with authoritative logit support."""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / 'external_fine8_multiseed_20261004'
OLD_BODY = HERE.parent / 'external_body_adaptation_20261003'
EXT_BODY = HERE.parent / 'external_body_extension20_20261004'
SEEDS = (2024, 2025)
sys.path.insert(0, str(OLD_BODY))
import body_common as c

def load_local(name, filename):
    local_spec = importlib.util.spec_from_file_location(name, HERE / filename)
    local_module = importlib.util.module_from_spec(local_spec)
    sys.modules[name] = local_module
    local_spec.loader.exec_module(local_module)
    return local_module


representation = load_local('authoritative_body_representation', 'representation_authoritative.py')
c.describe = representation.describe
spec = importlib.util.spec_from_file_location('multiseed_body_prepare_recovery', HERE / 'prepare_bodies.py')
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
import run_body as runner
runner.p = p

BASE_CONFIG = copy.deepcopy(c.CONFIG)
OLD_SOURCES = dict(c.read(EXT_BODY / 'run_manifest.json')['identity']['sources'])
LOCAL_SOURCES = ('run_heads.py', 'PROTOCOL.md', 'prepare_bodies.py',
                 'representation_authoritative.py', 'test_recovery.py')


def case(seed):
    if seed not in SEEDS:
        raise ValueError('Only seeds 2024 and 2025')
    return HERE / 'cases' / f'seed{seed}'


def stop_check():
    for folder in (c.ROOT, HERE, c.HERE):
        for name in ('STOP.txt', 'STOP_AFTER_EPOCH.txt'):
            if (folder / name).exists():
                raise InterruptedError(f'User stop marker: {folder / name}')


def source_hashes():
    for path, digest in OLD_SOURCES.items():
        assert c.sha(path) == digest, path
    return {**OLD_SOURCES, **{str(HERE / name): c.sha(HERE / name)
                            for name in LOCAL_SOURCES}}


def selected_for(seed):
    up = c.read(UPSTREAM / 'run_manifest.json')
    if up['status'] != 'completed':
        raise RuntimeError('All upstream seeds must complete before body adaptation')
    case_root = UPSTREAM / 'cases' / f'seed{seed}'
    root = c.read(case_root / 'run_manifest.json')
    assert root['status'] == 'completed' and root['external_test_pixels_read'] is False
    assert str(seed) in up['completed_seeds']
    assert c.sha(case_root / 'summary.json') == up['completed_seeds'][str(seed)]
    result = {}
    for domain in c.DATASETS:
        folder = case_root / f'outputs/{domain}_seed{seed}/native_detail'
        meta = c.read(folder / 'run_manifest.json')
        chosen = c.read(folder / 'selected_validation.json')
        assert meta['status'] == 'completed' and meta['identity'] == root['identity']
        assert len(meta['history']) == len(meta['epochs']) == 21
        best = max(range(21), key=lambda i:
            meta['history'][i]['validation']['metrics']['canonical512']['eightclass']['mIoU'])
        assert meta['best_epoch'] == chosen['selected_epoch'] == best
        assert chosen['checkpoint'] == meta['epochs'][best]
        assert chosen['metrics'] == meta['history'][best]['validation']['metrics']
        assert chosen['saved_pixel_replay_exact'] and chosen['split'] == 'val'
        assert c.sha(folder / 'selected_validation.json') == meta['selected_sha256']
        checkpoint = folder / 'epochs' / chosen['checkpoint']['file']
        assert c.sha(checkpoint) == chosen['checkpoint']['sha256']
        result[domain] = dict(dataset=domain, folder=str(folder),
            checkpoint=str(checkpoint), checkpoint_sha256=c.sha(checkpoint),
            epoch=best, frozen_sha256=meta['frozen_sha256'],
            selected_sha256=meta['selected_sha256'])
    return root, result


def identity():
    seed = c.CONFIG['seed']
    upstream, selected = selected_for(seed)
    main_head = c.u.c.locked.paths(seed)['separate_checkpoint']
    return dict(sources=source_hashes(), config=copy.deepcopy(c.CONFIG),
        upstream_identity=upstream['identity'], selected_upstream=selected,
        upstream_root_sha256=c.sha(UPSTREAM / 'run_manifest.json'),
        upstream_case_sha256=c.sha(UPSTREAM / f'cases/seed{seed}/run_manifest.json'),
        main_head_checkpoint=str(main_head), main_head_sha256=c.sha(main_head),
        frozen_label_sha256=c.sha(c.UPSTREAM / 'label_freeze.json'),
        frozen_cache_sha256=c.sha(c.UPSTREAM / 'cache/manifest.json'))


def source_head(arm):
    if arm != 'separate':
        raise ValueError('This replication trains only the complete separate head')
    seed = c.CONFIG['seed']
    files = c.u.c.locked.paths(seed)
    meta = c.read(files['body_manifest'])
    assert meta['status'] == 'completed'
    checkpoint = files['separate_checkpoint']
    assert c.sha(checkpoint) == meta['arms']['separate']['checkpoint_sha256']
    model = c.BodyInterfaceDecoder('separate')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    assert sum(x.numel() for x in model.parameters()) == 25573
    return model


def order_for(indices, epoch):
    return np.random.default_rng(c.CONFIG['seed'] * 1000 + epoch).permutation(indices)


def stream(epoch, device='cuda'):
    return torch.Generator(device=device).manual_seed(c.CONFIG['seed'] * 10000 + epoch)


def load_upstream_checkpoint(upstream, ids):
    assert ids['selected_upstream'][upstream['dataset']] == upstream
    path = Path(upstream['checkpoint'])
    assert c.sha(path) == upstream['checkpoint_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['identity'] == ids['upstream_identity']
    assert saved['dataset'] == upstream['dataset'] and saved['arm'] == 'native_detail'
    assert saved['epoch'] == upstream['epoch']
    assert saved['frozen_sha256'] == upstream['frozen_sha256']
    assert all(torch.isfinite(value).all() for value in saved['model'].values())
    assert c.u.c.state_hash(c.u.legacy.frozen_from_state(saved['model'])) == upstream['frozen_sha256']
    return saved


def configure(seed):
    folder = case(seed)
    c.HERE = folder
    c.CONFIG = {**BASE_CONFIG, 'protocol': 'external_fine8_multiseed_body_logit_support_recovery_v1',
                'seed': seed, 'initialization': f'matching main-domain seed{seed} separate head',
                'upstream': 'matching seed first-best native within epochs0..20, frozen'}
    c.ARMS = ('separate',)
    c.stop_check = stop_check
    c.identity = identity
    c.source_head = source_head
    c.order_for = order_for
    c.stream = stream
    c.load_upstream_checkpoint = load_upstream_checkpoint
    c.u.c.SEED = seed  # AdaptationModel constructor used only to reload the frozen upstream.
    return folder


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


def contracts():
    assert not torch.cuda.is_initialized()
    source_hashes()
    for seed in SEEDS:
        configure(seed)
        model = source_head('separate')
        assert all(torch.isfinite(x).all() for x in model.state_dict().values())
        assert order_for(np.arange(10), 3).shape == (10,)
    c.write(HERE / 'preparation_checks.json', dict(status='PASS', seeds=list(SEEDS),
        sources=source_hashes(), cuda_initialized=False, test_pixels_read=False,
        completed_utc=c.utc()))
    print('MULTISEED_BODY_CPU_CONTRACTS_PASS', flush=True)


def smoke():
    stop_check()
    assert c.read(HERE / 'preparation_checks.json')['sources'] == source_hashes()
    from data8 import verify as verify_pixels
    verify_pixels()
    c.setup()
    reports = {}
    for seed in SEEDS:
        configure(seed)
        case(seed).mkdir(parents=True, exist_ok=True)
        assert not (c.HERE / 'run_manifest.json').exists()
        ids = identity()
        for domain in c.DATASETS:
            meta = p.prepare(domain, ids, smoke=True)
            data = p.Bodies(p.cache_folder(domain, True), 'train')
            assert len(data.indices) > 0
            index = np.resize(data.indices, 512)
            model = source_head('separate').cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=c.CONFIG['lr'],
                                          weight_decay=c.CONFIG['weight_decay'])
            before = c.u.c.state_hash(model.state_dict())
            points, valid, summary = data.inputs(index)
            target = data.targets(index)
            optimizer.zero_grad(set_to_none=True)
            loss = c.body_loss(model(c.augment(points, stream(1)), valid, summary),
                target, torch.tensor(c.weights(domain), device='cuda'))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
            optimizer.step()
            assert c.u.c.state_hash(model.state_dict()) != before
            reports[f'{seed}_{domain}'] = dict(images=meta['images_by_split'],
                source_head_sha256=ids['main_head_sha256'], smoke_weights_discarded=True)
            model.cpu(); del model, optimizer, points, valid, summary, target, loss
            torch.cuda.empty_cache()
        c.write(c.HERE / 'smoke_report.json', dict(status='PASS', seed=seed,
            identity=ids, test_pixels_read=False, completed_utc=c.utc()))
    c.write(HERE / 'smoke_report.json', dict(status='PASS', reports=reports,
        source_sha256=c.sha(HERE / 'run_heads.py'), test_pixels_read=False,
        completed_utc=c.utc()))
    print('MULTISEED_BODY_GPU_SMOKE_PASS', flush=True)


def run_case(seed, resume):
    configure(seed)
    ids = identity()
    assert c.read(c.HERE / 'smoke_report.json')['identity'] == ids
    path = c.HERE / 'run_manifest.json'
    if path.exists():
        assert resume
        meta = c.read(path)
        assert meta['identity'] == ids and meta['status'] != 'completed'
    else:
        assert not resume
        meta = dict(status='prepared', identity=ids, seed=seed, completed_arms={},
                    test_pixels_read=False, created_utc=c.utc())
    meta.update(status='running', pid=os.getpid(), updated_utc=c.utc())
    c.write(path, meta)
    try:
        c.setup()
        report = dict(status='completed', seed=seed, stage='frozen_upstream_plus_separate_head',
                      test_pixels_read=False, datasets={})
        for domain in c.DATASETS:
            stop_check()
            p.prepare(domain, ids)
            val = p.Bodies(p.cache_folder(domain), 'val')
            key = domain + '_separate'
            folder = c.HERE / f'outputs/{domain}/separate'
            if key in meta['completed_arms']:
                assert c.sha(folder / 'selected_validation.json') == meta['completed_arms'][key]
                selected = c.read(folder / 'selected_validation.json')
                runner.verify_assignments(folder, selected, val, domain)
            else:
                selected = runner.train_arm(domain, 'separate', ids,
                                            resume=(folder / 'run_manifest.json').exists())
                meta['completed_arms'][key] = c.sha(folder / 'selected_validation.json')
                c.write(path, meta)
            report['datasets'][domain] = dict(
                native_detail={g: c.u.metrics(np.asarray(h)) for g, h in val.frozen.items()},
                complete=selected)
        c.write(c.HERE / 'summary.json', report)
        meta.update(status='completed', summary_sha256=c.sha(c.HERE / 'summary.json'),
                    completed_utc=c.utc())
        c.write(path, meta)
        runner.verify()
        return c.sha(c.HERE / 'summary.json')
    except BaseException as error:
        meta.update(status='paused' if isinstance(error, InterruptedError) else 'failed',
                    error=repr(error), traceback=traceback.format_exc(), stopped_utc=c.utc())
        c.write(path, meta)
        raise


def run(resume):
    stop_check()
    smoke_report = c.read(HERE / 'smoke_report.json')
    assert smoke_report['status'] == 'PASS'
    assert smoke_report['source_sha256'] == c.sha(HERE / 'run_heads.py')
    path = HERE / 'run_manifest.json'
    if path.exists():
        assert resume
        meta = c.read(path)
        assert meta['status'] != 'completed'
    else:
        assert not resume
        meta = dict(status='prepared', completed_seeds={}, test_pixels_read=False,
                    command=sys.argv, created_utc=c.utc())
    meta.update(status='running', pid=os.getpid(), updated_utc=c.utc())
    c.write(path, meta)
    try:
        for seed in SEEDS:
            configure(seed)
            stop_check()
            meta.update(current_seed=seed, updated_utc=c.utc())
            c.write(path, meta)
            if str(seed) not in meta['completed_seeds']:
                digest = run_case(seed, resume=(case(seed) / 'run_manifest.json').exists())
                meta['completed_seeds'][str(seed)] = digest
                c.write(path, meta)
            else:
                assert c.sha(case(seed) / 'summary.json') == meta['completed_seeds'][str(seed)]
        meta.update(status='completed', current_seed=None, completed_utc=c.utc())
        print('MULTISEED_BODY_COMPLETED', flush=True)
    except BaseException as error:
        meta.update(status='paused' if isinstance(error, InterruptedError) else 'failed',
                    error=repr(error), traceback=traceback.format_exc(), stopped_utc=c.utc())
        raise
    finally:
        c.write(path, meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('contracts', 'smoke', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert not args.resume or args.stage == 'run'
    with job_lock():
        if args.stage == 'contracts':
            contracts()
        elif args.stage == 'smoke':
            smoke()
        elif args.stage == 'run':
            run(args.resume)
        else:
            for seed in SEEDS:
                configure(seed)
                runner.verify()
