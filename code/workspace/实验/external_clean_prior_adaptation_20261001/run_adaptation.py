"""Matched target-domain adaptation, bounded to two seed42 TRAIN/VAL pairs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
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

import common as c
from model_adapter import AdaptationModel, phase_loss, to_device, frozen_from_state
from prepare_data import Cached, verify_cache


def primary(report):
    return report['metrics']['canonical512']['collapse8']['mIoU']


def selection(history):
    # Python max retains the first item on ties, including initialization.
    return max(range(len(history)), key=lambda i: primary(history[i]['validation']))


def tokens(model, dataset):
    ids, attention = model.tokenizer.encode(dataset.description)
    return ids[None].cuda(), attention[None].cuda()


def close_model(model, optimizer=None):
    model.cpu()
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def counts():
    return {g: {r: np.zeros((4, 4), np.int64) for r in c.RULES} for g in c.GRIDS}


@torch.inference_mode()
def evaluate(model, dataset, save_to=None):
    model.eval()
    ids, attention = tokens(model, dataset)
    matrices = counts()
    saved = []
    started = time.monotonic()
    if save_to is not None:
        save_to.mkdir(exist_ok=False)
    for index in range(len(dataset)):
        c.stop_check()
        sample = dataset.get(index)
        image, low, target_gpu = to_device(sample)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            coarse, native = model(image, low, ids, attention)
        native_decisions = c.decisions(native)
        canonical = c.decisions(coarse) if model.arm == 'global' else {
            rule: c.data.nearest512(value) for rule, value in native_decisions.items()}
        predictions = {'native': native_decisions, 'canonical512': canonical}
        target = sample[2]
        for grid in c.GRIDS:
            truth = target if grid == 'native' else c.data.nearest512(target)
            for rule in c.RULES:
                matrices[grid][rule] += c.data.histogram(predictions[grid][rule], truth)
        if save_to is not None:
            path = save_to/f'{index:04d}.npz'
            values = {'target_native': target}
            values.update({f'{g}_{r}': predictions[g][r] for g in c.GRIDS for r in c.RULES})
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, **values)
            path.with_suffix('.tmp').replace(path)
            saved.append({'index': index, 'file': path.name, 'sha256': c.sha(path)})
        del image, low, target_gpu, coarse, native
    result = {'images': len(dataset), 'split': dataset.split,
              'metrics': {g: {r: c.data.metrics(matrices[g][r]) for r in c.RULES} for g in c.GRIDS},
              'seconds': time.monotonic()-started, 'external_test_pixels_read': False}
    if save_to is not None:
        result['saved_predictions'] = saved
        # Independent prediction-first encoding; no model/target feedback.
        verified = counts()
        for row in saved:
            path = save_to/row['file']
            assert c.sha(path) == row['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                for grid in c.GRIDS:
                    truth = arrays['target_native'] if grid == 'native' else c.data.nearest512(arrays['target_native'])
                    for rule in c.RULES:
                        h = np.bincount((arrays[f'{grid}_{rule}'].astype(np.int64)*4+truth).ravel(), minlength=16)
                        verified[grid][rule] += h.reshape(4, 4).T
        for grid in c.GRIDS:
            for rule in c.RULES:
                np.testing.assert_array_equal(verified[grid][rule], matrices[grid][rule])
        result['saved_pixel_replay_exact'] = True
    return result


def train_epoch(model, data, optimizer, epoch, progress_path=None, plan=None):
    model.train()
    plan = c.epoch_plan(len(data), epoch) if plan is None else plan
    c.order.base.set_seed(c.SEED+10007*epoch)
    ids, attention = tokens(model, data)
    weights = torch.tensor(data.entry['class_weights'], dtype=torch.float32, device='cuda')
    for group in optimizer.param_groups:
        group['lr'] = float(c.lr_at(epoch))
    frozen = c.state_hash(model.frozen_state())
    optimizer.zero_grad(set_to_none=True)
    running, updates = 0., 0
    started = time.monotonic()
    for step, (index, transform) in enumerate(plan):
        c.stop_check()
        image, low, target = to_device(data.get(index, transform))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            coarse, logits = model(image, low, ids, attention)
            loss = phase_loss(logits, target, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss {data.dataset} {model.arm} epoch{epoch} image{index}')
        (loss/c.window_size(step, len(plan))).backward()
        running += float(loss.detach())
        del image, low, target, coarse, logits, loss
        if (step+1)%c.CONFIG['accumulate'] == 0 or step+1 == len(plan):
            norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        if (step+1)%100 == 0 or step+1 == len(plan):
            seconds = time.monotonic()-started
            progress = {'stage': 'train', 'dataset': data.dataset, 'arm': model.arm, 'epoch': epoch,
                        'completed_images': step+1, 'images': len(plan), 'loss': running/(step+1),
                        'updates': updates, 'seconds': seconds, 'updated_utc': c.utc(), 'pid': os.getpid()}
            if progress_path is not None:
                c.write(progress_path, progress)
            print(f'TRAIN {data.dataset} {model.arm} epoch={epoch} {step+1}/{len(plan)} loss={running/(step+1):.6f} seconds={seconds:.1f}', flush=True)
    assert c.state_hash(model.frozen_state()) == frozen, 'Frozen text encoder or BN statistics changed'
    return {'loss': running/len(plan), 'images': len(plan), 'updates': updates, 'lr': float(c.lr_at(epoch)),
            'seconds': time.monotonic()-started, 'frozen_state_sha256': frozen,
            'plan_sha256': hashlib.sha256(json.dumps(plan).encode()).hexdigest()}


def optimizer_for(model):
    return torch.optim.AdamW(model.trainable_parameters(), lr=c.CONFIG['lr'], weight_decay=c.CONFIG['weight_decay'])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, value):
    assert not path.exists(), f'Preserve existing checkpoint: {path}'
    tmp = path.with_suffix('.tmp')
    torch.save(cpu_tree(value), tmp)
    tmp.replace(path)
    return c.sha(path)


@torch.inference_mode()
def legacy_id_gpu_parity(model, dataset):
    """Engineering check only: identical old token IDs must preserve source logits."""
    files = c.locked.paths(c.SEED)
    tokenizer = c.order.base.CharTokenizer.load(files['tokenizer'])
    original = c.order.build_model('fusion_before_ppm', tokenizer.vocab_size)
    saved = torch.load(files['global_checkpoint'], map_location='cpu', weights_only=False)
    original.load_state_dict(saved['model_state_dict'], strict=True)
    del saved
    original.cuda().to(memory_format=torch.channels_last).eval()
    model.eval()
    image, low, target = to_device(dataset.get(0))
    ids, attention = tokenizer.encode(c.data.description(dataset.dataset))
    ids, attention = ids[None].cuda(), attention[None].cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        before = original(low, ids, attention)
        after = model.global_model(low, ids, attention)
    assert torch.equal(before, after), 'Append-only embedding changed original-ID inference'
    del image, low, target, before, after, ids, attention
    original.cpu()
    del original
    gc.collect()
    torch.cuda.empty_cache()
    return True


def smoke():
    c.stop_check()
    torch.set_num_threads(4)
    preparation = c.read(c.HERE/'preparation_report.json')
    assert preparation['status'] == 'passed' and preparation['identity'] == c.identity()
    # Imported upstream folders also have test_contracts.py; load this file
    # explicitly so sys.path ordering cannot silently run a different suite.
    spec = importlib.util.spec_from_file_location(
        'external_clean_prior_contract_checks', c.HERE/'test_contracts.py')
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    tests = unittest.defaultTestLoader.loadTestsFromModule(test_module)
    checked = unittest.TextTestRunner(verbosity=2).run(tests)
    assert checked.wasSuccessful()
    identity = c.identity()
    verify_cache()
    c.setup()
    reports = {}
    for name in c.DATASETS:
        data = Cached(name, 'train')
        initial_global = None
        for arm in c.ARMS:
            model = AdaptationModel(arm).cuda().to(memory_format=torch.channels_last)
            parity = legacy_id_gpu_parity(model, data)
            optimizer = optimizer_for(model)
            initial_rows = model.global_model.text_encoder.token_embed.new_weight.detach().clone()
            initial = c.state_hash(model.global_model.state_dict())
            if initial_global is None:
                initial_global = initial
            assert initial == initial_global
            native_initial = c.state_hash(model.native_model.state_dict()) if model.native_model is not None else None
            count = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.trainable_parameters())
            torch.cuda.reset_peak_memory_stats()
            # Exactly one accumulated update; discard all smoke weights.
            plan = [(0, 0), (len(data)-1, 1), (0, 4), (len(data)-1, 8)]
            result = train_epoch(model, data, optimizer, 1, plan=plan)
            assert result['images'] == 4 and result['updates'] == 1
            assert c.state_hash(model.global_model.state_dict()) != initial
            assert not torch.equal(initial_rows, model.global_model.text_encoder.token_embed.new_weight)
            if model.native_model is not None:
                assert c.state_hash(model.native_model.state_dict()) != native_initial
            model.eval()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                image, low, target = to_device(data.get(0))
                output = model(image, low, *tokens(model, data))
                assert output[1].shape == (1, 8, *target.shape[-2:]) and torch.isfinite(output[1]).all()
            reports[f'{name}_{arm}'] = {**result, 'parameters': count, 'trainable_parameters': trainable,
                'peak_allocated_MiB': torch.cuda.max_memory_allocated()/2**20,
                'initial_global_sha256': initial, 'initial_native_sha256': native_initial,
                'smoke_weights_reused': False, 'legacy_id_gpu_parity_exact': parity,
                'new_character_rows_updated': True}
            print('SMOKE_PASSED', name, arm, reports[f'{name}_{arm}']['peak_allocated_MiB'], flush=True)
            del image, low, target, output
            close_model(model, optimizer); del model, optimizer
    assert c.identity() == identity
    c.write(c.HERE/'smoke_report.json', {'status': 'passed', 'identity': identity, 'reports': reports,
            'cpu_tests': checked.testsRun, 'external_test_pixels_read': False, 'completed_utc': c.utc()})


def arm_folder(dataset, arm):
    return c.HERE/f'outputs/{dataset}_seed42/{arm}'


def register_checkpoint(folder, manifest, path, saved):
    epoch = saved['epoch']
    assert saved['identity'] == manifest['identity'] and saved['dataset'] == manifest['dataset'] and saved['arm'] == manifest['arm']
    assert len(saved['history']) == epoch+1
    assert [row['epoch'] for row in saved['history']] == list(range(epoch+1))
    assert saved['history'][:-1] == manifest['history']
    frozen = frozen_from_state(saved['model'])
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    assert c.state_hash(frozen) == saved['frozen_sha256'] == manifest['frozen_sha256']
    if epoch:
        expected_count = 630 if saved['dataset'] == 'bentheimer' else 1600
        row = saved['history'][-1]['training']
        assert row['images'] == expected_count and row['updates'] == (expected_count+3)//4
        assert row['plan_sha256'] == hashlib.sha256(json.dumps(c.epoch_plan(expected_count, epoch)).encode()).hexdigest()
    chosen = selection(saved['history'])
    entry = {'epoch': epoch, 'file': path.name, 'sha256': c.sha(path)}
    manifest['epochs'].append(entry)
    manifest['history'] = saved['history']
    manifest.update(last_epoch=epoch, last_checkpoint=entry, best_epoch=chosen,
                    best_checkpoint=manifest['epochs'][chosen], updated_utc=c.utc(), status='running')
    c.write(folder/'history.json', manifest['history'])
    c.write(folder/'run_manifest.json', manifest)


def run_epoch(dataset_name, arm, epoch, identity, resume):
    folder = arm_folder(dataset_name, arm)
    if not folder.exists():
        assert epoch == 0
        folder.mkdir(parents=True)
        (folder/'epochs').mkdir()
        temporary = AdaptationModel(arm)
        manifest = {'identity': identity, 'dataset': dataset_name, 'arm': arm, 'history': [], 'epochs': [],
                    'last_epoch': -1, 'frozen_sha256': c.state_hash(temporary.frozen_state()), 'status': 'prepared'}
        del temporary
        c.write(folder/'run_manifest.json', manifest)
    manifest = c.read(folder/'run_manifest.json')
    assert manifest['identity'] == identity
    if manifest['last_epoch'] >= epoch:
        for entry in manifest['epochs']:
            assert c.sha(folder/'epochs'/entry['file']) == entry['sha256']
        return
    assert manifest['last_epoch'] == epoch-1
    path = folder/'epochs'/f'epoch_{epoch:03d}.pth'
    if path.exists():
        assert resume, 'An orphan completed checkpoint requires explicit --resume'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        register_checkpoint(folder, manifest, path, saved)
        print(f'RECOVERED completed checkpoint {dataset_name} {arm} epoch={epoch}', flush=True)
        return
    data = Cached(dataset_name, 'train') if epoch else None
    val = Cached(dataset_name, 'val')
    model = AdaptationModel(arm).cuda().to(memory_format=torch.channels_last)
    optimizer = optimizer_for(model)
    if epoch:
        previous = manifest['last_checkpoint']
        assert c.sha(folder/'epochs'/previous['file']) == previous['sha256']
        saved = torch.load(folder/'epochs'/previous['file'], map_location='cpu', weights_only=False)
        assert saved['identity'] == identity and saved['epoch'] == epoch-1
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        assert saved['history'] == manifest['history']
        del saved
    assert c.state_hash(model.frozen_state()) == manifest['frozen_sha256']
    started = time.monotonic()
    try:
        c.write(c.HERE/'progress.json', {'dataset': dataset_name, 'arm': arm, 'epoch': epoch,
                'stage': 'train' if epoch else 'initial_validation', 'pid': os.getpid(), 'updated_utc': c.utc()})
        training = train_epoch(model, data, optimizer, epoch, c.HERE/'progress.json') if epoch else None
        c.write(c.HERE/'progress.json', {'dataset': dataset_name, 'arm': arm, 'epoch': epoch,
                'stage': 'validation', 'pid': os.getpid(), 'updated_utc': c.utc()})
        validation = evaluate(model, val)
        if epoch == 0:
            validation['initialization'] = 'Original main-domain checkpoints; cleaned priors with new untrained character rows'
            validation['previous_prompt_metrics_not_a_parity_target'] = True
        assert c.state_hash(model.frozen_state()) == manifest['frozen_sha256']
        history = manifest['history'] + [{'epoch': epoch, 'training': training, 'validation': validation,
                                         'total_seconds': time.monotonic()-started}]
        save_checkpoint(path, {'identity': identity, 'dataset': dataset_name, 'arm': arm, 'epoch': epoch,
                              'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history,
                              'frozen_sha256': manifest['frozen_sha256'], 'completed_utc': c.utc()})
        saved = {'identity': identity, 'dataset': dataset_name, 'arm': arm, 'epoch': epoch,
                 'model': model.state_dict(), 'history': history, 'frozen_sha256': manifest['frozen_sha256']}
        register_checkpoint(folder, manifest, path, saved)
        print(f'EPOCH_COMPLETE {dataset_name} {arm} epoch={epoch} mIoU={primary(validation):.8f} best_epoch={manifest["best_epoch"]} seconds={time.monotonic()-started:.1f}', flush=True)
    finally:
        close_model(model, optimizer)


def finish_arm(dataset, arm, identity):
    folder = arm_folder(dataset, arm)
    meta = c.read(folder/'run_manifest.json')
    assert meta['identity'] == identity and meta['last_epoch'] == c.CONFIG['epochs']
    assert meta['best_epoch'] == selection(meta['history'])
    if meta['status'] == 'completed':
        assert c.sha(folder/'selected_validation.json') == meta['selected_sha256']
        return c.read(folder/'selected_validation.json')
    entry = meta['best_checkpoint']
    assert c.sha(folder/'epochs'/entry['file']) == entry['sha256']
    saved = torch.load(folder/'epochs'/entry['file'], map_location='cpu', weights_only=False)
    model = AdaptationModel(arm).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(saved['model'], strict=True)
    try:
        # Unique folder preserves any partial prior verification attempt.
        stamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        destination = folder/f'selected_predictions_{stamp}'
        result = evaluate(model, Cached(dataset, 'val'), save_to=destination)
        expected = meta['history'][entry['epoch']]['validation']
        for grid in c.GRIDS:
            for rule in c.RULES:
                np.testing.assert_array_equal(result['metrics'][grid][rule]['confusion_matrix'],
                                              expected['metrics'][grid][rule]['confusion_matrix'])
        result.update(selected_epoch=entry['epoch'], checkpoint=entry, predictions_directory=destination.name,
                      frozen_sha256=c.state_hash(model.frozen_state()))
        assert result['frozen_sha256'] == meta['frozen_sha256']
        c.write(folder/'selected_validation.json', result)
        meta.update(status='completed', selected_sha256=c.sha(folder/'selected_validation.json'),
                    selected_validation_verified=True, completed_utc=c.utc())
        c.write(folder/'run_manifest.json', meta)
        return result
    finally:
        close_model(model)


def summary(reports):
    result = {'status': 'completed', 'seed': 42, 'scope': 'Single-seed target adaptation with cleaned priors and append-only vocabulary',
              'datasets': {}, 'external_test_pixels_read': False, 'main_model_promoted': False}
    lines = ['# 清洗先验后的目标域监督适配：seed42', '', '只使用目标域train拟合、val选模，非零样本、非测试、非三种子结果。', '',
             '| 数据集 | Global mIoU | Native mIoU | 差值（百分点） | 最佳轮数 G/N |', '|---|---:|---:|---:|---|']
    all_usable, positive = [], []
    for dataset in c.DATASETS:
        a, b = reports[f'{dataset}_global'], reports[f'{dataset}_native_detail']
        ga, nb = primary(a), primary(b)
        foreground = b['metrics']['canonical512']['collapse8']['class_IoU'][1:]
        checks = {'native_mIoU_at_least_0.70': nb >= .70, 'all_foreground_IoU_at_least_0.50': min(foreground) >= .50,
                  'no_more_than_0.005_regression': nb-ga >= -.005, 'gain_at_least_0.005': nb-ga >= .005}
        all_usable += [checks[k] for k in ('native_mIoU_at_least_0.70', 'all_foreground_IoU_at_least_0.50', 'no_more_than_0.005_regression')]
        positive.append(checks['gain_at_least_0.005'])
        result['datasets'][dataset] = {'global': a, 'native_detail': b, 'difference_pp': 100*(nb-ga), 'screen': checks}
        lines.append(f'| {dataset} | {ga:.6f} | {nb:.6f} | {100*(nb-ga):+.4f} | {a["selected_epoch"]}/{b["selected_epoch"]} |')
    result['replication_screen_passed'] = all(all_usable) and any(positive)
    lines += ['', f'预定投入三种子复验筛选：{result["replication_screen_passed"]}。工程门槛不代表统计显著性。',
              '全部逐类、双网格、双决策结果见 summary.json；首次最大主指标选模，包括epoch0。',
              '清洗与新增字符学习联合实施；不能单独归因于措辞，亦不证明新术语语义理解。',
              '油体形态头未用四相标签训练，本表不证明外部五形态分类。额外输入/主域预训练参数/计算量差异保留。', '']
    c.write(c.HERE/'summary.json', result)
    (c.HERE/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def run(resume):
    c.stop_check()
    identity = c.identity()
    smoke_report = c.read(c.HERE/'smoke_report.json')
    assert smoke_report['status'] == 'passed' and smoke_report['identity'] == identity
    verify_cache()
    c.setup()
    path = c.HERE/'run_manifest.json'
    if path.exists():
        meta = c.read(path)
        assert meta['identity'] == identity
        if meta['status'] == 'completed':
            print('ALREADY_COMPLETE preserved', flush=True); return
        assert resume, 'Existing run requires --resume after checking why it stopped'
        meta.setdefault('resumptions', []).append({'time_utc': c.utc(), 'previous_status': meta['status'], 'pid': os.getpid()})
    else:
        assert not resume
        snapshot = c.HERE/'source_snapshot'
        snapshot.mkdir(exist_ok=False)
        mapping = {}
        for index, (source, digest) in enumerate(identity['sources'].items()):
            target = snapshot/f'{index:02d}_{Path(source).name}'
            shutil.copy2(source, target)
            assert c.sha(target) == digest
            mapping[source] = target.name
        meta = {'identity': identity, 'source_snapshot': mapping, 'created_utc': c.utc(), 'completed_arms': {},
                'command': sys.argv, 'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                     'device': torch.cuda.get_device_name(), 'gpu_memory_fraction_cap': c.CONFIG['gpu_memory_fraction']},
                'external_test_pixels_read': False}
    meta.update(status='running', pid=os.getpid())
    c.write(path, meta)
    reports = {}
    try:
        for dataset in c.DATASETS:
            for epoch in range(c.CONFIG['epochs']+1):
                for arm in c.ARMS:
                    meta.update(current_dataset=dataset, current_arm=arm, current_epoch=epoch, updated_utc=c.utc())
                    c.write(path, meta)
                    run_epoch(dataset, arm, epoch, identity, resume)
            for arm in c.ARMS:
                key = f'{dataset}_{arm}'
                reports[key] = finish_arm(dataset, arm, identity)
                meta['completed_arms'][key] = c.sha(arm_folder(dataset, arm)/'selected_validation.json')
                c.write(path, meta)
        assert c.identity() == identity
        report = summary(reports)
        meta.update(status='completed', completed_utc=c.utc(), replication_screen_passed=report['replication_screen_passed'])
        c.write(c.HERE/'progress.json', {'stage': 'completed', 'time_utc': c.utc(), 'pid': os.getpid()})
        print('ALL_PAIRS_COMPLETED', flush=True)
    except BaseException as error:
        meta.update(status='paused' if isinstance(error, InterruptedError) else 'failed', error=repr(error),
                    traceback=traceback.format_exc(), stopped_utc=c.utc())
        raise
    finally:
        c.write(path, meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert not (args.smoke and args.resume)
    with c.lock():
        smoke() if args.smoke else run(args.resume)
