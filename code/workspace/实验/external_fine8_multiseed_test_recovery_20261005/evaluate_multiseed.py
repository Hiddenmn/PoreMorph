"""Locked new-seed fine8 TEST replication; no test-based model decisions."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
UP = EXP / 'external_fine8_multiseed_20261004'
BODY = EXP / 'external_fine8_multiseed_body_recovery_20261005'
OLD_TEST = EXP / 'external_full_vs_global_fine8_test_20261004'
SEEDS = (2024, 2025)
DATASETS = ('bentheimer', 'wf1')
GRIDS = ('canonical512', 'native')
spec = importlib.util.spec_from_file_location('locked_seed42_evaluator', OLD_TEST / 'evaluate_locked.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
sys.path.insert(0, str(BODY))
import run_heads as heads
c = heads.c
assert old.c is c
assert old.h.ORIGINAL_CONFIG['protocol'] == 'external_fine8_body_stage2_seed42_v1'
# The old evaluator's import installs its seed42 extension config globally.
# Restore the exact base config used by the standalone new-seed body process.
heads.BASE_CONFIG = copy.deepcopy(old.h.ORIGINAL_CONFIG)


def case(seed):
    assert seed in SEEDS
    return HERE / 'cases' / f'seed{seed}'


def stop_check():
    for folder in (c.ROOT, HERE, UP, BODY, c.HERE):
        for name in ('STOP.txt', 'STOP_AFTER_EPOCH.txt'):
            if (folder / name).exists():
                raise InterruptedError(f'User stop marker: {folder / name}')


def selected_global(seed, dataset, upstream_identity):
    folder = UP / f'cases/seed{seed}/outputs/{dataset}_seed{seed}/global'
    meta = c.read(folder / 'run_manifest.json')
    selected = c.read(folder / 'selected_validation.json')
    assert meta['status'] == 'completed' and meta['identity'] == upstream_identity
    assert len(meta['history']) == len(meta['epochs']) == 21
    best = max(range(21), key=lambda i:
        meta['history'][i]['validation']['metrics']['canonical512']['eightclass']['mIoU'])
    assert best == meta['best_epoch'] == selected['selected_epoch']
    assert selected['checkpoint'] == meta['epochs'][best]
    assert selected['metrics'] == meta['history'][best]['validation']['metrics']
    assert selected['split'] == 'val' and selected['saved_pixel_replay_exact']
    assert c.sha(folder / 'selected_validation.json') == meta['selected_sha256']
    checkpoint = folder / 'epochs' / selected['checkpoint']['file']
    assert c.sha(checkpoint) == selected['checkpoint']['sha256']
    return dict(folder=str(folder), checkpoint=str(checkpoint),
                checkpoint_sha256=selected['checkpoint']['sha256'], epoch=best,
                selected_sha256=meta['selected_sha256'], frozen_sha256=meta['frozen_sha256'])


def source_identity(seed):
    heads.configure(seed)
    stop_check()
    upstream_root = c.read(UP / 'run_manifest.json')
    body_root = c.read(BODY / 'run_manifest.json')
    assert upstream_root['status'] == body_root['status'] == 'completed'
    assert upstream_root['test_pixels_read'] is False
    assert body_root['test_pixels_read'] is False
    assert set(upstream_root['completed_seeds']) == set(body_root['completed_seeds']) == {'2024', '2025'}
    upstream_case = c.read(UP / f'cases/seed{seed}/run_manifest.json')
    body_case = c.read(BODY / f'cases/seed{seed}/run_manifest.json')
    body_ids = heads.identity()
    assert upstream_case['status'] == body_case['status'] == 'completed'
    assert upstream_case['external_test_pixels_read'] is False
    assert body_case['test_pixels_read'] is False
    assert body_case['identity'] == body_ids
    assert c.sha(BODY / f'cases/seed{seed}/summary.json') == body_root['completed_seeds'][str(seed)]
    assert c.read(BODY / f'cases/seed{seed}/verification.json')['status'] == 'PASS'
    selected = {}
    for dataset in DATASETS:
        global_model = selected_global(seed, dataset, upstream_case['identity'])
        native = body_ids['selected_upstream'][dataset]
        folder = BODY / f'cases/seed{seed}/outputs/{dataset}/separate'
        meta = c.read(folder / 'run_manifest.json')
        chosen = c.read(folder / 'selected_validation.json')
        assert meta['status'] == 'completed' and meta['identity'] == body_ids
        assert len(meta['history']) == len(meta['checkpoints']) == 13
        assert meta['best_epoch'] == c.selected_epoch(meta['history']) == chosen['selected_epoch']
        assert chosen['checkpoint'] == meta['checkpoints'][chosen['selected_epoch']]
        assert chosen['validation'] == meta['history'][chosen['selected_epoch']]['validation']
        assert c.sha(folder / 'selected_validation.json') == meta['selected_sha256']
        checkpoint = folder / 'epochs' / chosen['checkpoint']['file']
        assert c.sha(checkpoint) == chosen['checkpoint']['sha256']
        selected[dataset] = dict(global_model=global_model, native=native,
            global_folder=global_model['folder'], native_epoch=native['epoch'],
            body_folder=str(folder), body_checkpoint=str(checkpoint),
            body_checkpoint_sha256=chosen['checkpoint']['sha256'],
            body_epoch=chosen['selected_epoch'], body_selected_sha256=meta['selected_sha256'])
    frozen = c.read(old.FINE / 'label_freeze.json')
    assert frozen['status'] == 'frozen_after_user_reported_expert_acceptance'
    old_test = c.read(OLD_TEST / 'run_manifest.json')
    assert old_test['status'] == 'completed'
    old_cache = c.read(old.OLDTEST / 'cache/manifest.json')
    assert old_cache['status'] == 'completed'
    return dict(protocol='external_fine8_multiseed_locked_test_logit_support_recovery_v1', seed=seed,
        sources={str(HERE / name): c.sha(HERE / name)
                 for name in ('PROTOCOL.md', 'evaluate_multiseed.py')},
        inputs={str(path): c.sha(path) for path in
                (UP / 'run_manifest.json', UP / f'cases/seed{seed}/run_manifest.json',
                 BODY / 'run_manifest.json', BODY / f'cases/seed{seed}/run_manifest.json',
                 BODY / f'cases/seed{seed}/verification.json',
                 OLD_TEST / 'run_manifest.json', old.OLDTEST / 'cache/manifest.json',
                 old.FINE / 'label_freeze.json')},
        upstream_identity=upstream_case['identity'], body_identity=body_ids,
        selected=selected, label_fingerprint=frozen['dataset_fingerprint'],
        same_source_spatial_test_blocks=True,
        test_split_previously_exposed_for_four_phase=True)


def load_upstream(selected, ids, dataset, arm):
    checkpoint = Path(selected['checkpoint'])
    assert c.sha(checkpoint) == selected['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['identity'] == ids['upstream_identity']
    assert saved['dataset'] == dataset and saved['arm'] == arm
    assert saved['epoch'] == selected['epoch']
    assert saved['frozen_sha256'] == selected['frozen_sha256']
    assert c.u.c.state_hash(c.u.legacy.frozen_from_state(saved['model'])) == selected['frozen_sha256']
    assert all(torch.isfinite(value).all() for value in saved['model'].values())
    return saved


class Pipeline(old.Pipeline):
    def __init__(self, dataset, ids):
        chosen = ids['selected'][dataset]
        global_state = load_upstream(chosen['global_model'], ids, dataset, 'global')
        native_state = heads.load_upstream_checkpoint(chosen['native'], ids['body_identity'])
        head_state = torch.load(chosen['body_checkpoint'], map_location='cpu', weights_only=False)
        assert c.sha(chosen['body_checkpoint']) == chosen['body_checkpoint_sha256']
        assert head_state['identity'] == ids['body_identity']
        assert head_state['dataset'] == dataset and head_state['arm'] == 'separate'
        assert head_state['epoch'] == chosen['body_epoch']
        self.global_model = c.u.legacy.AdaptationModel('global')
        self.native_model = c.u.legacy.AdaptationModel('native_detail')
        self.head = c.BodyInterfaceDecoder('separate')
        self.global_model.load_state_dict(global_state['model'], strict=True)
        self.native_model.load_state_dict(native_state['model'], strict=True)
        self.head.load_state_dict(head_state['model'], strict=True)
        for model in (self.global_model, self.native_model, self.head):
            model.eval().requires_grad_(False).cuda()
        self.global_model.to(memory_format=torch.channels_last)
        self.native_model.to(memory_format=torch.channels_last)
        self.features = None
        self.hook = self.native_model.native_model.up_fine.register_forward_hook(
            lambda module, args, output: setattr(self, 'features', output))
        self.description = c.u.prior_schema.PROMPTS[dataset]
        self.g_ids, self.g_attention = self.global_model.tokenizer.encode(self.description)
        self.n_ids, self.n_attention = self.native_model.tokenizer.encode(self.description)
        self.g_ids, self.g_attention = self.g_ids.cuda()[None], self.g_attention.cuda()[None]
        self.n_ids, self.n_attention = self.n_ids.cuda()[None], self.n_attention.cuda()[None]

    @torch.inference_mode()
    def predict(self, image, low, diagnostics=False):
        """Same frozen inference, with authoritative logit-argmax support."""
        assert image.ndim == low.ndim == 2 and low.shape == (512, 512)
        image = np.asarray(image, np.float32)
        ni = torch.from_numpy(image.copy()).cuda()[None, None]
        li = torch.from_numpy(np.asarray(low, np.float32).copy()).cuda()[None, None].repeat(1, 3, 1, 1)
        li = li.to(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            g512_logits, gn_logits = self.global_model(ni, li, self.g_ids, self.g_attention)
            _, n_logits = self.native_model(ni, li, self.n_ids, self.n_attention)
        g512 = g512_logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        gn = gn_logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        native = n_logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        probabilities = n_logits.float().softmax(1)[0].cpu().numpy()
        features = self.features.float()[0].cpu().numpy()
        self.features = None
        descriptors, components, descriptor_pred = c.describe(
            image, probabilities, features, prediction=native)
        np.testing.assert_array_equal(native, descriptor_pred)
        n = len(descriptors['points'])
        if n:
            points = torch.from_numpy(np.asarray(descriptors['points'], np.float32)).cuda()
            valid = torch.from_numpy(np.asarray(descriptors['valid'], bool)).cuda()
            summary = torch.from_numpy(np.asarray(descriptors['summary'], np.float32)).cuda()
            assigned = self.head(points, valid, summary).argmax(1).cpu().numpy().astype(np.uint8)
        else:
            assigned = np.empty(0, np.uint8)
        complete = c.assign_bodies(native, components, assigned.astype(np.int64) + 3)
        np.testing.assert_array_equal(complete >= 3, native >= 3)
        np.testing.assert_array_equal(complete[native < 3], native[native < 3])
        result = {'native_global': gn, 'canonical512_global': g512, 'native_upstream': native,
            'native_complete': complete,
            'canonical512_complete': old.s.nearest512_ids(complete).astype(np.uint8)}
        if diagnostics:
            result.update(assigned=assigned, descriptors=descriptors, components=components)
        return result


def configure(seed):
    heads.configure(seed)
    old.HERE = case(seed)
    old.BODY = BODY / f'cases/seed{seed}'
    old.Pipeline = Pipeline
    old.stop_check = stop_check


def contracts():
    assert not torch.cuda.is_initialized()
    assert c.read(OLD_TEST / 'run_manifest.json')['status'] == 'completed'
    cache = c.read(old.OLDTEST / 'cache/manifest.json')
    assert cache['status'] == 'completed' and set(cache['entries']) == set(DATASETS)
    assert c.read(BODY / 'preparation_checks.json')['status'] == 'PASS'
    for seed in SEEDS:
        configure(seed)
        assert c.CONFIG['seed'] == seed and c.CONFIG['epochs'] == 12
        source_identity(seed)
    assert not torch.cuda.is_initialized()
    c.write(HERE / 'preparation_checks.json', dict(status='PASS', seeds=list(SEEDS),
        source_sha256=c.sha(HERE / 'evaluate_multiseed.py'),
        test_pixels_read=False, cuda_initialized=False, checked_utc=c.utc()))
    print('MULTISEED_TEST_CPU_CONTRACTS_PASS; NO_TEST_PIXELS_READ', flush=True)


def smoke():
    assert c.read(HERE / 'preparation_checks.json')['source_sha256'] == c.sha(HERE / 'evaluate_multiseed.py')
    assert c.read(UP / 'run_manifest.json')['status'] == 'completed'
    assert c.read(BODY / 'run_manifest.json')['status'] == 'completed'
    c.setup('cuda')
    reports = {}
    for seed in SEEDS:
        configure(seed)
        ids = source_identity(seed)
        report = {}
        for dataset in DATASETS:
            pipeline = Pipeline(dataset, ids)
            try:
                report[dataset] = old.val_parity(dataset, pipeline, ids)
            finally:
                pipeline.close()
        path = case(seed)
        path.mkdir(parents=True, exist_ok=True)
        c.write(path / 'smoke_report.json', dict(status='PASS', seed=seed,
            identity=ids, reports=report, test_pixels_read=False, completed_utc=c.utc()))
        reports[str(seed)] = c.sha(path / 'smoke_report.json')
    c.write(HERE / 'smoke_report.json', dict(status='PASS', reports=reports,
        test_pixels_read=False, completed_utc=c.utc()))
    print('MULTISEED_TEST_VAL_PARITY_PASS; NO_TEST_PIXELS_READ', flush=True)


def verify_dataset_readonly(dataset, ids):
    """Recount saved TEST predictions without rewriting the locked report."""
    folder = old.HERE / 'outputs' / dataset
    meta = c.read(folder / 'run_manifest.json')
    assert meta['status'] == 'completed' and meta['identity'] == ids
    assert c.sha(folder / 'report.json') == meta['report_sha256']
    rows, _ = old.test_records(dataset)
    assert len(meta['records']) == len(rows)
    matrices = {grid: {arm: np.zeros((8, 8), np.int64) for arm in ('global', 'complete')}
                for grid in GRIDS}
    for index, saved in enumerate(meta['records']):
        assert saved['index'] == index and saved['label_id'] == rows[index]['label_id']
        path = folder / 'predictions' / f'{index:04d}.npz'
        assert c.sha(path) == saved['sha256']
        with np.load(path, allow_pickle=False) as file:
            np.testing.assert_array_equal(file['target_native'],
                old.read_label(dataset, rows[index], tuple(file['target_native'].shape)))
        values = old.histograms(path)
        for grid in GRIDS:
            for arm in ('global', 'complete'):
                matrices[grid][arm] += values[grid + '_' + arm]
    metrics = {grid: {arm: c.u.metrics(matrices[grid][arm])
                      for arm in ('global', 'complete')} for grid in GRIDS}
    report = c.read(folder / 'report.json')
    assert report['source_identity'] == ids and report['metrics'] == metrics
    assert report['images'] == len(rows) and report['independently_replayed']
    return report


def run_case(seed, resume):
    configure(seed)
    ids = source_identity(seed)
    assert c.read(case(seed) / 'smoke_report.json')['identity'] == ids
    root = case(seed) / 'run_manifest.json'
    if root.exists():
        assert resume
        meta = c.read(root)
        assert meta['identity'] == ids and meta['status'] != 'completed'
    else:
        assert not resume
        meta = dict(status='prepared', seed=seed, identity=ids,
                    test_pixels_read=False, created_utc=c.utc())
    meta.update(status='running', pid=os.getpid(), updated_utc=c.utc())
    c.write(root, meta)
    try:
        reports = {}
        for dataset in DATASETS:
            stop_check()
            path = case(seed) / 'outputs' / dataset / 'run_manifest.json'
            if path.exists() and c.read(path)['status'] == 'completed':
                reports[dataset] = verify_dataset_readonly(dataset, ids)
            else:
                meta.update(current_dataset=dataset, test_pixels_read=True, updated_utc=c.utc())
                c.write(root, meta)
                reports[dataset] = old.evaluate_dataset(dataset, ids, resume=path.exists())
            meta.update(current_dataset=dataset, test_pixels_read=True, updated_utc=c.utc())
            c.write(root, meta)
        report = dict(status='completed', seed=seed, datasets=reports,
            scope='target-domain adapted fine8; historical same-core TEST blocks',
            test_used_for_selection=False)
        c.write(case(seed) / 'summary.json', report)
        meta.update(status='completed', current_dataset=None,
                    summary_sha256=c.sha(case(seed) / 'summary.json'), completed_utc=c.utc())
        c.write(root, meta)
        return meta['summary_sha256']
    except BaseException as error:
        meta.update(status='paused' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
                    error=repr(error), traceback=traceback.format_exc(), stopped_utc=c.utc())
        c.write(root, meta)
        raise


def aggregate():
    seed42 = c.read(OLD_TEST / 'summary.json')
    assert seed42['status'] == 'completed' and seed42['seed'] == 42
    assert c.sha(OLD_TEST / 'summary.json') == c.read(OLD_TEST / 'run_manifest.json')['summary_sha256']
    reports = {42: seed42, **{seed: c.read(case(seed) / 'summary.json') for seed in SEEDS}}
    table = []
    summary = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for grid in GRIDS:
            pairs = []
            for seed in (42, *SEEDS):
                scores = reports[seed]['datasets'][dataset]['metrics'][grid]
                a, b = scores['global']['mIoU'], scores['complete']['mIoU']
                pairs.append(dict(seed=seed, global_mIoU=a, complete_mIoU=b,
                    delta_pp=100 * (b - a), global_metrics=scores['global'],
                    complete_metrics=scores['complete']))
                table.append((dataset, grid, seed, a, b, 100 * (b - a)))
            deltas = [row['delta_pp'] for row in pairs]
            summary[dataset][grid] = dict(pairs=pairs, mean_delta_pp=statistics.mean(deltas),
                sample_sd_delta_pp=statistics.stdev(deltas), positive_count=sum(x > 0 for x in deltas))
    out = dict(status='completed', seeds=[42, *SEEDS], datasets=summary,
        scope='target-adapted; same-source spatial TEST blocks; not independent new cores',
        test_used_for_selection=False, complete_has_additional_head_training=True,
        v7_annotation_provenance_is_recorded=True)
    c.write(HERE / 'summary.json', out)
    lines = ['# 外部八类多种子锁定测试：完整模型 vs 自创 Global', '',
        '目标域有标签适配；同一来源的空间分块，不是独立新岩心。seed42 取自已完成的原始锁定测试。',
        '测试结果未用于改训练、选种子或选检查点；完整系统相对 Global 多了油体头适配预算。', '',
        '| 域 | 网格 | 种子 | Global mIoU | 完整模型 mIoU | 差值/百分点 |',
        '|---|---|---:|---:|---:|---:|']
    for dataset, grid, seed, a, b, delta in table:
        lines.append(f'| {dataset} | {grid} | {seed} | {a:.8f} | {b:.8f} | {delta:+.4f} |')
    lines += ['', '| 域 | 网格 | 配对均值/百分点 | 样本标准差 | 正增益种子数 |',
              '|---|---|---:|---:|---:|']
    for dataset in DATASETS:
        for grid in GRIDS:
            row = summary[dataset][grid]
            lines.append(f'| {dataset} | {grid} | {row["mean_delta_pp"]:+.4f} | '
                         f'{row["sample_sd_delta_pp"]:.4f} | {row["positive_count"]}/3 |')
    lines += ['', '逐类 IoU、Dice 和混淆矩阵保存在 summary.json；不隐藏负增益。']
    (HERE / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return c.sha(HERE / 'summary.json')


def run(resume):
    stop_check()
    # Match the standalone seed-42 evaluator and VAL smoke in this process.
    # Each CLI stage starts a new interpreter; smoke's resource setup is not inherited.
    c.setup('cuda')
    pre = c.read(HERE / 'preparation_checks.json')
    assert pre['status'] == 'PASS' and pre['source_sha256'] == c.sha(HERE / 'evaluate_multiseed.py')
    smoke_report = c.read(HERE / 'smoke_report.json')
    assert smoke_report['status'] == 'PASS' and not smoke_report['test_pixels_read']
    assert set(smoke_report['reports']) == {'2024', '2025'}
    for seed in SEEDS:
        configure(seed)
        ids = source_identity(seed)
        report = c.read(case(seed) / 'smoke_report.json')
        assert report['status'] == 'PASS' and report['identity'] == ids
        assert c.sha(case(seed) / 'smoke_report.json') == smoke_report['reports'][str(seed)]
    root = HERE / 'run_manifest.json'
    if root.exists():
        assert resume
        meta = c.read(root)
        assert meta['status'] != 'completed'
        assert meta['smoke_sha256'] == c.sha(HERE / 'smoke_report.json')
    else:
        assert not resume
        meta = dict(status='prepared', completed_seeds={},
            test_pixels_read=False, smoke_sha256=c.sha(HERE / 'smoke_report.json'),
            created_utc=c.utc(), command=sys.argv)
    meta.update(status='running', pid=os.getpid(), updated_utc=c.utc())
    c.write(root, meta)
    try:
        for seed in SEEDS:
            configure(seed)
            stop_check()
            meta.update(current_seed=seed, updated_utc=c.utc())
            c.write(root, meta)
            if str(seed) not in meta['completed_seeds']:
                meta['test_pixels_read'] = True
                c.write(root, meta)
                digest = run_case(seed, resume=(case(seed) / 'run_manifest.json').exists())
                meta['completed_seeds'][str(seed)] = digest
                c.write(root, meta)
            else:
                assert c.sha(case(seed) / 'summary.json') == meta['completed_seeds'][str(seed)]
        meta.update(status='completed', current_seed=None,
            summary_sha256=aggregate(), completed_utc=c.utc())
        c.write(root, meta)
        print('MULTISEED_FINE8_LOCKED_TEST_COMPLETE', flush=True)
    except BaseException as error:
        meta.update(status='paused' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
                    error=repr(error), traceback=traceback.format_exc(), stopped_utc=c.utc())
        c.write(root, meta)
        raise


def verify():
    meta = c.read(HERE / 'run_manifest.json')
    assert meta['status'] == 'completed' and set(meta['completed_seeds']) == {'2024', '2025'}
    for seed in SEEDS:
        configure(seed)
        ids = source_identity(seed)
        case_meta = c.read(case(seed) / 'run_manifest.json')
        assert case_meta['status'] == 'completed' and case_meta['identity'] == ids
        assert c.sha(case(seed) / 'summary.json') == meta['completed_seeds'][str(seed)]
        for dataset in DATASETS:
            verify_dataset_readonly(dataset, ids)
    assert c.sha(HERE / 'summary.json') == meta['summary_sha256']
    print('MULTISEED_FINE8_TEST_INDEPENDENT_REPLAY_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('contracts', 'smoke', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert not args.resume or args.stage == 'run'
    with heads.job_lock():
        # The body's own lock prevents a simultaneous training/test process.
        if args.stage == 'contracts':
            contracts()
        elif args.stage == 'smoke':
            smoke()
        elif args.stage == 'run':
            run(args.resume)
        else:
            verify()
