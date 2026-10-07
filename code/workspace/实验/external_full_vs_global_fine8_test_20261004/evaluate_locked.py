"""One locked eight-class external TEST comparison: complete model vs own Global."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
UP = EXP / 'external_eightclass_extension20_recovery_20261004'
BODY = EXP / 'external_body_extension20_20261004'
FINE = EXP / 'external_eightclass_adaptation_20261003'
OLDTEST = EXP / 'external_clean_prior_test_20261001'
sys.path.insert(0, str(BODY))
import run_extension_heads as h
h.install()
c, s = h.c, h.p.s
DATASETS = ('bentheimer', 'wf1')
GRIDS = ('canonical512', 'native')


def stop_check():
    h.stop_check()
    for name in ('STOP.txt', 'STOP_AFTER_EPOCH.txt'):
        if (HERE / name).exists():
            raise InterruptedError(f'User pause marker: {HERE / name}')


def key(path):
    return os.path.normcase(str(Path(path).resolve()))


def source_identity():
    stop_check()
    body_ids = h.identity()
    body = c.read(BODY / 'run_manifest.json')
    assert body['status'] == 'completed' and body['identity'] == body_ids
    assert c.read(BODY / 'verification.json')['status'] == 'PASS'
    up = c.read(UP / 'run_manifest.json')
    assert up['status'] == 'completed' and c.read(UP / 'verification.json')['status'] == 'PASS'
    assert up['identity'] == body_ids['upstream_identity']
    old = c.read(OLDTEST / 'cache/manifest.json')
    assert old['status'] == 'completed' and set(old['entries']) == set(DATASETS)
    frozen = c.read(FINE / 'label_freeze.json')
    assert frozen['status'] == 'frozen_after_user_reported_expert_acceptance'
    assert len(frozen['records']) == 3750
    selected = {}
    for dataset in DATASETS:
        gfolder = UP / f'outputs/{dataset}_seed42/global'
        gmeta = c.read(gfolder / 'run_manifest.json')
        gsel = c.read(gfolder / 'selected_validation.json')
        h.validate_selection(gmeta, gsel, up['identity'])
        assert c.sha(gfolder / 'selected_validation.json') == gmeta['selected_sha256']
        assert gsel['saved_pixel_replay_exact']
        ge = gsel['checkpoint']
        assert c.sha(ge['path']) == ge['sha256']
        bf = BODY / f'outputs/{dataset}/separate'
        bm, bs = c.read(bf / 'run_manifest.json'), c.read(bf / 'selected_validation.json')
        assert bm['status'] == 'completed' and bm['identity'] == body_ids
        assert c.sha(bf / 'selected_validation.json') == bm['selected_sha256']
        assert bs['selected_epoch'] == bm['best_epoch'] == c.selected_epoch(bm['history'])
        be = bs['checkpoint']
        bp = bf / 'epochs' / be['file']
        assert c.sha(bp) == be['sha256']
        selected[dataset] = dict(global_folder=str(gfolder), global_selected_sha256=gmeta['selected_sha256'],
            global_checkpoint=ge, native=body_ids['selected_upstream'][dataset],
            body_folder=str(bf), body_selected_sha256=bm['selected_sha256'],
            body_checkpoint=str(bp), body_checkpoint_sha256=be['sha256'],
            global_epoch=gsel['selected_epoch'], native_epoch=body_ids['selected_upstream'][dataset]['epoch'],
            body_epoch=bs['selected_epoch'])
    return dict(protocol='external_full_vs_global_fine8_test_seed42_v1', seed=42,
        sources={str(HERE / name): c.sha(HERE / name) for name in ('PROTOCOL.md', 'evaluate_locked.py')},
        inputs={str(p): c.sha(p) for p in (BODY / 'run_manifest.json', BODY / 'verification.json',
            UP / 'run_manifest.json', UP / 'verification.json', FINE / 'label_freeze.json',
            OLDTEST / 'cache/manifest.json')}, body_identity=body_ids,
        selected=selected, label_fingerprint=frozen['dataset_fingerprint'],
        test_split_previously_exposed_for_four_phase=True)


def test_records(dataset):
    old = c.read(OLDTEST / 'cache/manifest.json')['entries'][dataset]
    frozen = c.read(FINE / 'label_freeze.json')
    lookup = {key(r['raw_source']): r for r in frozen['records'] if r['split'] == 'test'}
    assert len(lookup) == 1130
    rows = []
    for record in old['records']:
        assert record['split'] == 'test' and record['dataset'] == dataset
        found = lookup[key(record['raw_path'])]
        assert found['raw_sha256'] == record['raw_sha256']
        assert found['split'] == 'test'
        rows.append(dict(label_id=found['id'], label=str(FINE / found['snapshot_label']),
            label_sha256=found['label_sha256'], raw_path=record['raw_path'], raw_sha256=record['raw_sha256']))
    assert len(rows) == (180 if dataset == 'bentheimer' else 950)
    assert len({r['label_id'] for r in rows}) == len(rows)
    for field in ('image', 'low'):
        info = old['files'][field]
        assert c.sha(info['path']) == info['sha256']
    return rows, old


def read_label(dataset, record, shape):
    assert c.sha(record['label']) == record['label_sha256']
    with Image.open(record['label']) as handle:
        target = np.asarray(handle).copy()
    if dataset == 'bentheimer':
        target = target[20:1000, 10:990]
    assert target.shape == shape and target.dtype == np.uint8
    assert 0 <= target.min() <= target.max() <= 7
    return target


class Pipeline:
    def __init__(self, dataset, ids):
        chosen = ids['selected'][dataset]
        global_state = h.recovery.e.load_saved(chosen['global_checkpoint'],
            ids['body_identity']['upstream_identity'], dataset, 'global')
        native_state = c.load_upstream_checkpoint(chosen['native'], ids['body_identity'])
        head_state = torch.load(chosen['body_checkpoint'], map_location='cpu', weights_only=False)
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
        descriptors, components, descriptor_pred = c.describe(image, probabilities, features)
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
            'native_complete': complete, 'canonical512_complete': s.nearest512_ids(complete).astype(np.uint8)}
        if diagnostics:
            result.update(assigned=assigned, descriptors=descriptors, components=components)
        return result

    def close(self):
        self.hook.remove()
        for model in (self.global_model, self.native_model, self.head):
            model.cpu()
        torch.cuda.empty_cache()


def val_parity(dataset, pipeline, ids):
    data = c.CachedFine(dataset, 'val')
    selected = ids['selected'][dataset]
    gsel = c.read(Path(selected['global_folder']) / 'selected_validation.json')
    nsel = c.read(Path(selected['native']['folder']) / 'selected_validation.json')
    bfolder = Path(selected['body_folder'])
    bsel = c.read(bfolder / 'selected_validation.json')
    with np.load(bfolder / bsel['assignments_file'], allow_pickle=False) as array:
        indices, assignments = array['indices'].copy(), array['assigned'].copy()
    records = c.read(BODY / 'cache' / dataset / 'records.json')
    val = {r['row']: r for r in records if r['split'] == 'val'}
    for row in (0, len(data) - 1):
        out = pipeline.predict(data.images[row], data.low[row], diagnostics=True)
        for ref, name in ((gsel, 'global'), (nsel, 'native')):
            record = ref['saved_predictions'][row]
            with np.load(Path(selected['global_folder'] if name == 'global' else selected['native']['folder']) /
                         ref['predictions_directory'] / record['file'], allow_pickle=False) as file:
                expected = file['canonical512_eightclass'] if name == 'global' else file['native_eightclass']
                actual = out['canonical512_global'] if name == 'global' else out['native_upstream']
                np.testing.assert_array_equal(actual, expected)
        record = val[row]
        segment = (indices >= record['start']) & (indices < record['end'])
        assert segment.sum() == record['bodies']
        np.testing.assert_array_equal(out['assigned'], assignments[segment])
        with np.load(BODY / 'cache' / dataset / 'shards' / record['file'], allow_pickle=False) as file:
            np.testing.assert_array_equal(out['components'], file['components'])
            for name in ('points', 'valid', 'summary'):
                np.testing.assert_array_equal(out['descriptors'][name], file[name])
            expected_dense = c.assign_bodies(file['prediction'], file['components'], assignments[segment] + 3)
            np.testing.assert_array_equal(out['native_complete'], expected_dense)
    return {'rows': [0, len(data)-1], 'upstream_and_head_pixel_exact': True}


def histograms(record_path):
    with np.load(record_path, allow_pickle=False) as file:
        truth = file['target_native']
        result = {}
        for grid in GRIDS:
            target = truth if grid == 'native' else s.nearest512_ids(truth)
            for arm in ('global', 'complete'):
                pred = file[f'{grid}_{arm}']
                assert pred.shape == target.shape
                result[grid + '_' + arm] = c.u.histogram(pred, target)
        np.testing.assert_array_equal(file['native_complete'] >= 3, file['native_upstream'] >= 3)
        np.testing.assert_array_equal(file['native_complete'][file['native_upstream'] < 3],
                                      file['native_upstream'][file['native_upstream'] < 3])
    return result


def evaluate_dataset(dataset, ids, resume=False):
    folder = HERE / 'outputs' / dataset
    rows, entry = test_records(dataset)
    if folder.exists():
        meta = c.read(folder / 'run_manifest.json')
        assert resume and meta['identity'] == ids
        assert meta['status'] in ('running', 'failed', 'stopped')
    else:
        folder.mkdir(parents=True)
        (folder / 'predictions').mkdir()
        meta = dict(identity=ids, dataset=dataset, status='prepared', records=[], test_pixels_read=False,
            created_utc=c.utc())
        c.write(folder / 'run_manifest.json', meta)
    pipeline = Pipeline(dataset, ids)
    try:
        parity = val_parity(dataset, pipeline, ids)
        meta['val_parity'] = parity
        c.write(folder / 'run_manifest.json', meta)
        images = np.load(entry['files']['image']['path'], mmap_mode='r')
        low = np.load(entry['files']['low']['path'], mmap_mode='r')
        assert images.shape == (len(rows), *entry['shape'][1:])
        assert low.shape == (len(rows), 512, 512)
        meta.update(status='running', test_pixels_read=True, updated_utc=c.utc())
        c.write(folder / 'run_manifest.json', meta)
        for index, record in enumerate(rows):
            stop_check()
            path = folder / 'predictions' / f'{index:04d}.npz'
            if index < len(meta['records']):
                saved = meta['records'][index]
                assert saved['label_id'] == record['label_id'] and saved['sha256'] == c.sha(path)
                continue
            assert index == len(meta['records']) and not path.exists()
            prediction = pipeline.predict(images[index], low[index])
            target = read_label(dataset, record, images[index].shape)
            values = {'target_native': target, 'native_upstream': prediction['native_upstream']}
            for grid in GRIDS:
                for arm in ('global', 'complete'):
                    values[f'{grid}_{arm}'] = prediction[f'{grid}_{arm}']
            c.save_npz(path, values)
            histograms(path)
            meta['records'].append(dict(index=index, label_id=record['label_id'],
                label_sha256=record['label_sha256'], sha256=c.sha(path)))
            meta['updated_utc'] = c.utc()
            c.write(folder / 'run_manifest.json', meta)
            if len(meta['records']) % 25 == 0 or len(meta['records']) == len(rows):
                print(f'TEST {dataset} {len(meta["records"])}/{len(rows)}', flush=True)
        report = verify_dataset(dataset, ids, allow_running=True)
        meta.update(status='completed', report_sha256=c.sha(folder / 'report.json'),
            completed_utc=c.utc())
        c.write(folder / 'run_manifest.json', meta)
        return report
    except BaseException as error:
        meta.update(status='stopped' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
                    error=repr(error), traceback=traceback.format_exc(), updated_utc=c.utc())
        c.write(folder / 'run_manifest.json', meta)
        raise
    finally:
        pipeline.close()


def verify_dataset(dataset, ids, allow_running=False):
    folder = HERE / 'outputs' / dataset
    meta = c.read(folder / 'run_manifest.json')
    assert meta['identity'] == ids and (meta['status'] == 'completed' or allow_running)
    rows, _ = test_records(dataset)
    assert len(meta['records']) == len(rows)
    matrices = {g: {arm: np.zeros((8, 8), np.int64) for arm in ('global', 'complete')} for g in GRIDS}
    for index, saved in enumerate(meta['records']):
        assert saved['index'] == index and saved['label_id'] == rows[index]['label_id']
        path = folder / 'predictions' / f'{index:04d}.npz'
        assert c.sha(path) == saved['sha256']
        with np.load(path, allow_pickle=False) as file:
            np.testing.assert_array_equal(file['target_native'],
                read_label(dataset, rows[index], tuple(file['target_native'].shape)))
        h = histograms(path)
        for grid in GRIDS:
            for arm in ('global', 'complete'):
                matrices[grid][arm] += h[f'{grid}_{arm}']
    metrics = {grid: {arm: c.u.metrics(matrices[grid][arm]) for arm in ('global', 'complete')}
               for grid in GRIDS}
    report = dict(dataset=dataset, images=len(rows), metrics=metrics, source_identity=ids,
        test_pixels_read=True, independently_replayed=True, completed_utc=c.utc())
    c.write(folder / 'report.json', report)
    return report


def run(resume=False):
    ids = source_identity()
    root = HERE / 'run_manifest.json'
    if root.exists():
        meta = c.read(root)
        assert resume and meta['identity'] == ids and meta['status'] in ('running', 'failed', 'stopped')
    else:
        assert not resume
        meta = dict(status='prepared', identity=ids, created_utc=c.utc(), test_pixels_read=False)
    c.setup('cuda')
    meta.update(status='running', pid=os.getpid(), updated_utc=c.utc())
    c.write(root, meta)
    try:
        results = {}
        for dataset in DATASETS:
            if (HERE / 'outputs' / dataset / 'run_manifest.json').exists() and \
                    c.read(HERE / 'outputs' / dataset / 'run_manifest.json')['status'] == 'completed':
                results[dataset] = verify_dataset(dataset, ids)
            else:
                results[dataset] = evaluate_dataset(dataset, ids, resume)
            meta['current_dataset'] = dataset
            meta['test_pixels_read'] = True
            c.write(root, meta)
        out = {'status': 'completed', 'seed': 42, 'datasets': results,
            'scope': 'target-domain fine8 adapted; historically exposed same-core test blocks',
            'checkpoint_selection': 'first canonical512 validation max only',
            'test_used_for_selection': False}
        c.write(HERE / 'summary.json', out)
        lines = ['# 外部八类固定测试：完整模型 vs 自创Global', '',
            '目标域有标签适配；只使用已锁定验证选点。此前同归档分段曾用于四相测试，不称独立新岩心。', '',
            '| 数据/网格 | Global mIoU | 完整模型 mIoU | 差值/百分点 | Global五油宏IoU | 完整五油宏IoU |',
            '|---|---:|---:|---:|---:|---:|']
        for dataset in DATASETS:
            for grid in GRIDS:
                a = results[dataset]['metrics'][grid]['global']
                b = results[dataset]['metrics'][grid]['complete']
                lines.append(f'| {dataset}/{grid} | {a["mIoU"]:.8f} | {b["mIoU"]:.8f} | '
                    f'{100*(b["mIoU"]-a["mIoU"]):+.4f} | {a["oil_macro_IoU"]:.8f} | {b["oil_macro_IoU"]:.8f} |')
        lines += ['', '仅seed42；所有逐类IoU/Dice与完整混淆矩阵见summary.json。完整模型另有后端训练预算，不能归因于单一模块。',
            '测试后不凭胜负调参或更换检查点；如需更多种子须依预先固定的TRAIN/VAL协议重复，并独立报告。']
        (HERE / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        meta.update(status='completed', summary_sha256=c.sha(HERE / 'summary.json'),
            completed_utc=c.utc())
        c.write(root, meta)
        print('LOCKED_FINE8_EXTERNAL_TEST_COMPLETE', flush=True)
    except BaseException as error:
        meta.update(status='stopped' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
            error=repr(error), traceback=traceback.format_exc(), updated_utc=c.utc())
        c.write(root, meta)
        raise


def smoke():
    ids = source_identity()
    c.setup('cuda')
    reports = {}
    for dataset in DATASETS:
        pipeline = Pipeline(dataset, ids)
        try:
            reports[dataset] = val_parity(dataset, pipeline, ids)
        finally:
            pipeline.close()
    c.write(HERE / 'smoke_report.json', dict(status='PASS', reports=reports,
        identity=ids, test_pixels_read=False, completed_utc=c.utc()))
    print('VAL_PARITY_PASS; NO_TEST_PIXELS_READ', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('smoke', 'run', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'smoke':
        smoke()
    elif args.stage == 'run':
        assert c.read(HERE / 'smoke_report.json')['status'] == 'PASS'
        assert c.read(HERE / 'smoke_report.json')['identity'] == source_identity()
        run(args.resume)
    else:
        ids = source_identity()
        reports = {d: verify_dataset(d, ids) for d in DATASETS}
        assert c.read(HERE / 'run_manifest.json')['status'] == 'completed'
        print('INDEPENDENT_TEST_REPLAY_PASS', {d: r['images'] for d, r in reports.items()})
