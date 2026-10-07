"""Read-only archived external TRAIN/VAL inputs, never test pixels."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP, ROOT = HERE.parent, HERE.parent.parent
UPGRADE = EXP / 'architecture_upgrade_20260928'
sys.path.insert(0, str(UPGRADE))
import coarse_data as old

CACHE = UPGRADE / 'outputs/external_float32_roi_v2_cache'
LEGACY = {name: ROOT / directory / '模型微调测试2.py'
          for name, directory in (('wf1', '扩展数据1'), ('bentheimer', '扩展数据2'))}
EXPECTED = {'wf1': 300, 'bentheimer': 90}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def description(dataset):
    tree = ast.parse(LEGACY[dataset].read_text(encoding='utf-8-sig'))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'EXTERNAL_TEXT' for t in node.targets)]
    assert len(matches) == 1
    value = ast.literal_eval(matches[0])
    assert isinstance(value, str) and value
    return value


def normalize_native(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Expected finite grayscale image')
    lo, hi = np.percentile(raw, (0.5, 99.5))
    if hi - lo <= 1e-8:
        lo, hi = float(raw.min()), float(raw.max())
    return (np.zeros_like(raw) if hi - lo <= 1e-8 else
            np.clip((raw - lo) / (hi - lo), 0, 1).astype(np.float32))


def low_image(native):
    if native.dtype != np.float32 or native.ndim != 2 or not np.isfinite(native).all():
        raise ValueError('Float32 finite native image required')
    if native.min() < 0 or native.max() > 1:
        raise ValueError('Image must be within [0,1]')
    return np.array(Image.fromarray(native).resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)


def nearest512(mask):
    return np.array(Image.fromarray(mask).resize((512, 512), Image.Resampling.NEAREST))


class Inputs:
    def __init__(self, dataset, split):
        if split not in ('train', 'val'):
            raise ValueError('External test access is deliberately forbidden')
        self.dataset, self.split = dataset, split
        self.geometry = old.geometry_contract(dataset)
        self.records = [r for r in old.membership(dataset) if r['split'] == split]
        prior = read(CACHE / 'manifest.json')
        assert prior['status'] == 'complete' and prior['test_data_read'] is False
        assert prior['identity']['source_manifests'][dataset] == sha(old.SOURCES[dataset])
        self.entry = prior['datasets'][f'{dataset}_{split}']
        assert len(self.records) == self.entry['count']
        self.lows = np.load(CACHE / self.entry['images'], mmap_mode='r')
        self.labels512 = np.load(CACHE / self.entry['labels'], mmap_mode='r')
        self.text = description(dataset)
        for record, cached in zip(self.records, self.entry['records']):
            assert all(record[k] == cached[k] for k in record)

    def image(self, index):
        r, cached = self.records[index], self.entry['records'][index]
        assert sha(r['raw_path']) == cached['raw_sha256']
        with Image.open(r['raw_path']) as file:
            raw = np.array(file)
        assert list(raw.shape) == self.geometry['raw_shape']
        if self.geometry['crop_yx'] is not None:
            (y0, y1), (x0, x1) = self.geometry['crop_yx']
            raw = raw[y0:y1, x0:x1]
        assert list(raw.shape) == self.geometry['label_shape']
        native = normalize_native(raw)
        low = low_image(native)
        np.testing.assert_array_equal(low, self.lows[index])
        return native, low

    def target(self, index):
        r, cached = self.records[index], self.entry['records'][index]
        assert sha(r['seg_path']) == cached['seg_sha256']
        with Image.open(r['seg_path']) as file:
            raw = np.array(file)
        assert list(raw.shape) == self.geometry['label_shape']
        assert raw.dtype == np.uint8 and raw.min() >= 0 and raw.max() <= 3
        mapped = np.asarray(old.RAW_TO_PHASE[self.dataset], np.uint8)[raw]
        np.testing.assert_array_equal(nearest512(mapped), self.labels512[index])
        return mapped


def histogram(pred, target):
    if pred.shape != target.shape or min(pred.min(), target.min()) < 0 or max(pred.max(), target.max()) > 3:
        raise ValueError('Expected aligned four-phase arrays')
    return np.bincount((target.astype(np.int64)*4 + pred).ravel(), minlength=16).reshape(4, 4)


def metrics(hist):
    h = np.asarray(hist, np.int64)
    diag = np.diag(h).astype(np.float64)
    support, predicted = h.sum(1), h.sum(0)
    iou = diag / (support + predicted - diag + 1e-8)
    return {'mIoU': float(iou.mean()), 'nonbackground_mIoU': float(iou[1:].mean()),
            'class_IoU': iou.tolist(), 'gt_pixels': support.tolist(), 'pred_pixels': predicted.tolist(),
            'confusion_matrix': h.tolist()}
