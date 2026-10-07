"""External-size bookkeeping only; model architecture and descriptors unchanged."""
import numpy as np
from PIL import Image

import body_common as c


def nearest512_ids(values):
    """Keep arbitrary component IDs; identical PIL grid to stage1 class masks."""
    a = np.asarray(values)
    if a.ndim != 2 or a.dtype.kind not in 'iu' or a.size == 0 or a.min() < 0:
        raise ValueError('Nonnegative integer 2D map required')
    if a.max() > np.iinfo(np.int32).max:
        raise ValueError('Component ID exceeds int32')
    return np.array(Image.fromarray(a.astype(np.int32)).resize((512, 512), Image.Resampling.NEAREST))


def targets(labels, prediction, truth):
    if labels.shape != prediction.shape or truth.shape != prediction.shape:
        raise ValueError('Aligned maps required')
    if truth.size == 0 or truth.min() < 0 or truth.max() > 7:
        raise ValueError('Eight-class truth required')
    if not np.array_equal(labels > 0, prediction >= 3):
        raise ValueError('Components must match predicted oil only')
    n = int(labels.max())
    counts, outside, frozen = {}, {}, {}
    for grid in c.GRIDS:
        lab, pred, gt = (labels, prediction, truth) if grid == 'native' else tuple(
            nearest512_ids(x) for x in (labels, prediction, truth))
        counts[grid] = np.bincount((lab.astype(np.int64) * 8 + gt).ravel(),
                                   minlength=(n + 1) * 8).reshape(n + 1, 8)[1:].astype(np.uint32)
        keep = lab == 0
        outside[grid] = np.bincount(gt[keep].astype(np.int64) * 8 + pred[keep], minlength=64).reshape(8, 8)
        frozen[grid] = c.u.histogram(pred, gt)
        assert int(counts[grid].sum() + outside[grid].sum()) == gt.size
    return counts, outside, frozen


def verify_dense(labels, prediction, truth, assignments, counts, outside):
    """Reconstruct actual pixel predictions to verify sparse bookkeeping."""
    dense = c.assign_bodies(prediction, labels, np.asarray(assignments) + 3)
    for grid in c.GRIDS:
        pix, gt = (dense, truth) if grid == 'native' else (nearest512_ids(dense), nearest512_ids(truth))
        expected = c.u.histogram(pix, gt)
        np.testing.assert_array_equal(c.aggregate(counts[grid], assignments, outside[grid]), expected)
    return dense


def assert_fixed_support(matrices, frozen):
    for grid, h in matrices.items():
        base = np.asarray(frozen[grid], np.int64)
        np.testing.assert_array_equal(h[:, :3], base[:, :3])
        np.testing.assert_array_equal(h[:, 3:].sum(1), base[:, 3:].sum(1))


def recount(assignments, counts, outside):
    """Class-by-class implementation independent of representation.aggregate."""
    h = np.asarray(outside, np.int64).copy()
    for oil_class in range(5):
        h[:, oil_class + 3] += np.asarray(counts[np.asarray(assignments) == oil_class], np.int64).sum(0)
    return h
