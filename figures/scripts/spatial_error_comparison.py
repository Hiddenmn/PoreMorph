"""Pixel-exact relative error displays shared by the two qualitative plates."""
from __future__ import annotations
from figure_io import load_snapshot, asset_directory

import numpy as np
from matplotlib.colors import to_rgb

ERROR_COLORS = ("#008D94", "#E679A8", "#6B4C9A")
ERROR_NAMES = ("Corrected by complete", "Introduced by complete", "Both incorrect")


def relative_errors(reference, baseline, complete):
    """0: both correct; 1: corrected; 2: introduced; 3: both incorrect."""
    assert reference.shape == baseline.shape == complete.shape
    baseline_wrong, complete_wrong = baseline != reference, complete != reference
    codes = np.zeros(reference.shape, dtype=np.uint8)
    codes[baseline_wrong & ~complete_wrong] = 1
    codes[~baseline_wrong & complete_wrong] = 2
    codes[baseline_wrong & complete_wrong] = 3
    assert np.array_equal(codes != 0, baseline_wrong | complete_wrong)
    assert np.array_equal(np.isin(codes, (1, 3)), baseline_wrong)
    assert np.array_equal(np.isin(codes, (2, 3)), complete_wrong)
    return codes


def error_rgb(raw, codes, *, input_max):
    """The same lightened raw background; every error pixel keeps its exact size."""
    assert raw.shape == codes.shape and input_max > 0
    gray = .62 + .32 * raw.astype(np.float32) / input_max
    assert gray.min() >= .62 - 1e-6 and gray.max() <= .94 + 1e-6
    image = np.repeat(gray[..., None], 3, axis=2)
    for code, color in enumerate(ERROR_COLORS, 1):
        image[codes == code] = to_rgb(color)
    return image
