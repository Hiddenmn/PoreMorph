"""Stage-level evidence from the completed three-seed main-data evaluation.

Question: which classes respond to native refinement and body labeling?
One heatmap retains the three class-wise contrasts not shown in main Figure 2.
A matched-head size view then localizes attention gains across component areas.
The body-labeling rules are alternative predictors on shared native support,
not three classifiers applied consecutively at inference time.
No image pixels, checkpoints, or running experiments are modified or evaluated.
"""
from __future__ import annotations
from figure_io import load_snapshot, asset_directory

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / "source_data/main_test_summary.json"
SIZE_SOURCE = HERE.parent / "source_data/body_size.json"
STYLE = json.loads((HERE / "visual_style.json").read_text(encoding="utf-8"))
from audit_panel_alignment import require_matplotlib_panel_alignment

SEEDS = (42, 2024, 2025)
KEYS = ("global", "native_detail", "body_mean", "pointwise", "separate")
NAMES = ("Global baseline", "Native-detail output", "Body-mean assignment", "Pointwise classifier", "Complete (PoreMorph)")
DISPLAY_CONTRASTS = (
    ("native_detail", "global"),
    ("body_mean", "native_detail"),
    ("pointwise", "body_mean"),
)
CONTRAST_LABELS = (
    "Native detail\n− Global",
    "Body mean\n− Native detail",
    "Pointwise\n− Body mean",
)
CLASSES = ("Background", "Rock", "Water", "Film", "Pore-corner", "Droplet", "Columnar", "Clustered")
INK = STYLE["text"]
WIDTH_MM, HEIGHT_MM = 114.0, 149.0


def data():
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))
    scores, classes, rows = [], [], []
    for seed in SEEDS:
        result = summary["test_results"][str(seed)]
        assert result["status"] == "completed" and result["images"] == 340
        m = result["metrics"]["canonical512"]
        scores.append([m[k]["mIoU"] * 100 for k in KEYS])
        classes.append([np.asarray(m[k]["class_IoU"]) * 100 for k in KEYS])
        for name, key in zip(NAMES, KEYS):
            np.testing.assert_allclose(np.mean(m[key]["class_IoU"]), m[key]["mIoU"], atol=1e-12)
            for ci, cname in enumerate(CLASSES):
                rows.append(dict(seed=seed, predictor=name, internal_key=key,
                                 grid="canonical512", class_id=ci, class_name=cname,
                                 IoU_percent=100 * m[key]["class_IoU"][ci],
                                 mIoU_percent=100 * m[key]["mIoU"]))
    scores, classes = np.asarray(scores), np.asarray(classes)
    assert scores.shape == (3, 5) and classes.shape == (3, 5, 8)
    assert np.isfinite(scores).all() and np.isfinite(classes).all()
    # All oil-body classifiers preserve non-oil labels and oil support.
    np.testing.assert_array_equal(classes[:, 1:2, :3].repeat(4, axis=1), classes[:, 1:, :3])
    return scores, np.diff(classes, axis=1), rows


def style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.serif": ["Times New Roman"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.5, "axes.linewidth": .5, "svg.fonttype": "none",
        "pdf.fonttype": 42, "savefig.facecolor": "white", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})




