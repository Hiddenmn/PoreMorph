"""Native-grid qualitative examples after labelled external adaptation.

Question: what spatial differences underlie favorable Complete-versus-Global
cases in two target domains? A full field anchors a shared local enlargement.
The user requested performance-selected examples. Selection is restricted to
seed 42 and completed native predictions; it never selects checkpoints, changes
labels or alters the full-test aggregate. Existing semantic colors are reused.
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from spatial_error_comparison import ERROR_COLORS, ERROR_NAMES, relative_errors, error_rgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "实验"
FINE = EXP / "external_eightclass_adaptation_20261003"
CACHE = EXP / "external_clean_prior_test_20261001/cache/manifest.json"
FOLDERS = {
    "bentheimer": EXP / "external_full_vs_global_fine8_test_20261004/outputs/bentheimer",
    "wf1": EXP / "external_fine8_wf1_repeat_seeds42_2024_20261005/evaluation/cases/seed42/outputs/wf1",
}
STYLE = json.loads((HERE / "visual_style.json").read_text(encoding="utf-8"))
COLORS = STYLE["semantic_colors"]
CLASS_NAMES = ("Background", "Rock", "Water", "Film oil", "Pore-corner", "Droplet", "Columnar", "Clustered")
INK = STYLE["text"]
WIDTH_MM, HEIGHT_MM = 114., 142.
ROI_SIDE = 256
from audit_panel_alignment import require_matplotlib_panel_alignment


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(target, prediction):
    confusion = np.bincount((target.astype(np.int64) * 8 + prediction).ravel(), minlength=64).reshape(8, 8)
    denom = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
    iou = np.divide(np.diag(confusion), denom, out=np.full(8, np.nan), where=denom > 0)
    # Local empty classes are omitted only from the local display-selection score.
    return float(np.nanmean(iou)), float(np.nanmean(iou[3:])), iou








def load_case(dataset, candidate):
    arrays, metadata = load_snapshot("external_" + dataset)
    assert all(metadata[key] == value for key, value in candidate.items())
    return arrays, metadata




def draw(output_dir, selection_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    choices = read(selection_path)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    left, side, gap = 4.2, 20.6, 1.45
    bottoms = (103., 79.2, 45., 21.2)
    sources, axes, panel_ids = [], [], []
    for col, label in enumerate(("CT input", "Reference", "Global\nbaseline", "PoreMorph\ncomplete", "Error\ncomparison")):
        x = left + col * (side + gap)
        fig.text((x + side/2) / WIDTH_MM, 137 / HEIGHT_MM, label, ha="center", va="center", fontsize=7.5, color=INK)
    for group, dataset in enumerate(("bentheimer", "wf1")):
        arrays, source = load_case(dataset, choices[dataset])
        codes = relative_errors(arrays[1], arrays[2], arrays[3])
        source["error_outcome_pixels_full"] = np.bincount(codes.ravel(), minlength=4).tolist()
        arrays.append(error_rgb(arrays[0], codes, input_max=1))
        sources.append(source)
        r = source["roi"]
        crop = (slice(r["y0"], r["y1"]), slice(r["x0"], r["x1"]))
        source["error_outcome_pixels_crop"] = np.bincount(codes[crop].ravel(), minlength=4).tolist()
        title = "Bentheimer" if dataset == "bentheimer" else "WF1 · repeated adaptation"
        y_title = 128 if group == 0 else 70
        fig.text(1 / WIDTH_MM, y_title / HEIGHT_MM, "ab"[group], weight="bold", fontsize=9, color=INK)
        fig.text(6 / WIDTH_MM, y_title / HEIGHT_MM, title, fontsize=8, color=INK)
        for zoom in range(2):
            row = group * 2 + zoom
            for col, array in enumerate(arrays):
                view = array[crop] if zoom else array
                ax = fig.add_axes([(left + col * (side + gap))/WIDTH_MM, bottoms[row]/HEIGHT_MM,
                                   side/WIDTH_MM, side/HEIGHT_MM])
                axes.append(ax)
                panel_ids.append(f"r{row}c{col}")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#A3ADB4")
                    spine.set_linewidth(.35)
                h, w = view.shape[:2]
                if col == 0:
                    ax.imshow(view, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                elif col == 4:
                    ax.imshow(view, interpolation="nearest")
                else:
                    ax.imshow(view, cmap=ListedColormap(COLORS), vmin=-.5, vmax=7.5, interpolation="nearest")
                # Slightly non-square WF1 fields use square limits, never stretched pixels.
                span = max(h, w)
                ax.set(xlim=(-.5, span-.5), ylim=(span-.5, -.5), aspect="equal")
                if not zoom:
                    for color, line in (("black", 1.2), ("white", .55)):
                        ax.add_patch(Rectangle((r["x0"]-.5, r["y0"]-.5), ROI_SIDE, ROI_SIDE,
                            fill=False, edgecolor=color, linewidth=line))
                if col == 0 and zoom:
                    for color, line in (("black", 3), ("white", 1.3)):
                        ax.plot([12, 76], [238, 238], color=color, linewidth=line, solid_capstyle="butt")
                    ax.text(44, 217, "64 px", fontsize=7, color="white", ha="center", va="bottom",
                            bbox=dict(facecolor="black", edgecolor="none", pad=.25))
            fig.text(1.9 / WIDTH_MM, (bottoms[row]+side/2) / HEIGHT_MM,
                     "Zoom" if zoom else "Field", rotation=90, rotation_mode="anchor",
                     ha="center", va="center", fontsize=7, color=INK)
    for i, (name, color) in enumerate(zip(ERROR_NAMES, ERROR_COLORS)):
        x, y = 2 + i * 38, 15.8
        fig.add_artist(Rectangle((x/WIDTH_MM, (y-.7)/HEIGHT_MM), 1.6/WIDTH_MM, 1.6/HEIGHT_MM,
            transform=fig.transFigure, facecolor=color, edgecolor="none"))
        fig.text((x+2.7)/WIDTH_MM, y/HEIGHT_MM, name, fontsize=7, va="center", color=INK)
    for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        row, col = divmod(i, 4)
        x, y = 2 + col * 28, 10.1 - row * 5.1
        fig.add_artist(Rectangle((x/WIDTH_MM, (y-.7)/HEIGHT_MM), 1.6/WIDTH_MM, 1.6/HEIGHT_MM,
            transform=fig.transFigure, facecolor=color, edgecolor="#70777D", linewidth=.25))
        fig.text((x+2.7)/WIDTH_MM, y/HEIGHT_MM, name, fontsize=7, va="center", color=INK)
    fig.canvas.draw()
    stem = output_dir / "figS4_external_qualitative"
    # Separate dataset groups retain a wider group break; within-group zoom gutters match.
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=panel_ids,
        row_groups=[[f"r{r}c{c}" for c in range(5)] for r in range(4)],
        column_groups=[[f"r{r}c{c}" for r in range(4)] for c in range(5)],
        exemptions=[dict(panels=panel_ids, checks=["vertical-gutter"],
                         reason="Two dataset blocks each contain a full field and its zoom; their separating gutter also contains the second dataset heading.")],
        require_panel_labels=False, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=stem.with_suffix(".alignment.json"), strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    record = dict(cases=sources, selection_file=str(selection_path), selection_sha256=sha(selection_path),
        performance_selected=True, seed=42, statistical_inference=False,
        display_mm=[WIDTH_MM, HEIGHT_MM], input_display_limits=[0, 1], input_gamma=1,
        masks_modified=False, interpolation="nearest", scale_bar_units="native pixels",
        scale_bar_length_pixels=64, pseudo_color=COLORS, local_contrast_adjustment=False,
        error_names=["Both correct", *ERROR_NAMES], error_colors=ERROR_COLORS,
        error_background="0.62 + 0.32 * normalized input", error_pixels_unmodified=True)
    stem.with_suffix(".source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": stem.name, "cases": [s["label_id"] for s in sources]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    draw(args.output_dir, args.selection)
