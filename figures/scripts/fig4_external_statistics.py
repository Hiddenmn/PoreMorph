"""External adaptation: overall performance, native-grid stability and class response.

One quantitative figure, not one chart per metric. Original and repeated WF1
attempts are distinct groups; every seed, grid and class remains represented.
Only completed summary records are read. No images, inference or training.
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
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
PROJECT = PAPER.parent
EXP = PROJECT / "实验"
ORIGINAL = EXP / "external_fine8_multiseed_test_recovery_20261005"
REPEAT = EXP / "external_fine8_wf1_repeat_seeds42_2024_20261005"
from audit_panel_alignment import require_matplotlib_panel_alignment

WIDTH_MM, HEIGHT_MM = 114.0, 160.0
SEEDS = (42, 2024, 2025)
GRIDS = ("canonical512", "native")
GROUPS = ("Bentheimer", "WF1 initial", "WF1 repeat")
TICKS = ("Bentheimer", "WF1\ninitial", "WF1\nrepeat")
CLASSES = ("Background", "Rock", "Water", "Film", "Pore-corner", "Droplet", "Columnar", "Clustered")
STYLE = json.loads((HERE / "visual_style.json").read_text(encoding="utf-8"))
PINK, TEAL = (STYLE["model_colors"][x] for x in ("Global", "Complete"))
INK = STYLE["text"]
MARKERS = ("o", "s", "^")


def read(path):
    return path.read_text(encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_values():
    return load_snapshot("external")


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    values, rows, metadata = load_values()
    miou = values.mean(axis=-1)
    deltas = values[:, :, :, 1] - values[:, :, :, 0]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": .5, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), dpi=300)

    def axes_mm(x, y, w, h):
        return fig.add_axes([x/WIDTH_MM, y/HEIGHT_MM, w/WIDTH_MM, h/HEIGHT_MM])

    a = axes_mm(10.2, 110, 47.5, 34)
    b = axes_mm(74, 110, 37.5, 34)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#69737B")
        ax.tick_params(length=2, width=.5, pad=2, colors=INK)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=.35, color="#E5E5E5")
    x = np.arange(3)
    for ai, color in enumerate((PINK, TEAL)):
        xx = x + (-.18 if ai == 0 else .18)
        mean, sd = miou[:, 0, :, ai].mean(axis=1), miou[:, 0, :, ai].std(axis=1, ddof=1)
        a.bar(xx, mean, width=.31, color=color, edgecolor="none", alpha=.9, zorder=2)
        a.errorbar(xx, mean, yerr=sd, fmt="none", ecolor=INK, linewidth=.65,
                   capsize=1.5, capthick=.65, zorder=4)
        for si, marker in enumerate(MARKERS):
            a.scatter(xx + (-.065, 0, .065)[si], miou[:, 0, si, ai],
                      s=10, marker=marker, facecolor="white", edgecolor=INK, linewidth=.45, zorder=5)
    a.set(xlim=(-.58, 2.58), ylim=(0, 75), yticks=[0, 25, 50, 75], ylabel="mIoU (%)")
    a.set_xticks(x, TICKS)
    a.set_title("512-grid performance", fontsize=8, pad=5)

    native = miou[:, 1, :, 1] - miou[:, 1, :, 0]
    mean, sd = native.mean(axis=1), native.std(axis=1, ddof=1)
    b.bar(x, mean, width=.59, color=TEAL, edgecolor="none", alpha=.85, zorder=2)
    b.axhline(0, color="#70777D", linewidth=.65)
    b.errorbar(x, mean, yerr=sd, fmt="none", ecolor=INK, linewidth=.7,
               capsize=2, capthick=.7, zorder=4)
    for si, marker in enumerate(MARKERS):
        b.scatter(x + (-.12, 0, .12)[si], native[:, si], marker=marker,
                  s=13, facecolor="white", edgecolor=INK, linewidth=.5, zorder=5)
    high = np.maximum(mean+sd, native.max(axis=1))
    for xx, yy, score in zip(x, high, mean):
        b.text(xx, yy+.18, f"+{score:.2f}", fontsize=7, color=INK, ha="center", va="bottom")
    b.set(xlim=(-.6, 2.6), ylim=(-.65, 3.8), yticks=[0, 1, 2, 3], ylabel="ΔmIoU (pp)")
    b.set_xticks(x, ("Bentheimer", "WF1\ninitial", "WF1\nrepeat"))
    b.set_title("Native-grid mIoU gain", fontsize=8, pad=5)
    b.grid(False)
    for ax, letter in ((a, "a"), (b, "b")):
        ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-10, 7),
                    textcoords="offset points", fontsize=9, weight="bold", color=INK,
                    ha="left", va="bottom", annotation_clip=False)

    method_handles = [Patch(facecolor=PINK, label="Global baseline"), Patch(facecolor=TEAL, label="Complete model")]
    fig.legend(handles=method_handles, loc="center", bbox_to_anchor=(.5, 154.5/HEIGHT_MM),
               ncol=2, frameon=False, fontsize=8, handlelength=1.2, handletextpad=.4)
    seed_handles = [Line2D([0], [0], linestyle="none", marker=m, markerfacecolor="white",
                           markeredgecolor=INK, markersize=3.2, markeredgewidth=.55) for m in MARKERS]
    fig.legend(seed_handles, ["Seed 42", "Seed 2024", "Seed 2025"], loc="center",
               bbox_to_anchor=(.5, 98/HEIGHT_MM), ncol=3, frameon=False, fontsize=7,
               handletextpad=.3, columnspacing=1.3)

    # Every class and both grids, one shared zero-centred difference scale.
    class_mean = deltas.mean(axis=2).reshape(6, 8)
    class_sd = deltas.std(axis=2, ddof=1).reshape(6, 8)
    norm_limit = float(np.ceil(np.abs(class_mean).max()))
    cmap = LinearSegmentedColormap.from_list("paired_pink_teal", ["#B24F79", "#FAFAFA", "#277F74"])
    c = axes_mm(28, 23, 83.5, 60)
    im = c.imshow(class_mean, cmap=cmap, norm=TwoSlopeNorm(0, -norm_limit, norm_limit),
                  interpolation="nearest", aspect="auto")
    c.set_xticks(np.arange(8), ["Back-\nground", "Rock", "Water", "Film", "Pore-\ncorner", "Droplet", "Columnar", "Clustered"])
    c.xaxis.tick_top()
    c.tick_params(axis="x", length=0, pad=4)
    c.set_yticks(np.arange(6), [f"{g}\n{r}" for g in GROUPS for r in ("512 grid", "native grid")])
    c.tick_params(axis="y", length=0, pad=5)
    for spine in c.spines.values():
        spine.set_visible(False)
    for boundary in (1.5, 3.5):
        c.axhline(boundary, color="white", linewidth=1.3)
    for ri in range(6):
        for ci in range(8):
            score, spread = class_mean[ri, ci], class_sd[ri, ci]
            color = "white" if abs(score) > .62*norm_limit else INK
            c.text(ci, ri, f"{score:+.2f}\n±{spread:.2f}", ha="center", va="center",
                   fontsize=7, linespacing=1.15, color=color)
    fig.text(1.5/WIDTH_MM, 94/HEIGHT_MM, "c", fontsize=9, weight="bold", color=INK, va="center")
    fig.text(28/WIDTH_MM, 94/HEIGHT_MM, "Class-wise gain: Complete − Global", fontsize=8, color=INK, va="center")
    color_ax = axes_mm(58, 8, 43, 2.6)
    cb = fig.colorbar(im, cax=color_ax, orientation="horizontal", ticks=[-norm_limit, 0, norm_limit])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=1.5, width=.4, pad=1.5)
    fig.text(25/WIDTH_MM, 9.3/HEIGHT_MM, "ΔIoU (pp), mean ± SD", fontsize=7, ha="center", va="center", color=INK)

    fig.canvas.draw()
    stem = output_dir / "fig4_external_statistics"
    require_matplotlib_panel_alignment(
        fig, axes=[a, b, c], panel_ids=["a", "b", "c"], row_groups=[["a", "b"]], column_groups=[],
        require_panel_labels=False, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=stem.with_suffix(".alignment.json"), strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    metadata["heatmap_symmetric_limit_pp"] = norm_limit
    stem.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    with stem.with_suffix(".source.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output":stem.name, "source_rows":len(rows),
                      "native_mean_gain_pp":native.mean(axis=1).tolist(),
                      "groups":list(GROUPS), "excluded_records":0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    main(parser.parse_args().output_dir)
