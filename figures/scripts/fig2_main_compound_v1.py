"""One compound result figure: overall gain, matched class effect, real cases.

The composition follows the author's supplied multi-panel examples: two compact
quantitative panels above one visually dominant, aligned image comparison.
Four native fields show selected successful segmentation of characteristic
oil morphologies. Quantitative panels use all observations unchanged.
This script reads completed results and creates display files only.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

COLORS = ["#000000", "#969696", "#2D6494", "#E6C64A", "#D88A3C", "#83C8DF", "#CC5C54", "#8964AE"]
CLASS_NAMES = ["Background", "Rock", "Water", "Film oil", "Pore-corner", "Droplet", "Columnar", "Clustered"]
from spatial_error_comparison import ERROR_COLORS, ERROR_NAMES, relative_errors, error_rgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "实验"
LOCKED = EXP / "body_inference_locked_20260930"
from audit_panel_alignment import require_matplotlib_panel_alignment

SEEDS = (42, 2024, 2025)
SELECTION = HERE.parent / "source_data/morphology_advantages_20261006/selection.json"
WIDTH_MM, HEIGHT_MM = 114.0, 172.0
MODEL_NAMES = ["TransUNet", "Global", "Pointwise", "Complete"]
STYLE = json.loads((HERE / "visual_style.json").read_text(encoding="utf-8"))
MODEL_COLORS = [STYLE["model_colors"][name] for name in MODEL_NAMES]
assert COLORS == STYLE["semantic_colors"]
MODEL_MARKERS = ("o", "s", "^")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_sources():
    return load_snapshot("main_comparison")


def main(output_dir: Path = HERE) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    miou, deltas, images, source = read_sources()
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.5, "axes.linewidth": .45,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), dpi=300)

    def axes_mm(left, bottom, width, height):
        return fig.add_axes([left/WIDTH_MM, bottom/HEIGHT_MM,
                             width/WIDTH_MM, height/HEIGHT_MM])

    def clean_axes(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#505050")
        ax.tick_params(length=2, width=.45, pad=1.5, colors="#383838")

    a = axes_mm(10.4, 133.5, 44.0, 29.0)
    b = axes_mm(76.5, 133.5, 35.0, 29.0)
    for ax in (a, b):
        clean_axes(ax)
    for ax, label in ((a, "a"), (b, "b")):
        ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(-8, 4),
                    textcoords="offset points", fontsize=9.0, weight="bold",
                    va="bottom", ha="left", annotation_clip=False)

    a.set_title("Overall segmentation", fontsize=8.0, pad=4)
    x = np.arange(4)
    for si, marker in enumerate(MODEL_MARKERS):
        offset = [-.10, 0, .10][si]
        a.plot(x+offset, miou[si], color="#C5C8CA", linewidth=.45, zorder=1)
        for j, color in enumerate(MODEL_COLORS):
            a.scatter(j+offset, miou[si, j], marker=marker, s=10, facecolor="white",
                      edgecolor=color, linewidth=.6, zorder=4)
    for j, color in enumerate(MODEL_COLORS):
        a.errorbar(j, miou[:, j].mean(), yerr=miou[:, j].std(ddof=1),
                   fmt="D", markersize=2.3, color=color, elinewidth=.8,
                   capsize=2.0, capthick=.8, zorder=5)
    a.set_xticks(x, ["TransUNet", "Global\nbaseline", "Pointwise\ncontrol", "Complete\nmodel"])
    a.set(xlim=(-.45, 3.45), ylim=(70, 85), yticks=[70, 75, 80, 85], ylabel="mIoU (%)")
    a.yaxis.grid(True, color="#E5E5E5", linewidth=.35, zorder=0)
    a.set_axisbelow(True)

    b.set_title("Complete − Pointwise", fontsize=8.0, pad=4)
    yy = np.arange(5)[::-1]
    gain_mean = deltas.mean(0)
    gain_sd = deltas.std(0, ddof=1)
    # Bar lengths encode the same paired means as the previous point display.
    # All bars begin at zero; green consistently identifies the complete model.
    b.barh(yy, gain_mean, height=.64, color=MODEL_COLORS[-1], alpha=.72,
           edgecolor="none", zorder=2)
    for si, marker in enumerate(MODEL_MARKERS):
        b.scatter(deltas[si], yy+[-.11, 0, .11][si], marker=marker,
                  s=8, facecolor="white", edgecolor="#586D69", linewidth=.6, zorder=4)
    b.errorbar(gain_mean, yy, xerr=gain_sd, fmt="D",
               markersize=2.4, color="#276E65", elinewidth=.8,
               capsize=1.8, capthick=.8, zorder=5)
    for y, mean, sd, seed_values in zip(yy, gain_mean, gain_sd, deltas.T):
        label_x = max(mean+sd, seed_values.max())+.16
        b.text(label_x, y, f"{mean:.2f}", color="#26313A", fontsize=7.0,
               ha="left", va="center", zorder=6)
    b.set_yticks(yy, ["Film", "Pore-corner", "Droplet", "Columnar", "Clustered"])
    b.set(xlim=(0, 5.0), ylim=(-.55, 4.55), xticks=[0, 1, 2, 3, 4, 5],
          xlabel="ΔIoU (percentage points)")
    b.grid(False)
    b.set_axisbelow(True)

    seed_handles = [Line2D([0], [0], linestyle="none", marker=m, color="#686868",
                           markerfacecolor="white", markersize=3, markeredgewidth=.6)
                    for m in MODEL_MARKERS]
    seed_handles.append(Line2D([0], [0], linestyle="none", marker="D", color="#505050",
                              markersize=2.6))
    fig.legend(seed_handles, ["Seed 42", "Seed 2024", "Seed 2025", "Mean ± SD"],
               loc="center", bbox_to_anchor=(.52, 123.3/HEIGHT_MM),
               ncol=4, frameon=False, fontsize=7.0, handletextpad=.35, columnspacing=1.0)

    fig.text(1.2/WIDTH_MM, 117.9/HEIGHT_MM, "c", fontsize=9.0, weight="bold",
             ha="left", va="center")
    left_mm, side_mm, gap_mm = 4.2, 20.6, 1.45
    bottoms = [91.5, 68.8, 46.1, 23.4]
    image_axes, image_ids = [], []
    for col, heading in enumerate(("Raw image", "Reference", "Global\nbaseline", "PoreMorph\ncomplete", "Error\ncomparison")):
        xx = left_mm+col*(side_mm+gap_mm)
        color = ["#26313A", "#26313A", MODEL_COLORS[1], MODEL_COLORS[3], "#777777"][col]
        fig.text((xx+side_mm/2)/WIDTH_MM, 117.0/HEIGHT_MM, heading,
                 fontsize=7.5, color="#26313A", ha="center", va="center")
        if col >= 2:
            fig.add_artist(Rectangle((xx/WIDTH_MM, 112.6/HEIGHT_MM),
                                     side_mm/WIDTH_MM, .45/HEIGHT_MM,
                                     facecolor=color, edgecolor="none", transform=fig.transFigure))
    for row, (case, cells) in enumerate(zip(source["cases"], images)):
        fig.text(1.45/WIDTH_MM, (bottoms[row]+side_mm/2)/HEIGHT_MM, case["stage"],
                 rotation=90, rotation_mode="anchor", fontsize=7.0,
                 ha="center", va="center", color="#26313A")
        for col, array in enumerate(cells):
            ax = axes_mm(left_mm+col*(side_mm+gap_mm), bottoms[row], side_mm, side_mm)
            image_axes.append(ax)
            image_ids.append(f"c{row}{col}")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#B1B7BC")
                spine.set_linewidth(.3)
            if col == 0:
                ax.imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                bar = 1000/13.25
                ax.plot([13, 13+bar], [240, 240], color="black", lw=2.8,
                        solid_capstyle="butt")
                ax.plot([13, 13+bar], [240, 240], color="white", lw=1.2,
                        solid_capstyle="butt")
                ax.text(13+bar/2, 215, "1 mm", ha="center", va="bottom", fontsize=7.0,
                        color="white", bbox=dict(facecolor="black", edgecolor="none", pad=.35, alpha=.75))
            elif col == 4:
                ax.imshow(array, interpolation="nearest")
            else:
                ax.imshow(array, cmap=ListedColormap(COLORS), vmin=-.5, vmax=7.5,
                          interpolation="nearest")

    for k, (name, color) in enumerate(zip(ERROR_NAMES, ERROR_COLORS)):
        xx, yy = 2.2+k*38.0, 18.1
        fig.add_artist(Rectangle((xx/WIDTH_MM, (yy-.8)/HEIGHT_MM),
                                 1.65/WIDTH_MM, 1.65/HEIGHT_MM,
                                 facecolor=color, edgecolor="none", transform=fig.transFigure))
        fig.text((xx+2.8)/WIDTH_MM, yy/HEIGHT_MM, name, fontsize=7.0,
                 ha="left", va="center", color="#26313A")

    for k, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        rr, cc = divmod(k, 4)
        xx, yy = 2.2+cc*28.0, 10.1-rr*5.1
        fig.add_artist(Rectangle((xx/WIDTH_MM, (yy-.8)/HEIGHT_MM),
                                 1.65/WIDTH_MM, 1.65/HEIGHT_MM,
                                 facecolor=color, edgecolor="#69737B", linewidth=.25,
                                 transform=fig.transFigure))
        fig.text((xx+2.8)/WIDTH_MM, yy/HEIGHT_MM, name, fontsize=7.5,
                 ha="left", va="center", color="#26313A")

    fig.canvas.draw()
    stem = output_dir / "fig2_main_compound_v1"
    require_matplotlib_panel_alignment(
        fig, axes=[a, b, *image_axes], panel_ids=["a", "b", *image_ids],
        row_groups=[["a", "b"], *[[f"c{r}{c}" for c in range(5)] for r in range(4)]],
        column_groups=[[f"c{r}{c}" for r in range(4)] for c in range(5)],
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False,
        json_out=stem.with_suffix(".alignment.json"), strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    stem.with_suffix(".source.json").write_text(json.dumps(source, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"figure": stem.name, "miou_means_percent": miou.mean(0).tolist(),
                      "head_differences_pp": deltas.mean(0).tolist(), "test_cases": len(images)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    main(parser.parse_args().output_dir)
