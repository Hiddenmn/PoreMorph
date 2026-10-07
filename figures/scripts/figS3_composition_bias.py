"""Resolve average composition error into oil-form and displacement-stage bias.

The reference composition supplies domain context; matched pointwise/complete
heatmaps identify under- and over-allocation within a fixed predicted oil phase.
Reads completed voxel-count summaries only. No volume rendering, smoothing,
inference, changed sample selection, or new significance testing is performed.
"""
from __future__ import annotations
from figure_io import load_snapshot, asset_directory

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np

from figS2_stage_contributions import style, require_matplotlib_panel_alignment
from fig3_spatial_quantification_v2 import read_composition

HERE = Path(__file__).resolve().parent
STYLE = json.loads((HERE / "visual_style.json").read_text(encoding="utf-8"))
INK = STYLE["text"]
STAGES = ("Oil", "WI", "NPI-006", "NPI-012")
NAMES = ("Film oil", "Pore-corner", "Droplet", "Columnar", "Clustered")
SEEDS = (42, 2024, 2025)
WIDTH_MM, HEIGHT_MM = 114.0, 149.0


def data():
    _, rows, provenance = read_composition()
    assert len(rows) == 120
    ref, predictions = np.zeros((4, 5)), np.zeros((2, 3, 4, 5))
    for ti, stage in enumerate(STAGES):
        for ci in range(5):
            values = [r for r in rows if r["stage"] == stage and r["class_id"] == ci + 3]
            assert len(values) == 6
            reference = np.asarray([r["reference_fraction"] for r in values])
            np.testing.assert_allclose(reference, reference[0], atol=0, rtol=0)
            ref[ti, ci] = reference[0] * 100
            for mi, method in enumerate(("Pointwise", "Complete")):
                for si, seed in enumerate(SEEDS):
                    row = next(r for r in values if r["method"] == method and r["seed"] == seed)
                    predictions[mi, si, ti, ci] = row["predicted_fraction"] * 100
    np.testing.assert_allclose(ref.sum(-1), 100, atol=1e-12)
    np.testing.assert_allclose(predictions.sum(-1), 100, atol=1e-12)
    bias = predictions - ref[None, None, :, :]
    np.testing.assert_allclose(bias.sum(-1), 0, atol=1e-12)
    for row in rows:
        row["signed_error_pp"] = (row["predicted_fraction"] - row["reference_fraction"]) * 100
    return ref, bias, rows, provenance


def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    ref, bias, rows, provenance = data()
    style()
    plt.rcParams.update({"font.family": "Times New Roman", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "svg.hashsalt": "poremorph-s3-horizontal-preview"})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM, w / WIDTH_MM, h / HEIGHT_MM])

    a = ax_mm(17, 106, 94, 28)
    b = ax_mm(17, 65, 94, 27)
    c = ax_mm(17, 24, 94, 27)
    for label, title, y in (("a", "Reference oil-form composition", 143),
                             ("b", "Pointwise control: allocation bias", 98),
                             ("c", "PoreMorph: allocation bias", 57)):
        fig.text(1 / WIDTH_MM, y / HEIGHT_MM, label, fontsize=9, weight="bold")
        fig.text(9 / WIDTH_MM, y / HEIGHT_MM, title, fontsize=8)
    colors = STYLE["semantic_colors"][3:]
    fig.legend([Patch(facecolor=color, edgecolor="none") for color in colors], NAMES,
               loc="center", bbox_to_anchor=(.53, 137 / HEIGHT_MM), ncol=5,
               frameon=False, fontsize=7.3, handlelength=.9, handletextpad=.3, columnspacing=.65)
    left = np.zeros(4)
    yy = np.arange(4)
    for ci, color in enumerate(colors):
        a.barh(yy, ref[:, ci], left=left, height=.67, color=color, edgecolor="white", linewidth=.35)
        left += ref[:, ci]
    a.set_yticks(yy, STAGES)
    a.set(xlim=(0, 100), ylim=(3.5, -.5))
    a.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    a.tick_params(axis="y", length=0, pad=4, labelsize=7.5)
    a.tick_params(axis="x", length=2, pad=2, width=.45, labelsize=7.3)
    a.spines[["top", "right", "left"]].set_visible(False)
    a.spines["bottom"].set_color(INK)
    cmap = LinearSegmentedColormap.from_list("signed_allocation", ["#55A395", "#FCFCFC", "#CD7198"])
    bound = max(1., float(np.ceil(np.abs(bias.mean(1)).max() * 2) / 2))
    for mi, ax in enumerate((b, c)):
        mean, sd = bias[mi].mean(0), bias[mi].std(0, ddof=1)
        im = ax.imshow(mean, cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                       interpolation="nearest", aspect="auto")
        for row in range(4):
            for col in range(5):
                ax.text(col, row, f"{mean[row, col]:+.2f} ± {sd[row, col]:.2f}",
                        va="center", ha="center", fontsize=7, color=INK)
        ax.set_xticks(np.arange(5), NAMES)
        ax.set_yticks(np.arange(4), STAGES)
        ax.tick_params(axis="both", length=0, pad=4)
        ax.spines[:].set_visible(False)
    cb = ax_mm(32, 8.0, 64, 2.6)
    colorbar = fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[-bound, 0, bound])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7, length=1.7, width=.45, pad=1.5)
    fig.text(.5, 1.5 / HEIGHT_MM, "Predicted − reference fraction (percentage points)",
             fontsize=7, ha="center", va="center")
    fig.text(32 / WIDTH_MM, 13.1 / HEIGHT_MM, "Underestimated", fontsize=7, ha="left")
    fig.text(96 / WIDTH_MM, 13.1 / HEIGHT_MM, "Overestimated", fontsize=7, ha="right")
    stem = output_dir / "figS3_composition_bias"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[a, b, c], panel_ids=["a", "b", "c"],
        column_groups=[["a", "b", "c"]], row_groups=[], require_panel_labels=False,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=stem.with_suffix(".alignment.json"), strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with stem.with_suffix(".source.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance.update(reference_percent=ref.tolist(), signed_error_pp=bias.tolist(),
        signed_error_mean_pp=bias.mean(1).tolist(), signed_error_sample_sd_pp=bias.std(1, ddof=1).tolist(),
        class_order=NAMES, model_order=["Pointwise", "Complete"], display_mm=[WIDTH_MM, HEIGHT_MM],
        row_count=len(rows), signed_heatmap_range_pp=[-bound, bound],
        reference_display="Horizontal 100% stacked bars; 28-mm plot area; percentage tick labels",
        zero_sum_bias=True, reference_is_deterministic=True)
    stem.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": stem.name, "rows": len(rows), "reference_percent": ref.tolist(),
                      "mean_bias_pp": bias.mean(1).tolist()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    main(parser.parse_args().output_dir)
