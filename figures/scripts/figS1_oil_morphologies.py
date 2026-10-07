"""Recompose the retained reviewed morphology examples at 114-mm text width.

The five embedded RGB panels are reused without changing their pixels. Only
labels, spacing and the vector scale bar are redrawn. The bar-to-field ratio
comes from the source figure's calibrated vector geometry, not its resampled
display-raster dimensions. These are annotation examples, not predictions.
"""
from __future__ import annotations
from figure_io import load_snapshot, asset_directory

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = (PROJECT / "NeuralNetworks_submission_20260926_DRAFT" /
          "source/fig_typical_oil_morphologies.pdf")
from audit_panel_alignment import require_matplotlib_panel_alignment

WIDTH_MM, HEIGHT_MM = 114.0, 28.0
CLASS_NAMES = ("Film", "Pore-corner", "Droplet", "Columnar", "Clustered")


def read_panels():
    return load_snapshot("morphology_examples")


def main(output_dir: Path):
    arrays, bar_fraction, metadata = read_panels()
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), dpi=300)
    gap, side = 1.4, (WIDTH_MM - 2.0 - 4*1.4)/5
    axes, ids = [], list("abcde")
    for i, (name, array) in enumerate(zip(CLASS_NAMES, arrays)):
        left = 1.0 + i*(side+gap)
        ax = fig.add_axes([left/WIDTH_MM, 1.0/HEIGHT_MM,
                           side/WIDTH_MM, side/HEIGHT_MM])
        axes.append(ax)
        ax.imshow(array, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)
            spine.set_edgecolor("#969696")
        fig.text(left/WIDTH_MM, 25.0/HEIGHT_MM, ids[i], fontsize=9.0,
                 fontweight="bold", va="center", ha="left")
        fig.text((left+side*.56)/WIDTH_MM, 25.0/HEIGHT_MM, name,
                 fontsize=7.5, color="#26313A", va="center", ha="center")
        if i == 0:
            x0, yy = 0.09, 0.08
            ax.plot([x0, x0+bar_fraction], [yy, yy], transform=ax.transAxes,
                    color="black", lw=2.8, solid_capstyle="butt")
            ax.plot([x0, x0+bar_fraction], [yy, yy], transform=ax.transAxes,
                    color="white", lw=1.25, solid_capstyle="butt")
            ax.text(x0+bar_fraction/2, 0.155, "0.5 mm", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7.0, color="white",
                    bbox=dict(facecolor="black", edgecolor="none", pad=0.25, alpha=.8))
    fig.canvas.draw()
    stem = output_dir / "figS1_oil_morphologies"
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=ids, row_groups=[ids],
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=stem.with_suffix(".alignment.json"), strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    stem.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"figure":str(stem), "panels":len(arrays),
                      "retained_scale_fraction":bar_fraction}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    main(parser.parse_args().output_dir)
