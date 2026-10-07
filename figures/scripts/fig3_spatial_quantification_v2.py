"""Connect real spatial predictions to held-out oil-composition agreement.

Figure question: what spatial and quantitative oil-morphology information can
be obtained from the ordered predictions? Panels a-c exactly reuse the existing
WI validation illustration; panel d uses the already completed four-subvolume
test analysis. No inference, training, selection, or test-image access occurs.

Reuse contract: exact spatial renderer reuse from fig3_spatial_context_v1;
new quantitative panel from archived class counts, not rendered meshes.
Method -> pointwise / separate -> Pointwise / Complete, with fixed cross-figure
colors. Replicate -> training seed (42, 2024, 2025). Endpoint -> five-class
fraction MAE, multiplied by 100 to percentage points. Bars -> mean +/- sample
SD; all seed points retained. All-stages values average four stages per seed.
The four stages belong to one core, not independent specimens. No p-values.
Target -> PR, 114 x 143 mm, Times New Roman, minimum 7 pt; editable PDF/SVG.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import proj3d
import numpy as np

from fig3_spatial_context_v1 import (
    HERE, ROOT, STYLE, COLORS, NAMES, INK, PITCH_MM, FIRST, MID,
    read_volume, volume_panel, overlay, section, sha,
    require_matplotlib_panel_alignment,
)

WIDTH_MM, HEIGHT_MM = 114.0, 143.0
SPATIAL_OFFSET_MM = 55.0
ANALYSIS = ROOT / "实验/body_downstream_3d_20261001"
SEEDS = (42, 2024, 2025)
STAGES = ("Oil", "WI", "NPI-006", "NPI-012")
METHODS = (("pointwise", "Pointwise"), ("separate", "Complete"))
MARKERS = ("o", "s", "^")


def add_spatial_coordinates(ax, labels, render):
    """Locate the existing sections without changing volume pixels or camera."""
    lx, ly, lz = np.asarray(labels.shape)[[2, 1, 0]] * PITCH_MM
    xc, yc, zc = render["cut_planes_xyz_mm"]
    assert np.isclose(xc, (128+.5)*PITCH_MM)
    assert np.isclose(zc, (MID-FIRST+.5)*PITCH_MM)

    def screen(world):
        projected = proj3d.proj_transform(*world, ax.get_proj())[:2]
        return ax.transAxes.inverted().transform(ax.transData.transform(projected))

    # Positive array-index directions under the exact existing projection.
    start = np.array([.76, .055])
    zero = screen((0, 0, 0))
    tips = []
    for name, vector, offset in (("x", (.45, 0, 0), (.012, -.006)),
                                  ("y", (0, .45, 0), (.011, .010)),
                                  ("z", (0, 0, .45), (.029, -.004))):
        tip = start + screen(vector) - zero
        ax.annotate("", xy=tip, xytext=start, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=.6, mutation_scale=6))
        loc = tip + np.asarray(offset)
        ax.text2D(*loc, name, transform=ax.transAxes, fontsize=7.0,
                  ha="center", va="center", color=INK)
        tips.append({"axis": name, "world_delta_mm": list(vector), "tip_axes_fraction": tip.tolist()})
    for name, position in (("b", ((xc+lx)/2, yc/2, zc)),
                            ("c", (xc, yc*.42, zc+.40*(lz-zc)))):
        loc = screen(position)
        ax.text2D(*loc, name, transform=ax.transAxes, fontsize=7.5,
                  weight="bold", color=INK, ha="center", va="center", zorder=10000,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=1.0))
    return {"positive_index_triad": tips,
            "cut_face_labels": {"b": "xy at native slice 675", "c": "yz at native x 512"},
            "labels_are_annotations_only": True}


def section_directions(ax, horizontal, vertical):
    # A compact orientation marker in a common corner of both section images.
    start = np.array([.82, .12])
    for name, delta, text_offset in ((horizontal, (.11, 0), (.028, 0)),
                                     (vertical, (0, .14), (0, .045))):
        tip = start + delta
        ax.annotate("", xy=tip, xytext=start, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=.6, mutation_scale=5))
        ax.text(*(tip + text_offset), name, transform=ax.transAxes, color="white",
                fontsize=7.0, ha="center", va="center",
                bbox=dict(facecolor="black", edgecolor="none", alpha=.7, pad=.3))


def read_composition():
    return load_snapshot("composition")


def plot_composition(ax, values):
    x = np.arange(5)
    for mi, (_, display) in enumerate(METHODS):
        val = values[display]
        center = x + (-.18 if mi == 0 else .18)
        color = STYLE["model_colors"][display]
        mean, sd = val.mean(axis=1), val.std(axis=1, ddof=1)
        ax.bar(center, mean, width=.30, color=color, alpha=.82,
               edgecolor=color, linewidth=.6, zorder=2)
        ax.errorbar(center, mean, yerr=sd, fmt="none", ecolor=INK,
                    elinewidth=.55, capsize=1.5, capthick=.55, zorder=3)
        for si, marker in enumerate(MARKERS):
            ax.scatter(center+(si-1)*.068, val[:, si], marker=marker, s=9,
                       facecolors="white", edgecolors=INK, linewidths=.45, zorder=4)
    ax.set_ylim(0, .72)
    ax.set_xlim(-.6, 4.6)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6"])
    ax.set_xticks(x)
    ax.set_xticklabels([*STAGES, "All stages"])
    ax.set_ylabel("Fraction MAE (percentage points)", fontsize=7.0, labelpad=3)
    ax.tick_params(axis="both", labelsize=7, width=.5, length=2, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.5)
        ax.spines[side].set_color(INK)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E7EAED", linewidth=.4)
    ax.axvline(3.5, color="#B9C0C6", lw=.55, ls=(0, (2, 2)), zorder=1)


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, labels, spatial_source = read_volume()
    values, class_rows, quant_source = read_composition()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), dpi=300)
    def axes_mm(x, y, w, h, **kwargs):
        return fig.add_axes([x/WIDTH_MM, y/HEIGHT_MM, w/WIDTH_MM, h/HEIGHT_MM], **kwargs)
    offset = SPATIAL_OFFSET_MM
    a = axes_mm(0, 14+offset, 68, 68, projection="3d")
    spatial_source["volume_render"] = volume_panel(a, labels)
    spatial_source["coordinate_annotations"] = add_spatial_coordinates(a, labels, spatial_source["volume_render"])
    b = axes_mm(72, 44+offset, 39, 39)
    c = axes_mm(72, 13+offset, 39, 39*169/256)
    ly, lx = np.asarray(labels.shape[1:])*PITCH_MM
    lz = labels.shape[0]*PITCH_MM
    mid_z, mid_x = MID-FIRST, 128
    section(b, overlay(raw[mid_z], labels[mid_z]), lx, ly)
    section(c, overlay(raw[:, :, mid_x], labels[:, :, mid_x]), ly, lz)
    b.axvline((mid_x+.5)*PITCH_MM, color="white", lw=.6, ls=(0, (3, 2)), alpha=.85)
    c.axhline((mid_z+.5)*PITCH_MM, color="white", lw=.6, ls=(0, (3, 2)), alpha=.85)
    section_directions(b, "x", "y")
    section_directions(c, "y", "z")
    d = axes_mm(14, 12.5, 97, 32)
    plot_composition(d, values)
    for x, y, letter, title in [(1.5, 84.8+offset, "a", "Open-corner volume"),
                               (72, 85+offset, "b", "x–y section"),
                               (72, 40.8+offset, "c", "y–z section"),
                               (1.5, 49.5, "d", "Oil-fraction error")]:
        fig.text(x/WIDTH_MM, y/HEIGHT_MM, letter, fontsize=9, weight="bold", va="center")
        fig.text((x+5)/WIDTH_MM, y/HEIGHT_MM, title, fontsize=8, va="center")
    for j, cid in enumerate(range(1, 8)):
        row, col = divmod(j, 4)
        xx, yy = 2+28*col, 7.5-4.6*row+offset
        fig.add_artist(Rectangle((xx/WIDTH_MM, (yy-.65)/HEIGHT_MM),
            1.5/WIDTH_MM, 1.5/HEIGHT_MM, facecolor=COLORS[cid], edgecolor="#737D85",
            linewidth=.2, transform=fig.transFigure))
        fig.text((xx+2.6)/WIDTH_MM, yy/HEIGHT_MM, NAMES[cid], fontsize=7.5, va="center")
    display_labels = {"Pointwise": "Pointwise control", "Complete": "Complete model"}
    method_handles = [Patch(facecolor=STYLE["model_colors"][name], alpha=.82,
                            edgecolor="none", label=display_labels[name]) for _, name in METHODS]
    fig.legend(handles=method_handles, loc="center right", bbox_to_anchor=(.987, 49.5/HEIGHT_MM),
               frameon=False, ncol=2, fontsize=7.5, handlelength=1.0,
               handletextpad=.4, columnspacing=1.0, borderaxespad=0)
    seed_handles = [Line2D([], [], color="none", marker=marker, markerfacecolor="white",
                          markeredgecolor=INK, markeredgewidth=.45, markersize=3,
                          label=f"Seed {seed}") for marker, seed in zip(MARKERS, SEEDS)]
    fig.legend(handles=seed_handles, loc="center", bbox_to_anchor=(.55, 3.2/HEIGHT_MM),
               frameon=False, ncol=3, fontsize=7, handlelength=.8,
               handletextpad=.4, columnspacing=1.5, borderaxespad=0)
    spatial_source["orthogonal_sections"] = dict(xy_slice_id=MID, yz_native_x=512,
        input_limits=[0, 255], oil_overlay_alpha=.78, categorical_interpolation="nearest",
        orientation="positive indices right/up")
    fig.canvas.draw()
    stem = output_dir / "fig3_spatial_quantification_v2"
    require_matplotlib_panel_alignment(fig, axes=[a, b, c, d], panel_ids=["a", "b", "c", "d"],
        row_groups=[], column_groups=[["b", "c"]], require_panel_labels=False,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=stem.with_suffix(".alignment.json"), strict=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with stem.with_suffix(".class_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    plot_rows = [dict(grid="canonical512", stage=stage, seed=seed, method=name,
                     fraction_MAE_pp=float(values[name][ti, si]))
                 for ti, stage in enumerate([*STAGES, "All stages"])
                 for _, name in METHODS for si, seed in enumerate(SEEDS)]
    with stem.with_suffix(".plot_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)
    source = dict(display_size_mm=[WIDTH_MM, HEIGHT_MM],
        question="Spatial localization and oil-morphology composition from ordered predictions",
        spatial_panels_abc=spatial_source, quantitative_panel_d=quant_source,
        minimum_font_pt=7, script_sha256=sha(Path(__file__)),
        spatial_helper_sha256=sha(HERE / "fig3_spatial_context_v1.py"),
        source_labels_modified=False, model_or_protocol_changed=False)
    stem.with_suffix(".source.json").write_text(json.dumps(source, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(stem), class_rows=len(class_rows), plot_rows=len(plot_rows),
                          aggregate=quant_source["aggregate"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    main(parser.parse_args().output_dir)
