"""Separate the existing S2 panels according to their scientific roles.

Main Figure 5 explains size-dependent recognition versus volume weighting;
Supplementary Figure S2 retains the class-wise predictor-stage contrasts.
No observations, bins, seeds, metrics, or uncertainty definitions change.
The original two-panel source and assets remain available as the prior version.
The main figure pairs recognition gains with complementary size distributions;
the Supplement retains a single-panel heatmap at the 114-mm PR text width.
"""
from __future__ import annotations
from figure_io import load_snapshot, asset_directory

import argparse
import hashlib
import json
from pathlib import Path

from figS2_stage_contributions import (
    HERE, SOURCE, SIZE_SOURCE, STYLE, SEEDS, DISPLAY_CONTRASTS,
    CONTRAST_LABELS, CLASSES, INK, data,
    require_matplotlib_panel_alignment,
)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

WIDTH_MM = 114.0  # Actual PR text width; not the generic Nature column widths.


def style():
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.serif': ['Times New Roman'],
        'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7.0,
        'ytick.labelsize': 7.5, 'axes.linewidth': .5, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canvas(height):
    width = WIDTH_MM
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=300)

    def axes(x, y, w, h):
        return fig.add_axes([x / width, y / height, w / width, h / height])

    return fig, axes


def export(fig, stem, record, panels=None):
    fig.canvas.draw()
    if panels is None:
        require_matplotlib_panel_alignment(
            fig, axes=[fig.axes[0]], panel_ids=['main'],
            require_panel_labels=False, strict=True,
            json_out=stem.with_suffix('.alignment.json'))
    else:
        require_matplotlib_panel_alignment(
            fig, axes=panels, panel_ids=['a', 'b'], row_groups=[],
            column_groups=[['a', 'b']], require_panel_labels=True, strict=True,
            json_out=stem.with_suffix('.alignment.json'))
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    fig.savefig(stem.with_suffix('.tiff'), dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    stem.with_suffix('.source.json').write_text(
        json.dumps(record, indent=2) + '\n', encoding='utf-8')


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, changes, rows = data()
    size = json.loads(SIZE_SOURCE.read_text(encoding='utf-8'))
    assert size['seeds'] == list(SEEDS) and size['images_per_seed'] == 340
    assert size['summary_sha256'] == sha(SOURCE)
    assert size['exact_native_confusion_recount']
    gains = np.asarray(size['paired_gain_pp'])
    assert gains.shape == (3, 6) and np.isfinite(gains).all()
    body_counts = np.asarray([r['component_count'] for r in size['rows']]).reshape(3, 6)
    pixel_counts = np.asarray([r['support_pixels'] for r in size['rows']]).reshape(3, 6)
    support_percent = 100 * pixel_counts / pixel_counts.sum(1, keepdims=True)
    body_percent = 100 * body_counts / body_counts.sum(1, keepdims=True)
    np.testing.assert_allclose(support_percent.sum(1), 100)
    np.testing.assert_allclose(body_percent.sum(1), 100)
    mean, sd = gains.mean(0), gains.std(0, ddof=1)
    style()

    height = 104.0
    fig, axes = canvas(height)
    ax = axes(22, 62, 90, 32)
    distribution = axes(22, 17, 90, 31)
    for panel, letter, title in (
        (ax, 'a', 'Subtype recognition: PoreMorph − Pointwise'),
        (distribution, 'b', 'Body frequency versus oil-volume weight'),
    ):
        panel.text(-.23, 1.13, letter, transform=panel.transAxes,
                   fontsize=9, fontweight='bold', va='bottom', ha='left')
        panel.text(0, 1.13, title, transform=panel.transAxes,
                   fontsize=8, va='bottom', ha='left')
    xx = np.arange(6)
    ax.bar(xx, mean, width=.63, color=STYLE['model_colors']['Complete'],
           alpha=.88, edgecolor='none', zorder=2)
    ax.errorbar(xx, mean, yerr=sd, fmt='none', color=INK,
                elinewidth=.6, capsize=2, capthick=.6, zorder=3)
    for si, marker in enumerate(('o', 's', '^')):
        ax.scatter(xx + (si - 1) * .12, gains[si], marker=marker, s=10,
                   facecolor='white', edgecolor=INK, linewidth=.45, zorder=4)
    ax.axhline(0, color=INK, linewidth=.55)
    ax.set(xlim=(-.5, 5.5), ylim=(-.3, 5.25), xticks=xx,
           yticks=[0, 1, 2, 3, 4, 5], xticklabels=size['bin_labels'],
           ylabel='Subtype accuracy gain\n(percentage points)')
    for offset, values, color, label in (
        (-.18, body_percent, '#DFA3BE', 'Body-count share'),
        (.18, support_percent, STYLE['model_colors']['Complete'], 'Oil-pixel share'),
    ):
        centers = xx + offset
        distribution.bar(centers, values.mean(0), width=.32, color=color,
                         alpha=.9, edgecolor='none', label=label, zorder=2)
        distribution.errorbar(
            centers, values.mean(0), yerr=values.std(0, ddof=1), fmt='none',
            color=INK, elinewidth=.55, capsize=1.8, capthick=.55, zorder=3)
        for si, marker in enumerate(('o', 's', '^')):
            distribution.scatter(centers + (si - 1) * .075, values[si],
                marker=marker, s=8, facecolor='white', edgecolor=INK,
                linewidth=.4, zorder=4)
    distribution.set(xlim=(-.5, 5.5), ylim=(0, 46), xticks=xx,
        xticklabels=size['bin_labels'], yticks=[0, 10, 20, 30, 40],
        xlabel='Predicted oil-body area (native pixels)', ylabel='Share (%)')
    distribution.legend(loc='upper left', ncol=2, frameon=False,
        handlelength=1.2, handletextpad=.4, columnspacing=1.3, fontsize=7,
        borderaxespad=.25)
    for panel in (ax, distribution):
        panel.grid(axis='y', color='#E5E8EA', linewidth=.35, zorder=0)
        panel.set_axisbelow(True)
        panel.spines[['top', 'right']].set_visible(False)
        panel.tick_params(axis='both', length=2.1, width=.45, pad=2.5)
    handles = [Line2D([0], [0], marker=m, markersize=3, linestyle='none',
                      color=INK, markerfacecolor='white', markeredgewidth=.5)
               for m in ('o', 's', '^')]
    fig.legend(handles, ['Seed 42', 'Seed 2024', 'Seed 2025'],
               loc='center', bbox_to_anchor=(.59, 3.5 / height), ncol=3,
               frameon=False, handletextpad=.4, columnspacing=1.1, fontsize=7)
    export(fig, output_dir / 'fig5_body_size', dict(
        source=str(SIZE_SOURCE), source_sha256=sha(SIZE_SOURCE),
        seeds=list(SEEDS), images_per_seed=340, grid='native',
        comparison='PoreMorph minus matched Pointwise body classifier',
        bins=size['bin_labels'], paired_gain_pp=gains.tolist(),
        mean_gain_pp=mean.tolist(), sample_sd_pp=sd.tolist(),
        body_counts=body_counts.tolist(), support_pixels=pixel_counts.tolist(),
        oil_support_share_percent=support_percent.tolist(), display_mm=[114, height],
        body_count_share_percent=body_percent.tolist(),
        panels={'a': 'Paired conditional subtype-accuracy gain',
                'b': 'Body-count share and oil-support pixel share by size'},
        uncertainty='Sample SD (ddof=1) across three training seeds',
        source_panel='figS2_stage_contributions panel b; numerical inputs unchanged'),
        panels=[ax, distribution])

    height = 80.0
    fig, axes = canvas(height)
    ax = axes(24, 24, 88, 51)
    shown = changes[:, :len(DISPLAY_CONTRASTS), :]
    cmean, csd = shown.mean(0), shown.std(0, ddof=1)
    bound = float(np.ceil(np.abs(cmean).max()))
    cmap = LinearSegmentedColormap.from_list(
        'stage_change', ['#C56686', '#FCFCFC', '#3E9F91'])
    image = ax.imshow(cmean.T, cmap=cmap,
                      norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                      interpolation='nearest', aspect='auto')
    for ci in range(len(CLASSES)):
        for ti in range(len(DISPLAY_CONTRASTS)):
            label = (f'{cmean[ti, ci]:+.2f} ± {csd[ti, ci]:.2f}'
                     if abs(cmean[ti, ci]) > 1e-12 else '0.00 ± 0.00')
            ax.text(ti, ci, label, ha='center', va='center', fontsize=7.5, color=INK)
    ax.set_yticks(np.arange(len(CLASSES)), CLASSES)
    ax.set_xticks(np.arange(len(DISPLAY_CONTRASTS)), CONTRAST_LABELS)
    ax.set_xticks(np.arange(-.5, len(DISPLAY_CONTRASTS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CLASSES), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.7)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(axis='both', length=0, pad=4)
    ax.spines[:].set_visible(False)
    colorbar = fig.colorbar(image, cax=axes(38, 7, 60, 2.6),
                           orientation='horizontal', ticks=[-bound, 0, bound])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7, length=1.7, width=.45, pad=1.8)
    fig.text(2 / 114, 8.3 / height, 'ΔIoU (pp)', fontsize=7.5, va='center')
    export(fig, output_dir / 'figS2_refinement_response', dict(
        source=str(SOURCE), source_sha256=sha(SOURCE), seeds=list(SEEDS),
        images_per_seed=340, grid='canonical512', class_order=CLASSES,
        displayed_contrasts=DISPLAY_CONTRASTS,
        paired_class_gain_pp=shown.tolist(), mean_pp=cmean.tolist(),
        sample_sd_pp=csd.tolist(), display_mm=[114, height],
        source_panel='figS2_stage_contributions panel a; numerical inputs unchanged'))
    print(json.dumps({'figures': ['fig5_body_size', 'figS2_refinement_response'],
                      'size_mean_gain_pp': mean.tolist(),
                      'all_seeds_and_bins_retained': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE)
    main(parser.parse_args().output_dir)
