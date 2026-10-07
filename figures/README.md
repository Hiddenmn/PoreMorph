# PoreMorph figure reproduction

PoreMorph v0.1.0 contains Python plotting code and numeric source data for
the five main figures and four supplementary figures in the associated
Pattern Recognition manuscript. The code reproduces the recorded figures.
The rendering calculations, selected cases, units, colors and layout are retained.

## Run

Use Python 3.12 with `requirements-figures.txt` and a licensed local installation
of Times New Roman. Fonts are not redistributed; the runner fails rather than
silently substituting a different face. Run in a new output directory:

```sh
python reproduce_figures.py --figures numeric --output output_numeric
```

This recreates F4, F5, S2 and S3 without any CT images, model weights or training
workspace. Numeric input arrays use JSON; row-wise source tables use CSV.

The other figures require the separately prepared **local image-asset companion**:

```sh
python reproduce_figures.py --figures all --assets /path/to/figure_assets --output output_all
```

That companion contains fixed image crops, archived predictions, feature arrays
and a saved validation volume, not a mirror of the raw datasets. It remains a
local reproduction aid until redistribution conditions are confirmed. The
code/numeric package deliberately contains no research-image pixels. Merely
listing the companion here does not claim it is publicly available.

Each figure is exported as PDF, SVG, PNG and TIFF. The runner never overwrites an
existing output directory. `figure_catalog.json` maps current figure numbers to
scripts, source-data groups, and exact reference exports. `SOURCE_DATA.md`
defines units, arrays, selection, and the scope of reproduction.

## Upstream data

Original micro-CT is credited at its original repositories, not republished:

- Main data: https://doi.org/10.6084/m9.figshare.c.4190738
- Bentheimer: https://doi.org/10.17612/KT0B-SZ28
- WF1: https://doi.org/10.17612/P7HT11

The project supplies derived annotations, trained states and analysis separately.
For raw-data preparation, inference and volume measurement use the accompanying
PoreMorph training/prediction code. This figure package reproduces
the retained displays from their source data; it does not rerun model evaluation
or choose a better seed or image.

## Release status

Author-written plotting code is
[MIT](../LICENSE); numerical source data are [CC BY 4.0](../LICENSE-DATA.txt).
Versioned source: <https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0>. The image-asset
companion remains local under its separate redistribution conditions.

The included panel-alignment helper is copied from the locally installed
Nature figure workflow, with its attribution retained in `THIRD_PARTY.md`.
Its exact Apache-2.0 source and licence copy are recorded in
[the combined licence notice](../LICENSE_STATUS.md).
