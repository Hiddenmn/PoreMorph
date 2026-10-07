# PoreMorph

**Prior-conditioned phase segmentation and fixed-support morphology recognition
of remaining oil.**

PoreMorph combines a sample-conditioned global predictor, native-resolution
detail recovery, and oil-body morphology recognition. The last stage assigns
one oil form to each predicted connected oil body while preserving the oil
support and non-oil labels. Ordered slice predictions also support
morphology-resolved volume measurements.

**Version 0.1.0.** [Download source, labels and weights](https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0).
Author-written code is MIT; author-generated data and selected model states
are CC BY 4.0. Third-party materials retain their own terms.

## Choose an entry point

| Task | Entry |
|---|---|
| Train all three stages or fine-tune on an external dataset | [Training guide](code/TRAINING.md) |
| Run selected models, recount metrics or measure ordered volumes | [Model guide](code/README.md), [measurement formats](code/MEASUREMENT.md) |
| Understand class IDs, sample priors and splits | [Data dictionary](code/DATA_DICTIONARY.md) |
| Reproduce the five main and four supplementary figures | [Figure guide](figures/README.md), [source-data dictionary](figures/SOURCE_DATA.md) |
| Locate the original images and the separate resources | [Data sources](DATA_SOURCES.md), [asset manifest](release_assets.json) |
| Check current sharing conditions | [Release preparation](RELEASE.md), [licence status](LICENSE_STATUS.md) |

## Folder layout

```text
PoreMorph/
  code/               # training, adaptation, inference and measurement
  figures/            # plotting code and numerical figure source data
  LICENSES/           # licence text for the unchanged third-party helper
  resources/          # labels, selected weights and authorized original images
  figure_assets/      # optional local-only image-derived figure inputs
  runs/               # new model outputs and prepared caches
  checks/             # new software-check outputs
  figure_outputs/     # newly rendered figures
```

The last five directories are inputs or generated outputs, not bundled source.
Do not commit raw images, large labels, weights or generated caches to Git.

## Installation

Use Python 3.12 in a separate environment. For model execution, install a
suitable PyTorch/CUDA build and `code/requirements.txt`. The recorded model
environment uses PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128; formal model
execution uses CUDA and BF16. CPU software checks use synthetic inputs.

For plotting alone, install `figures/requirements-figures.txt`; PyTorch and
model weights are not needed. The paper figures require a licensed local
Times New Roman installation. No font is redistributed. The shared numerical
dependencies have matching version pins in both requirement lists.

## Start from the repository root

Check the packaged files without loading a model or reading research pixels:

```sh
python verify_bundle.py
python code/reproduce.py verify
```

Check the model interfaces using synthetic CPU inputs:

```sh
python code/run_cpu_checks.py --output checks/cpu
```

Reproduce the statistics-only figures F4, F5, S2 and S3:

```sh
python figures/reproduce_figures.py --figures numeric --output figure_outputs
```

Other image-bearing panels require the separate local image-asset companion,
whose redistribution conditions are not yet confirmed. To render all nine
figures with that companion available:

```sh
python figures/reproduce_figures.py --figures all --assets figure_assets --output figure_outputs_all
```

Use a new output directory for each run. Commands inside the linked model
guides run from `code/`; figure-guide commands run from `figures/`. The commands
shown here run from the repository root. No command downloads data or publishes
anything automatically.

## Labels, weights and raw images

The two resource ZIPs are separate [v0.1.0 Release attachments](https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0).
`release_assets.json` gives their exact filenames, sizes, SHA-256 values
and direct download URLs.
Check each archive before installing it or loading trusted model weights:

```sh
python verify_bundle.py --labels-archive /path/to/labels.zip --weights-archive /path/to/weights.zip
python install_resources.py --labels-archive /path/to/labels.zip --weights-archive /path/to/weights.zip
```

The installer preserves the `resources/` tree, separates attachment metadata,
and never overwrites a file with different content. The label package contains
3,380 main-data and 3,750 external-data masks. The
weight package contains 39 selected model states and three tokenizers. These
states support inference and fresh fine-tuning, not optimizer-state resume.
Obtain the original CT images from the original repositories in
[DATA_SOURCES.md](DATA_SOURCES.md), using their applicable access conditions.

External models are target-adapted. Original experiments and the complete WF1
repeat are separately identified. The final body classifier operates on 2D
connected oil support; stacking predictions is downstream volume measurement,
not an end-to-end 3D segmentation network.

## Citation and sharing

The associated manuscript is *PoreMorph: Prior-conditioned phase segmentation
and fixed-support morphology recognition of remaining oil*. Cite software
version 0.1.0 at <https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0>; author metadata are in `CITATION.cff`.
No accepted-paper DOI is assigned by this software release.
Retain the original dataset citations when reusing the data. Author-written
software is under [MIT](LICENSE); author-generated annotations, metadata,
numerical source data and selected weights are under [CC BY 4.0](LICENSE-DATA.txt).
See [licence scope](LICENSE_STATUS.md) and [resource attribution](RESOURCE_LICENSE.md)
for the separate third-party terms and excluded image-derived assets.

