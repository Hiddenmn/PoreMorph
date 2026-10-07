# PoreMorph v0.1.0

Versioned release: <https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0>.

## Downloads

- `PoreMorph_source_v0.1.0.zip`: training, adaptation, inference, evaluation,
  ordered-volume measurement, plotting code and numerical source data.
- `PoreMorph_labels_v0.1.0.zip`: 7,130 eight-class masks, sample correspondences,
  splits, prior descriptions and annotation metadata.
- `PoreMorph_weights_v0.1.0.zip`: 39 selected model states and three tokenizers.
- `SHA256SUMS.txt`: checksums for these three archives.

The two resource archives and their direct download links are recorded in
`release_assets.json`. Use one release version for source and resources.
Original CT volumes remain with their original providers; see
[DATA_SOURCES.md](DATA_SOURCES.md).

## Installation and use

Extract the source archive or check out the `v0.1.0` tag. From the repository root:

```sh
python verify_bundle.py
python install_resources.py --labels-archive /path/to/PoreMorph_labels_v0.1.0.zip --weights-archive /path/to/PoreMorph_weights_v0.1.0.zip
```

The installer verifies archive/member hashes, merges scientific files into
`resources/`, and puts attachment-specific documentation in
`resource_metadata/labels/` and `resource_metadata/weights/`. It refuses to
overwrite differing files. Obtain original images separately and preserve
the catalogued sample correspondence and preprocessing.

Follow [the main README](README.md), [training guide](code/TRAINING.md),
[model guide](code/README.md) or [figure guide](figures/README.md).
Selected model states support inference and fresh fine-tuning; they do not
contain optimizer state for resuming interrupted historical training.

## Licences and scope

Author-written code is MIT; author-generated labels, metadata, numerical source
data and selected states are CC BY 4.0. The unchanged Nature Skills alignment
helper retains Apache 2.0. Original data retain the providers' applicable terms.
See [licence scope](LICENSE_STATUS.md) and [resource credits](RESOURCE_LICENSE.md).

Image-derived figure assets are not included in this release. The numerical
figures F4, F5, S2 and S3 can be reproduced directly from the included values;
image-bearing figures require the separately described image inputs.

The release covers the current three-stage PoreMorph model. Original external
experiments and the complete WF1 repeat are provided as separate case groups.
The model predicts 2D slices; its volume utility assembles ordered predictions.
This software release does not imply journal acceptance or a new experiment.
