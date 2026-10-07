# PoreMorph

Prior-conditioned phase segmentation and fixed-support morphology recognition
of remaining oil. The three stages are a global image/prior predictor,
native-resolution detail recovery, and an oil-body morphology classifier.
The final classifier changes oil subtype assignments while preserving the
upstream oil support and all non-oil labels.

This is the **PoreMorph v0.1.0** model implementation. Author-written software is
[MIT](../LICENSE); author-generated data and states are [CC BY 4.0](../LICENSE-DATA.txt).
The current code is distinct from the earlier MM-PoreFormer v0.1.1 release.

## Contents

- Fresh three-stage training on the main dataset, including preparation from
  original image/label files and selected-stage feature caches.
- External fine-tuning of the upstream predictors and the complete oil-body head.
- Selected-model inference, eight-class metric recounting and ordered-volume
  measurements.
- Fixed sample IDs, priors, seeds, configurations and resource identities.
- CPU synthetic tests for model constraints, preprocessing and training control.

Use the documented root-level entry points. The source modules under
`workspace/` retain the computational implementations used by those entry
points; their old standalone drivers are not the public launch interface.
Paper-figure assembly is supplied in the sibling `../figures/` directory;
see [Figure reproduction](../figures/README.md). No raw third-party images, labels or weights are
embedded in the source archive.

## Environment

Use Python 3.12 in a separate environment. Install a suitable PyTorch/CUDA build
and the packages in `requirements.txt`. The verification environment is recorded
in `environment.json` (PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128). The formal
training and frozen inference entry points require a CUDA GPU with BF16 support.
Synthetic CPU tests need PyTorch but do not require a GPU. Different library or
GPU environments are not claimed to reproduce all floating-point results exactly.

On Windows, use a short checkout/output path. Keep all outputs outside the code
and input trees. Commands refuse existing outputs except documented explicit
resume of the same new run; stop markers are never removed automatically.

```text
python reproduce.py verify
python -m unittest test_shared_resources -v
python reproduce.py smoke --output ../checks/model_checks.json
python run_cpu_checks.py --output ../checks/all_cpu_checks
```

Create `../checks/` first. See [TRAINING.md](TRAINING.md) for the data-to-training
chain, memory/disk requirements and CPU checks. No training, dependency
installation, download or network publication starts on import.

## Data and resources

Original micro-CT images remain with their original providers:

- Main dataset: <https://doi.org/10.6084/m9.figshare.c.4190738>
- Bentheimer: <https://doi.org/10.17612/KT0B-SZ28>
- WF1: <https://doi.org/10.17612/P7HT11>

Our labels are a separate derived resource: 3,380 main masks and 3,750 external
masks. The label archive and selected-weight archive are intended as versioned
[v0.1.0 Release attachments](https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0). Their author-resource
licence is CC BY 4.0; original-provider terms remain separate.
Do not replace the labels with the providers' different label spaces.

Extract the label and model-state archives into a common parent, preserving
their `resources/` tree. Obtain authorized original images separately and place
them at the relative paths in the data catalogs. Example layout:

```text
PoreMorph/
  code/                 # this source package
  resources/            # images, our labels, selected weight states
    实验/...
    扩展数据1/...
    扩展数据2/...
  runs/                 # new caches and training outputs
  checks/
```

`main_raw_preparation.json` and `external_training.json` identify the required
TRAIN/VAL image/label files. Original file and decoded-pixel identities are
checked, rather than guessing matches from similar filenames. Test labels
belong to evaluation only. [DATA_DICTIONARY.md](DATA_DICTIONARY.md) defines
classes, priors, splits, geometry and label construction.

## Shared weights and full training checkpoints

The public model-state files contain the exact selected parameters and inference
metadata, without optimizer states or historical path metadata.
`resource_exports.json` explicitly maps each original checkpoint hash to its
verified model-state export; original scientific catalogs retain their original
checkpoint identities. A file is accepted only if it matches either the exact
original or the named export at the same relative path.

These selected states support inference and **fresh external fine-tuning**.
They are not optimizer-resume checkpoints. Resume a new training run from that
run's own full checkpoint. The fresh main training chain below does not require
archived model checkpoints. Historical-cache replay options require the original
full resources and are not part of the shared quick-start route.

## Predict, evaluate and quantify

From the code directory, using a fresh output for every command:

```text
python reproduce.py predict-main --seed 42 --resource-root ../resources --image ORIGINAL_MAIN_IMAGE.tif --description-file prior.txt --output ../runs/main_prediction --trusted-checkpoints
python reproduce.py predict-external --case original_bentheimer_seed42 --resource-root ../resources --image ORIGINAL_EXTERNAL_IMAGE.tif --output ../runs/external_prediction --trusted-checkpoints
python reproduce.py recount --manifest evaluation_pairs.json --output ../runs/metrics.json
python reproduce.py quantify --manifest ordered_stack.json --output ../runs/geometry.json
```

Use the recorded sample prior for main data. For an original-batch replay, replace
`--image` and `--description-file` with `--manifest main_batch.json`; supply 1–4
ordered records with `id`, resource-relative `image`, and `description` fields.
Batching and precision must match when checking exact historical predictions.
Only load weights from a trusted source. The `separate` prediction is complete
PoreMorph; `global` and other outputs are explicitly named comparison variants.

External inference uses the matching target-adapted model and preprocessing,
not a main-domain checkpoint alone. The six original cases and the three WF1
whole-repeat cases are distinct in `external_checkpoints.json`. Do not assemble
a new three-seed group by choosing each attempt's better result.

Metrics pool pixel counts before computing IoU. Five-oil macro IoU averages
class IDs 3–7; it is not binary-oil IoU. Predictions are class-ID images, not
recoloured visualizations. Manifest formats and voxel conventions are in
[MEASUREMENT.md](MEASUREMENT.md).

## Sharing status

Original third-party data and dependencies retain their own licences.
Confirmed author licences and separate third-party terms are described in
[the licence scope](../LICENSE_STATUS.md). Use the [versioned release](https://github.com/Hiddenmn/PoreMorph/releases/tag/v0.1.0)
for matching resources. No accepted-paper DOI is implied.
