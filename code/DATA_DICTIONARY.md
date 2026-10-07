# Data dictionary

## Native reference labels

Labels are integer class-ID images, not RGB masks. Main masks are palette TIFFs;
external masks are PNGs. Read pixel indices without converting a palette to RGB.

| ID | Class | Meaning |
|---:|---|---|
| 0 | Background | Outside the imaged sample/support |
| 1 | Rock | Rock skeleton |
| 2 | Water | Water/brine phase |
| 3 | Film oil | Oil assigned to the film morphology |
| 4 | Pore-corner oil | Oil assigned to pore-corner morphology |
| 5 | Droplet oil | Oil assigned to droplet morphology |
| 6 | Columnar oil | Oil assigned to columnar morphology |
| 7 | Clustered oil | Oil assigned to clustered morphology |

## Samples, splits and spatial grids

| Dataset | Train | Validation | Test | Original label grid |
|---|---:|---:|---:|---|
| Main, four stages | 2,364 | 676 | 340 | 1024 by 1024 |
| Bentheimer | 630 | 90 | 180 | 1014 by 992 |
| WF1 | 1,600 | 300 | 950 | 1033 by 1030 |

Each main stage (Oil, WI, NPI-006 and NPI-012) has 845 axial slices. The fixed
split has 591 training, 169 validation and 85 test slices per stage. Slice IDs
760–844 form the held-out block. The external sample records preserve the
original block-specific correspondences; raw-image and phase-label file numbers
must not be assumed identical.

Bentheimer uses the crop `y[20:1000], x[10:990]` for a 980 by 980 native model
grid. WF1 retains its full native grid. These source masks are not already
cropped or resampled. Use the r17 preprocessing and recorded nearest-neighbour
class-ID mapping for canonical512 scores. Do not substitute generic interpolation.

## Metadata fields

`main_samples.json` is a list of 3,380 records:

- `sample`, `slice_id`: original stage and ordered axial identifier.
- `split`: `train`, `val` or `test`.
- `image.path`, `image.sha256`: expected original image location and byte identity.
- `label.path`, `label.sha256`: reference mask location and byte identity.
- `prior`: exact description paired with the selected main-domain model.

`external_samples.json` is a list of 3,750 records:

- `id`, `group`, `split`: stable sample, block and partition identifiers.
- `image`, `label`: project-relative paths and SHA-256 values.
- `shape_yx`: original row/column dimensions.
- `metadata`: normalized per-image metadata and its new byte hash.
- `original_metadata_sha256`: identity of the historical metadata before path normalization.

External per-image metadata retain phase thresholds, class counts, edit records,
confidence summaries and original references. Only path representations change.
Their pre-acceptance status fields are historical. The subsequent whole-set
acceptance is supplied separately in `external_expert_acceptance.json` and
applies to the frozen v7 masks; it does not overwrite the earlier record.

Paths under `resources/` are relative to the extraction resource root. Metadata
paths outside that subtree are relative to the label archive root. Checkpoint
catalogs retain `original` and `whole_repeat` attempt names, seed, model role and
selected epoch. All hashes are SHA-256 of file bytes, not image-array checksums.

## Metrics and units

The eight-class mIoU is the unweighted class mean after pooling the pixel
confusion matrix over the specified evaluation split. Five-oil macro IoU uses
class IDs 3–7 only. Neither quantity is binary oil IoU. A difference expressed in
percentage points is not a relative percentage change.

The 3D utility assembles ordered 2D predictions; it is not a 3D neural model.
Provide calibrated voxel spacing in z,y,x order in micrometres. Outputs explicitly
identify volume (mm3), contact area (mm2), class voxel counts and oil-form fractions.
The small CPU example uses synthetic spacing and is not a physical measurement.

## Original sources

- Pak et al., X-ray computed micro-tomography dataset for nanoparticle-based oil
  removal from porous media: <https://doi.org/10.6084/m9.figshare.c.4190738>.
- Jackson et al., A large scale X-Ray micro-tomography dataset of steady-state
  multiphase flow: <https://doi.org/10.17612/KT0B-SZ28>.
- Scanziani et al., Water-wet three-phase flow micro-CT tomograms:
  <https://doi.org/10.17612/P7HT11>.

The eight-class morphology masks are derived study outputs, not evidence that
the manuscript authors acquired the original micro-CT data. Author-generated
annotations and metadata are CC BY 4.0; original inputs retain their own terms.
See [licence scope and source attribution](../RESOURCE_LICENSE.md).
