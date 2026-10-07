# Evaluation and ordered-volume measurements

## Paired label maps

Paths are relative to the manifest's directory, stay within it, and refer to
aligned uint8 class-ID maps with IDs 0–7. Sample IDs must be unique. No implicit
resizing is performed.

```json
[
  {"id":"example-000", "prediction":"pred/000.png", "reference":"ref/000.png"},
  {"id":"example-001", "prediction":"pred/001.png", "reference":"ref/001.png"}
]
```

Confusion-matrix rows are reference classes and columns are predictions. Pixel
counts are pooled before computing the fixed eight-class macro mean. Zero-union
classes contribute zero with the recorded 1e-8 denominator epsilon. Five-oil
macro IoU averages classes 3–7, not a merged binary-oil class.

## Ordered volume

```json
{
  "first_slice_id":760,
  "last_slice_id":762,
  "spacing_um_zyx":[13.25,26.5,26.5],
  "slices":[
    {"slice_id":760,"prediction":"pred/760.png"},
    {"slice_id":761,"prediction":"pred/761.png"},
    {"slice_id":762,"prediction":"pred/762.png"}
  ]
}
```

This is a three-slice interface example, not the paper's 85-slice test subvolume.
That analysis uses slice IDs 760–844 for each stage. Main canonical spacing is
13.25 × 26.5 × 26.5 μm in z,y,x order; main native spacing is 13.25 μm on each
axis. Use each external dataset's own calibration, not these main-data spacings.

Assembly is post-processing of ordered 2D labels, not an end-to-end 3D predictor
or a new tomographic reconstruction. Connectivity is 4 in the slice-wise body
classifier and 26 in volume geometry. The small-component reporting threshold
equals 27 canonical voxels (108 native voxels) in physical volume; components
are not removed from the label maps. Missing, duplicate or unordered slice IDs
are rejected by the measurement entry point.
