# Figure source data

All inputs describe the current manuscript, not an earlier submission. Missing
values are not imputed. Seeds 42, 2024 and 2025 are retained. Original external
adaptation and the whole WF1 repetition are separate groups, never seed-mixed.
Means and sample SD use the same calculations as the manuscript (`ddof=1`).

| Figure | Evidence and source group | Units / shape / interpretation |
|---|---|---|
| F1 | Real architecture inputs, `fig1_real_assets` | WI validation slice 675; actual activations, masks, body/contact sample and retained 3D illustration. Shared semantic colors; per-map 2nd–99th percentile display for features. |
| F2 | `main_comparison.json`, main qualitative snapshot | mIoU in percent, shape 3 seeds × 4 methods; paired subtype-IoU differences in percentage points, 3 × 5. Four selected native 256-square crops from 340 test slices. |
| F3 | `composition.json`, spatial snapshot | 5 stage summaries × 3 seeds for each method; five-class oil-fraction MAE in percentage points. The fifth summary averages four stages within each seed. Spatial panels: WI validation slices 591–759, central 256-square region, 13.25 μm voxels. |
| F4 | `external.json`, `external_classes.csv` | 3 domains/attempts × 2 grids × 3 seeds × 2 models × 8 classes. JSON plotting array uses percent; CSV `IoU`/`mIoU` use fractions in [0,1]. 288 CSV rows. |
| F5 | `main_test_summary.json`, `body_size.json` | Native two-dimensional body area; six fixed bins. Conditional subtype-accuracy gains in percentage points; body and oil-pixel shares each normalize to 100% per seed. |
| S1 | Retained morphology-example snapshot | Five reviewed annotation illustrations, unchanged RGB pixels. Scale from the original figure's vector calibration: 0.5 mm. |
| S2 | `main_test_summary.json`, `stage_classes.csv` | Three paired class-wise contrasts; all eight classes and three seeds. CSV IoU and mIoU are percentages. Body alternatives are not successive inference stages. |
| S3 | `composition.json`, `composition_classes.csv` | Reference and predicted fractions are in [0,1]. Displayed signed errors are percentage points, and sum to zero over the five oil forms for each stage/seed/method. |
| S4 | External qualitative snapshots and `external_selection.json` | Bentheimer slice 814 (initial), WF1 top-block slice 251 (whole repetition), seed 42. Fixed native fields, crops and relative error maps. No reranking. |

Main data comprise four displacement stages of one core, not four independent
specimens. The spatial illustration is ordered slice-wise inference, not an
end-to-end 3D predictor. The displayed morphology examples are explanatory;
F2 and S4 are performance-selected illustrations and do not replace aggregate
held-out comparisons. Their exact selection and crop fields remain in the
source-data records. Error maps include corrected, introduced and shared errors.

## File encoding and preservation

`source_data/*.json` contains exact decimal numeric values and scientific
metadata. Typed numeric arrays record dtype and shape explicitly. Larger image
arrays are referenced by dtype, shape, array digest and a separate NPZ member;
they are not embedded in this code/statistics archive. NPZ loading disables
pickle. Workstation paths are replaced by logical recorded-input identifiers;
source checksums, sample IDs, scientific values, units and array bytes remain.

`contents.json` verifies the packaged files. `asset_catalog.json` verifies the
separate image companion. `derivation.json` records the original source hashes
and which input-loading functions were adapted. Drawing/statistical code is
unchanged except for relocatable input routes. Archived intermediate inputs
are used instead of requiring access to the authors' training workstation.

The source tables retain all reported numerical observations, even when the
corresponding qualitative plate shows selected examples. No figures are
generated from mock observations. No significance tests are added.
