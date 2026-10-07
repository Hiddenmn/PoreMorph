# Training and data preparation

Run from the code directory. Replace example paths with new, short directories;
never point an output at code, input resources or a completed experiment. Use
the same seed at every stage. Supported fixed seeds are 42, 2024 and 2025.

## Main dataset: original files to the three-stage model

Place the original TIFFs and our eight-class masks under the relative paths in
`main_raw_preparation.json`. All 3,040 TRAIN/VAL pairs are checked against fixed
file/pixel identities. Preprocessing uses the original 1024-grid image, bilinear
512-grid images and nearest-neighbour 512-grid masks. There is no fitted
normalization or test-data loader in this preparation route.

```text
python prepare_main_data.py --resource-root ../resources --output ../runs/main_pixels
python train_main_global.py check --resource-root ../runs/main_pixels --seed 42 --report ../checks/main_pixels.json
python train_main_global.py run --resource-root ../runs/main_pixels --seed 42 --output ../runs/main_global42
python prepare_main_native.py --resource-root ../runs/main_pixels --seed 42 --global-run ../runs/main_global42 --output ../runs/main_coarse42 --trusted-checkpoints
python train_main_native.py run --resource-root ../runs/main_pixels --seed 42 --prepared-cache ../runs/main_coarse42 --output ../runs/main_native42
python prepare_main_bodies.py --resource-root ../runs/main_pixels --seed 42 --native-run ../runs/main_native42 --prepared-cache ../runs/main_coarse42 --output ../runs/main_bodies42 --trusted-checkpoints
python train_main_body.py run --resource-root ../runs/main_pixels --seed 42 --prepared-cache ../runs/main_bodies42 --output ../runs/main_head42
```

The global stage trains from scratch with the fixed 60-epoch/patience-10
protocol. The detail stage freezes the selected global predictor, trains for
12 epochs, then the body stage trains for 12 epochs from the selected native
representation. Each selection uses its documented validation rule; later
stages consume the selected previous run, not a checkpoint chosen by test score.
Inspect `--help` before using optional resume or smoke modes.

Main pixel resources require approximately 8.33 GiB plus a 512 MiB reserve.
The global array keeps its historical 3,380-slot row layout, but 340 non-training
slots are zero placeholders, not usable TEST data. Native arrays hold only
3,040 TRAIN/VAL rows. Three copied metadata bindings use project-relative paths;
`preparation.json` records the new reconstruction. A new coarse cache and body
descriptors need additional disk space. Allow roughly 6 GiB per seed for body
shards plus merged descriptors, with extra margin for changed predicted counts.

## External datasets: original files to complete-model fine-tuning

Use the original external grayscale images and our external eight-class labels.
Do not resize, invert intensity, or apply a generic normalization beforehand.
Bentheimer's recorded crop is applied before normalization; WF1 retains its
original grid. Source weights and labels remain read-only.

```text
python prepare_external_data.py --dataset all --resource-root ../resources --output ../runs/external_pixels
python train_external.py check --dataset bentheimer --seed 42 --resource-root ../resources --cache-root ../runs/external_pixels --report ../checks/bentheimer_pixels.json
python train_external.py run --dataset bentheimer --seed 42 --resource-root ../resources --cache-root ../runs/external_pixels --output ../runs/bentheimer_upstream42 --trusted-checkpoints
python prepare_external_bodies.py --case original_bentheimer_seed42 --upstream-run ../runs/bentheimer_upstream42 --resource-root ../resources --cache-root ../runs/external_pixels --output ../runs/bentheimer_bodies42 --trusted-checkpoints
python train_external_body.py run --prepared-cache ../runs/bentheimer_bodies42 --resource-root ../resources --cache-root ../runs/external_pixels --output ../runs/bentheimer_head42 --trusted-checkpoints
```

The upstream command performs the paired `global` and `native_detail` adaptation;
it does not omit the final body stage. The body-preparation/head commands finish
the complete model. For WF1, replace the dataset and case by `wf1` and
`original_wf1_seed42`. Repeat the matched chain for seeds 2024 and 2025 using
separate output paths. The original and whole-repeat WF1 cases remain distinct.

This route initializes from the selected main-domain parameters. Model-state
exports are valid initialization files because their state tensors are unchanged;
optimization starts fresh, not from an exported optimizer state. Cached
descriptors are regenerated from the selected new upstream run. A descriptor
smoke subset is not accepted as a full training cache.

## CPU checks

These use synthetic inputs and validate interfaces, data routing, selection and
training control, not a new reproduction of the paper's benchmark scores.
Create `../checks/` first. Each output must be new.

```text
python -m unittest test_shared_resources -v
python test_main_data_preparation.py --output ../checks/main_data.json
python test_main_global_training.py --output ../checks/main_global.json
python test_main_native.py --output ../checks/main_native.json
python test_main_body_preparation.py --output ../checks/main_body_data.json
python test_main_body.py --output ../checks/main_body.json
python test_external_runtime.py --output ../checks/external_runtime.json
python test_external_preparation.py --output ../checks/external_data.json
python test_external_training.py --output ../checks/external_upstream.json
python test_external_body_preparation.py --output ../checks/external_body_data.json
python test_external_body_training.py --output ../checks/external_body.json
```

GPU smoke commands are available separately. They must write to separate new
directories and do not authorize a change of scientific protocol. For stopped
work, retain its logs/checkpoints and stop marker until the operator decides
whether to resume.
