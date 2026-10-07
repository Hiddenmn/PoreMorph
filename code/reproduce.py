"""Portable entry points over the unchanged manuscript-linked source snapshot.

No training starts on import. Outputs must be new paths. Main-domain frozen
inference uses the historical CUDA/BF16 path; CPU smoke checks use synthetic
inputs and are not research results. Load checkpoints only from trusted sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import unittest

import numpy as np
from PIL import Image

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE / "workspace"
EXPERIMENTS = WORKSPACE / "实验"
LABEL_NAMES = ("background", "rock", "water", "film_oil", "pore_corner_oil",
               "droplet_oil", "columnar_oil", "clustered_oil")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = new_output(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def new_output(path):
    path = Path(path).resolve()
    if path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("Outputs must stay outside the archived workspace")
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    return path


def sha256(path):
    path = Path(path).resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        path = Path("\\\\?\\" + str(path))
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(base, relative):
    path = (Path(base) / relative).resolve()
    path.relative_to(Path(base).resolve())
    return path


def module(name, relative):
    path = local_path(EXPERIMENTS, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def pipeline_modules():
    for folder in ("body_inference_locked_20260930", "body_pointwise_control_20260930",
                   "body_interface_decoder_20260930", "native_detail_context_control_20260930",
                   "native_detail_decoder_20260930"):
        sys.path.insert(0, str(EXPERIMENTS / folder))
    sys.path.insert(0, str(WORKSPACE / "MM-PoreFormer/scripts"))
    import inference
    import locked_common
    return inference, locked_common


def verify_contents():
    manifest = read_json(PACKAGE / "contents.json")
    for entry in manifest["files"]:
        path = local_path(PACKAGE, entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Source changed: {entry['path']}")
    return {"status": "PASS", "files_verified": len(manifest["files"])}


def load_ids(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as im:
            value = np.array(im)
    if value.ndim != 2 or value.dtype != np.uint8 or value.size == 0 or value.max() > 7:
        raise ValueError(f"Expected a nonempty 2D uint8 label map in [0,7]: {path}")
    return value


def confusion(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction/target grids differ; implicit resizing is forbidden")
    return np.bincount((8 * target.astype(np.int64) + prediction).ravel(),
                       minlength=64).reshape(8, 8)


def scores(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (8, 8) or (matrix < 0).any() or matrix.sum() == 0:
        raise ValueError("Expected a nonempty 8x8 confusion matrix")
    diagonal = matrix.diagonal().astype(np.float64)
    actual, predicted = matrix.sum(1), matrix.sum(0)
    iou = diagonal / (actual + predicted - diagonal + 1e-8)
    dice = 2 * diagonal / (actual + predicted + 1e-8)
    return {"class_names": list(LABEL_NAMES), "confusion_matrix": matrix.tolist(),
            "class_IoU": iou.tolist(), "class_Dice": dice.tolist(),
            "mIoU": float(iou.mean()), "mDice": float(dice.mean()),
            "five_oil_macro_IoU": float(iou[3:].mean()), "pixels": int(matrix.sum())}


def run_smoke(output):
    verify_contents()
    import torch
    torch.set_num_threads(2)
    pipeline_modules()
    sys.path.insert(0, str(EXPERIMENTS / "body_downstream_3d_20261001"))
    tests = (
        ("pack_test_inference", "body_inference_locked_20260930/test_inference.py"),
        ("pack_test_body", "body_interface_decoder_20260930/test_model.py"),
        ("pack_test_control", "body_pointwise_control_20260930/test_control.py"),
        ("pack_test_geometry", "body_downstream_3d_20261001/test_metrics_3d.py"),
    )
    suite = unittest.TestSuite()
    for name, relative in tests:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module(name, relative)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    # The entry-point metric implementation must reproduce the archived helper.
    _, original = pipeline_modules()
    matrix = np.arange(64, dtype=np.int64).reshape(8, 8)
    actual, expected = scores(matrix), original.metrics(matrix)
    for key in ("class_IoU", "class_Dice", "mIoU", "mDice"):
        np.testing.assert_array_equal(actual[key], expected[key])
    if not result.wasSuccessful():
        raise RuntimeError("Synthetic source tests failed")
    report = {"status": "PASS", "original_tests": result.testsRun,
              "metric_helper_exact": True, "training_started": False,
              "research_images_or_labels_read": False,
              "scope": "Synthetic unit/invariant tests, not paper-metric reproduction"}
    write_json(output, report)
    return report


def predict_main(args):
    if not args.trusted_checkpoints:
        raise ValueError("Use --trusted-checkpoints only for trusted locally verified weights")
    verify_contents()
    out = new_output(args.output)
    if args.manifest:
        if args.description_file:
            raise ValueError("Batch descriptions are part of the manifest")
        inputs = read_json(args.manifest)
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 4:
            raise ValueError("Provide 1--4 aligned records in their original batch order")
        ids = [item["id"] for item in inputs]
        if len(set(ids)) != len(ids) or any(
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                for name in ids):
            raise ValueError("Use distinct, simple alphanumeric record IDs")
        inputs = [dict(item, path=local_path(args.resource_root, item["image"])) for item in inputs]
    else:
        if not args.description_file:
            raise ValueError("Single-image inference requires --description-file")
        inputs = [{"id": None, "path": Path(args.image).resolve(),
                   "description": Path(args.description_file).read_text(encoding="utf-8").strip()}]
    images, descriptions = [], []
    for item in inputs:
        with Image.open(item["path"]) as im:
            # External float images deliberately use a different entry point.
            if im.size != (1024, 1024) or im.mode not in ("L", "RGB", "RGBA"):
                raise ValueError("Use 1024x1024 8-bit main-domain slices; external data have a separate protocol")
            images.append(np.array(im.convert("L"), dtype=np.uint8))
        description = item["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError("A nonempty sample-prior description is required")
        descriptions.append(description)
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The recorded full inference path requires a CUDA/BF16 GPU")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.35)
    config = read_json(PACKAGE / "main_checkpoints.json")[str(args.seed)]
    files = {}
    for role, item in config.items():
        from shared_resources import checked
        path = checked(args.resource_root, item)
        files[role] = path
    inference, _ = pipeline_modules()
    predictor = inference.FrozenPipeline(args.seed, files)
    reports = []
    try:
        predictions = predictor.predict_batch(images, descriptions)
        out.mkdir(parents=True)
        for index, (item, description, prediction) in enumerate(zip(inputs, descriptions, predictions)):
            destination = out / item["id"] if args.manifest else out
            if args.manifest:
                destination.mkdir()
            saved = {}
            for grid in ("native", "canonical512"):
                upstream = prediction[grid]["native_detail"]
                for method, labels in prediction[grid].items():
                    if method in ("body_mean", "pointwise", "separate"):
                        np.testing.assert_array_equal(labels >= 3, upstream >= 3)
                        np.testing.assert_array_equal(labels[upstream < 3], upstream[upstream < 3])
                    name = f"{grid}_{method}.png"
                    Image.fromarray(labels).save(destination / name)
                    saved[name] = {"sha256": sha256(destination / name), "shape": list(labels.shape)}
            report = {"seed": args.seed, "global_epoch": predictor.global_epoch,
                      "native_epoch": predictor.native_epoch, "body_epochs": predictor.head_epochs,
                      "input_sha256": sha256(item["path"]), "description": description,
                      "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
                      "batch_size": len(inputs), "batch_index": index,
                      "precision": "Global/native BF16; heads FP32",
                      "oil_bodies": prediction["bodies"], "outputs": saved,
                      "oil_support_and_non_oil_labels_preserved": True,
                      "reference_labels_read": False, "source": "unchanged frozen manuscript pipeline"}
            write_json(destination / "prediction.json", report)
            reports.append({"id": item["id"], "oil_bodies": report["oil_bodies"],
                            "prediction_metadata_sha256": sha256(destination / "prediction.json")})
        if args.manifest:
            write_json(out / "batch_prediction.json", {"seed": args.seed, "batch_size": len(inputs),
                       "input_manifest_sha256": sha256(args.manifest), "items": reports})
    finally:
        predictor.close()
    return {"status": "PASS", "output": str(out), "batch_size": len(inputs), "items": reports}


def recount(args):
    verify_contents()
    new_output(args.output)
    manifest_path = Path(args.manifest).resolve()
    rows = read_json(manifest_path)
    if not rows:
        raise ValueError("Empty evaluation manifest")
    matrix, seen = np.zeros((8, 8), np.int64), set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample: {row['id']}")
        seen.add(row["id"])
        pred = load_ids(local_path(manifest_path.parent, row["prediction"]))
        target = load_ids(local_path(manifest_path.parent, row["reference"]))
        matrix += confusion(pred, target)
    result = scores(matrix)
    result.update(samples=len(rows), aggregation="pooled-pixel confusion, fixed eight-class macro",
                  zero_union_convention="zero, with denominator epsilon 1e-8",
                  manifest_sha256=sha256(manifest_path))
    write_json(args.output, result)
    return result


def quantify(args):
    verify_contents()
    new_output(args.output)
    manifest_path = Path(args.manifest).resolve()
    config = read_json(manifest_path)
    rows = config["slices"]
    ids = [row["slice_id"] for row in rows]
    expected = list(range(config["first_slice_id"], config["last_slice_id"] + 1))
    if not expected or ids != expected:
        raise ValueError("Require every expected slice once, in original axial order")
    spacing = np.asarray(config["spacing_um_zyx"], dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or not (spacing > 0).all():
        raise ValueError("Specify positive calibrated z,y,x voxel spacings in micrometres")
    planes = [load_ids(local_path(manifest_path.parent, row["prediction"])) for row in rows]
    if len({plane.shape for plane in planes}) != 1:
        raise ValueError("All axial slices must share one spatial grid")
    volume = np.stack(planes)
    m = module("pack_metrics_3d", "body_downstream_3d_20261001/metrics_3d.py")
    counts = m.counts(volume)
    oil = int(counts[3:].sum())
    result = {"class_voxels": counts.tolist(), "class_names": list(LABEL_NAMES),
              "oil_form_fractions": (counts[3:] / oil).tolist() if oil else None,
              "shape_zyx": list(volume.shape), "spacing_um_zyx": spacing.tolist(),
              "geometry": m.geometry(volume, spacing, counts),
              "assembly": "ordered 2D labels; no cross-slice neural inference",
              "geometry_connectivity": 26,
              "component_reference_min_volume_um3": m.MIN_VOLUME_UM3,
              "manifest_sha256": sha256(manifest_path)}
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Check archived source bytes")
    smoke = sub.add_parser("smoke", help="CPU synthetic tests; no training or research-data reads")
    smoke.add_argument("--output", type=Path, required=True)
    predict = sub.add_parser("predict-main", help="Frozen complete main-domain prediction")
    predict.add_argument("--seed", type=int, choices=(42, 2024, 2025), required=True)
    predict.add_argument("--resource-root", type=Path, required=True)
    input_group = predict.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path)
    input_group.add_argument("--manifest", type=Path)
    predict.add_argument("--description-file", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--trusted-checkpoints", action="store_true")
    for command in ("recount", "quantify"):
        task = sub.add_parser(command)
        task.add_argument("--manifest", type=Path, required=True)
        task.add_argument("--output", type=Path, required=True)
    external = sub.add_parser("predict-external", help="Frozen target-adapted complete/Global inference")
    external.add_argument("--case", required=True)
    external.add_argument("--resource-root", type=Path, required=True)
    external.add_argument("--image", type=Path, required=True)
    external.add_argument("--output", type=Path, required=True)
    external.add_argument("--trusted-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_contents()
    elif args.command == "smoke":
        result = run_smoke(args.output)
    elif args.command == "predict-main":
        result = predict_main(args)
    elif args.command == "predict-external":
        from external_runtime import run
        result = run(args)
    elif args.command == "recount":
        result = recount(args)
    else:
        result = quantify(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
