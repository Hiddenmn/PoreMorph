"""Synthetic CPU contracts for external raw-image cache preparation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import reproduce as r
import external_runtime as e
import train_external as t
import prepare_external_data as p


class PreparationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pm_raw_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.inputs = self.base / "inputs"
        self.inputs.mkdir()
        self.output = self.base / "generated"

    def fixture(self, counts=(3, 2)):
        geometry = {"raw_grid_yx": [9, 12], "native_grid_yx": [6, 8], "crop_yx": [[1, 7], [2, 10]]}
        catalog = {"descriptions": {"bentheimer": "synthetic prior"}, "entries": {}}
        weight = e.definitions("external_eightclass_adaptation_20261003/bridge.py", ("fine_weights",), {"np": np}).fine_weights
        for split, count in zip(("train", "val"), counts):
            key = "bentheimer_" + split
            rows, values = [], {role: [] for role in p.ROLES}
            for index in range(count):
                row = {"label_id": f"{split}_{index}"}
                raw = (np.indices((9, 12)).sum(0) ** 2 + index * 7 + (split == "val") * 100).astype(np.uint16)
                lab = ((np.indices((9, 12)).sum(0) + index) % 8).astype(np.uint8)
                for role, value in (("raw", raw), ("label", lab)):
                    path = self.inputs / role / f"{split}_{index}.png"
                    path.parent.mkdir(exist_ok=True)
                    Image.fromarray(value).save(path)
                    row[role] = {"path": path.relative_to(self.inputs).as_posix(), "sha256": r.sha256(path)}
                sample = p.read_sample(row, geometry, self.inputs)
                for role in p.ROLES:
                    values[role].append(sample[role])
                rows.append(row)
            resources = {}
            for role in p.ROLES:
                array = np.stack(values[role])
                path = self.inputs / "old_cache" / f"{key}_{role}.npy"
                path.parent.mkdir(exist_ok=True)
                np.save(path, array)
                resources[role] = {"path": path.relative_to(self.inputs).as_posix(),
                    "sha256": r.sha256(path), "bytes": path.stat().st_size}
            pixel_counts = np.bincount(np.stack(values["mask"]).ravel(), minlength=8)
            catalog["entries"][key] = {"dataset": "bentheimer", "split": split, "shape": [count, 6, 8],
                "count": count, "records": rows, "resources": resources,
                "class_pixel_counts": pixel_counts.tolist(),
                "class_weights": weight(pixel_counts).tolist() if split == "train" else None}
        return catalog, {"bentheimer": geometry}

    def build(self, catalog, geometries, resume=False):
        return p.build(catalog, geometries, "bentheimer", self.inputs, self.output, resume)

    def test_all_exact_and_training_cache_handoff(self):
        catalog, geometry = self.fixture()
        result = self.build(catalog, geometry)
        self.assertEqual(result["cache_files_exact"], 6)
        self.assertEqual(result["images"], 5)
        self.assertFalse(result["training_started"] or result["test_pixels_read"] or result["gpu_initialized"])
        for split in ("train", "val"):
            old = t.Cached("bentheimer", split, catalog, self.inputs, lambda a, code: a.copy())
            new = t.Cached("bentheimer", split, catalog, self.output, lambda a, code: a.copy())
            for i in range(len(old)):
                for a, b in zip(old.get(i), new.get(i)):
                    np.testing.assert_array_equal(a, b)
            for data in (old, new):
                p.close_arrays({"image": data.image, "low": data.low, "mask": data.mask})

    def test_image_crop_precedes_normalization(self):
        catalog, geometry = self.fixture()
        row = catalog["entries"]["bentheimer_train"]["records"][0]
        with Image.open(self.inputs / row["raw"]["path"]) as handle:
            raw = np.array(handle)
        ops = e.preprocessing()
        expected = ops.normalize_native(raw[1:7, 2:10])
        actual = p.read_sample(row, geometry["bentheimer"], self.inputs)
        np.testing.assert_array_equal(actual["image"], expected)
        np.testing.assert_array_equal(actual["low"], ops.low_image(expected))
        self.assertFalse(np.array_equal(expected, ops.normalize_native(raw)[1:7, 2:10]))

    def test_mask_only_cropped_no_resize_or_relabel(self):
        catalog, geometry = self.fixture()
        row = catalog["entries"]["bentheimer_train"]["records"][0]
        with Image.open(self.inputs / row["label"]["path"]) as handle:
            label = np.array(handle)
        result = p.read_sample(row, geometry["bentheimer"], self.inputs)
        np.testing.assert_array_equal(result["mask"], label[1:7, 2:10])

    def test_constant_and_nonfinite_images(self):
        ops = e.preprocessing()
        np.testing.assert_array_equal(ops.normalize_native(np.ones((6, 8), np.uint16)), np.zeros((6, 8), np.float32))
        with self.assertRaises(ValueError):
            ops.normalize_native(np.full((6, 8), np.nan))

    def test_original_shape_required(self):
        catalog, geometry = self.fixture()
        row = catalog["entries"]["bentheimer_train"]["records"][0]
        geometry["bentheimer"]["raw_grid_yx"] = [6, 8]
        with self.assertRaises(ValueError):
            p.read_sample(row, geometry["bentheimer"], self.inputs)

    def test_source_hash_mismatch_rejected(self):
        catalog, geometry = self.fixture()
        row = copy.deepcopy(catalog["entries"]["bentheimer_train"]["records"][0])
        row["label"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            p.read_sample(row, geometry["bentheimer"], self.inputs)

    def test_illegal_label_rejected(self):
        catalog, geometry = self.fixture()
        row = catalog["entries"]["bentheimer_train"]["records"][0]
        path = self.inputs / row["label"]["path"]
        Image.fromarray(np.full((9, 12), 8, np.uint8)).save(path)
        row["label"]["sha256"] = r.sha256(path)
        with self.assertRaises(ValueError):
            p.read_sample(row, geometry["bentheimer"], self.inputs)

    def test_validation_weights_rejected(self):
        catalog, geometry = self.fixture()
        catalog["entries"]["bentheimer_val"]["class_weights"] = [1] * 8
        with self.assertRaises(ValueError):
            self.build(catalog, geometry)

    def test_no_test_route(self):
        catalog, geometry = self.fixture()
        with self.assertRaises(ValueError):
            p.entries_for(catalog, "test")
        catalog["entries"]["bentheimer_train"]["split"] = "test"
        with self.assertRaises(ValueError):
            self.build(catalog, geometry)

    def test_raw_and_label_tree_protection(self):
        catalog, _ = self.fixture()
        for out in (self.inputs, self.inputs / "raw/new", self.inputs / "label/new", self.inputs / "old_cache/new"):
            with self.assertRaises(ValueError):
                p.destination(out, self.inputs, catalog)

    def test_explicit_identical_resume_and_idempotency(self):
        catalog, geometry = self.fixture()
        self.build(catalog, geometry)
        with self.assertRaises(FileExistsError):
            self.build(catalog, geometry)
        self.build(catalog, geometry, resume=True)
        catalog["entries"]["bentheimer_train"]["records"][0]["label_id"] = "changed"
        with self.assertRaises(ValueError):
            self.build(catalog, geometry, resume=True)

    def test_interrupted_cache_resume_exact(self):
        catalog, geometry = self.fixture((27, 2))
        original = p.read_sample
        calls = []
        def interrupted(row, geom, root):
            calls.append(row["label_id"])
            if row["label_id"] == "train_26":
                raise InterruptedError("synthetic interruption")
            return original(row, geom, root)
        with patch.object(p, "read_sample", interrupted):
            with self.assertRaises(InterruptedError):
                self.build(catalog, geometry)
        state = r.read_json(self.output / "preparation.json")
        self.assertEqual(state["status"], "paused")
        self.assertEqual(state["entries"]["bentheimer_train"]["completed_rows"], 25)
        calls.clear()
        def resumed(row, geom, root):
            calls.append(row["label_id"])
            return original(row, geom, root)
        with patch.object(p, "read_sample", resumed):
            result = self.build(catalog, geometry, resume=True)
        self.assertEqual(calls, ["train_25", "train_26", "val_0", "val_1"])
        self.assertTrue(result["all_rebuilt_cache_bytes_exact"])

    def test_final_hash_mismatch_preserves_source(self):
        catalog, geometry = self.fixture()
        old = catalog["entries"]["bentheimer_train"]["resources"]["image"]
        path, before = self.inputs / old["path"], old["sha256"]
        old["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.build(catalog, geometry)
        self.assertEqual(r.sha256(path), before)

    def test_stop_marker_retained(self):
        catalog, geometry = self.fixture()
        marker = self.inputs / "STOP.txt"
        marker.touch()
        with self.assertRaises(InterruptedError):
            self.build(catalog, geometry)
        self.assertTrue(marker.exists())
        self.assertFalse(self.output.exists())

    def test_separate_cache_root_routes_without_changing_weights(self):
        catalog = t.load_catalog()
        accesses = []
        def checked(root, entry):
            accesses.append((Path(root), entry["path"]))
            return Path(root) / entry["path"]
        with patch.object(e, "checked_resource", checked):
            t.verify_resources(catalog, self.inputs, "wf1", 42, cache_root=self.output)
        self.assertEqual([root for root, _ in accesses[:3]], [self.inputs] * 3)
        self.assertEqual([root for root, _ in accesses[3:]], [self.output] * 6)
        engine = t.build_engine(catalog, 42, {}, self.inputs, self.base / "run", cache_root=self.output)
        with patch.object(t, "Cached") as cached:
            engine.original.run_epoch.__globals__["Cached"]("wf1", "train")
            self.assertEqual(cached.call_args.args[3], self.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r.verify_contents()
    output = r.new_output(args.output)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationContracts))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
              "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
              "gpu_initialized": torch.cuda.is_initialized(), "research_pixels_read": False,
              "training_started": False, "test_pixels_read": False}
    r.write_json(output, report)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
