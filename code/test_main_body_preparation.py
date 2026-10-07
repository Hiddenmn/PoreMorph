"""CPU contracts for main native-to-body preparation; synthetic pixels only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import reproduce as r
import main_body as b
import main_native as n
import prepare_main_bodies as p


def pixels():
    image = np.arange(256, dtype=np.float32).reshape(16, 16)/255
    pred = np.ones((16, 16), np.uint8)
    pred[2:7, 2:7] = 3
    pred[2:7, 7] = 2
    pred[10, 10] = 7
    pred[11, 11] = 4  # Diagonal contact is not 4-connected.
    probs = np.eye(8, dtype=np.float32)[pred].transpose(2, 0, 1)
    fine = np.arange(16*256, dtype=np.float32).reshape(16, 16, 16)/4096
    truth = (np.arange(256).reshape(16, 16) % 8).astype(np.uint8)
    return image, probs, fine, truth


class PreparationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c, cls.nc = b.load_catalog(), n.load_catalog()
        cls.fingerprint = p.implementation()
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pm-mainprep-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def binding(self, seed=42):
        case = self.nc["cases"][str(seed)]
        return {"seed": seed, "coarse_sha256": case["coarse"]["sha256"],
            "selected_global": {"kind": "archived-selected", "seed": seed,
                "checkpoint_sha256": case["global_checkpoint"]["sha256"]}}

    def selection(self):
        case = self.nc["cases"]["42"]
        val = case["archived_validation"]
        history = [{"epoch": i, "validation": copy.deepcopy(val)} for i in range(1, 13)]
        ident = {"stage": "main native-detail", "seed": 42, "config": case["config"],
            "selection": self.nc["selection"], "native_catalog_sha256": r.sha256(r.PACKAGE/"main_native_training.json"),
            "source_contents_sha256": self.c["compatible_native_contents_sha256"][0], "fresh_run": True,
            "full_initial_sha256": case["full_initial_sha256"], **self.binding()}
        meta = {"identity": ident, "status": "completed", "selected_replay_exact": True,
            "test_pixels_read": False, "completed_epochs": 12, "best_epoch": 1,
            "best_native_mIoU": val["native"]["mIoU"]}
        selected = {"epoch": 1, "validation": val, "selection": self.nc["selection"],
            "checkpoint": "best.pth", "sha256": "1"*64, "test_pixels_read": False}
        return meta, selected, history

    def prepared(self):
        case = self.nc["cases"]["42"]
        info = {"kind": "archived-selected", "seed": 42,
            "checkpoint_sha256": case["native_checkpoint"]["sha256"],
            "epoch": case["archived_selected_epoch"], "validation": case["archived_validation"]}
        ident = p.cache_identity(self.c, self.nc, self.binding(), info, False)
        return {"identity": ident, "status": "completed", "seed": 42, "smoke": False,
            "trainable": True, "frozen_validation_replayed": True, "test_pixels_read": False,
            "research_training_started": False, "completed_images": 3040,
            "selected_rows": list(range(3040)), "images_by_split": {"train": 2364, "val": 676}}

    def small_cache(self, fail=False):
        records = [{"sample": "synthetic", "slice_id": i, "split": "train" if i == 0 else "val",
                    "shared_index": i} for i in range(2)]
        calls = []
        def producer(row):
            calls.append(row)
            if fail and row == 1:
                raise InterruptedError("Synthetic interruption")
            return p.row_values(*pixels(), row, row)
        identity = {"seed": 42, "smoke": True, "version": "synthetic-only"}
        return records, producer, identity, calls

    def test_original_descriptor_exact(self):
        image, probs, fine, truth = pixels()
        values = p.row_values(image, probs, fine, truth, 9, 17)
        desc, labels, pred = p.representation.describe(image, probs, fine)
        counts, outside = p.representation.targets(labels, pred, truth)
        for key, value in desc.items():
            np.testing.assert_array_equal(values[key], value)
        for grid in p.GRIDS:
            np.testing.assert_array_equal(values[grid+"_counts"], counts[grid])
            np.testing.assert_array_equal(values[grid+"_outside"], outside[grid])
        self.assertEqual((int(values["row"]), int(values["shared_index"])), (9, 17))

    def test_targets_never_change_forward_descriptors(self):
        args = list(pixels())
        first = p.row_values(*args, 0, 0)
        args[-1] = 7 - args[-1]
        second = p.row_values(*args, 0, 0)
        for key in ("points", "valid", "summary"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.array_equal(first["native_counts"], second["native_counts"]))

    def test_unfiltered_four_connected_single_pixels_retained(self):
        values = p.row_values(*pixels(), 0, 0)
        self.assertEqual(len(values["points"]), 3)
        self.assertEqual(sorted(values["native_counts"].sum(1).tolist()), [1, 1, 25])

    def test_original_shapes_and_dtypes(self):
        values = p.row_values(*pixels(), 0, 0)
        for key, (dtype, tail) in b.LAYOUT.items():
            self.assertEqual(values[key].shape, (3, *tail))
            self.assertEqual(values[key].dtype, dtype)

    def test_both_grid_pixel_accounting(self):
        values = p.row_values(*pixels(), 0, 0)
        for grid, total in (("native", 256), ("canonical512", 64)):
            self.assertEqual(int(values[grid+"_counts"].sum()+values[grid+"_outside"].sum()), total)
            self.assertEqual(int(values[grid+"_frozen"].sum()), total)
            self.assertFalse(values[grid+"_outside"][:, 3:].any())

    def test_deterministic_angular_sampling(self):
        yy, xx = np.indices((9, 9))
        a = p.representation.sample_group(yy.ravel(), xx.ravel(), 4, 4)
        expected = np.lexsort((xx.ravel(), yy.ravel(), ((yy-4)**2+(xx-4)**2).ravel(),
                               np.arctan2(yy-4, xx-4).ravel()))[np.linspace(0, 80, 16, dtype=np.int64)]
        np.testing.assert_array_equal(a, expected)

    def test_smoke_fixed_by_stage_and_split(self):
        rows = p.selected_rows(self.nc["records"], True)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({(self.nc["records"][i]["sample"], self.nc["records"][i]["split"]) for i in rows}), 8)
        self.assertEqual(p.selected_rows(self.nc["records"], False), list(range(3040)))

    def test_test_rows_rejected(self):
        with self.assertRaises(ValueError):
            p.selected_rows([{"split": "test"}], True)

    def test_native_first_tie_selection(self):
        self.assertEqual(p.selection_contract(*self.selection(), self.c, self.nc, 42, self.binding()), 1)

    def test_native_selection_rejects_incomplete_wrong_seed_and_protocol(self):
        for key, value in (("status", "running"), ("completed_epochs", 11),
                           ("test_pixels_read", True), ("selected_replay_exact", False)):
            meta, selected, history = self.selection(); meta[key] = value
            with self.assertRaises(ValueError):
                p.selection_contract(meta, selected, history, self.c, self.nc, 42, self.binding())
        for key, value in (("seed", 2024), ("fresh_run", False), ("coarse_sha256", "0"*64),
                           ("source_contents_sha256", "0"*64), ("config", {}), ("full_initial_sha256", "0"*64)):
            meta, selected, history = self.selection(); meta["identity"][key] = value
            with self.assertRaises(ValueError):
                p.selection_contract(meta, selected, history, self.c, self.nc, 42, self.binding())

    def test_native_later_tie_and_partial_history_rejected(self):
        meta, selected, history = self.selection()
        selected["epoch"] = 2
        with self.assertRaises(ValueError):
            p.selection_contract(meta, selected, history, self.c, self.nc, 42, self.binding())
        with self.assertRaises(ValueError):
            p.selection_contract(meta, selected, history[:-1], self.c, self.nc, 42, self.binding())

    def test_native_metric_corruption_rejected(self):
        value = copy.deepcopy(self.nc["cases"]["42"]["archived_validation"])
        value["native"]["confusion_matrix"][0][0] += 1
        with self.assertRaises(ValueError):
            p.validate_metrics(value)

    def test_archived_native_rejects_changed_coarse_before_weight_read(self):
        binding = self.binding(); binding["coarse_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            p.selected_native(self.c, self.nc, self.root, 42, binding)

    def test_trust_required_before_any_read(self):
        with self.assertRaisesRegex(ValueError, "trust"):
            p.run(argparse.Namespace(trusted_checkpoints=False))

    def test_prepared_full_archived_identity(self):
        p.prepared_contract(self.prepared(), self.c, 42)

    def test_smoke_incomplete_and_wrong_prepared_seed_rejected(self):
        for key, value in (("status", "preparing"), ("smoke", True), ("trainable", False),
                           ("frozen_validation_replayed", False), ("completed_images", 8), ("seed", 2024)):
            meta = self.prepared(); meta[key] = value
            with self.assertRaises(ValueError):
                p.prepared_contract(meta, self.c, 42)

    def test_prepared_source_and_body_protocol_changes_rejected(self):
        for key in ("implementation", "body_protocol_sha256", "records_sha256", "native_catalog_sha256"):
            meta = self.prepared(); meta["identity"][key] = "0"*64
            with self.assertRaises(ValueError):
                p.prepared_contract(meta, self.c, 42)

    def test_prepared_archived_pixel_coarse_and_model_changes_rejected(self):
        for section, key in (("shared_pixels", "image"), ("native", "checkpoint_sha256"),
                             ("selected_global", "checkpoint_sha256")):
            meta = self.prepared(); meta["identity"][section][key] = "0"*64
            with self.assertRaises(ValueError):
                p.prepared_contract(meta, self.c, 42)
        meta = self.prepared(); meta["identity"]["native"]["epoch"] = 1
        with self.assertRaises(ValueError):
            p.prepared_contract(meta, self.c, 42)

    def test_completed_portable_native_identity_accepted(self):
        meta = self.prepared()
        info = meta["identity"]["native"]
        info.update(kind="new-portable-selected", checkpoint_sha256="1"*64,
            run_manifest_sha256="2"*64, selected_validation_sha256="3"*64,
            source_contents_sha256=self.c["compatible_native_contents_sha256"][0])
        p.prepared_contract(meta, self.c, 42)
        info["source_contents_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            p.prepared_contract(meta, self.c, 42)

    def test_preparation_merge_and_completed_resume_no_forward(self):
        rows, producer, identity, calls = self.small_cache()
        out = self.root/"out"
        meta = p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,))
        self.assertEqual((meta["status"], meta["n"], calls), ("completed", 6, [0, 1]))
        self.assertFalse(meta["trainable"])
        self.assertFalse(meta["frozen_validation_replayed"])
        p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,), True)
        self.assertEqual(calls, [0, 1])
        for key in b.LAYOUT:
            merged = np.load(out/(key+".npy"), allow_pickle=False)
            np.testing.assert_array_equal(merged[:3], p.row_values(*pixels(), 0, 0)[key])

    def test_interruption_resume_only_unfinished_images(self):
        rows, producer, identity, calls = self.small_cache(True)
        out = self.root/"out"
        with self.assertRaises(InterruptedError):
            p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,))
        first = r.sha256(out/"shards/0000.npz")
        self.assertEqual(r.read_json(out/"manifest.json")["status"], "stopped")
        _, producer, _, calls = self.small_cache()
        p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,), True)
        self.assertEqual(calls, [1])
        self.assertEqual(first, r.sha256(out/"shards/0000.npz"))

    def test_changed_resume_identity_rejected(self):
        rows, producer, identity, _ = self.small_cache()
        out = self.root/"out"
        p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,))
        with self.assertRaises(ValueError):
            p.prepare_cache(out, rows, [0, 1], dict(identity, seed=2024), producer, None, (self.root,), True)
        with self.assertRaises(FileExistsError):
            p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,))

    def test_changed_committed_shard_rejected(self):
        rows, producer, identity, _ = self.small_cache(True)
        out = self.root/"out"
        with self.assertRaises(InterruptedError):
            p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,))
        (out/"shards/0000.npz").write_bytes(b"synthetic corruption")
        with self.assertRaisesRegex(ValueError, "shard changed"):
            p.prepare_cache(out, rows, [0, 1], identity, producer, None, (self.root,), True)

    def test_full_frozen_mismatch_cannot_be_trainable(self):
        rows, producer, identity, _ = self.small_cache()
        identity["smoke"] = False
        out = self.root/"out"
        with self.assertRaisesRegex(ValueError, "validation changed"):
            p.prepare_cache(out, rows, [0, 1], identity, producer, {}, (self.root,))
        self.assertFalse(r.read_json(out/"manifest.json")["trainable"])

    def test_stop_marker_retained_without_inference(self):
        rows, producer, identity, calls = self.small_cache()
        (self.root/"STOP_TRAINING.txt").touch()
        with self.assertRaises(InterruptedError):
            p.prepare_cache(self.root/"out", rows, [0, 1], identity, producer, None, (self.root,))
        self.assertEqual(calls, [])
        self.assertTrue((self.root/"STOP_TRAINING.txt").exists())

    def test_prepared_input_cannot_be_training_output(self):
        source = self.root/"input"; source.mkdir()
        with self.assertRaises(ValueError):
            b.output_path(source/"new", self.root, self.c, source)

    def test_root_stop_applies_to_prepared_training_route(self):
        (self.root/"STOP_TRAINING.txt").touch()
        with patch("prepare_main_bodies.bind_prepared") as mocked:
            with self.assertRaises(InterruptedError):
                b.bind(self.c, self.root, 42, self.root/"input")
            mocked.assert_not_called()

    def test_new_prepared_binding_does_not_inherit_archived_metrics(self):
        meta = self.prepared()
        meta.update(n=17, bodies_by_split={"train": 9, "val": 8},
                    records_sha256="1"*64, outside_sha256="2"*64, frozen_validation_sha256="3"*64,
                    array_sha256={key: "4"*64 for key in b.LAYOUT})
        with patch.object(p, "prepared_contract"), patch.object(p, "verify_prepared_files"), \
             patch.object(b, "validate_records"), patch.object(b, "validate_arrays"), \
             patch.object(r, "read_json", side_effect=[meta, [], meta["identity"]["native"]["validation"]]), \
             patch.object(r, "sha256", return_value="5"*64):
            case = p.bind_prepared(self.c, self.root, 42)["case"]
        self.assertEqual(case["n"], 17)
        self.assertNotIn("archived_validation", case)
        self.assertNotIn("archived_checkpoint", case)
        self.assertEqual(case["upstream"], meta["identity"]["native"])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(); r.new_output(args.output)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationContracts))
    ok = result.wasSuccessful() and not torch.cuda.is_initialized()
    report = {"status": "PASS" if ok else "FAIL", "tests": result.testsRun,
        "source_contents_sha256": r.sha256(r.PACKAGE/"contents.json"),
        "gpu_initialized": torch.cuda.is_initialized(), "research_pixels_read": False,
        "research_training_started": False, "test_pixels_read": False}
    r.write_json(args.output, report)
    print(json.dumps(report))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
