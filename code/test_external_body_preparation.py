"""Synthetic CPU contracts for upstream-to-body preparation and handoff."""
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
import train_external as t
import train_external_body as b
import prepare_external_bodies as p


class PreparationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = t.load_catalog()
        cls.bodies = b.load_catalog()
        cls.case = cls.bodies["cases"]["original_bentheimer_seed42"]

    def folder(self):
        return Path(tempfile.mkdtemp(prefix="case_", dir=self.fixtures))

    def selection(self):
        entries = self.catalog["entries"]["bentheimer_val"]["records"]
        history = [{"epoch": epoch, "validation": {"metrics": {
            "canonical512": {"eightclass": {"mIoU": .7 if epoch in (8, 19) else .5}}}}}
            for epoch in range(21)]
        selected = {"selected_epoch": 8, "split": "val", "label_space": "fine8",
            "external_test_pixels_read": False, "saved_pixel_replay_exact": True,
            "frozen_sha256": "synthetic-frozen", "images": 90,
            "metrics": history[8]["validation"]["metrics"], "predictions_directory": "selected_predictions",
            "saved_predictions": [{"index": i, "label_id": x["label_id"], "file": f"{i:04d}.npz", "sha256": "synthetic"}
                                  for i, x in enumerate(entries)]}
        meta = {"status": "completed", "dataset": "bentheimer", "arm": "native_detail", "last_epoch": 20,
                "history": history, "best_epoch": 8, "frozen_sha256": "synthetic-frozen"}
        return meta, selected, entries

    def upstream_fixture(self):
        root = self.folder()
        folder = root / "outputs/bentheimer_seed42/native_detail"
        (folder / "epochs").mkdir(parents=True)
        checkpoint = folder / "epochs/epoch_008.pth"
        checkpoint.write_bytes(b"SYNTHETIC-CHECKPOINT-NOT-A-RESEARCH-MODEL")
        meta, selected, _ = self.selection()
        selected["checkpoint"] = {"epoch": 8, "file": checkpoint.name, "sha256": r.sha256(checkpoint)}
        identity = t.run_identity(self.catalog, 42, "bentheimer")
        r.write_json(folder / "selected_validation.json", selected)
        meta.update(identity=identity, best_checkpoint=selected["checkpoint"],
                    selected_sha256=r.sha256(folder / "selected_validation.json"))
        r.write_json(folder / "run_manifest.json", meta)
        r.write_json(root / "portable_run.json", {"identity": identity, "status": "completed", "test_pixels_read": False})
        return root, folder

    def prepared_fixture(self):
        root = self.folder()
        folder = root / "cache/bentheimer"
        folder.mkdir(parents=True)
        identity = {"entry_point": "portable-external-body-preparation-v1", "smoke": False,
            "template_case": "original_bentheimer_seed42", "dataset": "bentheimer", "seed": 42,
            "config": self.case["config"], "descriptor_support": self.case["descriptor_support"],
            "selected_upstream": {"bentheimer": {"sha256": "SYNTHETIC-UPSTREAM"}},
            "package_contents_sha256": r.sha256(r.PACKAGE / "contents.json")}
        records = []
        for split in ("train", "val"):
            for row, item in enumerate(self.catalog["entries"]["bentheimer_" + split]["records"]):
                start = len(records)
                records.append({"split": split, "row": row, "label_id": item["label_id"],
                    "file": f"{split}_{row:04d}.npz", "bodies": 1, "start": start, "end": start + 1})
        for key, (dtype, shape) in b.LAYOUT.items():
            np.save(folder / (key + ".npy"), np.zeros((720, *shape), dtype))
        r.write_json(folder / "records.json", records)
        r.write_json(folder / "matrices.json", {"synthetic": True})
        meta = {"status": "completed", "smoke": False, "test_pixels_read": False,
            "identity": identity, "n": 720, "images_by_split": {"train": 630, "val": 90},
            "bodies_by_split": {"train": 630, "val": 90}, "dense_sparse_exact": True,
            "selected_validation_pixel_replay_exact": True,
            "array_sha256": {k: r.sha256(folder / (k + ".npy")) for k in b.LAYOUT},
            "records_sha256": r.sha256(folder / "records.json"), "matrices_sha256": r.sha256(folder / "matrices.json")}
        r.write_json(folder / "manifest.json", meta)
        p.export_case(self.case, "original_bentheimer_seed42", folder, root, identity, False)
        return root

    def change_case(self, root, edit):
        value = r.read_json(root / "body_case.json")
        edit(value)
        t.atomic_writer(root)(root / "body_case.json", value)

    def test_first_maximum_and_epoch_zero(self):
        meta, selected, entries = self.selection()
        self.assertEqual(p.selection_contract(meta, selected, self.case, entries), 8)
        meta["history"][0]["validation"]["metrics"] = copy.deepcopy(selected["metrics"])
        meta["best_epoch"] = selected["selected_epoch"] = 0
        self.assertEqual(p.selection_contract(meta, selected, self.case, entries), 0)

    def test_later_tie_rejected(self):
        meta, selected, entries = self.selection()
        meta["best_epoch"] = selected["selected_epoch"] = 19
        with self.assertRaises(ValueError):
            p.selection_contract(meta, selected, self.case, entries)

    def test_incomplete_history_rejected(self):
        meta, selected, entries = self.selection()
        meta["history"].pop(3)
        with self.assertRaises(ValueError):
            p.selection_contract(meta, selected, self.case, entries)

    def test_nonfinite_score_rejected(self):
        meta, selected, entries = self.selection()
        meta["history"][4]["validation"]["metrics"]["canonical512"]["eightclass"]["mIoU"] = float("nan")
        with self.assertRaises(ValueError):
            p.selection_contract(meta, selected, self.case, entries)

    def test_test_split_rejected(self):
        meta, selected, entries = self.selection()
        selected["split"] = "test"
        with self.assertRaises(ValueError):
            p.selection_contract(meta, selected, self.case, entries)

    def test_validation_order_rejected(self):
        meta, selected, entries = self.selection()
        selected["saved_predictions"].reverse()
        with self.assertRaises(ValueError):
            p.selection_contract(meta, selected, self.case, entries)

    def test_new_portable_upstream_resolves(self):
        root, _ = self.upstream_fixture()
        result = p.selected_upstream(self.case, self.catalog, root, root)
        self.assertEqual(result["kind"], "new-portable-selected")
        self.assertEqual(result["epoch"], 8)

    def test_wrong_seed_rejected(self):
        root, _ = self.upstream_fixture()
        value = r.read_json(root / "portable_run.json")
        value["identity"]["seed"] = 2024
        t.atomic_writer(root)(root / "portable_run.json", value)
        with self.assertRaises(ValueError):
            p.selected_upstream(self.case, self.catalog, root, root)

    def test_incomplete_upstream_rejected(self):
        root, _ = self.upstream_fixture()
        value = r.read_json(root / "portable_run.json")
        value["status"] = "running"
        t.atomic_writer(root)(root / "portable_run.json", value)
        with self.assertRaises(ValueError):
            p.selected_upstream(self.case, self.catalog, root, root)

    def test_changed_checkpoint_rejected(self):
        root, folder = self.upstream_fixture()
        (folder / "epochs/epoch_008.pth").write_bytes(b"CHANGED-SYNTHETIC-CHECKPOINT")
        with self.assertRaises(ValueError):
            p.selected_upstream(self.case, self.catalog, root, root)

    def test_changed_selection_rejected(self):
        root, folder = self.upstream_fixture()
        (folder / "selected_validation.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            p.selected_upstream(self.case, self.catalog, root, root)

    def test_output_overlap_rejected(self):
        root = self.folder()
        for output in (root, root.parent, root / "child"):
            with self.assertRaises(ValueError):
                p.independent_output(output, root)
        self.assertEqual(p.independent_output(root.parent / "independent", root), root.parent / "independent")

    def test_prepared_case_handoff(self):
        root = self.prepared_fixture()
        case = b.prepared_case(root, self.bodies)
        self.assertEqual(case["n"], 720)
        self.assertNotIn("archived_head", case)
        identity = b.run_identity(case, "prepared/original_bentheimer_seed42")
        self.assertIn("prepared_case_sha256", identity)

    def test_smoke_handoff_rejected(self):
        root = self.prepared_fixture()
        self.change_case(root, lambda value: value.update(trainable=False))
        with self.assertRaises(ValueError):
            b.prepared_case(root, self.bodies)

    def test_different_protocol_rejected(self):
        root = self.prepared_fixture()
        self.change_case(root, lambda value: value["case"]["config"].update(epochs=13))
        with self.assertRaises(ValueError):
            b.prepared_case(root, self.bodies)

    def test_inherited_result_rejected(self):
        root = self.prepared_fixture()
        self.change_case(root, lambda value: value["case"].update(archived_head={}))
        with self.assertRaises(ValueError):
            b.prepared_case(root, self.bodies)

    def test_stale_descriptors_rejected(self):
        root = self.prepared_fixture()
        self.change_case(root, lambda value: value["case"].update(selected_upstream={"sha256": "OTHER-UPSTREAM"}))
        with self.assertRaises(ValueError):
            b.prepared_case(root, self.bodies)

    def test_partial_coverage_rejected(self):
        root = self.prepared_fixture()
        self.change_case(root, lambda value: value["case"]["images_by_split"].update(train=2))
        with self.assertRaises(ValueError):
            b.prepared_case(root, self.bodies)

    def test_inputs_are_not_augmented_or_test_loaded(self):
        case = copy.deepcopy(self.case)
        initial = {"tokenizer": "synthetic", "global_checkpoint": "synthetic", "native_checkpoint": "synthetic"}
        root = self.folder()
        for support in ("probability_argmax", "logit_argmax"):
            case["descriptor_support"] = support
            engine = p.build_engine(case, self.catalog, root, root, root, initial, {"folder": str(root)})
            with self.assertRaises(ValueError):
                engine.c.CachedFine("bentheimer", "test")
            self.assertIn("describe", engine.c.__dict__)

    def test_stop_marker_is_retained(self):
        root = self.folder()
        (root / "STOP.txt").write_text("synthetic pause", encoding="utf-8")
        with self.assertRaises(InterruptedError):
            t.stop_check(root)
        self.assertTrue((root / "STOP.txt").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = r.new_output(args.output)
    with tempfile.TemporaryDirectory(prefix="pm_body_preparation_") as folder:
        PreparationContracts.fixtures = Path(folder)
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationContracts))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "gpu_initialized": torch.cuda.is_initialized(), "research_pixels_read": False,
        "research_training_started": False, "test_pixels_read": False,
        "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json")}
    r.write_json(output, report)
    print(json.dumps(report))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
