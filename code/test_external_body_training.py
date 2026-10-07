"""Synthetic CPU contracts for the portable body-head adaptation interface."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import reproduce as r
import train_external as t
import train_external_body as b


class BodyTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.catalog = b.load_catalog()
        cls.case = cls.catalog["cases"]["original_bentheimer_seed42"]
        cls.c, cls.s = b.mechanics(cls.case)

    def folder(self):
        return Path(tempfile.mkdtemp(prefix="body_", dir=self.fixtures))

    def test_nine_cases_and_scope(self):
        self.assertEqual(len(self.catalog["cases"]), 9)
        self.assertFalse(self.catalog["test_loader"])
        self.assertTrue(self.catalog["upstream_frozen"])
        for case in self.catalog["cases"].values():
            self.assertEqual(case["config"]["epochs"], 12)
            self.assertEqual(case["config"]["batch_size"], 512)
            self.assertEqual(case["config"]["precision"], "FP32")
            self.assertEqual(case["bodies_by_split"]["train"] + case["bodies_by_split"]["val"], case["n"])

    def test_attempts_are_not_merged(self):
        for seed in (42, 2024, 2025):
            a = self.catalog["cases"][f"original_wf1_seed{seed}"]
            z = self.catalog["cases"][f"whole_repeat_wf1_seed{seed}"]
            self.assertNotEqual(a["cache_folder"], z["cache_folder"])
            self.assertNotEqual(a["archived_head"], z["archived_head"])
            self.assertEqual(a["initial_head"], z["initial_head"])
        with self.assertRaises(ValueError):
            b.choose(self.catalog, "best_available")

    def test_support_conventions_are_explicit(self):
        for key, case in self.catalog["cases"].items():
            expected = "probability_argmax" if key in ("original_bentheimer_seed42", "original_wf1_seed42") else "logit_argmax"
            self.assertEqual(case["descriptor_support"], expected)

    def test_first_validation_maximum(self):
        history = lambda values: [{"validation": {"canonical512": {"mIoU": x}}} for x in values]
        self.assertEqual(self.c.selected_epoch(history([.7, .6, .7])), 0)
        self.assertEqual(self.c.selected_epoch(history([.6, .7, .7])), 1)
        with self.assertRaises(ValueError):
            self.c.selected_epoch([])

    def test_seeded_order_and_stream(self):
        for seed in (42, 2024, 2025):
            case = self.catalog["cases"][f"original_wf1_seed{seed}"]
            c, _ = b.mechanics(case)
            for epoch in (1, 7, 12):
                np.testing.assert_array_equal(c.order_for(np.arange(19), epoch), np.random.default_rng(seed * 1000 + epoch).permutation(19))
                self.assertEqual(c.stream(epoch, "cpu").initial_seed(), seed * 10000 + epoch)

    def test_augmentation_preserves_other_features(self):
        points = torch.arange(3 * 48 * 31, dtype=torch.float32).reshape(3, 48, 31)
        before = points.clone()
        output = self.c.augment(points, self.c.stream(2, "cpu"))
        torch.testing.assert_close(points, before, rtol=0, atol=0)
        torch.testing.assert_close(output[:, :, 4:], before[:, :, 4:], rtol=0, atol=0)
        for first, second in ((0, 1), (2, 3)):
            torch.testing.assert_close(output[:, :, first] ** 2 + output[:, :, second] ** 2,
                                       before[:, :, first] ** 2 + before[:, :, second] ** 2, rtol=0, atol=0)

    def test_count_loss_and_gradient(self):
        logits = torch.linspace(-1, 1, 20).reshape(4, 5).requires_grad_()
        counts = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        weights = torch.tensor([.5, 1., 2., 3., 1.5])
        value = self.c.body_loss(logits, counts, weights)
        target, prob = counts[:, 3:], logits.softmax(-1)
        ce = -(target * weights * logits.log_softmax(-1)).sum() / (target * weights).sum()
        dice = 1 - (2 * (prob * target).sum(0) + 1e-6) / ((prob * counts.sum(1, keepdim=True)).sum(0) + target.sum(0) + 1e-6)
        expected = .5 * ce + .5 * (dice * weights).sum() / weights.sum()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weights_derived_from_train_counts(self):
        for case in self.catalog["cases"].values():
            c, _ = b.mechanics(case)
            np.testing.assert_array_equal(c.weights(case["dataset"]), np.asarray(case["training_weights"], np.float32)[3:])
            with self.assertRaises(ValueError):
                c.weights("test")

    def test_sparse_recount_and_support(self):
        counts = np.arange(24, dtype=np.uint32).reshape(3, 8)
        outside = np.zeros((8, 8), np.int64)
        outside[1, 1] = 100
        assigned = np.array([1, 4, 1])
        actual = self.c.aggregate(counts, assigned, outside)
        np.testing.assert_array_equal(actual, self.s.recount(assigned, counts, outside))
        self.s.assert_fixed_support({"native": actual}, {"native": actual})
        bad = actual.copy()
        bad[1, 1] += 1
        with self.assertRaises(AssertionError):
            self.s.assert_fixed_support({"native": bad}, {"native": actual})

    def test_reader_is_read_only_and_test_rejected(self):
        folder = self.folder()
        cache = folder / "cache"
        cache.mkdir()
        n = 6
        r.write_json(cache / "manifest.json", {"status": "completed"})
        records = [{"split": "train", "start": 0, "end": 4}, {"split": "val", "start": 4, "end": 6}]
        r.write_json(cache / "records.json", records)
        r.write_json(cache / "matrices.json", {kind: {split: {grid: np.zeros((8, 8), np.int64).tolist() for grid in b.GRIDS}
                       for split in ("train", "val")} for kind in ("outside", "frozen")})
        for key, (dtype, shape) in b.LAYOUT.items():
            np.save(cache / (key + ".npy"), np.zeros((n, *shape), dtype=dtype))
        engine = b.build_engine(self.case, folder, folder)
        data = engine.data.Bodies(cache, "train")
        self.assertTrue(all(not array.flags.writeable for array in data.arrays.values()))
        self.assertEqual(data.indices.tolist(), [0, 1, 2, 3])
        points, _, _ = data.inputs(data.indices, "cpu")
        points.fill_(7)
        self.assertEqual(data.arrays["points"].sum(), 0)
        with self.assertRaises(ValueError):
            engine.data.Bodies(cache, "test")
        with self.assertRaises(ValueError):
            engine.c.CachedFine(self.case["dataset"], "test")

    def test_path_protection(self):
        root = self.folder()
        for path in (root, root.parent, root / "实验" / "new", r.PACKAGE / "new"):
            with self.assertRaises(ValueError):
                b.output_path(path, root, self.catalog)

    def test_binary_writer_does_not_escape(self):
        folder = self.folder()
        save = b.binary_writer(folder, lambda f, value: np.savez_compressed(f, **value))
        with self.assertRaises(ValueError):
            save(folder.parent / "outside.npz", {"x": np.arange(2)})
        path = folder / "inside.npz"
        save(path, {"x": np.arange(2)})
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["x"], np.arange(2))

    def test_stop_markers_retained(self):
        folder = self.folder()
        for name in ("STOP.txt", "STOP_AFTER_EPOCH.txt"):
            marker = folder / name
            marker.touch()
            with self.assertRaises(InterruptedError):
                t.stop_check(folder)
            self.assertTrue(marker.exists())

    def test_lock_excludes_second_writer(self):
        folder = self.folder()
        with t.output_lock(folder):
            with self.assertRaises(OSError):
                with t.output_lock(folder):
                    self.fail("Duplicate lock")

    def test_resume_identity_guard(self):
        folder = self.folder()
        with self.assertRaises(FileExistsError):
            t.start_output(folder, {"case": "a"}, False)
        r.write_json(folder / "portable_run.json", {"identity": {"case": "a"}})
        t.start_output(folder, {"case": "a"}, True)
        with self.assertRaises(ValueError):
            t.start_output(folder, {"case": "b"}, True)

    def test_run_requires_explicit_checkpoint_trust(self):
        with self.assertRaises(ValueError):
            b.run(types.SimpleNamespace(trusted_checkpoints=False))

    def state_engine(self, folder):
        engine = b.build_engine(self.case, folder, folder)
        c, p = engine.c, engine.data
        cache = folder / "cache"
        cache.mkdir()
        r.write_json(cache / "manifest.json", {"synthetic": True})
        p.cache_folder = lambda dataset: cache
        p.Bodies = lambda directory, split: types.SimpleNamespace(indices=np.arange(4))
        def model():
            layer = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(layer.weight)
            layer.cuda = lambda: layer
            return layer
        c.source_head = lambda arm: model()
        calls = []
        def train(layer, data, optimizer, dataset, arm, epoch):
            calls.append(epoch)
            optimizer.zero_grad()
            loss = (layer(torch.ones(2, 1)) - 1).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.)
            optimizer.step()
            return {"loss": float(loss.detach()), "updates": 1}
        def evaluate(layer, data, save_to=None):
            if save_to is not None:
                c.save_npz(save_to, {"indices": np.arange(4), "assigned": np.zeros(4, np.uint8)})
            return {grid: {"mIoU": float(layer.weight.detach()[0, 0])} for grid in b.GRIDS}
        p.evaluate = evaluate
        namespace = engine.runner.train_arm.__globals__
        namespace["train_epoch"] = train
        namespace["verify_assignments"] = lambda *args: {"synthetic": True}
        return engine, calls

    def test_resume_recovers_saved_unregistered_epoch(self):
        full, resumed = self.folder(), self.folder()
        engine, calls = self.state_engine(full)
        identity = {"synthetic": True}
        chosen = engine.runner.train_arm("bentheimer", "separate", identity, False)
        self.assertEqual(calls, list(range(1, 13)))
        self.assertEqual(chosen["selected_epoch"], 12)
        other, resumed_calls = self.state_engine(resumed)
        writer = other.c.write
        failed = False
        def interrupted(path, value):
            nonlocal failed
            if Path(path).name == "history.json" and len(value) == 4 and not failed:
                failed = True
                raise InterruptedError("synthetic crash after atomic epoch save")
            return writer(path, value)
        other.c.write = interrupted
        with self.assertRaises(InterruptedError):
            other.runner.train_arm("bentheimer", "separate", identity, False)
        self.assertEqual(resumed_calls, [1, 2, 3])
        other.c.write = writer
        result = other.runner.train_arm("bentheimer", "separate", identity, True)
        self.assertEqual(resumed_calls, list(range(1, 13)))
        self.assertEqual(result["selected_epoch"], chosen["selected_epoch"])
        a = torch.load(full / "outputs/bentheimer/separate/epochs/epoch_012.pth", weights_only=False)
        z = torch.load(resumed / "outputs/bentheimer/separate/epochs/epoch_012.pth", weights_only=False)
        torch.testing.assert_close(a["model"], z["model"], rtol=0, atol=0)
        torch.testing.assert_close(a["optimizer"], z["optimizer"], rtol=0, atol=0)
        self.assertEqual(a["schedule"], z["schedule"])
        with self.assertRaises(AssertionError):
            other.runner.completed_checkpoint(resumed / "outputs/bentheimer/separate/epochs/epoch_012.pth", {"wrong": True}, "wrong", "bentheimer", "separate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = r.new_output(args.output)
    with tempfile.TemporaryDirectory(prefix="pm_body_") as folder:
        BodyTrainingContracts.fixtures = Path(folder)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BodyTrainingContracts)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
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
