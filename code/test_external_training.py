"""CPU engineering checks for the portable external training interface."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import types
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import reproduce as r
import train_external as t


class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = t.load_catalog()
        cls.c, cls.b = t.mechanics(cls.catalog, 42)

    def folder(self):
        methods = sorted(name for name in dir(self) if name.startswith("test_"))
        path = self.fixtures / f"t{methods.index(self.id().split('.')[-1]):02d}"
        path.mkdir()
        return path

    def test_protocol_config(self):
        c = t.config_for(self.catalog, 42)
        self.assertEqual(c["epochs"], 20)
        self.assertEqual(c["accumulate"], 4)
        self.assertEqual(c["microbatch"], 1)
        self.assertEqual(c["primary_rule"], "eightclass")
        self.assertEqual(c["selection"], "first maximum validation mIoU, including epoch0")
        self.assertTrue(c["freeze_existing_text_encoder"] and c["freeze_bn_running_statistics"])
        with self.assertRaises(ValueError):
            t.config_for(self.catalog, 13)

    def test_schedule_preserved(self):
        rates = [self.c.lr_at(i) for i in range(1, 21)]
        np.testing.assert_array_equal(rates[:12], [5e-6 + .5 * (5e-5 - 5e-6) * (1 + np.cos(np.pi * i / 11)) for i in range(12)])
        self.assertEqual(rates[12:], [5e-6] * 8)
        for epoch in (0, 21):
            with self.assertRaises(ValueError):
                self.c.lr_at(epoch)

    def test_paired_augmentation_plans(self):
        for seed in (42, 2024, 2025):
            c, _ = t.mechanics(self.catalog, seed)
            for epoch in (1, 12, 13, 20):
                rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 129]))
                expected = list(zip(rng.permutation(17).tolist(), rng.integers(0, 16, size=17).tolist()))
                self.assertEqual(c.epoch_plan(17, epoch), expected)
                self.assertEqual(sorted(i for i, _ in expected), list(range(17)))

    def test_d4_alignment(self):
        a = np.arange(35).reshape(5, 7)
        for code in range(16):
            expected = a[::-1] if code & 1 else a
            expected = expected[:, ::-1] if code & 2 else expected
            expected = np.rot90(expected, code >> 2)
            np.testing.assert_array_equal(self.c.d4(a, code), expected)
            np.testing.assert_array_equal(self.c.d4(a + 100, code), expected + 100)

    def test_partial_accumulation(self):
        self.assertEqual([self.c.window_size(i, 6) for i in range(6)], [4, 4, 4, 4, 2, 2])

    def test_weights_are_train_only(self):
        for key, entry in self.catalog["entries"].items():
            if key.endswith("_val"):
                self.assertIsNone(entry["class_weights"])
            else:
                np.testing.assert_array_equal(self.b.fine_weights(entry["class_pixel_counts"]),
                                             np.asarray(entry["class_weights"], np.float32))

    def test_loss_and_backward(self):
        torch.manual_seed(41006)
        logits = torch.randn(1, 8, 5, 7, requires_grad=True)
        target = torch.arange(35).reshape(1, 5, 7) % 8
        weights = torch.linspace(.3, 3., 8)
        value = self.b.fine_loss(logits, target, weights)
        logp = torch.nn.functional.log_softmax(logits.float(), 1)
        probs = logp.exp()
        hot = torch.nn.functional.one_hot(target, 8).permute(0, 3, 1, 2)
        dice = (2 * (probs * hot).sum((0, 2, 3)) + 1e-5) / ((probs + hot).sum((0, 2, 3)) + 1e-5)
        expected = .5 * torch.nn.functional.nll_loss(logp, target, weight=weights) + .5 * (1 - (dice * weights).sum() / (weights.sum() + 1e-8))
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            self.b.fine_loss(logits, target.float(), weights)

    def test_first_validation_maximum_including_zero(self):
        engine = t.build_engine(self.catalog, 42, {}, self.fixtures, self.fixtures)
        def history(values):
            return [{"epoch": i, "validation": {"metrics": {"canonical512": {"eightclass": {"mIoU": value}}}}}
                    for i, value in enumerate(values)]
        self.assertEqual(engine.original.selection(history([.6, .6, .5])), 0)
        self.assertEqual(engine.original.selection(history([.6, .7, .7])), 1)

    def test_cache_is_read_only_and_no_test_route(self):
        folder = self.folder()
        images = np.arange(70, dtype=np.float32).reshape(2, 5, 7) / 70
        low = np.zeros((2, 512, 512), np.float32)
        labels = (np.arange(70).reshape(2, 5, 7) % 8).astype(np.uint8)
        resources = {}
        for name, value in (("image", images), ("low", low), ("mask", labels)):
            path = folder / (name + ".npy")
            np.save(path, value)
            resources[name] = {"path": path.name, "sha256": r.sha256(path)}
        catalog = {"entries": {"wf1_train": {"resources": resources, "shape": [2, 5, 7], "count": 2}},
                   "descriptions": {"wf1": "synthetic"}}
        data = t.Cached("wf1", "train", catalog, folder, self.c.d4)
        self.assertFalse(data.image.flags.writeable)
        np.testing.assert_array_equal(data.get(1, 4)[2], np.rot90(labels[1]))
        data.get(0)[0].fill(99)
        np.testing.assert_array_equal(data.image, images)
        with self.assertRaises(ValueError):
            t.Cached("wf1", "test", catalog, folder, self.c.d4)

    def test_output_protection(self):
        root = self.folder() / "resources"
        root.mkdir()
        for candidate in (root, root.parent, r.PACKAGE / "new_run", root / "实验" / "new_run"):
            with self.assertRaises(ValueError):
                t.destination(candidate, root, self.catalog)
        # The suite may itself be run from inside a fresh extraction. Its
        # valid output example still has to sit outside the package tree.
        safe = Path(tempfile.gettempdir()) / ("pm_" + hashlib.sha256(str(self.fixtures).encode()).hexdigest()[:8])
        self.assertEqual(t.destination(safe, root, self.catalog), safe.resolve())

    def test_existing_and_foreign_resume_rejected(self):
        folder = self.folder()
        with self.assertRaises(FileExistsError):
            t.start_output(folder, {"run": 1}, False)
        t.atomic_writer(folder)(folder / "portable_run.json", {"identity": {"run": 1}})
        t.start_output(folder, {"run": 1}, True)
        with self.assertRaises(ValueError):
            t.start_output(folder, {"run": 2}, True)
        with self.assertRaises(FileNotFoundError):
            t.start_output(folder / "missing", {}, True)

    def test_atomic_writer_cannot_escape(self):
        folder = self.folder()
        with self.assertRaises(ValueError):
            t.atomic_writer(folder)(folder.parent / "escaped.json", {})
        self.assertFalse((folder.parent / "escaped.json").exists())

    def test_stop_marker_is_preserved(self):
        folder = self.folder()
        marker = folder / "STOP.txt"
        marker.touch()
        with self.assertRaises(InterruptedError):
            t.stop_check(folder)
        self.assertTrue(marker.exists())

    def test_lock_excludes_second_writer(self):
        folder = self.folder()
        with t.output_lock(folder):
            with self.assertRaises(OSError):
                with t.output_lock(folder):
                    self.fail("Two concurrent writers obtained the lock")
        with t.output_lock(folder):
            pass

    def test_explicit_checkpoint_trust_required(self):
        with self.assertRaises(ValueError):
            t.run(types.SimpleNamespace(trusted_checkpoints=False))
        self.assertFalse(torch.cuda.is_initialized())

    def test_epoch_registration_resume_and_orphan_recovery(self):
        # Keep the original state-machine definitions. Substitute a tiny CPU
        # stand-in for expensive network/data operations, not a research run.
        folder = self.folder()
        engine = t.build_engine(self.catalog, 42, {}, self.fixtures, folder)
        ns = engine.original.run_epoch.__globals__
        frozen_from_state = engine.adapter.frozen_from_state
        class TinyModel:
            def __init__(self, arm):
                self.arm = arm
                self.value = torch.nn.Parameter(torch.tensor([0.]))
                self.frozen = torch.tensor([3.])
            def cuda(self): return self
            def to(self, **kwargs): return self
            def cpu(self): return self
            def trainable_parameters(self): return [self.value]
            def state_dict(self):
                return {"value": self.value.detach(), "global_model.text_encoder.weight": self.frozen}
            def frozen_state(self): return frozen_from_state(self.state_dict())
            def load_state_dict(self, state, strict=True):
                with torch.no_grad(): self.value.copy_(state["value"])
                self.frozen = state["global_model.text_encoder.weight"].clone()
        def train(model, data, optimizer, epoch, progress_path=None):
            with torch.no_grad(): model.value.fill_(epoch)
            return {"images": 630, "updates": 158,
                    "plan_sha256": hashlib.sha256(json.dumps(engine.c.epoch_plan(630, epoch)).encode()).hexdigest()}
        def evaluate(model, data, save_to=None):
            value = .9 if int(model.value.item()) in (1, 2) else .5
            metrics = {g: {"eightclass": {"mIoU": value, "confusion_matrix": np.eye(8, dtype=int).tolist()}}
                       for g in engine.c.GRIDS}
            if save_to is not None: save_to.mkdir()
            return {"metrics": metrics}
        ns.update(AdaptationModel=TinyModel, Cached=lambda *a: None,
                  train_epoch=train, evaluate=evaluate, close_model=lambda *a: None)
        identity = {"synthetic_only": True}
        for epoch in (0, 1):
            engine.original.run_epoch("bentheimer", "global", epoch, identity, False)
        arm = folder / "outputs/bentheimer_seed42/global"
        old = r.read_json(arm / "run_manifest.json")
        before = r.sha256(arm / "epochs/epoch_001.pth")
        engine.original.run_epoch("bentheimer", "global", 1, identity, True)
        self.assertEqual(before, r.sha256(arm / "epochs/epoch_001.pth"))
        # A checkpoint completed just before process failure, but not registered.
        with patch.dict(ns, {"register_checkpoint": lambda *a: None}):
            engine.original.run_epoch("bentheimer", "global", 2, identity, True)
        self.assertEqual(r.read_json(arm / "run_manifest.json"), old)
        engine.original.run_epoch("bentheimer", "global", 2, identity, True)
        resumed = r.read_json(arm / "run_manifest.json")
        self.assertEqual(resumed["last_epoch"], 2)
        self.assertEqual(resumed["best_epoch"], 1)
        # Set a two-epoch test horizon only in this isolated CPU namespace.
        engine.c.CONFIG["epochs"] = 2
        chosen = engine.original.finish_arm("bentheimer", "global", identity)
        self.assertEqual(chosen["selected_epoch"], 1)
        self.assertEqual(r.read_json(arm / "run_manifest.json")["status"], "completed")
        self.assertEqual(engine.original.finish_arm("bentheimer", "global", identity), chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = r.new_output(args.output)
    fixtures = out.with_name(out.stem + "_fixtures")
    fixtures.mkdir(parents=True, exist_ok=False)
    TrainingContracts.fixtures = fixtures
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrainingContracts))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
              "research_pixels_read": False, "research_training_started": False,
              "synthetic_cpu_backward": True, "gpu_initialized": torch.cuda.is_initialized()}
    r.write_json(out, report)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
