"""CPU-only native training and handoff tests, with synthetic pixels only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import reproduce as r
import main_native as n
import train_main_global as g
import train_main_native as m
import prepare_main_native as p


class NativeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c, cls.gc = n.load_catalog(), g.load_catalog()
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pm-native-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, seed=42):
        files = {}
        generator = np.random.default_rng(710)
        for key, side in (("image", 1024), ("mask", 1024), ("low", 512)):
            a = generator.integers(0, 8 if key == "mask" else 256, (4, side, side), dtype=np.uint8)
            path = self.root / (key + ".dat")
            a.tofile(path)
            files[key] = str(path)
        path = self.root / "coarse.dat"
        torch.randn(4, 8, 512, 512).bfloat16().view(torch.uint16).numpy().tofile(path)
        files["coarse"] = str(path)
        rows = [{"split": "train" if i < 3 else "val", "shared_index": i} for i in range(4)]
        return {"seed": seed, "records": rows, "files": files, "n": 4, "shared_n": 4}

    def selection(self):
        history = [{"epoch": i, "val": {"mIoU": .7}} for i in range(1, 12)]
        ident = {"seed": 42, "config": self.gc["config"], "selection": self.gc["selection"],
            "main_catalog_sha256": r.sha256(r.PACKAGE / "main_training.json"),
            "source_contents_sha256": self.c["compatible_global_contents_sha256"][0], "fresh_run": True}
        meta = {"status": "completed", "identity": ident, "test_pixels_read": False,
            "completed_epochs": 11, "best_epoch": 1, "best_val_miou": .7, "stopped_by": "early_stopping"}
        selected = {"epoch": 1, "val_mIoU": .7, "selection": self.gc["selection"], "checkpoint": "best_model.pth"}
        return meta, selected, history

    def prepared(self):
        return {"status": "completed", "smoke": False, "n": 3040, "seed": 42,
            "test_pixels_read": False, "research_training_started": False,
            "records": self.c["records"], "batch_size": 4, "completed_records": 3040,
            "native_catalog_sha256": r.sha256(r.PACKAGE / "main_native_training.json"),
            "selected_global": {"kind": "archived-selected", "seed": 42,
                "checkpoint_sha256": self.c["cases"]["42"]["global_checkpoint"]["sha256"]}}

    def loop(self, scores=None, fail_epoch=None, stop_epoch=None, replay_fail=False):
        model = torch.nn.Conv2d(1, 8, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
        sampler = types.SimpleNamespace(epoch=0)
        scores = scores or [i / 100 for i in range(1, 13)]
        evaluations = []
        def update(*args):
            if sampler.epoch == fail_epoch:
                with torch.no_grad():
                    next(model.parameters()).add_(100)
                raise KeyboardInterrupt("Synthetic mid-epoch stop")
            opt.step()
            sched.step()
            return 1.
        def evaluate(*args):
            if len(evaluations) == 12:
                return {"native": {"mIoU": max(scores) + (.1 if replay_fail else 0)}}
            if sampler.epoch == stop_epoch:
                (self.root / "STOP_AFTER_EPOCH.txt").touch()
            value = {"native": {"mIoU": scores[sampler.epoch-1]}}
            evaluations.append(value)
            return value
        with patch.object(n, "train_epoch", update), patch.object(n.ops, "evaluate", evaluate):
            return m.run_epochs(model, opt, sched, [], [], sampler, torch.ones(8), self.root,
                                {"selection": self.c["selection"]}, (self.root,))

    def test_protocol_and_shared_weights(self):
        self.assertEqual(self.c["geometry"], {"native": 1024, "coarse": 512, "crop": 384, "halo": 32})
        for case in self.c["cases"].values():
            self.assertEqual(case["config"]["class_weights"], self.gc["seeds"]["42"]["weights"])
            self.assertEqual(case["config"]["epochs"], 12)

    def test_three_original_initializations(self):
        for case in self.c["cases"].values():
            model = n.build_model(case)
            self.assertEqual(model.full_initial_sha256, case["full_initial_sha256"])
            self.assertEqual(sum(x.numel() for x in model.parameters()), 118744)
            self.assertFalse(hasattr(model, "context_gate"))
            self.assertFalse(hasattr(model, "context_projection"))

    def test_context_not_read(self):
        model = n.build_model(self.c["cases"]["42"]).eval()
        image, coarse = torch.rand(1, 1, 32, 32), torch.rand(1, 8, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(image, coarse, None), model(image, coarse, torch.full((1,), float("nan")))))

    def test_halo_loss_matches_original_formula(self):
        logits = torch.randn(1, 8, 80, 80, requires_grad=True)
        target = torch.randint(8, (1, 80, 80))
        weights = torch.tensor(self.c["cases"]["42"]["config"]["class_weights"])
        loss = n.ops.loss_function(logits, target, weights)
        inside, y = logits[:, :, 32:-32, 32:-32], target[:, 32:-32, 32:-32]
        expected = .5 * torch.nn.functional.cross_entropy(inside, y, weight=weights) + .5 * g.base.WeightedDiceLoss(8, weights)(inside, y)
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad[:, :, :32]).item(), 0)
        self.assertEqual(torch.count_nonzero(logits.grad[:, :, :, :32]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[:, :, 32:-32, 32:-32]).item(), 0)

    def test_seeded_crop_and_augmentation(self):
        binding = self.fixture()
        for seed in (42, 2024, 2025):
            binding["seed"] = seed
            data = n.NativeDataset(binding, "train", True)
            for epoch in (1, 7, 12):
                a, b = data[(1, epoch)], data[(1, epoch)]
                rng = np.random.default_rng(seed + 1000003 * epoch + 97)
                self.assertEqual(a["origin"].tolist(), (4 * rng.integers(0, 161, 2)).tolist())
                self.assertEqual(a["aug"].tolist(), rng.integers(0, [2, 2, 4]).tolist())
                self.assertEqual(list(a["image"].shape), [384, 384])
                self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_validation_native_grid(self):
        data = n.NativeDataset(self.fixture(), "val", False)
        item = data[0]
        self.assertEqual(list(item["image"].shape), [1024, 1024])
        self.assertEqual(item["origin"].tolist(), [0, 0])
        self.assertEqual(item["aug"].tolist(), [0, 0, 0])

    def test_dataset_rejects_test_and_val_training(self):
        for split, training in (("test", False), ("val", True)):
            with self.assertRaises(ValueError):
                n.NativeDataset({"seed": 42, "records": []}, split, training)

    def test_pickle_memmap_readonly(self):
        data = n.NativeDataset(self.fixture(), "train", True)
        a = data[(0, 1)]
        restored = pickle.loads(pickle.dumps(data))
        self.assertIsNone(restored.arrays)
        self.assertTrue(torch.equal(a["image"], restored[(0, 1)]["image"]))
        self.assertTrue(all(x.mode == "r" for x in restored.arrays.values()))

    def test_windows_spawn_explicit_paths(self):
        data = n.NativeDataset(self.fixture(), "train", True)
        hashes = {k: r.sha256(p) for k, p in data.binding["files"].items()}
        a = list(DataLoader(data, batch_size=2, num_workers=0))
        b = list(DataLoader(data, batch_size=2, num_workers=2, multiprocessing_context="spawn"))
        self.assertTrue(all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x))
        self.assertEqual(hashes, {k: r.sha256(p) for k, p in data.binding["files"].items()})

    def test_sampler_permutation_and_epoch(self):
        for seed in (42, 2024, 2025):
            sampler = n.sampler_ops.SeedSampler(list(range(18)), seed=seed)
            sampler.epoch = 3
            self.assertEqual(list(sampler), [(int(x), 3) for x in torch.randperm(18, generator=torch.Generator().manual_seed(seed*1000+3))])

    def test_native_selection_first_tie(self):
        history = [{"epoch": i, "validation": {"native": {"mIoU": s}}} for i, s in enumerate((.6, .7, .7, .65), 1)]
        self.assertEqual(m.choose(history)["epoch"], 2)
        for changed in ([], [{"epoch": 0}], [{"epoch": 1, "validation": {"native": {"mIoU": float("nan")}}}]):
            with self.assertRaises(ValueError):
                m.choose(changed)

    def test_native_twelve_epochs_and_reload(self):
        state = self.loop([.7] * 12)
        self.assertEqual(state["completed_epochs"], 12)
        self.assertEqual(state["best_epoch"], 1)
        self.assertTrue(state["selected_replay_exact"])
        self.assertEqual(r.read_json(self.root / "history.json")[-1]["lr_after_epoch"], 0.)
        selected = r.read_json(self.root / "selected_validation.json")
        self.assertEqual(selected["sha256"], r.sha256(self.root / "best.pth"))

    def test_mid_epoch_does_not_replace_completed(self):
        with self.assertRaises(KeyboardInterrupt):
            self.loop(fail_epoch=2)
        saved = torch.load(self.root / "last_completed_epoch.pth", map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertLess(next(iter(saved["model"].values())).abs().max(), 2)
        self.assertEqual(r.read_json(self.root / "run_manifest.json")["completed_epochs"], 1)
        self.assertFalse((self.root / "selected_validation.json").exists())

    def test_stop_marker_retained(self):
        with self.assertRaises(InterruptedError):
            self.loop(stop_epoch=1)
        self.assertTrue((self.root / "STOP_AFTER_EPOCH.txt").exists())
        self.assertEqual(r.read_json(self.root / "run_manifest.json")["status"], "stopped")

    def test_reloaded_validation_must_match(self):
        with self.assertRaises(ValueError):
            self.loop(replay_fail=True)
        self.assertFalse((self.root / "selected_validation.json").exists())

    def test_global_selection_completed_rule(self):
        meta, selected, history = self.selection()
        self.assertEqual(p.selection_contract(meta, selected, history, self.c, self.gc, 42), 1)

    def test_global_selection_rejects_partial_and_late_tie(self):
        for mode in ("partial", "tie", "nan", "stale", "seed", "source"):
            meta, selected, history = self.selection()
            if mode == "partial":
                history.pop(); meta["completed_epochs"] = 10
            elif mode == "tie":
                selected["epoch"] = meta["best_epoch"] = 2
            elif mode == "nan":
                history[0]["val"]["mIoU"] = float("nan")
            elif mode == "stale":
                history.append({"epoch": 12, "val": {"mIoU": .8}}); meta["completed_epochs"] = 12
            elif mode == "seed":
                meta["identity"]["seed"] = 2024
            else:
                meta["identity"]["source_contents_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                p.selection_contract(meta, selected, history, self.c, self.gc, 42)

    def test_prepared_cache_accepts_full_matching_only(self):
        n.prepared_contract(self.prepared(), self.c, 42)
        for key, value in (("smoke", True), ("n", 8), ("status", "failed"), ("seed", 2024),
                           ("completed_records", 3036), ("test_pixels_read", True), ("batch_size", 8)):
            meta = self.prepared(); meta[key] = value
            with self.assertRaises(ValueError):
                n.prepared_contract(meta, self.c, 42)

    def test_prepared_teacher_identity(self):
        for key, value in (("seed", 2025), ("kind", "other"), ("checkpoint_sha256", "0" * 64)):
            meta = self.prepared(); meta["selected_global"][key] = value
            with self.assertRaises(ValueError):
                n.prepared_contract(meta, self.c, 42)

    def test_output_overwrite_and_overlap_rejected(self):
        for path in (self.root, r.PACKAGE, self.root / "实验/new"):
            with self.assertRaises((ValueError, FileExistsError)):
                n.output_path(path, self.root, self.c)
        with self.assertRaises(ValueError):
            n.output_path(self.root / "new/inside", self.root, self.c, self.root / "new")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r.new_output(args.output)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeContracts))
    passed = result.wasSuccessful() and not torch.cuda.is_initialized()
    report = {"status": "PASS" if passed else "FAIL", "tests": result.testsRun,
        "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "gpu_initialized": torch.cuda.is_initialized(), "research_pixels_read": False,
        "research_training_started": False, "test_pixels_read": False}
    r.write_json(args.output, report)
    if not passed:
        raise SystemExit(1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
