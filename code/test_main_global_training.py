"""CPU-only main training contracts; all generated pixels are synthetic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import reproduce as r
import train_main_global as m


class MainContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = m.load_catalog()
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pm-main-")
        self.root = Path(self.tmp.name)
        m.configure(self.catalog, 42)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, augment=False):
        shape = (4, 512, 512)
        image = (np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) % 256).astype(np.uint8)
        mask = (image % 8).astype(np.uint8)
        files = {}
        for key, pixels in (("image", image), ("mask", mask)):
            p = self.root / f"{key}.dat"
            pixels.tofile(p)
            files[key] = str(p)
        records = [{"split": "val", "cache_index": i, "text": "Oil rock water",
                    "sample": "Oil", "slice_id": i, "image_filename": f"{i}.tif"} for i in range(4)]
        tok = m.base.CharTokenizer(["Oil rock water"])
        return m.MainDataset(records, tok, {"files": files, "shape": shape}, augment), image, mask

    def engine(self):
        model = torch.nn.Conv2d(3, 8, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        tok = m.base.CharTokenizer(["synthetic"])
        return model, opt, sched, tok

    def loop(self, val_scores, fail_epoch=None, stop_epoch=None):
        model, opt, sched, tok = self.engine()
        epoch_count = 0
        def train(*args):
            nonlocal epoch_count
            epoch_count += 1
            if epoch_count == fail_epoch:
                with torch.no_grad():
                    next(model.parameters()).add_(50)
                raise KeyboardInterrupt("synthetic mid-epoch interruption")
            opt.step()
            sched.step()
            return {"loss": 1., "mIoU": .1}
        scores = iter(val_scores)
        def val(*args):
            if epoch_count == stop_epoch:
                (self.root / "STOP_TRAINING.txt").touch()
            return {"loss": 1., "mIoU": next(scores)}
        with patch.object(m.base, "train_one_epoch", train), patch.object(m.base, "evaluate_model", val):
            return m.run_epochs(model, opt, sched, None, tok, [], [], torch.ones(8),
                torch.device("cpu"), self.root, {"selection": self.catalog["selection"]}, (self.root,))

    def test_catalog_scope_and_order(self):
        self.assertIs(m.validate_catalog(self.catalog), self.catalog)
        for seed in (42, 2024, 2025):
            train = m.records_for(self.catalog, seed, "train")
            self.assertEqual(len(train), 2364)
            self.assertEqual([x["id"] for x in train], self.catalog["seeds"][str(seed)]["train_ids"])
            self.assertEqual(len(m.records_for(self.catalog, seed, "val")), 676)

    def test_reject_test_and_other_seed(self):
        with self.assertRaises(ValueError):
            m.records_for(self.catalog, 42, "test")
        with self.assertRaises(ValueError):
            m.records_for(self.catalog, 5, "train")
        with self.assertRaises(ValueError):
            m.MainDataset([{"split": "test"}], None, {})

    def test_reject_overlap_wrong_split_and_config(self):
        for mutation in (lambda c: c["records"][0].update(split="test"),
                         lambda c: c["records"][0].update(cache_index=c["records"][1]["cache_index"]),
                         lambda c: c["config"].update(num_epochs=20)):
            c = copy.deepcopy(self.catalog)
            mutation(c)
            with self.assertRaises(ValueError):
                m.validate_catalog(c)

    def test_tokenizer_train_only(self):
        tok = m.tokenizer_for(self.catalog)
        c = copy.deepcopy(self.catalog)
        for row in c["records"]:
            if row["split"] == "val":
                row["text"] = "\U0001f984"
        self.assertEqual(m.tokenizer_for(c).idx2token, tok.idx2token)
        self.assertEqual(tok.max_len, 192)

    def test_exact_weights_from_record_counts(self):
        by_id = {x["id"]: x for x in self.catalog["records"]}
        for info in self.catalog["seeds"].values():
            counts = sum((np.array(by_id[k]["counts"], dtype=np.float64) for k in info["weight_sample_ids"]), np.zeros(8))
            self.assertEqual(counts.tolist(), info["train_pixel_counts"])
            weights = 1 / np.sqrt(counts / (counts.sum() + 1e-8) + 1e-8)
            weights = np.clip(weights / weights.mean(), .3, 6).astype(np.float32)
            np.testing.assert_array_equal(weights, np.array(info["weights"], dtype=np.float32))

    def test_dataset_matches_original_item_and_augmentation(self):
        data, _, _ = self.fixture(True)
        original = m.base.MultimodalRemainingOilDataset(data.records, data.tokenizer, augment=True)
        original._images_mm = np.memmap(data.binding["files"]["image"], mode="r", dtype=np.uint8, shape=data.binding["shape"])
        original._masks_mm = np.memmap(data.binding["files"]["mask"], mode="r", dtype=np.uint8, shape=data.binding["shape"])
        for seed in range(8):
            random.seed(seed)
            a = data[seed % 4]
            random.seed(seed)
            b = original[seed % 4]
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]) if torch.is_tensor(a[key]) else a[key] == b[key])

    def test_pickle_closes_memmaps_and_uses_explicit_path(self):
        data, _, _ = self.fixture()
        first = data[0]
        restored = pickle.loads(pickle.dumps(data))
        self.assertIsNone(restored._images_mm)
        with patch.object(m.base, "open_cache_memmaps", side_effect=AssertionError("legacy routing used")):
            self.assertTrue(torch.equal(restored[0]["image"], first["image"]))
        self.assertEqual(restored._images_mm.mode, "r")

    def test_windows_spawn_worker_routing(self):
        data, _, _ = self.fixture()
        hashes = {k: r.sha256(p) for k, p in data.binding["files"].items()}
        expected = list(DataLoader(data, batch_size=2, num_workers=0))
        actual = list(DataLoader(data, batch_size=2, num_workers=2, multiprocessing_context="spawn"))
        for a, b in zip(expected, actual):
            for key in ("image", "mask", "input_ids", "attention_mask"):
                self.assertTrue(torch.equal(a[key], b[key]))
        self.assertEqual(hashes, {k: r.sha256(p) for k, p in data.binding["files"].items()})

    def test_selection_first_max_no_epoch_zero(self):
        value = (-1., -1, 0)
        for epoch, score in enumerate((.5, .6, .6, .55), 1):
            result = m.advance_selection(epoch, score, *value)
            value = result[:3]
        self.assertEqual(value, (.6, 2, 2))
        for epoch, score in ((0, .9), (61, .9), (1, float("nan"))):
            with self.assertRaises(ValueError):
                m.advance_selection(epoch, score, -1., -1, 0)

    def test_early_stopping_and_compatible_checkpoint(self):
        state = self.loop([.8] * 11)
        self.assertEqual(state["completed_epochs"], 11)
        self.assertEqual(state["best_epoch"], 1)
        self.assertEqual(state["stopped_by"], "early_stopping")
        saved = torch.load(self.root / "best_model.pth", map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["experiment"]["model_type"], "fusion_before_ppm")
        self.assertTrue(saved["completed_epoch"])

    def test_epoch_limit_and_scheduler(self):
        state = self.loop([x / 100 for x in range(1, 61)])
        self.assertEqual(state["completed_epochs"], 60)
        self.assertEqual(state["best_epoch"], 60)
        history = r.read_json(self.root / "history.json")
        self.assertEqual(history[-1]["lr_after_epoch"], 0.)

    def test_mid_epoch_interruption_preserves_completed_checkpoint(self):
        with self.assertRaises(KeyboardInterrupt):
            self.loop([.5, .6], fail_epoch=2)
        state = r.read_json(self.root / "run_manifest.json")
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["completed_epochs"], 1)
        saved = torch.load(self.root / "last_completed_epoch.pth", map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertLess(next(iter(saved["model_state_dict"].values())).abs().max(), 2)
        self.assertFalse((self.root / "selected_validation.json").exists())

    def test_stop_markers_retained(self):
        with self.assertRaises(InterruptedError):
            self.loop([.5, .6], stop_epoch=1)
        self.assertTrue((self.root / "STOP_TRAINING.txt").exists())
        self.assertEqual(r.read_json(self.root / "run_manifest.json")["completed_epochs"], 1)

    def test_nonfinite_validation_fails(self):
        with self.assertRaises(ValueError):
            self.loop([float("nan")])
        self.assertEqual(r.read_json(self.root / "run_manifest.json")["status"], "failed")
        self.assertFalse((self.root / "best_model.pth").exists())

    def test_output_protection(self):
        for path in (r.PACKAGE, self.root, self.root / "实验/new_training"):
            with self.assertRaises((ValueError, FileExistsError)):
                m.output_path(path, self.root, self.catalog)

    def test_model_and_loss_contract(self):
        model = m.order.build_model("fusion_before_ppm", m.tokenizer_for(self.catalog).vocab_size)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9023884)
        weights = torch.tensor(self.catalog["seeds"]["42"]["weights"])
        small = torch.nn.Conv2d(3, 8, 1)
        class Adapter(torch.nn.Module):
            def forward(self, image, ids, mask):
                return small(image)
        batch = {"image": torch.rand(2, 3, 8, 8), "mask": torch.randint(8, (2, 8, 8)),
                 "input_ids": torch.zeros(2, 192, dtype=torch.long), "attention_mask": torch.ones(2, 192, dtype=torch.long)}
        ce = torch.nn.CrossEntropyLoss(weight=weights)
        dice = m.base.WeightedDiceLoss(8, weights)
        # The archived dtype probe queries CUDA even when autocast is disabled.
        # Keep this CPU contract device-free; the CUDA smoke tests that probe.
        with patch.object(m.base, "autocast_dtype", return_value=torch.float32):
            loss, a, b, _, _ = m.base.forward_loss(Adapter(), batch, torch.device("cpu"), ce, dice, False)
        self.assertTrue(torch.equal(loss.detach(), .5 * a + .5 * b))
        loss.backward()
        self.assertTrue(torch.isfinite(small.weight.grad).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r.new_output(args.output)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MainContracts))
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
