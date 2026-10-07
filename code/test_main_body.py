"""CPU-only main body-head contracts using synthetic descriptors."""
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
import main_body as b
import train_main_body as m


def synthetic_data(count=520, device="cpu"):
    rng = np.random.default_rng(718)
    points = rng.normal(0, .2, (count, 48, 31)).astype(np.float16)
    valid = rng.integers(0, 2, (count, 48), dtype=np.uint8)
    valid[:, 0] = 1
    summary = rng.uniform(.05, .9, (count, 13)).astype(np.float32)
    summary[:, :8] /= summary[:, :8].sum(1, keepdims=True)
    targets = rng.integers(0, 99, (count, 8), dtype=np.uint32)
    data = object.__new__(b.Bodies)
    data.split, data.device, data.indices = "train", device, np.arange(count, dtype=np.int64)
    data.arrays = {"points": points, "valid": valid, "summary": summary,
                   "native_counts": targets, "canonical512_counts": targets // 4}
    return data


class BodyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = b.load_catalog()
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pm-body-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def loop(self, scores=None, fail=0, stop=0, replay_fail=False):
        model = torch.nn.Linear(2, 5)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
        history = []
        values = scores or [i / 100 for i in range(1, 13)]
        def train(*args):
            epoch = args[-1]
            if epoch == fail:
                with torch.no_grad():
                    next(model.parameters()).add_(100)
                raise KeyboardInterrupt("Synthetic partial epoch")
            history.append(epoch)
            opt.step(); scheduler.step()
            return 1.
        calls = []
        def evaluate(*args):
            if len(calls) == 12:
                return {"native": {"mIoU": max(values) + (.1 if replay_fail else 0)}}
            epoch = history[-1]
            if epoch == stop:
                (self.root / "STOP_AFTER_EPOCH.txt").touch()
            calls.append(epoch)
            return {"native": {"mIoU": values[epoch-1]}}
        with patch.object(b, "train_epoch", train), patch.object(b.ops, "evaluate", evaluate):
            return m.run_epochs(model, opt, scheduler, None, None, torch.ones(5), self.root,
                                {"seed": 42, "selection": self.c["selection"]}, (self.root,))

    def test_original_three_initial_states(self):
        for case in self.c["cases"].values():
            model = b.build_model(case)
            self.assertEqual(model.mode, "separate")
            self.assertEqual(b.ops.state_digest(model.state_dict()), case["initial_sha256"])
            self.assertEqual(sum(p.numel() for p in model.parameters()), 25573)

    def test_protocol_native_selection_not_external(self):
        for case in self.c["cases"].values():
            self.assertFalse(case["config"]["epoch_zero_eligible"])
            self.assertEqual(case["config"]["batch_size"], 512)
        self.assertIn("native", self.c["selection"])
        self.assertTrue(self.c["upstream_frozen"])

    def test_seeded_order_and_augmentation_stream(self):
        for seed in (42, 2024, 2025):
            for epoch in (1, 6, 12):
                np.testing.assert_array_equal(b.random_ops.order_for(np.arange(41), seed, epoch),
                    np.random.default_rng(seed * 1000 + epoch).permutation(np.arange(41)))
                self.assertEqual(b.random_ops.stream(seed, epoch, "cpu").initial_seed(), seed*10000+epoch)

    def test_forward_excludes_targets(self):
        data = synthetic_data(6)
        model = b.build_model(self.c["cases"]["42"]).eval()
        with torch.inference_mode():
            first = model(*data.input_batch(data.indices))
            data.arrays["native_counts"][:] = 777
            data.arrays["canonical512_counts"][:] = 888
            self.assertTrue(torch.equal(first, model(*data.input_batch(data.indices))))

    def test_input_dtype_and_copy(self):
        data = synthetic_data(3)
        p, v, s = data.input_batch(data.indices)
        self.assertEqual((p.dtype, v.dtype, s.dtype), (torch.float32, torch.bool, torch.float32))
        p.fill_(99)
        self.assertLess(np.abs(data.arrays["points"]).max(), 99)

    def test_empty_contact_groups_finite(self):
        data = synthetic_data(3)
        data.arrays["valid"][:, 16:] = 0
        model = b.build_model(self.c["cases"]["42"])
        self.assertTrue(torch.isfinite(model(*data.input_batch(data.indices))).all())

    def test_d4_only_coordinates_and_gradients(self):
        data = synthetic_data(6)
        points = data.input_batch(data.indices)[0]
        out = b.model_ops.augment(points, b.random_ops.stream(42, 1, "cpu"))
        self.assertTrue(torch.equal(points[:, :, 4:], out[:, :, 4:]))
        for k in (0, 2):
            self.assertTrue(torch.equal(points[:, :, k:k+2].square().sum(2), out[:, :, k:k+2].square().sum(2)))

    def test_original_conditional_loss_formula(self):
        logits, counts, weights = torch.randn(9, 5), torch.randint(0, 20, (9, 8)).float(), torch.rand(5)+.2
        weighted = counts[:, 3:]*weights
        ce = -(weighted*logits.log_softmax(-1)).sum()/weighted.sum().clamp_min(1)
        p = logits.softmax(-1)
        dice = 1-(2*(p*counts[:, 3:]).sum(0)+1e-6)/((p*counts.sum(1, keepdim=True)).sum(0)+counts[:, 3:].sum(0)+1e-6)
        expected = .5*ce + .5*(dice*weights).sum()/weights.sum()
        self.assertTrue(torch.equal(b.model_ops.body_loss(logits, counts, weights), expected))

    def test_training_one_cpu_epoch_preserves_descriptors(self):
        data = synthetic_data(17)
        hashes = {k: x.tobytes() for k, x in data.arrays.items()}
        model = b.build_model(self.c["cases"]["42"])
        initial = b.ops.state_digest(model.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12)
        loss = b.train_epoch(model, data, opt, sched, torch.ones(5), 42, 1)
        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(b.ops.state_digest(model.state_dict()), initial)
        self.assertEqual(sched.last_epoch, 1)
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters()))
        self.assertEqual(hashes, {k: x.tobytes() for k, x in data.arrays.items()})

    def test_validation_cannot_train(self):
        data = synthetic_data(1); data.split = "val"
        with self.assertRaises(ValueError):
            b.train_epoch(None, data, None, None, None, 42, 1)

    def test_no_test_loader(self):
        with self.assertRaises(ValueError):
            b.Bodies(self.root, "test")

    def test_aggregation_fixed_nonoil_and_oil_support(self):
        counts = np.arange(24, dtype=np.uint32).reshape(3, 8)
        outside = np.zeros((8, 8), np.int64); outside[:, :3] = 11
        h = b.aggregate(counts, [0, 3, 4], outside)
        np.testing.assert_array_equal(h[:, :3], outside[:, :3])
        np.testing.assert_array_equal(h[:, 3:].sum(1), counts.sum(0))

    def test_selection_first_native_tie(self):
        rows = [{"epoch": i, "validation": {"native": {"mIoU": v}, "canonical512": {"mIoU": i/10}}}
                for i, v in enumerate((.5, .7, .7, .65), 1)]
        self.assertEqual(m.choose(rows)["epoch"], 2)

    def test_selection_rejects_zero_missing_and_nonfinite(self):
        for rows in ([], [{"epoch": 0}], [{"epoch": 2}],
                     [{"epoch": 1, "validation": {"native": {"mIoU": float("nan")}}}]):
            with self.assertRaises(ValueError):
                m.choose(rows)

    def test_twelve_epochs_selected_reload(self):
        state = self.loop([.7]*12)
        self.assertEqual(state["best_epoch"], 1)
        self.assertEqual(state["completed_epochs"], 12)
        self.assertTrue(state["selected_replay_exact"])
        self.assertFalse(state["upstream_updated"])
        self.assertEqual(r.read_json(self.root / "history.json")[-1]["lr_after_epoch"], 0.)
        self.assertEqual(r.read_json(self.root / "selected_validation.json")["sha256"], r.sha256(self.root / "best.pth"))

    def test_partial_epoch_keeps_previous_checkpoint(self):
        with self.assertRaises(KeyboardInterrupt):
            self.loop(fail=2)
        saved = torch.load(self.root / "last_completed_epoch.pth", map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertLess(next(iter(saved["model"].values())).abs().max(), 2)
        self.assertFalse((self.root / "selected_validation.json").exists())

    def test_stop_marker_retained(self):
        with self.assertRaises(InterruptedError):
            self.loop(stop=1)
        self.assertTrue((self.root / "STOP_AFTER_EPOCH.txt").exists())
        self.assertEqual(r.read_json(self.root / "run_manifest.json")["status"], "stopped")

    def test_reload_failure_not_completed(self):
        with self.assertRaises(ValueError):
            self.loop(replay_fail=True)
        self.assertEqual(r.read_json(self.root / "run_manifest.json")["status"], "failed")

    def test_records_reject_shifted_split_and_offsets(self):
        rows = [dict(x, row=i, bodies=1, start=i, end=i+1) for i, x in enumerate(self.c["records"])]
        case = {"n": 3040, "images_by_split": {"train": 2364, "val": 676},
                "bodies_by_split": {"train": 2364, "val": 676}}
        b.validate_records(case, rows, self.c["records"])
        for field, value in (("split", "test"), ("start", 1), ("shared_index", 2)):
            altered = copy.deepcopy(rows); altered[0][field] = value
            with self.assertRaises(ValueError):
                b.validate_records(case, altered, self.c["records"])

    def test_output_overwrite_and_archived_roots_rejected(self):
        for path in (self.root, r.PACKAGE, self.root / "实验/new"):
            with self.assertRaises((ValueError, FileExistsError)):
                b.output_path(path, self.root, self.c)

    def test_incomplete_and_smoke_descriptors_rejected(self):
        case = self.c["cases"]["42"]
        meta = {"status": "completed", "smoke": False, "test_pixels_read": False,
            "frozen_validation_replayed": True, "completed_images": 3040,
            "selected_rows": list(range(3040)), "n": case["n"],
            "images_by_split": case["images_by_split"], "bodies_by_split": case["bodies_by_split"]}
        for key, value in (("status", "preparing"), ("smoke", True), ("test_pixels_read", True),
                           ("completed_images", 3039), ("selected_rows", [0]), ("n", 1),
                           ("frozen_validation_replayed", False)):
            changed = copy.deepcopy(meta); changed[key] = value
            with patch.object(b.e, "checked_resource", lambda *a: None), patch.object(b.r, "read_json", side_effect=[changed, []]):
                with self.assertRaises(ValueError):
                    b.bind(self.c, self.root, 42)

    def test_selected_native_mismatch_rejected_before_array_open(self):
        case = self.c["cases"]["42"]
        meta = {"status": "completed", "smoke": False, "test_pixels_read": False,
            "frozen_validation_replayed": True, "completed_images": 3040,
            "selected_rows": list(range(3040)), "n": case["n"],
            "images_by_split": case["images_by_split"], "bodies_by_split": case["bodies_by_split"],
            "identity": {"native_checkpoint": "0"*64}}
        with patch.object(b.e, "checked_resource", lambda *a: None), patch.object(b.r, "read_json", side_effect=[meta, []]):
            with self.assertRaisesRegex(ValueError, "Descriptor/native"):
                b.bind(self.c, self.root, 42)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); r.new_output(args.output)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BodyContracts))
    ok = result.wasSuccessful() and not torch.cuda.is_initialized()
    report = {"status": "PASS" if ok else "FAIL", "tests": result.testsRun,
        "source_contents_sha256": r.sha256(r.PACKAGE / "contents.json"),
        "gpu_initialized": torch.cuda.is_initialized(), "research_pixels_read": False,
        "research_training_started": False, "test_pixels_read": False}
    r.write_json(args.output, report)
    print(json.dumps(report))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
