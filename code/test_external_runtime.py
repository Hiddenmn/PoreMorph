"""Synthetic external-interface checks; no GPU or research pixels are read."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
from PIL import Image
import torch

import reproduce as r
import external_runtime as e


class ExternalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.verify_contents()
        cls.catalog = r.read_json(r.PACKAGE / "external_checkpoints.json")
        cls.ops = e.preprocessing()

    def test_case_space(self):
        cases = self.catalog["cases"]
        self.assertEqual(len(cases), 9)
        for seed in (42, 2024, 2025):
            for domain in ("bentheimer", "wf1"):
                self.assertIn(f"original_{domain}_seed{seed}", cases)
            self.assertIn(f"whole_repeat_wf1_seed{seed}", cases)
        self.assertFalse(any(key.startswith("whole_repeat_bentheimer") for key in cases))
        e.verify_sources(self.catalog)

    def test_descriptor_modes_not_silently_merged(self):
        for case in self.catalog["cases"].values():
            original42 = case["attempt"] == "original" and case["seed"] == 42
            self.assertEqual(case["descriptor_support"], "probability_argmax" if original42 else "logit_argmax")

    def test_float_normalization_and_constant_image(self):
        raw = (np.arange(77).reshape(7, 11) ** 2).astype(np.float32)
        low, high = np.percentile(raw, (0.5, 99.5))
        expected = np.clip((raw - low) / (high - low), 0, 1).astype(np.float32)
        np.testing.assert_array_equal(self.ops.normalize_native(raw), expected)
        np.testing.assert_array_equal(self.ops.normalize_native(np.full((5, 8), 321)), np.zeros((5, 8)))
        self.assertEqual(self.ops.low_image(expected).dtype, np.float32)
        self.assertEqual(self.ops.low_image(expected).shape, (512, 512))

    def test_crop_before_normalization(self):
        raw = (np.arange(1014 * 992).reshape(1014, 992) % 65535).astype(np.uint16)
        path = self.fixtures / "bentheimer.tif"
        Image.fromarray(raw).save(path)
        native, low = e.prepare_image(path, self.catalog["cases"]["original_bentheimer_seed42"])
        expected = self.ops.normalize_native(raw[20:1000, 10:990])
        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(low, self.ops.low_image(expected))
        self.assertEqual(native.shape, (980, 980))

    def test_wf1_geometry_and_no_polarity_inversion(self):
        raw = np.broadcast_to(np.arange(1030, dtype=np.uint16), (1033, 1030)).copy()
        path = self.fixtures / "wf1.tif"
        Image.fromarray(raw).save(path)
        native, _ = e.prepare_image(path, self.catalog["cases"]["original_wf1_seed42"])
        np.testing.assert_array_equal(native, self.ops.normalize_native(raw))
        self.assertEqual(native.shape, (1033, 1030))
        self.assertEqual(float(native[0, 0]), 0)
        self.assertEqual(float(native[0, -1]), 1)

    def test_bad_geometry_and_nonfinite_rejected(self):
        case = self.catalog["cases"]["original_wf1_seed42"]
        for name, raw in (("rgb", np.zeros((1033, 1030, 3), np.uint8)),
                          ("resize", np.zeros((512, 512), np.uint8)),
                          ("nan", np.full((1033, 1030), np.nan, np.float32))):
            path = self.fixtures / f"{name}.tif"
            Image.fromarray(raw).save(path)
            with self.assertRaises(ValueError):
                e.prepare_image(path, case)

    def test_tokenizer_extension_keeps_original_ids(self):
        core, _ = r.pipeline_modules()
        prior = r.module("pack_external_test_prior", "external_clean_prior_adaptation_20261001/prior_schema.py")
        for config in r.read_json(r.PACKAGE / "main_checkpoints.json").values():
            original = core.order.base.CharTokenizer.load(r.local_path(r.WORKSPACE, config["tokenizer"]["path"]))
            extended = prior.extend_tokenizer(original)
            self.assertEqual(extended.idx2token[:original.vocab_size], original.idx2token)
            for prompt in prior.PROMPTS.values():
                stats = prior.token_stats(extended, prompt)
                self.assertEqual(stats["unknown_active_tokens"], 0)
                self.assertEqual(stats["truncated_characters"], 0)

    def test_prediction_has_no_reference_input(self):
        method = e.definitions("external_fine8_multiseed_test_recovery_20261005/evaluate_multiseed.py",
                              ("predict",), {"torch": torch}, owner="Pipeline").predict
        self.assertEqual(list(inspect.signature(method).parameters), ["self", "image", "low", "diagnostics"])
        with self.assertRaises(ValueError):
            e.run(types.SimpleNamespace(trusted_checkpoints=False))
        self.assertFalse(torch.cuda.is_initialized())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Retain generated synthetic fixtures beside the report; do not delete anything.
    fixture_dir = args.output.with_suffix("").with_name(args.output.stem + "_fixtures").resolve()
    fixture_dir.mkdir(parents=True, exist_ok=False)
    ExternalContracts.fixtures = fixture_dir
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExternalContracts))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
              "gpu_initialized": torch.cuda.is_initialized(), "research_pixels_read": False,
              "training_started": False}
    r.write_json(args.output, report)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(report)


if __name__ == "__main__":
    main()
