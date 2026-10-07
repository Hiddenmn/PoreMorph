import inspect
import unittest
import numpy as np
from PIL import Image

import locked_common as c
from inference import FrozenPipeline, assign_bodies, low_image


class InferenceContracts(unittest.TestCase):
    def test_predictor_has_no_target_or_sample_identifier_input(self):
        self.assertEqual(list(inspect.signature(FrozenPipeline.predict_batch).parameters),
                         ["self", "images", "descriptions", "diagnostics"])

    def test_empty_and_mixed_bodies_preserve_support(self):
        original = np.array([[0, 1, 3, 4], [2, 1, 6, 7]], np.uint8)
        labels = np.array([[0, 0, 1, 1], [0, 0, 2, 2]], np.int32)
        result = assign_bodies(original, labels, np.array([5, 3]))
        np.testing.assert_array_equal(result, np.array([[0, 1, 5, 5], [2, 1, 3, 3]], np.uint8))
        empty = original[:, :2]
        np.testing.assert_array_equal(assign_bodies(empty, np.zeros_like(empty), []), empty)
        with self.assertRaises(AssertionError):
            assign_bodies(original, labels, [2, 3])

    def test_preprocessing_centres_and_type_contract(self):
        label = (np.indices((1024, 1024)).sum(0) % 8).astype(np.uint8)
        shrunk = np.asarray(Image.fromarray(label).resize((512, 512), Image.Resampling.NEAREST))
        np.testing.assert_array_equal(shrunk, label[1::2, 1::2])
        self.assertEqual(low_image(label).shape, (512, 512))
        with self.assertRaises(ValueError):
            low_image(label.astype(np.float32))

    def test_test_access_needs_all_full_validation_gates(self):
        valid = {str(seed): {"status": "completed", "split": "val", "images": 676,
                            "all_parity_exact": True, "test_pixels_read": False} for seed in c.SEEDS}
        self.assertTrue(c.allow_test(valid))
        del valid["2025"]
        with self.assertRaises(AssertionError):
            c.allow_test(valid)

    def test_confusion_orientation_and_counts(self):
        actual = np.array([[0, 3], [3, 4]], np.uint8)
        pred = np.array([[0, 4], [3, 4]], np.uint8)
        hist = c.histogram(pred, actual)
        self.assertEqual(hist[3, 4], 1)
        self.assertEqual(hist[4, 3], 0)
        self.assertEqual(hist.sum(), 4)
        self.assertAlmostEqual(c.metrics(hist)["class_IoU"][3], .5, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
