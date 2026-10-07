import unittest
import numpy as np
import metrics_3d as m


class GeometryContracts(unittest.TestCase):
    def test_anisotropic_faces_against_independent_loop(self):
        volume = np.random.default_rng(5).integers(0, 8, (3, 4, 5), dtype=np.uint8)
        expected = np.zeros((3, 4, 4), np.int64)
        for position in np.ndindex(volume.shape):
            for axis in range(3):
                neighbour = list(position)
                neighbour[axis] += 1
                if neighbour[axis] < volume.shape[axis]:
                    a, b = min(int(volume[position]), 3), min(int(volume[tuple(neighbour)]), 3)
                    expected[axis, a, b] += 1
        np.testing.assert_array_equal(m.contact_faces(volume), expected)
        result = m.geometry(volume, (2., 3., 5.))
        for phase, name in ((2, "water"), (1, "rock")):
            faces = expected[:, 3, phase] + expected[:, phase, 3]
            self.assertAlmostEqual(result[f"oil_{name}_contact_area_mm2"], float(faces @ [15., 10., 6.]) / 1e6)

    def test_26_connected_diagonal_voxels(self):
        mask = np.zeros((4, 4, 4), bool)
        mask[0, 0, 0] = mask[1, 1, 1] = mask[3, 3, 3] = True
        result = m.components(mask, m.SPACINGS["native"])
        self.assertEqual(result["component_count"], 2)
        self.assertAlmostEqual(result["largest_component_fraction"], 2/3)

    def test_equivalent_physical_component_threshold(self):
        canonical = m.components(np.ones((1, 1, 27), bool), m.SPACINGS["canonical512"])
        native = m.components(np.ones((1, 1, 108), bool), m.SPACINGS["native"])
        small_native = m.components(np.ones((1, 1, 107), bool), m.SPACINGS["native"])
        self.assertEqual(canonical["minimum_component_voxels"], 27)
        self.assertEqual(native["minimum_component_voxels"], 108)
        self.assertEqual(canonical["component_count_ge_reference_min_volume"], 1)
        self.assertEqual(native["component_count_ge_reference_min_volume"], 1)
        self.assertEqual(small_native["component_count_ge_reference_min_volume"], 0)
        self.assertAlmostEqual(canonical["p90_component_volume_um3"], native["p90_component_volume_um3"])

    def test_no_oil_has_undefined_relative_errors(self):
        volume = np.zeros((3, 4, 5), np.uint8)
        geometry = m.geometry(volume, m.SPACINGS["native"])
        self.assertEqual(geometry["component_count"], 0)
        self.assertEqual(geometry["oil_volume_mm3"], 0)
        result = m.comparison(m.counts(volume), m.counts(volume), geometry, geometry, m.SPACINGS["native"])
        self.assertIsNone(result["mean_oil_class_fraction_abs_error"])
        self.assertIsNone(result["agreement"]["oil_volume_mm3"]["absolute_relative_error_percent"])

    def test_subtype_change_is_not_phase_geometry_change(self):
        reference = np.zeros((3, 4, 5), np.uint8)
        reference[:, :, 1:3] = 3
        reference[:, :, 3] = 1
        reference[:, :, 4] = 2
        prediction = reference.copy()
        prediction[reference == 3] = 4
        m.same_phase_support(reference, prediction)
        original = m.geometry(reference, m.SPACINGS["canonical512"])
        changed = m.geometry(prediction, m.SPACINGS["canonical512"])
        self.assertEqual(original, changed)
        result = m.comparison(m.counts(prediction), m.counts(reference), changed, original, m.SPACINGS["canonical512"])
        self.assertEqual(result["mean_oil_class_fraction_abs_error"], 0.4)
        self.assertEqual(result["agreement"]["oil_volume_mm3"]["absolute_difference"], 0)
        prediction[0, 0, 1] = 1
        with self.assertRaises(AssertionError):
            m.same_phase_support(reference, prediction)

    def test_missing_duplicate_or_reordered_slice_is_rejected(self):
        ids = list(range(760, 845))
        m.check_ids(ids)
        for bad in (ids[:-1], ids[:-1] + [843], ids[::-1]):
            with self.assertRaises(AssertionError):
                m.check_ids(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
