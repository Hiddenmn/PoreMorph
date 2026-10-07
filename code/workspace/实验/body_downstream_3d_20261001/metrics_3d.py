"""CPU-only geometry and composition measurements; no prediction modification."""
from __future__ import annotations
import math
import numpy as np
from scipy import ndimage

SPACINGS = {"canonical512": (13.25, 26.5, 26.5), "native": (13.25, 13.25, 13.25)}
MIN_VOLUME_UM3 = 27 * float(np.prod(SPACINGS["canonical512"]))


def check_ids(ids):
    assert list(ids) == list(range(760, 845)), "Require all 85 original ordered test slices"


def counts(volume):
    assert volume.ndim == 3 and volume.dtype == np.uint8
    result = np.zeros(8, dtype=np.int64)
    for plane in volume:
        assert plane.max() < 8
        result += np.bincount(plane.ravel(), minlength=8)
    return result


def contact_faces(volume):
    """Positive-direction face matrices [z,y,x,phase_left,phase_right]."""
    matrices = np.zeros((3, 4, 4), np.int64)
    previous = None
    for labels in volume:
        plane = np.minimum(labels, 3).astype(np.uint16)
        if previous is not None:
            matrices[0] += np.bincount((previous * 4 + plane).ravel(), minlength=16).reshape(4, 4)
        matrices[1] += np.bincount((plane[:-1] * 4 + plane[1:]).ravel(), minlength=16).reshape(4, 4)
        matrices[2] += np.bincount((plane[:, :-1] * 4 + plane[:, 1:]).ravel(), minlength=16).reshape(4, 4)
        previous = plane
    z, y, x = volume.shape
    np.testing.assert_array_equal(matrices.sum((1, 2)), [(z-1)*y*x, z*(y-1)*x, z*y*(x-1)])
    return matrices


def components(mask, spacing):
    labels, number = ndimage.label(mask, structure=np.ones((3, 3, 3), np.uint8))
    sizes = np.zeros(number + 1, dtype=np.int64)
    for plane in labels:
        sizes += np.bincount(plane.ravel(), minlength=number + 1)
    del labels
    sizes = sizes[1:]
    assert int(sizes.sum()) == int(np.count_nonzero(mask))
    voxel = float(np.prod(spacing))
    minimum = int(math.ceil(MIN_VOLUME_UM3 / voxel - 1e-9))
    retained = sizes[sizes >= minimum]
    return {"component_count": int(number), "minimum_component_voxels": minimum,
            "component_count_ge_reference_min_volume": int(len(retained)),
            "largest_component_fraction": float(sizes.max() / sizes.sum()) if number else 0.0,
            "p90_component_volume_um3": float(np.percentile(sizes, 90) * voxel) if number else 0.0,
            "p90_retained_component_volume_um3": float(np.percentile(retained, 90) * voxel) if len(retained) else 0.0}


def geometry(volume, spacing, count=None):
    count = counts(volume) if count is None else np.asarray(count, dtype=np.int64)
    dz, dy, dx = spacing
    face_area = np.array([dy * dx, dz * dx, dz * dy])
    faces = contact_faces(volume)
    water = faces[:, 3, 2] + faces[:, 2, 3]
    rock = faces[:, 3, 1] + faces[:, 1, 3]
    water_area, rock_area = float(water @ face_area), float(rock @ face_area)
    oil = int(count[3:].sum())
    result = {"oil_saturation": oil / max(int(count[2:].sum()), 1),
              "oil_volume_mm3": oil * float(np.prod(spacing)) / 1e9,
              "oil_water_contact_area_mm2": water_area / 1e6,
              "oil_rock_contact_area_mm2": rock_area / 1e6,
              "oil_water_to_rock_area_ratio": water_area / max(rock_area, 1.0),
              "oil_water_faces_zyx": water.tolist(), "oil_rock_faces_zyx": rock.tolist()}
    result.update(components(volume >= 3, spacing))
    return result


def same_phase_support(first, second):
    assert first.shape == second.shape
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a >= 3, b >= 3)
        np.testing.assert_array_equal(a[a < 3], b[a < 3])


def comparison(count, reference_count, predicted_geometry, reference_geometry, spacing):
    count, reference_count = np.asarray(count), np.asarray(reference_count)
    oil, reference_oil = int(count[3:].sum()), int(reference_count[3:].sum())
    fractions = count[3:] / oil if oil else None
    reference_fractions = reference_count[3:] / reference_oil if reference_oil else None
    errors = np.abs(fractions - reference_fractions) if fractions is not None and reference_fractions is not None else None
    scalar_names = ("oil_saturation", "oil_volume_mm3", "oil_water_contact_area_mm2", "oil_rock_contact_area_mm2",
                    "oil_water_to_rock_area_ratio", "component_count", "component_count_ge_reference_min_volume",
                    "largest_component_fraction", "p90_component_volume_um3", "p90_retained_component_volume_um3")
    agreement = {}
    for name in scalar_names:
        p, r = predicted_geometry[name], reference_geometry[name]
        agreement[name] = {"prediction": p, "reference": r, "signed_difference": float(p-r),
                           "absolute_difference": float(abs(p-r)),
                           "absolute_relative_error_percent": float(100*abs(p-r)/abs(r)) if r else None}
    voxel_mm3 = float(np.prod(spacing)) / 1e9
    return {"predicted_class_voxels": count.tolist(), "reference_class_voxels": reference_count.tolist(),
            "oil_class_ids": list(range(3, 8)), "predicted_oil_class_fractions": None if fractions is None else fractions.tolist(),
            "reference_oil_class_fractions": None if reference_fractions is None else reference_fractions.tolist(),
            "oil_class_fraction_abs_error": None if errors is None else errors.tolist(),
            "mean_oil_class_fraction_abs_error": None if errors is None else float(errors.mean()),
            "predicted_oil_class_volume_mm3": (count[3:] * voxel_mm3).tolist(),
            "reference_oil_class_volume_mm3": (reference_count[3:] * voxel_mm3).tolist(),
            "oil_class_volume_abs_error_mm3": (np.abs(count[3:] - reference_count[3:]) * voxel_mm3).tolist(),
            "geometry": predicted_geometry, "agreement": agreement}
