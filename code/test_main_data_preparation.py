"""CPU contracts for exact main pixel reconstruction; synthetic TIFFs only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import reproduce as r
import prepare_main_data as p


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


class RawContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pm-mainraw-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base/"inputs"
        self.root.mkdir()
        self.out = self.base/"rebuilt"
        self.bindings = self.base/"references"
        self.bindings.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(r, "WORKSPACE", self.bindings).start()

    def fixture(self, count=4):
        c = {"geometry": {"native": 4, "global": 2}, "dtype": "uint8", "records": [],
            "arrays": {}, "historical_bindings": [], "unpopulated_global_indices": [0, count+1, count+2],
            "images_by_split": {"train": count-1, "val": 1},
            "unpopulated_global_policy": "zero placeholders, never training or evaluation data"}
        stacks = {role: [] for role in p.ROLES}
        for i in range(count):
            image = ((np.arange(16).reshape(4, 4)**2+i*3) % 256).astype(np.uint8)
            mask = ((np.arange(16).reshape(4, 4)+i) % 8).astype(np.uint8)
            low = np.array(Image.fromarray(image).resize((2, 2), Image.Resampling.BILINEAR))
            values = {"native_image": image, "native_mask": mask, "native_low": low,
                      "global_image": low, "global_mask": mask[1::2, 1::2]}
            row = {"id": f"synthetic_{i}", "split": "val" if i == count-1 else "train",
                   "shared_index": i, "cache_index": count-i, "raw": {},
                   "pixels": {key: digest(value) for key, value in values.items()},
                   "counts": np.bincount(values["global_mask"].ravel(), minlength=8).tolist()}
            for role, value in (("image", image), ("mask", mask)):
                path = self.root/role/f"{i}.tif"
                path.parent.mkdir(exist_ok=True)
                im = Image.fromarray(value)
                if role == "mask":
                    # Palette colours differ from the semantic IDs on purpose.
                    im.putpalette([v for k in range(256) for v in (255-k, k, (k*3) % 256)])
                im.save(path)
                row["raw"][role] = {"path": path.relative_to(self.root).as_posix(), "sha256": r.sha256(path)}
            for role, value in values.items():
                stacks[role].append(value)
            c["records"].append(row)
        for role in p.ROLES:
            side = 4 if role in ("native_image", "native_mask") else 2
            shape = [count+3 if role.startswith("global_") else count, side, side]
            c["arrays"][role] = {"path": "cache/"+role+".dat", "shape": shape,
                                  "bytes": int(np.prod(shape))}
            if role.startswith("native_"):
                c["arrays"][role]["sha256"] = digest(np.stack(stacks[role]))
        for name in ("global_identity.json", "native_identity.json", "native_rows.json"):
            path = self.bindings/"bindings"/name
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"historical_reference": name}), encoding="utf-8")
            c["historical_bindings"].append({"path": "bindings/"+name, "sha256": r.sha256(path)})
        return c

    def run_prepare(self, c, resume=False, reader=p.read_sample):
        return p.prepare(c, self.root, self.out, resume, reader)

    def interrupt(self, c):
        def reader(row, catalog, root):
            if row["shared_index"] == 26:
                raise InterruptedError("synthetic interruption")
            return p.read_sample(row, catalog, root)
        with self.assertRaises(InterruptedError):
            self.run_prepare(c, reader=reader)

    def test_palette_ids_preserved_and_nearest_centres_exact(self):
        c = self.fixture(); row = c["records"][0]
        values = p.read_sample(row, c, self.root)
        mask = np.arange(16, dtype=np.uint8).reshape(4, 4) % 8
        np.testing.assert_array_equal(values["native_mask"], mask)
        np.testing.assert_array_equal(values["global_mask"], [[5, 7], [5, 7]])
        p.check_row(row, values)

    def test_bilinear_not_nearest_or_rescaled(self):
        c = self.fixture(); row = c["records"][0]
        values = p.read_sample(row, c, self.root)
        image = np.array([[0, 1, 4, 9], [16, 25, 36, 49], [64, 81, 100, 121], [144, 169, 196, 225]], np.uint8)
        np.testing.assert_array_equal(values["native_image"], image)
        np.testing.assert_array_equal(values["native_low"], Image.fromarray(image).resize((2, 2), Image.Resampling.BILINEAR))
        self.assertFalse(np.array_equal(values["native_low"], image[1::2, 1::2]))
        np.testing.assert_array_equal(values["native_low"], values["global_image"])

    def test_constant_grayscale_is_not_normalized_to_zero(self):
        c = self.fixture(); row = c["records"][0]
        path = self.root/row["raw"]["image"]["path"]
        Image.fromarray(np.full((4, 4), 73, np.uint8)).save(path)
        row["raw"]["image"]["sha256"] = r.sha256(path)
        values = p.read_sample(row, c, self.root)
        self.assertTrue(np.all(values["native_image"] == 73))
        self.assertTrue(np.all(values["global_image"] == 73))

    def test_wrong_hash_rejected(self):
        c = self.fixture(); c["records"][0]["raw"]["image"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            p.read_sample(c["records"][0], c, self.root)

    def test_wrong_shape_rejected(self):
        c = self.fixture(); c["geometry"]["native"] = 8
        with self.assertRaises(ValueError):
            p.read_sample(c["records"][0], c, self.root)

    def test_wrong_label_dtype_rgb_and_class_rejected(self):
        c = self.fixture(); row = c["records"][0]
        path = self.root/row["raw"]["mask"]["path"]
        for invalid in (np.ones((4, 4), np.uint16), np.ones((4, 4, 3), np.uint8), np.full((4, 4), 8, np.uint8)):
            Image.fromarray(invalid).save(path)
            row["raw"]["mask"]["sha256"] = r.sha256(path)
            with self.assertRaises(ValueError):
                p.read_sample(row, c, self.root)

    def test_no_test_read_or_output_creation(self):
        c = self.fixture(); row = c["records"][0]; row["split"] = "test"
        with patch.object(p.e, "checked_resource") as checked:
            with self.assertRaises(ValueError):
                p.read_sample(row, c, self.root)
            with self.assertRaises(ValueError):
                self.run_prepare(c)
            checked.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_wrong_pixel_or_class_counts_rejected(self):
        c = self.fixture(); row = c["records"][0]
        values = p.read_sample(row, c, self.root)
        bad = copy.deepcopy(row); bad["pixels"]["native_mask"] = "0"*64
        with self.assertRaises(ValueError):
            p.check_row(bad, values)
        bad = copy.deepcopy(row); bad["counts"][0] += 1
        with self.assertRaises(ValueError):
            p.check_row(bad, values)

    def test_full_preparation_all_rows_slots_and_references(self):
        c = self.fixture(); result = self.run_prepare(c)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["gpu_initialized"])
        self.assertEqual(result["native_files_byte_identical"], 3)
        paths = {role: self.out/item["path"] for role, item in c["arrays"].items()}
        p.verify_arrays(c, paths)
        for item in c["historical_bindings"]:
            self.assertEqual(r.sha256(self.out/item["path"]), item["sha256"])
        self.assertEqual(r.read_json(self.out/"preparation.json")["status"], "completed")

    def test_old_global_indices_not_repacked(self):
        c = self.fixture(); self.run_prepare(c)
        item = c["arrays"]["global_mask"]
        a = np.fromfile(self.out/item["path"], np.uint8).reshape(item["shape"])
        for row in c["records"]:
            self.assertEqual(digest(a[row["cache_index"]]), row["pixels"]["global_mask"])
        self.assertFalse(a[c["unpopulated_global_indices"]].any())

    def test_completed_resume_does_not_read_tiffs(self):
        c = self.fixture(); self.run_prepare(c)
        with patch.object(p, "read_sample") as mocked:
            self.run_prepare(c, True, mocked)
            mocked.assert_not_called()

    def test_fresh_output_and_explicit_same_identity_required(self):
        c = self.fixture(); self.run_prepare(c)
        with self.assertRaises(FileExistsError):
            self.run_prepare(c)
        changed = copy.deepcopy(c); changed["geometry"]["global"] = 3
        with self.assertRaises(ValueError):
            self.run_prepare(changed, True)

    def test_absent_resume_rejected(self):
        c = self.fixture()
        with self.assertRaises(FileNotFoundError):
            self.run_prepare(c, True)

    def test_interruption_resume_only_uncommitted_rows(self):
        c = self.fixture(29); self.interrupt(c)
        meta = r.read_json(self.out/"preparation.json")
        self.assertEqual((meta["status"], meta["completed_rows"]), ("stopped", 25))
        self.assertFalse((self.out/"bindings").exists())
        calls = []
        def reader(row, catalog, root):
            calls.append(row["shared_index"])
            return p.read_sample(row, catalog, root)
        self.run_prepare(c, True, reader)
        self.assertEqual(calls, list(range(25, 29)))

    def test_committed_row_corruption_rejected(self):
        c = self.fixture(29); self.interrupt(c)
        path = self.out/(c["arrays"]["native_image"]["path"]+".partial")
        with path.open("r+b") as stream:
            stream.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_prepare(c, True)

    def test_truncated_partial_rejected(self):
        c = self.fixture(29); self.interrupt(c)
        path = self.out/(c["arrays"]["native_image"]["path"]+".partial")
        path.write_bytes(b"synthetic truncation")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.run_prepare(c, True)

    def test_nontraining_placeholder_corruption_rejected(self):
        c = self.fixture(); self.run_prepare(c)
        path = self.out/c["arrays"]["global_image"]["path"]
        with path.open("r+b") as stream:
            stream.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            self.run_prepare(c, True)

    def test_no_binding_metadata_if_native_hash_fails(self):
        c = self.fixture(); c["arrays"]["native_image"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "Native pixel bytes"):
            self.run_prepare(c)
        self.assertFalse((self.out/"bindings").exists())
        self.assertEqual(r.read_json(self.out/"preparation.json")["status"], "failed")

    def test_partial_finalization_resumes_without_raw_reads(self):
        c = self.fixture()
        original = Path.rename; calls = []
        def interrupt_rename(path, target):
            calls.append(path)
            if len(calls) == 2:
                raise InterruptedError("synthetic finalization interruption")
            return original(path, target)
        with patch.object(Path, "rename", interrupt_rename):
            with self.assertRaises(InterruptedError):
                self.run_prepare(c)
        self.assertFalse((self.out/"bindings").exists())
        with patch.object(p, "read_sample") as mocked:
            self.run_prepare(c, True, mocked)
            mocked.assert_not_called()

    def test_stop_markers_retained_without_output(self):
        c = self.fixture(); marker = self.root/"STOP_TRAINING.txt"; marker.touch()
        with self.assertRaises(InterruptedError):
            self.run_prepare(c)
        self.assertTrue(marker.exists())
        self.assertFalse(self.out.exists())

    def test_input_and_archive_trees_protected(self):
        c = self.fixture()
        for output in (self.root, self.root/"image/new", self.root/"mask/new", self.root/"cache/new", r.PACKAGE/"new"):
            with self.assertRaises(ValueError):
                p.destination(output, self.root, c)

    def test_binding_corruption_on_resume_rejected(self):
        c = self.fixture(); self.run_prepare(c)
        (self.out/c["historical_bindings"][0]["path"]).write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_prepare(c, True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(); r.new_output(args.output); r.verify_contents()
    p.load_catalog()  # Identity records only, no research pixels.
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RawContracts))
    ok = result.wasSuccessful() and not torch.cuda.is_initialized()
    report = {"status": "PASS" if ok else "FAIL", "tests": result.testsRun,
        "source_contents_sha256": r.sha256(r.PACKAGE/"contents.json"),
        "gpu_initialized": torch.cuda.is_initialized(), "research_pixels_read": False,
        "research_training_started": False, "test_pixels_read": False}
    r.write_json(args.output, report)
    print(json.dumps(report))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
