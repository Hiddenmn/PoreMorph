"""Synthetic file-routing checks; no model, research pixels or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import shared_resources as s


class SharedResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pm-shared-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "weights/model.pth"
        self.path.parent.mkdir()
        self.path.write_bytes(b"synthetic exported payload")
        self.old = hashlib.sha256(b"synthetic original payload").hexdigest()
        self.entry = {"path": "weights/model.pth", "sha256": self.old}
        self.row = dict(path=self.entry["path"], original_sha256=self.old,
            sha256=s.digest(self.path), bytes=self.path.stat().st_size,
            tensor_values_shapes_dtypes_strides_and_module_metadata_exact=True)
        self.mapping = self.root / "exports.json"
        self.write_mapping([self.row])

    def write_mapping(self, rows):
        self.mapping.write_text(json.dumps(dict(format="verified-model-state-exports-v1",
            optimizer_resume_supported=False, checkpoints=rows)), encoding="utf-8")

    def test_original_is_still_accepted(self):
        self.assertEqual(s.checked(self.root, dict(self.entry, sha256=s.digest(self.path)),
                                   self.root / "absent.json"), self.path)

    def test_only_explicit_same_path_same_original_export_is_accepted(self):
        self.assertEqual(s.checked(self.root, self.entry, self.mapping), self.path)

    def test_changed_export_bytes_rejected(self):
        self.path.write_bytes(b"altered")
        with self.assertRaises(ValueError):
            s.checked(self.root, self.entry, self.mapping)

    def test_wrong_original_binding_rejected(self):
        self.write_mapping([dict(self.row, original_sha256="0" * 64)])
        with self.assertRaises(ValueError):
            s.checked(self.root, self.entry, self.mapping)

    def test_ambiguous_mapping_rejected(self):
        self.write_mapping([self.row, self.row])
        with self.assertRaises(ValueError):
            s.checked(self.root, self.entry, self.mapping)

    def test_data_not_in_mapping_rejected(self):
        self.write_mapping([])
        with self.assertRaises(ValueError):
            s.checked(self.root, self.entry, self.mapping)

    def test_unsafe_paths_rejected(self):
        for name in ("../outside.pth", "/outside.pth", "weights/../model.pth", "weights//model.pth"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                s.checked(self.root, dict(self.entry, path=name), self.mapping)


if __name__ == "__main__":
    unittest.main(verbosity=2)
