"""Synthetic tests only; no research data or network calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from install_resources import install


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = self.root / "installed"

    def archive(self, role, entries=None, bad_member=False):
        entries = entries or {f"resources/{role}/item.bin": b"synthetic", "README.md": role.encode()}
        path = self.root / f"{role}.zip"
        manifest = {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name, data in entries.items()}
        if bad_member:
            manifest[next(iter(manifest))]["sha256"] = "0" * 64
        with zipfile.ZipFile(path, "w") as z:
            for name, data in entries.items():
                z.writestr(name, data)
            z.writestr("archive_manifest.json", json.dumps(dict(files=manifest)))
        spec = dict(role=role, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return path, spec

    def test_roles_keep_metadata_separate(self):
        a, sa = self.archive("labels")
        b, sb = self.archive("weights")
        self.dest.mkdir()
        (self.dest / "README.md").write_bytes(b"project readme")
        report = install(dict(labels=a, weights=b), self.dest, [sa, sb])
        self.assertEqual(report["files_written"], 6)
        self.assertEqual((self.dest / "README.md").read_bytes(), b"project readme")
        self.assertEqual((self.dest / "resource_metadata/weights/README.md").read_bytes(), b"weights")

    def test_identical_existing_is_reused(self):
        path, spec = self.archive("labels")
        install(dict(labels=path), self.dest, [spec])
        report = install(dict(labels=path), self.dest, [spec])
        self.assertEqual(report["files_written"], 0)
        self.assertEqual(report["identical_existing_files"], 3)

    def test_existing_difference_fails_before_writes(self):
        path, spec = self.archive("labels")
        target = self.dest / "resources/labels/item.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"other")
        with self.assertRaises(ValueError):
            install(dict(labels=path), self.dest, [spec])
        self.assertEqual(target.read_bytes(), b"other")
        self.assertFalse((self.dest / "resource_metadata").exists())

    def test_wrong_archive_hash(self):
        path, spec = self.archive("labels")
        spec["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            install(dict(labels=path), self.dest, [spec])
        self.assertFalse(self.dest.exists())

    def test_wrong_member_hash(self):
        path, spec = self.archive("labels", bad_member=True)
        with self.assertRaises(ValueError):
            install(dict(labels=path), self.dest, [spec])
        self.assertFalse(self.dest.exists())

    def test_path_traversal(self):
        path, spec = self.archive("labels", {"../escape.bin": b"bad"})
        with self.assertRaises(ValueError):
            install(dict(labels=path), self.dest, [spec])
        self.assertFalse(self.dest.exists())

    def test_cross_archive_conflict(self):
        a, sa = self.archive("labels", {"resources/common.bin": b"a"})
        b, sb = self.archive("weights", {"resources/common.bin": b"b"})
        with self.assertRaises(ValueError):
            install(dict(labels=a, weights=b), self.dest, [sa, sb])
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
