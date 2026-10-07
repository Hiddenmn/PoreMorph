"""Resolve exact original resources or explicitly mapped model-state exports.

The exported weights are suitable for inference and fresh optimization from
selected parameters. They never supply missing optimizer/resume metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(root, entry, exports=None):
    root = Path(root).resolve()
    name = entry["path"]
    posix = PurePosixPath(name)
    if (not name or posix.is_absolute() or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Use a contained, project-relative resource path")
    path = (root / name).resolve()
    path.relative_to(root)
    actual = digest(path)
    if actual == entry["sha256"]:
        return path
    exports = exports or Path(__file__).with_name("resource_exports.json")
    mapping = json.loads(Path(exports).read_text(encoding="utf-8"))
    if (mapping["format"] != "verified-model-state-exports-v1"
            or mapping["optimizer_resume_supported"] is not False):
        raise ValueError("Unsupported model-state export mapping")
    matches = [item for item in mapping["checkpoints"] if item["path"] == name
               and item["original_sha256"] == entry["sha256"]]
    if (len(matches) != 1 or matches[0]["sha256"] != actual
            or matches[0]["bytes"] != path.stat().st_size
            or matches[0]["tensor_values_shapes_dtypes_strides_and_module_metadata_exact"] is not True):
        raise ValueError(f"Resource differs from its original and verified export: {name}")
    return path
