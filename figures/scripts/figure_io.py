"""Load immutable figure source data without access to the training workspace."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or ":" in name or "\\" in name or ".." in p.parts:
        raise ValueError("Expected a contained relative resource path")
    target = (root / p).resolve()
    target.relative_to(root.resolve())
    return target


def asset_directory(name):
    root = Path(os.environ.get("POREMORPH_FIGURE_ASSETS", ROOT / "figure_assets")).resolve()
    directory = safe_path(root, name)
    catalog = json.loads((ROOT / "asset_catalog.json").read_text("utf-8"))
    matching = [r for r in catalog["files"] if r["path"].startswith(name.rstrip("/") + "/")]
    if not matching:
        raise ValueError("Unknown image-asset group")
    for record in matching:
        path = safe_path(root, record["path"])
        if not path.is_file():
            raise FileNotFoundError(
                "Image-derived figure inputs are separate from the code/statistics package. "
                "Set POREMORPH_FIGURE_ASSETS to the authorized asset directory."
            )
        if digest(path) != record["sha256"]:
            raise ValueError(f"Figure asset changed: {record['path']}")
    return directory


def load_snapshot(name):
    """Restore exact numeric dtypes/shapes and optional, separately stored images."""
    contents = json.loads((ROOT / "contents.json").read_text("utf-8"))
    records = {r["path"]: r for r in contents["files"]}
    member = f"source_data/{name}.json"
    path = safe_path(ROOT, member)
    if member not in records or digest(path) != records[member]["sha256"]:
        raise ValueError("Figure source data differ from the recorded package")
    payload = json.loads(path.read_text("utf-8"))
    arrays = None

    def restore(value):
        nonlocal arrays
        if isinstance(value, dict) and value.get("__kind__") == "array":
            result = np.asarray(value["values"], dtype=value["dtype"])
            if list(result.shape) != value["shape"]:
                raise ValueError("Source-data array shape differs")
            return result
        if isinstance(value, dict) and value.get("__kind__") == "image_array":
            if arrays is None:
                directory = asset_directory(f"snapshots/{name}")
                with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
            result = arrays[value["key"]].copy()
            if str(result.dtype) != value["dtype"] or list(result.shape) != value["shape"]:
                raise ValueError("Image source shape/dtype differs")
            if hashlib.sha256(result.tobytes()).hexdigest() != value["array_sha256"]:
                raise ValueError("Image source pixels differ")
            return result
        if isinstance(value, dict) and value.get("__kind__") == "tuple":
            return tuple(restore(item) for item in value["items"])
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return restore(payload)
