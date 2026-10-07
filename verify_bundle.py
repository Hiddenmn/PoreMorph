"""Read-only source and optional resource-archive checks; standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    require(isinstance(relative, str) and bool(relative), "Missing relative path")
    require("\\" not in relative and ":" not in relative, "Nonportable path")
    require(all(p not in ("", ".", "..") for p in relative.split("/")), "Unsafe path")
    require(not PurePosixPath(relative).is_absolute(), "Absolute path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes source root")
    return path


def read(path):
    return json.loads(path.read_text("utf-8"))


def check_file(path, item):
    require(path.is_file(), f"Missing file: {path.name}")
    require(path.stat().st_size == item["bytes"], f"Length differs: {path.name}")
    require(digest(path) == item["sha256"], f"Hash differs: {path.name}")


def verify(labels=None, weights=None):
    bundle = read(ROOT / "bundle.json")
    for item in bundle["files"]:
        check_file(contained(ROOT, item["path"]), item)
    component_counts = {}
    for component in ("code", "figures"):
        folder = ROOT / component
        manifest = read(folder / "contents.json")
        for item in manifest["files"]:
            check_file(contained(folder, item["path"]), item)
        component_counts[component] = len(manifest["files"])
    assets = read(ROOT / "release_assets.json")
    checked = []
    for role, archive in (("labels", labels), ("weights", weights)):
        if archive is not None:
            item = next(x for x in assets["assets"] if x["role"] == role)
            check_file(Path(archive).resolve(), item)
            checked.append(role)
    return dict(status="PASS", source_files=len(bundle["files"]),
                component_files=component_counts, archives_checked=checked,
                resource_archives_supplied=bool(checked),
                image_pixels_read=False, model_loaded=False, uploaded=False,
                publication_ready=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-archive", type=Path)
    parser.add_argument("--weights-archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.labels_archive, args.weights_archive), indent=2))

