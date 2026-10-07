"""Verify and install known local resource ZIPs without overwriting existing data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from verify_bundle import ROOT, check_file, contained, read, require


def plan_archive(archive, destination, spec):
    check_file(archive, spec)
    role = spec["role"]
    require(role in ("labels", "weights"), "Unknown resource role")
    plan = []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        records = json.loads(z.read("archive_manifest.json"))["files"]
        require(set(names) == set(records) | {"archive_manifest.json"}, "Unexpected archive membership")
        for name in names:
            # Validate the original name even when it will be namespaced.
            contained(destination, name)
            member = z.getinfo(name)
            require(not member.is_dir(), "Directory entries are not expected")
            require((member.external_attr >> 16) & 0o170000 != 0o120000, "Symlink member")
            with z.open(name) as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if name in records:
                require(sha == records[name]["sha256"] and member.file_size == records[name]["bytes"],
                        f"Member differs: {name}")
            relative = name if name.startswith("resources/") else f"resource_metadata/{role}/{name}"
            path = contained(destination, relative)
            prefix = "resources" if name.startswith("resources/") else "resource_metadata"
            require(path.relative_to(destination.resolve()).parts[0] == prefix,
                    "Resolved resource path leaves its intended subtree")
            item = dict(path=path, member=name, archive=archive, sha256=sha, bytes=member.file_size)
            if path.exists():
                check_file(path, item)
            plan.append(item)
    return plan


def install(archives, destination, specs):
    destination = destination.resolve()
    require(destination != ROOT / "code" and destination != ROOT / "figures", "Invalid installation root")
    plan = []
    for role, path in archives.items():
        spec = next(s for s in specs if s["role"] == role)
        plan.extend(plan_archive(Path(path).resolve(), destination, spec))
    expected = {}
    for row in plan:
        if row["path"] in expected:
            require(expected[row["path"]] == row["sha256"], "Conflicting resource paths")
        expected[row["path"]] = row["sha256"]
    # Complete preflight precedes all writes, including checks of existing files.
    written, reused = 0, 0
    for role, archive in archives.items():
        resolved = Path(archive).resolve()
        with zipfile.ZipFile(resolved) as z:
            for row in (r for r in plan if r["archive"] == resolved):
                target = row["path"]
                if target.exists():
                    check_file(target, row)
                    reused += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(row["member"]) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                check_file(target, row)
                written += 1
    return dict(status="PASS", files_written=written, identical_existing_files=reused,
                model_loaded=False, pixels_decoded=False, downloaded=False, uploaded=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-archive", type=Path)
    parser.add_argument("--weights-archive", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    archives = {key: value for key, value in (("labels", args.labels_archive), ("weights", args.weights_archive))
                if value is not None}
    if not archives:
        parser.error("Provide at least one resource archive")
    specs = read(ROOT / "release_assets.json")["assets"]
    print(json.dumps(install(archives, args.destination, specs), indent=2))
