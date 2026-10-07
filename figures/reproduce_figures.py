"""Render only the nine current PoreMorph manuscript figures from source data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def verify_contents():
    manifest = json.loads((ROOT / "contents.json").read_text("utf-8"))
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        path.relative_to(ROOT)
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != row["sha256"]:
            raise ValueError(f"Package source changed: {row['path']}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", choices=("all", "numeric", "F1", "F2", "F3", "F4", "F5", "S1", "S2", "S3", "S4"), default="numeric")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    args = p.parse_args()
    manifest = verify_contents()
    from matplotlib import font_manager
    # Silent font fallback would change line lengths and the accepted layout.
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing figure exports are preserved")
    output.mkdir(parents=True)
    catalog = json.loads((ROOT / "figure_catalog.json").read_text("utf-8"))
    wanted = set(catalog["figures"]) if args.figures == "all" else (
        {"F4", "F5", "S2", "S3"} if args.figures == "numeric" else {args.figures})
    # F5 and S2 are rendered by the same retained two-output script.
    if wanted & {"F5", "S2"}:
        wanted |= {"F5", "S2"}
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if args.assets:
        env["POREMORPH_FIGURE_ASSETS"] = str(args.assets.resolve())
    runs = []
    seen = set()
    for figure_id, entry in catalog["figures"].items():
        if figure_id not in wanted or entry["script"] in seen:
            continue
        seen.add(entry["script"])
        print(f"Rendering {', '.join(k for k, v in catalog['figures'].items() if v['script'] == entry['script'])}", flush=True)
        command = [sys.executable, str(ROOT / entry["script"]), entry["output_argument"], str(output)]
        if figure_id == "S4":
            command += ["--selection", str(ROOT / "source_data/external_selection.json")]
        done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                              text=True, encoding="utf-8", check=False)
        (output / (figure_id + ".stdout.log")).write_text(done.stdout, encoding="utf-8")
        (output / (figure_id + ".stderr.log")).write_text(done.stderr, encoding="utf-8")
        if done.returncode:
            raise RuntimeError(f"{figure_id} rendering failed; see its output logs")
        runs.append(dict(script=entry["script"], returncode=done.returncode))
    report = dict(status="PASS", figures=sorted(wanted), runs=runs,
                  source_contents_sha256=hashlib.sha256((ROOT / "contents.json").read_bytes()).hexdigest(),
                  model_inference=False, training=False,
                  source_version=manifest["version"], visual_check_required=True)
    (output / "render_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
