"""Run the synthetic CPU checks in isolated child processes; never train research models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
TESTS = ("test_shared_resources", "test_main_data_preparation", "test_main_global_training",
    "test_main_native", "test_main_body_preparation", "test_main_body",
    "test_external_runtime", "test_external_preparation", "test_external_training",
    "test_external_body_preparation", "test_external_body_training")


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")


def run(output, workers=2):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new check directory")
    if output.is_relative_to(PACKAGE) or PACKAGE.is_relative_to(output):
        raise ValueError("Checks must be outside the source directory")
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    env.pop("PYTHONPATH", None)
    started = time.monotonic()

    def child(name):
        if name == "model_contracts":
            args = [str(PACKAGE / "reproduce.py"), "smoke", "--output", str(output / (name + ".json"))]
        elif name == "test_shared_resources":
            args = [str(PACKAGE / (name + ".py"))]
        else:
            args = [str(PACKAGE / (name + ".py")), "--output", str(output / (name + ".json"))]
        result = subprocess.run([sys.executable, *args], cwd=output, env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=300)
        write(output / (name + ".stdout.log"), result.stdout)
        write(output / (name + ".stderr.log"), result.stderr)
        counts = re.findall(r"Ran (\d+) tests? in", result.stderr)
        item = dict(name=name, returncode=result.returncode,
                    tests=sum(map(int, counts)), status="PASS" if result.returncode == 0 else "FAIL")
        if result.returncode == 0 and name != "test_shared_resources":
            report = json.loads((output / (name + ".json")).read_text("utf-8"))
            if report.get("status") != "PASS" or report.get("gpu_initialized", False):
                item["status"] = "FAIL"
        print(f'{item["status"]} {name}: {item["tests"]} tests', flush=True)
        return item

    with ThreadPoolExecutor(max_workers=workers) as pool:
        checks = list(pool.map(child, [*TESTS, "model_contracts"]))
    with (PACKAGE / "contents.json").open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = dict(status="PASS" if all(x["status"] == "PASS" for x in checks) else "FAIL",
        source_contents_sha256=source_hash, suites=checks, tests=sum(x["tests"] for x in checks),
        gpu_enabled=False, research_training_started=False, research_pixels_read=False,
        seconds=round(time.monotonic() - started, 3))
    write(output / "report.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "suites"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.workers)["status"] == "PASS" else 1)
