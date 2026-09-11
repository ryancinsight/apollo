"""Run the production lane probe in two replicated, counterbalanced orders.

Writes a fixed set of eight reports and executable hashes. Each invocation has
a 60-second deadline; the complete campaign has the 300-second suite budget.
The destination must be new so a later run cannot overwrite its evidence.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time


def run(baseline: Path, candidate: Path, destination: Path) -> None:
    executables = {"baseline": baseline.resolve(strict=True),
                   "candidate": candidate.resolve(strict=True)}
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {name: {"path": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in executables.items()}
    manifest["priority"] = "high" if os.name == "nt" else "inherited"
    manifest["platform"] = platform.platform()
    manifest["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["order"] = ["baseline", "candidate", "candidate", "baseline"] * 2
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.monotonic()
    for index, name in enumerate(manifest["order"]):
        remaining = 300 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("lane probe campaign exceeded 300 seconds")
        flags = subprocess.HIGH_PRIORITY_CLASS if os.name == "nt" else 0
        report = destination / str(index)
        report.mkdir()
        if os.name == "nt":
            host = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,LoadPercentage | ConvertTo-Json"],
                capture_output=True, text=True, check=True, timeout=min(10, remaining))
            (report / "host.json").write_text(host.stdout, encoding="utf-8")
        remaining = 300 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("lane probe campaign exceeded 300 seconds")
        completed = subprocess.run([str(executables[name])], capture_output=True,
                                   timeout=min(60, remaining), creationflags=flags,
                                   check=False, text=True)
        (report / "lanes.csv").write_text(completed.stdout, encoding="utf-8")
        (report / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        completed.check_returncode()
        if time.monotonic() - started >= 300:
            raise TimeoutError("lane probe campaign exceeded 300 seconds")
        print(f"{index}: {name}: {report}", flush=True)
    manifest["duration_seconds"] = time.monotonic() - started
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if time.monotonic() - started >= 300:
        raise TimeoutError("lane probe campaign exceeded 300 seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    run(args.baseline, args.candidate, args.destination)
