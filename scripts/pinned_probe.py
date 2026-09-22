#!/usr/bin/env python3
"""Run a pinned probe in replicated processes and read it as a median.

A pinned probe (`small_sizes_against_the_references_by_core_type`,
`large_sizes_against_the_references_by_core_type`, ...) prints one CSV row
per arm and length with the minimum and the ordered samples of one process.
Past the caches one process is not a reading: at 262144 five quiet runs
moved apollo's and RustFFT's arms independently by 4 to 8% between processes
while each process's samples stayed within 5%, a per-run state that a longer
budget cannot narrow. This runner executes the probe `--runs` times, keeps
every process's output beside a manifest, and reports each arm as the median
of the per-run minima with its spread, and each ratio against a reference as
the median of the per-run ratios with its range.

The campaign has the 300-second suite budget; each run has a bounded share
of it. The destination must be new so a later campaign cannot overwrite its
evidence. Under a stack umbrella whose cargo overlay must not apply, pass a
`--cwd` outside it (the manifest path is absolute).
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

CAMPAIGN_SECONDS = 300
REFERENCES = ("rustfft", "phastft")


def parse_rows(report: str) -> dict[tuple[str, str, str, int], int]:
    """The per-run minimum in picoseconds by (core, arm, scalar, length)."""
    rows = {}
    for line in report.splitlines():
        head, sep, rest = line.partition(",")
        if not sep or head.count("/") != 2:
            continue
        core_arm, length = head.rsplit("/", 1)
        core, arm_scalar = core_arm.split("/", 1)
        arm, dash, scalar = arm_scalar.rpartition("-")
        minimum = rest.split(",", 1)[0]
        if not dash or not length.isdigit() or not minimum.isdigit():
            continue
        rows[(core, arm, scalar, int(length))] = int(minimum)
    return rows


def summarize(runs: list[dict[tuple[str, str, str, int], int]]) -> list[dict]:
    """One record per apollo case: the median arm minimum over runs, its
    spread (max over min minus one), and per reference the median and range
    of the per-run ratios of apollo's minimum to the reference's."""
    if not runs:
        raise ValueError("a campaign summarizes at least one run")
    keys = sorted(k for k in runs[0] if k[1] == "apollo" and all(k in run for run in runs))
    records = []
    for core, _arm, scalar, length in keys:
        apollo = [run[(core, "apollo", scalar, length)] for run in runs]
        record = {
            "core": core,
            "scalar": scalar,
            "length": length,
            "runs": len(runs),
            "apollo_min_ps_median": int(statistics.median(apollo)),
            "apollo_min_ps_spread": max(apollo) / min(apollo) - 1.0,
        }
        for reference in REFERENCES:
            key = (core, reference, scalar, length)
            if not all(key in run for run in runs):
                continue
            ratios = [run[(core, "apollo", scalar, length)] / run[key] for run in runs]
            record[f"{reference}_ratio_median"] = statistics.median(ratios)
            record[f"{reference}_ratio_min"] = min(ratios)
            record[f"{reference}_ratio_max"] = max(ratios)
        records.append(record)
    return records


def render(records: list[dict]) -> str:
    lines = ["core,scalar,length,runs,apollo_min_ps_median,apollo_min_ps_spread,"
             + ",".join(f"{r}_ratio_median,{r}_ratio_min,{r}_ratio_max" for r in REFERENCES)]
    for record in records:
        cells = [record["core"], record["scalar"], str(record["length"]), str(record["runs"]),
                 str(record["apollo_min_ps_median"]), f"{record['apollo_min_ps_spread']:.4f}"]
        for reference in REFERENCES:
            for field in ("median", "min", "max"):
                value = record.get(f"{reference}_ratio_{field}")
                cells.append("" if value is None else f"{value:.4f}")
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def run(test: str, runs: int, destination: Path, manifest_path: Path, cwd: Path,
        target_dir: Path | None) -> list[dict]:
    if runs < 1:
        raise ValueError("a campaign needs at least one run")
    destination.mkdir(parents=True, exist_ok=False)
    selection = ["--manifest-path", str(manifest_path.resolve(strict=True)),
                 "-p", "apollo-fft", "--release", "--features", "kernel-strategy-bench",
                 "--run-ignored", "ignored-only", "-E", f"test({test})"]
    command = ["cargo", "nextest", "run", *selection, "--no-capture"]
    # Same build and selection, no test executed: this is what moves nextest's
    # own build out of the campaign clock below.
    settle = ["cargo", "nextest", "list", *selection]
    environment = dict(os.environ)
    if target_dir is not None:
        environment["CARGO_TARGET_DIR"] = str(target_dir)
    # The release test binary is built before the campaign clock starts: a
    # cold build takes minutes the 300-second budget is not for.
    build = ["cargo", "build", "--tests", "--manifest-path", str(manifest_path.resolve(strict=True)),
             "-p", "apollo-fft", "--release", "--features", "kernel-strategy-bench"]
    built = subprocess.run(build, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=cwd, env=environment, timeout=1800, check=False)
    (destination / "build.log").write_text(built.stderr, encoding="utf-8")
    built.check_returncode()
    # nextest builds before it runs, and under a shared target directory that
    # build blocks on the cargo file lock for as long as peers hold it. That
    # wait is neither measurement nor bounded, so it is spent here, against the
    # build timeout, rather than inside the campaign budget.
    listed = subprocess.run(settle, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=cwd, env=environment, timeout=1800,
                            check=False)
    (destination / "settle.log").write_text(listed.stderr, encoding="utf-8")
    listed.check_returncode()
    manifest = {
        "test": test,
        "runs": runs,
        "command": command,
        "cwd": str(cwd),
        "platform": platform.platform(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.monotonic()
    parsed = []
    for index in range(runs):
        remaining = CAMPAIGN_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"probe campaign exceeded {CAMPAIGN_SECONDS} seconds")
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", cwd=cwd, env=environment,
                                   timeout=remaining, check=False)
        (destination / f"run{index}.txt").write_text(completed.stdout, encoding="utf-8")
        (destination / f"run{index}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        completed.check_returncode()
        rows = parse_rows(completed.stdout)
        if not rows:
            raise RuntimeError(f"run {index} printed no probe rows; see run{index}.stderr.log")
        parsed.append(rows)
        print(f"run {index}: {len(rows)} rows", flush=True)
    records = summarize(parsed)
    (destination / "summary.csv").write_text(render(records), encoding="utf-8")
    manifest["duration_seconds"] = time.monotonic() - started
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", help="the probe test's name, as nextest's filter matches it")
    parser.add_argument("destination", type=Path, help="a new directory for the campaign's evidence")
    parser.add_argument("--runs", type=int, default=5, help="processes to replicate (default 5)")
    parser.add_argument("--manifest-path", type=Path,
                        default=Path(__file__).resolve().parents[1] / "Cargo.toml")
    parser.add_argument("--cwd", type=Path, default=Path(__file__).resolve().parents[1],
                        help="where cargo runs; outside a stack umbrella whose overlay must not apply")
    parser.add_argument("--target-dir", type=Path, default=None,
                        help="CARGO_TARGET_DIR for the campaign (the stack's shared cache)")
    args = parser.parse_args()
    records = run(args.test, args.runs, args.destination, args.manifest_path, args.cwd,
                  args.target_dir)
    print(render(records), end="")


if __name__ == "__main__":
    main()
