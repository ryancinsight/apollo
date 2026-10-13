#!/usr/bin/env python3
"""Measure unchanged lane-controlled binaries on one native Linux CPU.

Two phase-reversed replications produce four A/B comparisons. The native
comparator evaluates the complete case family with its existing simultaneous
median intervals and between-run spread rule. Reverse comparison reports
supported improvements; an absence of regressions alone is not a retention
verdict. Raw reports and executable identities remain alongside both verdicts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


CAMPAIGN = (
    ("first", "baseline-first", "baseline"),
    ("first", "baseline-first", "candidate"),
    ("first", "candidate-first", "candidate"),
    ("first", "candidate-first", "baseline"),
    ("second", "candidate-first", "candidate"),
    ("second", "candidate-first", "baseline"),
    ("second", "baseline-first", "baseline"),
    ("second", "baseline-first", "candidate"),
)


def measure(executable: Path, directory: Path, environment: dict[str, str]) -> None:
    directory.mkdir()
    with (directory / "rustfft_comparison.csv").open("w") as report, (
        directory / "measurement.log"
    ).open("w") as log:
        subprocess.run(
            [str(executable)], env=environment, stdout=report, stderr=log,
            check=True, timeout=45,
        )
    with (directory / "rustfft_comparison.csv").open() as report:
        rows = list(csv.DictReader(report))
    assert len(rows) == 104, "the unchanged 26-size, four-engine sweep is required"
    assert len({row["case"] for row in rows}) == len(rows), "duplicate cases"
    for row in rows:
        samples = [int(value) for value in row["ordered_samples_ps"].split(";")]
        assert len(samples) == 100 and samples == sorted(samples), row["case"]
        assert int(row["iterations_per_sample"]) > 0, row["case"]


def compare(evidence: Path, reverse: bool) -> str:
    arguments = [str(evidence / "bin/compare"), "compare-replicated-counterbalanced"]
    for replication in ("first", "second"):
        for order in ("baseline-first", "candidate-first"):
            for role in ("baseline", "candidate"):
                source_order = ({"baseline-first": "candidate-first", "candidate-first": "baseline-first"}[order]
                                if reverse else order)
                source = ({"baseline": "candidate", "candidate": "baseline"}[role]
                          if reverse else role)
                arguments.extend((
                    f"--{replication}-{order}-{role}-directory",
                    str(evidence / f"{replication}-{source_order}-{source}"),
                ))
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    # Exit 1 represents a supported slowdown; malformed evidence is also
    # rejected by the comparator, and must never be called a timing verdict.
    if result.returncode not in (0, 1) or (
        result.returncode == 1 and "candidate is slower in all four comparisons" not in output
    ):
        raise RuntimeError(f"native comparison failed: {output}")
    name = "improvements" if reverse else "regressions"
    (evidence / f"{name}.txt").write_text(output)
    return output


def main(evidence: Path) -> None:
    evidence = evidence.resolve(strict=True)
    assert platform.machine() == "x86_64" and sys.platform == "linux"
    flags = Path("/proc/cpuinfo").read_text().split("flags", 1)[1].splitlines()[0].split()
    assert {"avx2", "fma"} <= set(flags), "native AVX2+FMA is required"
    available = sorted(os.sched_getaffinity(0))
    processor = available[1] if len(available) > 1 else available[0]
    os.sched_setaffinity(0, {processor})
    assert os.sched_getaffinity(0) == {processor}
    environment = dict(os.environ, APOLLO_BENCH_MODE="measurement")
    environment.pop("APOLLO_BENCH_FULL_SWEEP", None)
    # Hermes does not implement Linux binding at the locked revision. The
    # kernel affinity above pins this process and every child instead.
    environment.pop("APOLLO_BENCH_PROCESSOR", None)
    binaries = {role: evidence / "bin" / role for role in ("baseline", "candidate")}
    identities = {
        role: hashlib.sha256(binary.read_bytes()).hexdigest()
        for role, binary in binaries.items()
    }
    (evidence / "execution.json").write_text(json.dumps({
        "processor": processor, "affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(), "binary_sha256": identities,
        "mode": "measurement", "orders": ["ABBA", "BAAB"],
    }, indent=2) + "\n")
    for role, executable in binaries.items():
        measure(executable, evidence / f"discarded-warm-{role}", environment)
    summary = [
        "# Native FFT121 evaluation", "",
        f"Linux CPU {processor}; externally pinned affinity inherited by each binary.",
        "Address lanes and timed closures are unchanged from PR #649.", "",
        "| Run | Apollo f32 ns | RustFFT f32 ns | Ratio | Apollo f64 ns | RustFFT f64 ns |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for replication, order, role in CAMPAIGN:
        name = f"{replication}-{order}-{role}"
        measure(binaries[role], evidence / name, environment)
        with (evidence / name / "rustfft_comparison.csv").open() as report:
            target = {row["case"].split("/")[1]: int(row["median_ps"]) / 1000
                      for row in csv.DictReader(report) if row["case"].endswith("/121")}
        summary.append(
            f"| {name} | {target['apollo_f32']:.3f} | {target['rustfft_f32']:.3f} | "
            f"{target['apollo_f32'] / target['rustfft_f32']:.4f} | "
            f"{target['apollo_f64']:.3f} | {target['rustfft_f64']:.3f} |"
        )
    regressions = compare(evidence, reverse=False)
    improvements = compare(evidence, reverse=True)
    for label, output in (("Regressions", regressions), ("Improvements (roles reversed)", improvements)):
        summary.extend(("", f"## {label}", "", "```text", output.rstrip(), "```"))
    (evidence / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: fft121_evaluate.py <evidence-directory>")
    main(Path(sys.argv[1]))
