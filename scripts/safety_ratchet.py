#!/usr/bin/env python3
"""Ratchet the count of `unsafe` blocks that carry no `// SAFETY:` comment.

Every `unsafe {` block is expected to sit directly under a `// SAFETY:`
comment discharging its obligation. Brownfield debt is held by a committed
per-file baseline that may only decrease: `check` fails when any file exceeds
its baseline (a new file counts against a baseline of zero) and reports files
whose count fell below it so the baseline can be tightened with `baseline`.

An `unsafe fn` signature is not a site: the obligation lives at its callers'
`unsafe {}` blocks (edition 2024 `unsafe_op_in_unsafe_fn`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
BASELINE = REPOSITORY / "scripts" / "safety_ratchet_baseline.json"
SOURCE_ROOTS = tuple(sorted(REPOSITORY.glob("crates/*/src")))

UNSAFE_BLOCK = re.compile(r"\bunsafe\s*\{")
SAFETY_COMMENT = re.compile(r"^\s*//[/!]?\s*SAFETY\b")
ATTRIBUTE_OR_COMMENT = re.compile(r"^\s*(#\[|#!\[|//)")


def uncommented_sites(text: str) -> int:
    """Count `unsafe {` blocks whose preceding comment run carries no SAFETY."""
    lines = text.splitlines()
    count = 0
    for index, line in enumerate(lines):
        if not UNSAFE_BLOCK.search(line) or line.lstrip().startswith("//"):
            continue
        # A SAFETY comment on the same line, or in the contiguous run of
        # comment and attribute lines directly above, discharges the site.
        if "SAFETY" in line:
            continue
        probe = index - 1
        discharged = False
        while probe >= 0 and ATTRIBUTE_OR_COMMENT.match(lines[probe]):
            if SAFETY_COMMENT.match(lines[probe]):
                discharged = True
                break
            probe -= 1
        if not discharged:
            count += 1
    return count


def measure() -> dict[str, int]:
    """Count uncommented sites per source file, repository-relative, POSIX paths."""
    counts: dict[str, int] = {}
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.rs")):
            sites = uncommented_sites(path.read_text(encoding="utf-8"))
            if sites:
                counts[path.relative_to(REPOSITORY).as_posix()] = sites
    return counts


def load_baseline() -> dict[str, int]:
    """Read the committed baseline; a missing file is an empty baseline."""
    if not BASELINE.is_file():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def write_baseline(counts: dict[str, int]) -> None:
    """Write the baseline deterministically (sorted keys, trailing newline)."""
    BASELINE.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(counts: dict[str, int], baseline: dict[str, int]) -> int:
    """Fail on any file above its baseline; report tightening opportunities."""
    regressions = {
        path: (baseline.get(path, 0), sites)
        for path, sites in counts.items()
        if sites > baseline.get(path, 0)
    }
    slack = {
        path: (allowed, counts.get(path, 0))
        for path, allowed in baseline.items()
        if counts.get(path, 0) < allowed
    }
    total = sum(counts.values())
    allowed_total = sum(baseline.values())
    for path, (allowed, sites) in sorted(regressions.items()):
        print(f"error: {path}: {sites} uncommented unsafe blocks, baseline {allowed}", file=sys.stderr)
    for path, (allowed, sites) in sorted(slack.items()):
        print(f"note: {path}: {sites} uncommented unsafe blocks, baseline {allowed} (tighten)")
    print(f"SAFETY ratchet: {total} uncommented unsafe blocks against a baseline of {allowed_total}.")
    if regressions:
        print(
            "error: add a `// SAFETY:` comment above each new unsafe block; "
            "the baseline only decreases (`python scripts/safety_ratchet.py baseline`).",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    """Run the selected ratchet operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "baseline"))
    arguments = parser.parse_args()

    counts = measure()
    if arguments.mode == "baseline":
        baseline = load_baseline()
        raised = {path for path, sites in counts.items() if sites > baseline.get(path, 0)}
        if baseline and raised:
            for path in sorted(raised):
                print(f"error: {path} would raise the baseline", file=sys.stderr)
            return 1
        write_baseline(counts)
        print(f"SAFETY ratchet baseline written: {sum(counts.values())} sites in {len(counts)} files.")
        return 0
    return check(counts, load_baseline())


if __name__ == "__main__":
    sys.exit(main())
