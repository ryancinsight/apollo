#!/usr/bin/env python3
"""Sync `[workspace.dependencies]` from the candidate manifest onto the baseline.

# The trap this exists for

The benchmark-regression job measures the baseline's `apollo-fft` source with
the *candidate's* instrument, so it copies the candidate's member manifests
into the baseline tree and pins the candidate's `Cargo.lock`. A member manifest
that inherits a workspace dependency (`themis = { workspace = true }`) is then
resolved against the *baseline's* root manifest, which never declared it:

    error: failed to load manifest for workspace member `.../apollo-czt`
      error inheriting `themis` from workspace root manifest's
      `workspace.dependencies.themis`

The step used to sync a hand-maintained allowlist of dependency names. An
allowlist is a second source of truth for the dependency set, so it drifts the
moment a PR adds a workspace dependency — and it fails *every* such PR, in the
harness rather than in the code under test, roughly fifteen seconds in and
before anything compiles. That looks like a benchmark regression and is not
one.

This script derives the set from the candidate manifest instead, so there is
nothing to keep in step. Existing keys are rewritten in place; keys the
baseline lacks are appended to its table. Entries are single-line by
convention (`name = "..."` or `name = { ... }`); a multi-line entry is
rejected loudly rather than silently half-copied.

Idempotent: running twice changes nothing. Verifies its own postcondition —
every candidate key present in the baseline with an identical spec — and exits
non-zero if that does not hold.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TABLE = "[workspace.dependencies]"
ENTRY = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)\s*=\s*(?P<spec>.+)$")
SECTION = re.compile(r"^\[")


def table_bounds(lines: list[str], manifest: Path) -> tuple[int, int]:
    """Return the half-open line range of the `[workspace.dependencies]` body."""
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == TABLE) + 1
    except StopIteration:
        raise SystemExit(f"{manifest}: no {TABLE} table")
    end = start
    while end < len(lines) and not SECTION.match(lines[end]):
        end += 1
    return start, end


def entries(lines: list[str], start: int, end: int, manifest: Path) -> dict[str, str]:
    """Map dependency name to its verbatim spec text over `lines[start:end]`."""
    found: dict[str, str] = {}
    for offset, line in enumerate(lines[start:end], start=start):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENTRY.match(stripped)
        if match is None:
            raise SystemExit(
                f"{manifest}:{offset + 1}: {TABLE} entry is not a single "
                f"`name = spec` line, which this sync cannot copy safely: {stripped!r}"
            )
        found[match.group("name")] = match.group("spec")
    return found


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <candidate-manifest> <baseline-manifest>")
    candidate_path, baseline_path = Path(sys.argv[1]), Path(sys.argv[2])

    candidate_lines = candidate_path.read_text(encoding="utf-8").splitlines()
    baseline_lines = baseline_path.read_text(encoding="utf-8").splitlines()

    c_start, c_end = table_bounds(candidate_lines, candidate_path)
    candidate = entries(candidate_lines, c_start, c_end, candidate_path)
    if not candidate:
        raise SystemExit(f"{candidate_path}: {TABLE} is empty")

    b_start, b_end = table_bounds(baseline_lines, baseline_path)
    baseline = entries(baseline_lines, b_start, b_end, baseline_path)

    rewritten, appended = [], []
    for index in range(b_start, b_end):
        stripped = baseline_lines[index].strip()
        match = ENTRY.match(stripped) if stripped and not stripped.startswith("#") else None
        if match is None:
            continue
        name = match.group("name")
        if name in candidate and candidate[name] != match.group("spec"):
            baseline_lines[index] = f"{name} = {candidate[name]}"
            rewritten.append(name)

    missing = [name for name in candidate if name not in baseline]
    if missing:
        tail = b_end
        while tail > b_start and not baseline_lines[tail - 1].strip():
            tail -= 1
        additions = [f"{name} = {candidate[name]}" for name in missing]
        baseline_lines[tail:tail] = additions
        appended.extend(missing)

    baseline_path.write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    # Postcondition: the baseline now declares every candidate dependency
    # identically, so no inherited member entry can fail to resolve.
    verify_lines = baseline_path.read_text(encoding="utf-8").splitlines()
    v_start, v_end = table_bounds(verify_lines, baseline_path)
    final = entries(verify_lines, v_start, v_end, baseline_path)
    for name, spec in candidate.items():
        if final.get(name) != spec:
            raise SystemExit(
                f"{baseline_path}: {name} is {final.get(name)!r} after sync, "
                f"expected {spec!r}"
            )

    print(
        f"workspace.dependencies synced: {len(candidate)} declared, "
        f"{len(rewritten)} rewritten, {len(appended)} appended"
        + (f" ({', '.join(sorted(appended))})" if appended else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
