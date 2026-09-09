#!/usr/bin/env python3
"""Attribute emitted instructions per symbol, and diff two revisions.

Board items routinely carry acceptance clauses about emitted code — "no
attributed code-size growth", "no added production scratch", "the fused
specialization is unchanged". No CI job produces that evidence and none can:
benchmarks and codegen inspection are local instruments here. This makes the
measurement repeatable so the clause is discharged by a command rather than by
an argument.

Two modes:

    measure <asm>                       per-symbol instruction counts
    compare <base.s> <change.s> [name]  per-symbol delta, optionally filtered

Producing the assembly, per revision, from a clean export::

    git archive <rev> | tar -x -C <dir>
    find <dir> -type f -exec touch {} +          # see the stale-macro note
    cargo rustc --manifest-path <dir>/Cargo.toml --release -p apollo-fft \
        --lib -- --emit=asm -C debuginfo=0

Two traps this exists to route around, both of which cost a session:

* A proc-macro built for one export is reused for the next under a shared
  `CARGO_TARGET_DIR`, so the second build silently compiles the first
  revision's macro output. Run `cargo clean -p apollo-fft-macros` between
  revisions; the tell is an error naming syntax that is absent from the tree.
* Grepping a release PE for a function name proves nothing either way: MSVC
  images carry no internal symbol table. Attribute from the assembly.

Instruction counts, not bytes: the emitted `.s` carries no encoded lengths.
Identical counts across every symbol of a library, together with identical
bodies, is the strong form of "unchanged codegen".
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

SYMBOL = re.compile(r"^(_R[A-Za-z0-9_$.]+):")


def bodies(path: Path) -> dict[str, list[str]]:
    """Map each mangled symbol to the instruction lines of its body."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = SYMBOL.match(line)
        if match:
            if current is not None:
                out[current] = buffer
            current, buffer = match.group(1), []
            continue
        if current is not None and line.startswith("\t") and not line.lstrip().startswith("."):
            buffer.append(line)
    if current is not None:
        out[current] = buffer
    return out


def counts(path: Path) -> Counter[str]:
    return Counter({name: len(body) for name, body in bodies(path).items()})


def measure(path: Path, name: str | None) -> int:
    sizes = counts(path)
    selected = {k: v for k, v in sizes.items() if name is None or name in k}
    for symbol in sorted(selected, key=lambda k: -selected[k]):
        print(f"{selected[symbol]:>8}  {symbol}")
    print(f"\n{len(selected)} symbols, {sum(selected.values())} instructions")
    return 0


def compare(base: Path, change: Path, name: str | None) -> int:
    before, after = bodies(base), bodies(change)
    keys = sorted(
        {k for k in set(before) | set(after) if name is None or name in k}
    )
    if not keys:
        print(f"no symbol matches {name!r}", file=sys.stderr)
        return 2
    identical = True
    total_before = total_after = 0
    for symbol in keys:
        b, a = before.get(symbol), after.get(symbol)
        nb, na = len(b or []), len(a or [])
        total_before += nb
        total_after += na
        same = b == a
        identical &= same
        if not same or nb != na:
            verdict = "GONE" if a is None else "NEW" if b is None else "DIFFERS"
            print(f"{verdict:>9}  {nb:>7} -> {na:<7} {symbol}")
    delta = total_after - total_before
    pct = (100.0 * delta / total_before) if total_before else 0.0
    print(
        f"\n{len(keys)} symbols  {total_before} -> {total_after} instructions"
        f"  delta {delta:+} ({pct:+.4f}%)"
    )
    print(f"all bodies byte-identical: {identical}")
    return 0 if identical else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    one = sub.add_parser("measure", help="per-symbol instruction counts")
    one.add_argument("asm", type=Path)
    one.add_argument("name", nargs="?", help="substring filter on the symbol")

    two = sub.add_parser("compare", help="per-symbol delta between revisions")
    two.add_argument("base", type=Path)
    two.add_argument("change", type=Path)
    two.add_argument("name", nargs="?", help="substring filter on the symbol")

    args = parser.parse_args()
    if args.mode == "measure":
        return measure(args.asm, args.name)
    return compare(args.base, args.change, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
