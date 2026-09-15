#!/usr/bin/env python3
"""Attribute emitted instructions per symbol, and diff two revisions.

Board items routinely carry acceptance clauses about emitted code — "no
attributed code-size growth", "no added production scratch", "the fused
specialization is unchanged". No CI job produces that evidence and none can:
benchmarks and codegen inspection are local instruments here. This makes the
measurement repeatable so the clause is discharged by a command rather than by
an argument.

Three modes:

    measure <asm>                       per-symbol instruction counts
    compare <base.s> <change.s> [name]  per-symbol delta, optionally filtered
    loops <regex> <asm>...              per-loop census of every matching symbol

The loop census reads each symbol's body with its labels kept, takes every
label a backward jump targets as a loop, and reports the loop's instruction
count by class (vector add/sub, multiply and FMA, shuffles, loads with the
share that is spill traffic, stores, calls) with a latency-weighted
critical-path estimate against its issue bound — a loop whose path is far
below its bound is throughput-bound, and on a core that splits 256-bit
instructions into two micro-ops that bound doubles. Several assembly files
serve the codegen-unit split of a test harness.

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

LABEL = re.compile(r"^(\.?L[\w$.]+):")
JUMP = re.compile(r"^\s*j\w+\s+(\.?L[\w$.]+)")
CALL = re.compile(r"^\s*callq?\s+(\S+)")
# Instruction classes by mnemonic; a vector instruction is one on a v-prefixed
# packed or scalar-in-vector mnemonic.
CLASSES = (
    ("addsub", re.compile(r"^v(add|sub|addsub|subadd)p[sd]\b")),
    ("mul", re.compile(r"^vmulp[sd]\b")),
    ("fma", re.compile(r"^vf(n?m(add|sub)|maddsub|msubadd)\d*p[sd]\b")),
    ("shuffle", re.compile(r"^v(perm\w*|shufp[sd]|unpck[lh]p[sd]|insert[fi]128|extract[fi]128|blendp[sd]|blendv\w*|movs[lh]dup|movddup|broadcast\w*|palignr|pshuf\w*|punpck\w*)\b")),
    ("logic", re.compile(r"^v(and|andn|or|xor)p[sd]\b")),
)
# Golden Cove class latencies: add/sub 3, multiply and FMA 4, in-lane shuffle 1,
# cross-lane shuffle 3, load 5, logic 1; a store ends a chain.
LATENCY = {"addsub": 3, "mul": 4, "fma": 4, "shuffle": 1, "cross": 3, "logic": 1, "load": 5}
CROSS_LANE = re.compile(r"^v(perm2[fi]128|perm[dq]|permp[sd]|insert[fi]128|extract[fi]128|broadcast[sf]\w*)\b")

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


def labeled_bodies(path: Path) -> dict[str, list[str]]:
    """Map each mangled symbol to its body lines with labels kept in place."""
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
        if current is None:
            continue
        if LABEL.match(line) or (line.startswith("\t") and not line.lstrip().startswith(".")):
            buffer.append(line)
    if current is not None:
        out[current] = buffer
    return out


def instruction_parts(line: str) -> tuple[str, str]:
    """The mnemonic and its operand text."""
    text = line.strip()
    head, _, rest = text.partition("\t")
    if not rest:
        head, _, rest = text.partition(" ")
    return head, rest.strip()


def classify(mnemonic: str, operands: str) -> str | None:
    """The class of one instruction, or none for a non-vector one."""
    for name, pattern in CLASSES:
        if pattern.match(mnemonic):
            return name
    if mnemonic.startswith("vmov") or mnemonic.startswith("vbroadcast"):
        parts = top_level_operands(operands)
        source = parts[0]
        target = parts[-1]
        if "(" in target:
            return "store"
        if "(" in source:
            return "load"
        return "move"
    return None


def top_level_operands(operands: str) -> list[str]:
    """The operands split at the commas outside parentheses, so an indexed
    address `(%rax,%rcx,8)` stays one operand."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in operands:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return parts


def loops_of(body: list[str]) -> list[tuple[int, int]]:
    """Each loop as the instruction index range a backward jump closes."""
    labels: dict[str, int] = {}
    instructions: list[str] = []
    for line in body:
        match = LABEL.match(line)
        if match:
            labels[match.group(1)] = len(instructions)
            continue
        instructions.append(line)
    found = []
    for index, line in enumerate(instructions):
        match = JUMP.match(line)
        if match and match.group(1) in labels and labels[match.group(1)] <= index:
            found.append((labels[match.group(1)], index))
    return found


def census(instructions: list[str]) -> dict[str, int]:
    """Class counts of one instruction range, spill traffic apart."""
    tally: Counter[str] = Counter()
    for line in instructions:
        mnemonic, operands = instruction_parts(line)
        if CALL.match(line):
            tally["call"] += 1
            continue
        kind = classify(mnemonic, operands)
        if kind is None:
            continue
        if kind in ("load", "store") and "(%rsp)" in operands:
            tally["spill"] += 1
        tally[kind] += 1
        if kind == "shuffle" and CROSS_LANE.match(mnemonic):
            tally["cross"] += 1
    tally["vector"] = sum(tally[k] for k in ("addsub", "mul", "fma", "shuffle", "logic", "load", "store", "move"))
    return dict(tally)


def register_keys(operands: str) -> list[str]:
    """The vector registers and stack slots an operand list names."""
    keys = re.findall(r"%[xyz]mm\d+", operands)
    keys += [f"stack{m}" for m in re.findall(r"(-?\d*)\(%rsp\)", operands)]
    return keys


def critical_path(instructions: list[str]) -> int:
    """A latency-weighted longest path over the def-use graph of one
    iteration in program order; stores end chains, non-vector instructions
    carry no latency."""
    ready: dict[str, int] = {}
    longest = 0
    for line in instructions:
        mnemonic, operands = instruction_parts(line)
        kind = classify(mnemonic, operands)
        if kind is None:
            continue
        keys = register_keys(operands)
        if not keys:
            continue
        sources, target = keys[:-1] or keys, keys[-1]
        start = max((ready.get(k, 0) for k in sources), default=0)
        if kind == "store":
            longest = max(longest, start)
            continue
        latency = LATENCY.get("cross" if kind == "shuffle" and CROSS_LANE.match(mnemonic) else kind, 1)
        ready[target] = start + latency
        longest = max(longest, ready[target])
    return longest


def loops(pattern: str, paths: list[Path]) -> int:
    matcher = re.compile(pattern)
    shown = 0
    for path in paths:
        for symbol, body in labeled_bodies(path).items():
            if not matcher.search(symbol):
                continue
            instructions = [line for line in body if not LABEL.match(line)]
            print(f"{symbol}\n  instructions={len(instructions)}  file={path.name}")
            for start, end in loops_of(body):
                region = instructions[start : end + 1]
                tally = census(region)
                path_cycles = critical_path(region)
                vector = tally.get("vector", 0)
                print(
                    f"  loop {start:>5}-{end:<5} len={len(region):<5} vector={vector:<4}"
                    f" addsub={tally.get('addsub', 0):<3} mul={tally.get('mul', 0):<3} fma={tally.get('fma', 0):<3}"
                    f" shuffle={tally.get('shuffle', 0):<3} (cross {tally.get('cross', 0):<2})"
                    f" load={tally.get('load', 0):<3} store={tally.get('store', 0):<3} spill={tally.get('spill', 0):<3}"
                    f" call={tally.get('call', 0):<2} critical~{path_cycles:<4} issue~{(vector + 3) // 4}/{(vector + 2) // 3} (4/3-wide)"
                )
            shown += 1
    if not shown:
        print(f"no symbol matches {pattern!r}", file=sys.stderr)
        return 2
    return 0


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

    three = sub.add_parser("loops", help="per-loop census of every matching symbol")
    three.add_argument("pattern", help="regex on the mangled symbol")
    three.add_argument("asm", type=Path, nargs="+", help="assembly files (a harness splits over codegen units)")

    args = parser.parse_args()
    if args.mode == "measure":
        return measure(args.asm, args.name)
    if args.mode == "loops":
        return loops(args.pattern, args.asm)
    return compare(args.base, args.change, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
