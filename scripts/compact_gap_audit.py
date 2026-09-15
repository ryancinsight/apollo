#!/usr/bin/env python3
"""Compact the risk artifact to its findings.

`gap_audit.md` holds one `## ` section per finding. A finding with an anchor
(`<a id="..">` in its heading or body) is a record other artifacts cite — the
board, the ADRs, the memory notes — so it stays, cut to its heading, its
lead paragraph and every line that names a re-open trigger, within
`--section-lines`. A section without an anchor is the report-file genre that
grew here — delivery ledgers, sprint notes, benchmark reruns, "Closed Gaps"
— whose narrative lives in git and whose durable residue already moved to
its owner, so it is dropped whole; the sections named by `--keep` (the slop
pattern library) stay untrimmed. Anchors never change, so every inbound
link survives; the run reports the counts, and a second run over its own
output changes nothing.

Usage: python scripts/compact_gap_audit.py gap_audit.md [--section-lines 10]
       [--keep "Slop patterns"] [--out gap_audit.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INLINE_ANCHOR = re.compile(r'<a id="([^"]+)"></a>')
REOPEN = re.compile(r"re-?open", re.I)
NL = "\n"


def sections(text: str) -> tuple[list[str], list[list[str]]]:
    """The head lines before the first `## ` and each section's lines."""
    lines = text.split(NL)
    head: list[str] = []
    out: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("## "):
            if current is not None:
                out.append(current)
            current = [line]
        elif current is None:
            head.append(line)
        else:
            current.append(line)
    if current is not None:
        out.append(current)
    return head, out


def lead_paragraph(body: list[str]) -> list[str]:
    """The first run of non-blank lines after the heading."""
    lead: list[str] = []
    started = False
    for line in body:
        if line.strip():
            lead.append(line)
            started = True
        elif started:
            break
    return lead


def trim(section: list[str], limit: int) -> list[str]:
    """Heading, lead paragraph, and the re-open and anchor lines, within `limit`.

    A lead paragraph longer than its budget is cut at the last sentence end
    inside the budget rather than mid-sentence."""
    heading, body = section[0], section[1:]
    lead = lead_paragraph(body)
    marked = [line for line in body if REOPEN.search(line) or INLINE_ANCHOR.search(line)]
    budget = max(1, limit - 1 - len([line for line in marked if line not in lead]))
    if len(lead) > budget:
        cut = lead[:budget]
        ends = [i for i, line in enumerate(cut) if line.rstrip().endswith((".", ":", ")", "`"))]
        lead = cut[: ends[-1] + 1] if ends else cut[:1]
    kept = [heading] + lead + [line for line in marked if line not in lead]
    return kept[:limit] + [""]


def compact(text: str, limit: int, keep: list[str]) -> tuple[str, dict]:
    head, parts = sections(text)
    kept: list[str] = []
    report = {"kept": 0, "dropped": 0, "kept_whole": 0}
    for section in parts:
        heading = section[0]
        if any(name in heading for name in keep):
            kept.extend(section if section[-1] == "" else section + [""])
            report["kept_whole"] += 1
        elif any(INLINE_ANCHOR.search(line) for line in section):
            kept.extend(trim(section, limit))
            report["kept"] += 1
        else:
            report["dropped"] += 1
    while head and not head[-1].strip():
        head.pop()
    out = NL.join(head + [""] + kept).rstrip(NL) + NL
    before = set(INLINE_ANCHOR.findall(text))
    after = set(INLINE_ANCHOR.findall(out))
    report["anchors"] = (len(before), len(after), sorted(before - after))
    report["lines"] = (text.count(NL), out.count(NL))
    return out, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--section-lines", type=int, default=10)
    parser.add_argument("--keep", action="append", default=["Slop patterns"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    text = args.artifact.read_text(encoding="utf-8")
    out, report = compact(text, args.section_lines, args.keep)
    (args.out or args.artifact).write_text(out, encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(errors="replace")
    lost = report["anchors"][2]
    print(
        f"lines {report['lines'][0]} -> {report['lines'][1]}; kept {report['kept']} anchored,"
        f" {report['kept_whole']} whole, dropped {report['dropped']}; anchors"
        f" {report['anchors'][0]} -> {report['anchors'][1]}; lost {lost}"
    )
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
