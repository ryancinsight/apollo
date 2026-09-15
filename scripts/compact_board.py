#!/usr/bin/env python3
"""Compact the board to its budget.

Every `## ` heading is an item. Each keeps (or gains) a stable anchor, a
status from the closed set (todo, in-progress, blocked, review, done), and
a record within the budget: an open item at most fifteen lines, a done
item one line (anchor, identity, outcome) in the closing `# Done` section.
Narrative beyond that is dropped — git is the archive, and the commit that
lands a compaction names the pre-compaction revision as its parent. A
stale in-progress claim (last update before `--release-before`) is
released to todo with a note. Sprint and session sections (the report-file
genre inside the board) are dropped whole. Anchors never change, so every
inbound link survives; the run reports the before and after line counts,
the anchors, and what it released, normalized and dropped.

Usage: python scripts/compact_board.py backlog.md --today 2026-09-15
       [--release-before 2026-09-14] [--out backlog.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CLOSED = ("todo", "in-progress", "blocked", "review", "done")
STATUS_MAP = {
    "downgraded": "done", "measured": "done", "superseded": "done", "closed": "done",
    "landed": "done", "merged": "done", "in_progress": "in-progress", "wip": "in-progress",
}
SECTION = re.compile(
    r"^## (Closed in this sprint|Open in this sprint|Planned next|Delivered|Sprint|Session|Wave|Tier"
    r"|Watchpoints|Backlog|Open items|Done items|Notes)",
    re.I,
)
GLYPHS = re.compile("[←-⇿⌀-⏿─-➿⬀-⯿\U0001f000-\U0001faff]")
ANCHOR = re.compile(r'^<a id="([^"]+)"></a>\s*$')
STATUS = re.compile(r" — ([A-Za-z_-]+)(\s+\d{4}-\d{2}(-\d{2})?)?\s*$")
LANDED = re.compile(r"PR #\d+|pull/\d+|\b[0-9a-f]{8,40}\b|[Ll]anded|[Mm]erged|\[x\]|Outcome:\*\* built")
ITEM_ID = re.compile(r"^## [A-Z0-9][A-Z0-9-]{6,} ")
OPEN_LIMIT = 14
NL = "\n"


def split_items(text: str) -> tuple[list[str], list[dict]]:
    """The head lines and the items, each with the anchor line before it."""
    lines = text.split(NL)
    head: list[str] = []
    i = 0
    while i < len(lines) and not lines[i].startswith("## "):
        head.append(lines[i])
        i += 1
    pending: str | None = None
    while head and (not head[-1].strip() or ANCHOR.match(head[-1])):
        last = head.pop()
        match = ANCHOR.match(last)
        if match:
            pending = match.group(1)
    items: list[dict] = []
    current: dict | None = None
    for line in lines[i:]:
        if line.startswith("## "):
            if current:
                items.append(current)
            current = {"anchor": pending, "heading": line, "body": []}
            pending = None
            continue
        match = ANCHOR.match(line)
        if match:
            pending = match.group(1)
            continue
        if current is not None:
            current["body"].append(line)
    if current:
        items.append(current)
    return head, items


def assign_anchors(items: list[dict]) -> None:
    """Existing anchors stay; a missing one is the ID lowercased without its
    trailing date, kept unique."""
    used = {it["anchor"] for it in items if it["anchor"]}
    for it in items:
        if it["anchor"]:
            continue
        match = re.match(r"^## ([A-Z0-9][A-Z0-9-]*)", it["heading"])
        base = (match.group(1) if match else it["heading"][3:20]).lower()
        base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base)
        candidate, n = base, 2
        while candidate in used:
            candidate, n = f"{base}-{n}", n + 1
        used.add(candidate)
        it["anchor"] = candidate


def status_of(heading: str) -> tuple[str | None, str | None]:
    match = STATUS.search(heading)
    return (match.group(1).lower(), match.group(2)) if match else (None, None)


def records_of(body: list[str]) -> list[str]:
    """The top-level record lines: bullets with continuation lines folded
    in; blank lines dropped; a sub-heading ends the record."""
    out: list[str] = []
    for line in body:
        if not line.strip():
            continue
        if re.match(r"^#{3,} ", line):
            break
        if line.startswith("- ") or line.startswith("* "):
            out.append(line)
        elif out and (line.startswith("  ") or line.startswith("\t")):
            out[-1] = out[-1] + " " + line.strip()
        else:
            out.append(line)
    return out


def compact(text: str, today: str, release_before: str) -> tuple[str, dict]:
    """The compacted board and a report of what changed."""
    head, items = split_items(text)
    assign_anchors(items)
    report: dict = {"released": [], "normalized": [], "dropped": [], "items": len(items)}
    open_lines: list[str] = []
    done_lines: list[str] = []
    for it in items:
        heading = GLYPHS.sub("", it["heading"]).replace("  ", " ").rstrip()
        status, date = status_of(heading)
        body_text = " ".join(line for line in it["body"] if line.strip())
        checklist_only = bool(body_text.strip()) and all(
            re.match(r"^\s*(- \[[x ]\]|\* \[[x ]\])", line) for line in it["body"] if line.strip()
        )
        if status is None and (SECTION.match(heading) or (checklist_only and not ITEM_ID.match(heading))):
            report["dropped"].append(heading[3:80])
            continue
        if status is None:
            status = "done" if LANDED.search(body_text[:600]) else "todo"
            heading = heading + " — " + status
            report["normalized"].append((it["anchor"], f"none -> {status}"))
        elif status not in CLOSED:
            new = STATUS_MAP.get(status, "todo")
            heading = re.sub(" — " + re.escape(status) + r"(\s+\d{4}-\d{2}(-\d{2})?)?\s*$", " — " + new, heading)
            report["normalized"].append((it["anchor"], f"{status} -> {new}"))
            status = new
        elif date:
            heading = re.sub(r"(\s+\d{4}-\d{2}(-\d{2})?)\s*$", "", heading)
        records = records_of(it["body"])
        if status == "in-progress":
            match = re.search(r"last-update:\*\*\s*(\d{4}-\d{2}-\d{2})", " ".join(records))
            last = match.group(1) if match else None
            if last is not None and last < release_before:
                heading = heading.replace(" — in-progress", " — todo")
                records = [r for r in records if "Integrator:" not in r]
                records.append(
                    f"- **Claim released:** {today}, the last update {last} older than the stale-claim window;"
                    " reclaim by re-syncing the board."
                )
                report["released"].append((it["anchor"], last))
                status = "todo"
        if status == "done":
            match = re.match(r"^## (\S+) — (.*) — done$", heading)
            ident, title = (match.group(1), match.group(2)) if match else (heading[3:], "")
            record = re.sub(r"^[-*] ", "", records[0]).strip() if records else ""
            done_lines.append(f'<a id="{it["anchor"]}"></a>- **{ident}** — {title}. {record}')
            continue
        open_lines.append(f'<a id="{it["anchor"]}"></a>')
        open_lines.append(heading)
        open_lines.extend(records[:OPEN_LIMIT])
        open_lines.append("")
    closing = [
        "# Done",
        "",
        "One line an item: the anchor, the identity, the outcome with its commit or PR; the narrative lives in git.",
        "",
    ]
    out = NL.join(head + [""] + open_lines + closing + done_lines).rstrip(NL) + NL
    report["lines"] = (text.count(NL), out.count(NL))
    before = set(re.findall(r'<a id="([^"]+)"></a>', text))
    after = set(re.findall(r'<a id="([^"]+)"></a>', out))
    report["anchors"] = (len(before), len(after), sorted(before - after))
    return out, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("board", type=Path)
    parser.add_argument("--today", required=True, help="the date written into released claims")
    parser.add_argument("--release-before", default=None, help="release in-progress claims last updated before this date")
    parser.add_argument("--out", type=Path, default=None, help="destination (default: the board in place)")
    args = parser.parse_args()
    text = args.board.read_text(encoding="utf-8")
    out, report = compact(text, args.today, args.release_before or args.today)
    (args.out or args.board).write_text(out, encoding="utf-8", newline="\n")
    lost = report["anchors"][2]
    sys.stdout.reconfigure(errors="replace")
    print(
        f"lines {report['lines'][0]} -> {report['lines'][1]}; items {report['items']};"
        f" anchors {report['anchors'][0]} -> {report['anchors'][1]}; lost {lost}"
    )
    print(f"released claims: {report['released']}")
    print(f"normalized statuses: {len(report['normalized'])}")
    print(f"dropped sections: {len(report['dropped'])}")
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
