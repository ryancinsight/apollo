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

The done section is ordered by the SHA-256 of each entry's anchor. Every
record adds one line there, and when records were appended at the end, any
two branches recording different items touched the same last line and
conflicted -- every landing made every other open record `DIRTY`. A
hashed order gives each entry its own place, and unlike alphabetical order
it scatters the related IDs a stream of work files together: over the
board's 276 entries, `git merge-file` conflicted on 275 of 275
consecutively recorded pairs appended at the end, 34 in alphabetical
order, and 3 in hashed order (`backlog.md#apollo-board-completed-section-union`).
Two conflicts remain possible: two insertions at adjacent places in that
order, and two branches deleting open blocks that sit next to each other.
Both keep every line on resolution; the second is what per-item files
would remove. `scripts/replay_board.py` resolves either during a rebase by
re-applying the stopped commit's item changes to the upstream board.

It also fails the board on what makes a link unsafe: an anchor defined twice,
and an in-file link (`](#id)`) whose anchor is absent.

Legacy heading forms fold into the closed set: `— done 2026-09-09
(rejected)`, `— done, by deletion`, `— closed 2026-08-25, premise false`,
`— measured: loses to batched`, `— complete`, `— provider done` are done
with the note kept beside the title; `— in progress` is in-progress;
`— blocked: no hardware` is blocked with the reason as its first record.
An anchor written inline in a heading is the item's anchor.

Usage: python scripts/compact_board.py backlog.md --today 2026-09-15
       [--release-before 2026-09-14] [--out backlog.md]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

CLOSED = ("todo", "in-progress", "blocked", "review", "done")
STATUS_MAP = {
    "done": "done", "downgraded": "done", "measured": "done", "superseded": "done", "closed": "done",
    "landed": "done", "merged": "done", "complete": "done", "completed": "done", "provider": "done",
    "todo": "todo", "open": "todo", "blocked": "blocked", "review": "review",
    "in-progress": "in-progress", "in_progress": "in-progress", "wip": "in-progress", "in": "in-progress",
}
SECTION = re.compile(
    r"^## (Closed in this sprint|Open in this sprint|Planned next|Delivered|Sprint|Session|Wave|Tier"
    r"|Watchpoints|Backlog|Open items|Done items|Notes)",
    re.I,
)
GLYPHS = re.compile("[←-⇿⌀-⏿─-➿⬀-⯿\U0001f000-\U0001faff]")
ANCHOR = re.compile(r'^<a id="([^"]+)"></a>\s*$')
DONE_LINE = re.compile(r'^(?:<a id="[^"]+"></a>)+- \*\*')
LINE_ANCHORS = re.compile(r'<a id="([^"]+)"></a>')
DONE_HEADING = "# Done"
DONE_NOTE = (
    "One line an item: the anchor, the identity, the outcome with its commit or PR; the narrative lives in git."
    " Ordered by the hash of the anchor so concurrent records land apart; find an item by its ID."
)
IN_FILE_LINK = re.compile(r"\]\(#([^)\s]+)\)")
INLINE_ANCHOR = re.compile(r'\s*<a id="([^"]+)"></a>\s*')
DATE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b")
LANDED = re.compile(r"PR #\d+|pull/\d+|\b[0-9a-f]{8,40}\b|[Ll]anded|[Mm]erged|\[x\]|Outcome:\*\* built")
ITEM_ID = re.compile(r"^## [A-Z0-9][A-Z0-9-]{6,} ")
OPEN_LIMIT = 14
NL = "\n"


def done_order(line: str) -> str:
    """The sort key of a done entry: the SHA-256 of its first anchor."""
    return hashlib.sha256(LINE_ANCHORS.search(line).group(1).encode("utf-8")).hexdigest()


def link_faults(text: str) -> tuple[list[str], list[str]]:
    """Anchors defined more than once, and in-file link targets never defined."""
    defined = re.findall(r'<a id="([^"]+)"></a>', text)
    counts: dict[str, int] = {}
    for anchor in defined:
        counts[anchor] = counts.get(anchor, 0) + 1
    duplicates = sorted(anchor for anchor, n in counts.items() if n > 1)
    dangling = sorted({target for target in IN_FILE_LINK.findall(text) if target not in counts})
    return duplicates, dangling


def split_items(text: str) -> tuple[list[str], list[dict], list[str]]:
    """The head lines, the items (each with the anchor line before it or the
    anchor written inline in its heading), and the closing section's
    one-line done entries of a previous compaction, kept as they are."""
    lines = text.split(NL)
    head: list[str] = []
    i = 0
    while i < len(lines) and not lines[i].startswith("## ") and lines[i].strip() != DONE_HEADING:
        head.append(lines[i])
        i += 1
    pending: str | None = None
    while head and (not head[-1].strip() or ANCHOR.match(head[-1])):
        last = head.pop()
        match = ANCHOR.match(last)
        if match:
            pending = match.group(1)
    items: list[dict] = []
    done_lines: list[str] = []
    current: dict | None = None
    in_done = False
    for line in lines[i:]:
        if line.strip() == DONE_HEADING:
            in_done = True
            continue
        if in_done:
            if DONE_LINE.match(line):
                done_lines.append(line.rstrip())
            elif line.startswith("## "):
                in_done = False
            else:
                continue
        if in_done:
            continue
        if line.startswith("## "):
            if current:
                items.append(current)
            inline = INLINE_ANCHOR.search(line)
            heading = INLINE_ANCHOR.sub(" ", line).rstrip() if inline else line
            # Both an anchor line and an inline anchor: the line's names the
            # item and the inline one stays beside it, so neither link breaks.
            extra = [inline.group(1)] if (inline and pending and inline.group(1) != pending) else []
            current = {"anchor": pending or (inline.group(1) if inline else None), "heading": heading, "body": [], "extra": extra}
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
    return head, items, done_lines


def assign_anchors(items: list[dict]) -> None:
    """Existing anchors stay; a missing one is the ID lowercased without its
    trailing date, kept unique."""
    used = {it["anchor"] for it in items if it["anchor"]}
    used |= set(getattr(assign_anchors, "reserved", ()))
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


def parse_heading(heading: str) -> tuple[str, str | None, str]:
    """The heading without its status segments, the status from the closed
    set (none when the heading carries no status), and the note the legacy
    form carried ('rejected', 'no AVX-512 hardware', ...)."""
    segments = [s.strip() for s in heading.split(" — ")]
    title_segments = [segments[0]]
    status: str | None = None
    note = ""
    for segment in segments[1:]:
        words = segment.split()
        token = words[0].lower().rstrip(":,") if words else ""
        if token == "in" and len(words) > 1 and words[1].lower().startswith("progress"):
            token, words = "in-progress", ["in-progress"] + words[2:]
        mapped = STATUS_MAP.get(token)
        if mapped is None:
            title_segments.append(segment)
            continue
        rest = " ".join(words[1:])
        rest = DATE.sub("", rest).strip(" :,()")
        if mapped == "todo" and status is not None and status != "todo":
            continue  # a stray todo appended after a real status
        status = mapped
        if rest and mapped != "todo":
            note = rest if not note else note
    return " — ".join(title_segments), status, note


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
    head, items, kept_done = split_items(text)
    reserved = {anchor for line in kept_done for anchor in LINE_ANCHORS.findall(line)}
    for it in items:
        if it["anchor"] in reserved:
            it["anchor"] = None  # a fresh item cannot take a kept entry's anchor
    assign_anchors.reserved = reserved
    assign_anchors(items)
    report: dict = {"released": [], "normalized": [], "dropped": [], "items": len(items)}
    open_lines: list[str] = []
    done_lines: list[str] = []
    for it in items:
        raw = GLYPHS.sub("", it["heading"]).replace("  ", " ").rstrip()
        title, status, note = parse_heading(raw)
        body_text = " ".join(line for line in it["body"] if line.strip())
        checklist_only = bool(body_text.strip()) and all(
            re.match(r"^\s*(- \[[x ]\]|\* \[[x ]\])", line) for line in it["body"] if line.strip()
        )
        if status is None and (SECTION.match(title) or (checklist_only and not ITEM_ID.match(title))):
            report["dropped"].append(title[3:80])
            continue
        if status is None:
            status = "done" if LANDED.search(body_text[:600]) else "todo"
            report["normalized"].append((it["anchor"], f"none -> {status}"))
        elif f" — {status}" != raw[len(title):].rstrip():
            report["normalized"].append((it["anchor"], f"{raw[len(title):].strip(' —')} -> {status}"))
        records = records_of(it["body"])
        if status == "blocked" and note:
            records.insert(0, f"- **Blocked:** {note}.")
        if status == "in-progress":
            match = re.search(r"last[- ]update:\*\*\s*(\d{4}-\d{2}-\d{2})", " ".join(records), re.I)
            last = match.group(1) if match else None
            if last is not None and last < release_before:
                records = [r for r in records if "Integrator:" not in r]
                records.append(
                    f"- **Claim released:** {today}, the last update {last} older than the stale-claim window;"
                    " reclaim by re-syncing the board."
                )
                report["released"].append((it["anchor"], last))
                status = "todo"
        if status == "done":
            match = re.match(r"^## (\S+)(?: — (.*))?$", title)
            ident, rest = (match.group(1), match.group(2) or "") if match else (title[3:], "")
            record = re.sub(r"^[-*] ", "", records[0]).strip() if records else ""
            noted = f"{rest} ({note})" if note else rest
            extra = "".join(f'<a id="{a}"></a>' for a in it.get("extra", []))
            done_lines.append(f'<a id="{it["anchor"]}"></a>{extra}- **{ident}** — {noted}. {record}'.replace(" — . ", ". "))
            continue
        for a in it.get("extra", []):
            open_lines.append(f'<a id="{a}"></a>')
        open_lines.append(f'<a id="{it["anchor"]}"></a>')
        open_lines.append(f"{title} — {status}")
        open_lines.extend(records[:OPEN_LIMIT])
        open_lines.append("")
    closing = [DONE_HEADING, "", DONE_NOTE, ""]
    while head and not head[-1].strip():
        head.pop()
    ordered = sorted(kept_done + done_lines, key=done_order)
    out = NL.join(head + [""] + open_lines + closing + ordered).rstrip(NL) + NL
    report["lines"] = (text.count(NL), out.count(NL))
    before = set(re.findall(r'<a id="([^"]+)"></a>', text))
    after = set(re.findall(r'<a id="([^"]+)"></a>', out))
    report["anchors"] = (len(before), len(after), sorted(before - after))
    report["duplicates"], report["dangling"] = link_faults(out)
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
    print(f"duplicate anchors: {report['duplicates']}; dangling links: {report['dangling']}")
    return 1 if lost or report["duplicates"] or report["dangling"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
