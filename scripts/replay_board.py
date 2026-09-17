#!/usr/bin/env python3
"""Re-apply one commit's board change onto the working board, item by item.

A rebase that stops on `backlog.md` is resolved here instead of by hand:

    git checkout --ours -- backlog.md
    python scripts/replay_board.py backlog.md "$(git rev-parse REBASE_HEAD)"
    python scripts/compact_board.py backlog.md --today <date>
    git add backlog.md && git rebase --continue

The commit's change is read against its own parent -- the done entries it
added or removed, and the open item blocks it added, removed or rewrote --
and applied to the upstream board one item at a time, so no conflict hunk is
ever merged textually. That matters because a hunk can span a whole
reordered section: keeping both of its sides duplicates every entry in it.
The compactor then restores the canonical form.

An open block is the anchor line that starts it and everything up to the
next anchor line; a done entry is one anchored line after `# Done`. A new
open block goes after the block that precedes it in the commit, or first
when none of its predecessors is on the working board. An item the result
records as done keeps no open block.

Usage: python scripts/replay_board.py backlog.md <commit>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DONE_HEADING = "# Done"
ANCHOR = re.compile(r'^<a id="([^"]+)"></a>\s*$')
DONE_ENTRY = re.compile(r'^<a id="([^"]+)"></a>(?:<a id="[^"]+"></a>)*- \*\*')


class Board:
    """A board split into its head, its open blocks in order, and its done section."""

    def __init__(self, text: str) -> None:
        before, _, done_part = text.partition(f"\n{DONE_HEADING}\n")
        self.head: list[str] = []
        self.blocks: list[tuple[str, list[str]]] = []
        for line in before.split("\n"):
            match = ANCHOR.match(line)
            if match:
                self.blocks.append((match.group(1), [line]))
            elif self.blocks:
                self.blocks[-1][1].append(line)
            else:
                self.head.append(line)
        self.notes = [line for line in done_part.split("\n") if line and not DONE_ENTRY.match(line)]
        self.done: dict[str, str] = {}
        for line in done_part.split("\n"):
            match = DONE_ENTRY.match(line)
            if match:
                self.done[match.group(1)] = line

    def index(self, anchor: str) -> int | None:
        return next((i for i, (a, _) in enumerate(self.blocks) if a == anchor), None)

    def render(self) -> str:
        opened = "\n".join(self.head + [line for _, block in self.blocks for line in block]).rstrip("\n")
        return (
            f"{opened}\n\n{DONE_HEADING}\n\n"
            + "\n".join(self.notes)
            + "\n\n"
            + "\n".join(self.done.values())
            + "\n"
        )


def replay(working: Board, parent: Board, child: Board) -> int:
    """Apply the parent-to-child change onto `working`; the number of items changed."""
    changed = 0
    before_blocks, after_blocks = dict(parent.blocks), dict(child.blocks)
    child_order = [anchor for anchor, _ in child.blocks]
    for anchor in sorted(before_blocks.keys() | after_blocks.keys()):
        before, after = before_blocks.get(anchor), after_blocks.get(anchor)
        if before == after:
            continue
        changed += 1
        at = working.index(anchor)
        if after is None:
            if at is not None:
                del working.blocks[at]
        elif at is not None:
            working.blocks[at] = (anchor, after)
        else:
            position = 0
            for previous in reversed(child_order[: child_order.index(anchor)]):
                found = working.index(previous)
                if found is not None:
                    position = found + 1
                    break
            working.blocks.insert(position, (anchor, after))
    for anchor in sorted(parent.done.keys() | child.done.keys()):
        if parent.done.get(anchor) == child.done.get(anchor):
            continue
        changed += 1
        if anchor in child.done:
            working.done[anchor] = child.done[anchor]
        else:
            working.done.pop(anchor, None)
    working.blocks = [(anchor, block) for anchor, block in working.blocks if anchor not in working.done]
    return changed


def board_at(repo: Path, revision: str, path: str) -> Board:
    shown = subprocess.run(
        ["git", "show", f"{revision}:{path}"], capture_output=True, cwd=repo, check=True
    )
    return Board(shown.stdout.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("board", type=Path)
    parser.add_argument("commit", help="the commit whose board change to re-apply")
    args = parser.parse_args()
    board = args.board.resolve()
    repo = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, cwd=board.parent, check=True
        ).stdout.decode("utf-8").strip()
    )
    relative = board.relative_to(repo.resolve()).as_posix()
    text = board.read_text(encoding="utf-8")
    if re.search(r"(?m)^(<<<<<<<|>>>>>>>) ", text):
        print("the working board still holds conflict markers; take a side first", file=sys.stderr)
        return 1
    working = Board(text)
    changed = replay(
        working,
        board_at(repo, f"{args.commit}^", relative),
        board_at(repo, args.commit, relative),
    )
    board.write_text(working.render(), encoding="utf-8", newline="\n")
    print(f"replayed {changed} item change(s) from {args.commit[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
