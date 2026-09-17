"""The board compaction on a handcrafted board: anchors kept, derived and
read inline, legacy status forms folded into the closed set, stale claims
released, sections dropped, budgets held, a second pass idempotent, the done
section in anchor-hash order so concurrent records merge, and link faults
reported."""

import hashlib
import importlib.util
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("compact_board", Path(__file__).parents[1] / "compact_board.py")
compaction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compaction)

BOARD = """# Backlog — sample

<a id="sample-open"></a>
## SAMPLE-OPEN-2026-09-01 — An open item [patch] — todo
- **Scope:** one.
- **Acceptance:** two.

## SAMPLE-STALE-2026-09-02 — A stale claim [patch] — in-progress
- **Integrator:** peer/x; **last-update:** 2026-09-01; lane none.
- **Scope:** three.
### A narrative section
Lines of narrative that the record does not keep.

## Closed in this sprint (Closure I phase)
- [x] something landed
- [x] something else landed

## SAMPLE-LEGACY-2026-09-03 — A legacy item without a status [patch]
- Landed as PR #12: the outcome.
  continued on the next line.
- A second record.

<a id="sample-done"></a>
## SAMPLE-DONE-2026-09-04 — A done item [patch] — done 2026-09-04
- [#13](https://example.invalid/pull/13): the outcome.
- More narrative.
- Even more.

## SAMPLE-ODD-2026-09-05 — An off-set status [minor] — measured: loses to batched as-built
- The reading.

## SAMPLE-REJECTED-2026-09-06 — A rejected trial [patch] [perf] — done 2026-09-06 (rejected)
- The trial and its reading.

## SAMPLE-INLINE-2026-09-07 — An inline anchor [patch] — in-progress <a id="sample-inline-anchor"></a>
- **Integrator:** me; **last-update:** 2026-09-15.

## SAMPLE-BLOCKED-2026-09-08 — A blocked item [arch] — blocked: no AVX-512 hardware
- **Scope:** four.

## SAMPLE-SPACED-2026-09-09 — An old spelling [patch] — in progress
- **Integrator:** peer/y; **last-update:** 2026-09-15.

## SAMPLE-CAPS-2026-09-10 — A capitalized claim [patch] — in-progress
- **Integrator:** peer/z. **Last update:** 2026-09-02.
- **Scope:** five.
"""


class Compaction(unittest.TestCase):
    def setUp(self):
        self.out, self.report = compaction.compact(BOARD, "2026-09-15", "2026-09-14")

    def test_keeps_derives_and_reads_inline_anchors_without_losing_any(self):
        self.assertEqual(self.report["anchors"][2], [])
        for anchor in ("sample-open", "sample-stale", "sample-legacy", "sample-done", "sample-odd", "sample-inline-anchor"):
            self.assertIn(f'<a id="{anchor}"></a>', self.out)
        self.assertIn("## SAMPLE-INLINE-2026-09-07 — An inline anchor [patch] — in-progress\n", self.out)
        self.assertNotIn("in-progress <a id=", self.out)

    def test_releases_a_stale_claim_and_drops_its_narrative(self):
        self.assertIn("## SAMPLE-STALE-2026-09-02 — A stale claim [patch] — todo", self.out)
        self.assertIn("**Claim released:** 2026-09-15, the last update 2026-09-01", self.out)
        self.assertNotIn("Integrator:** peer/x", self.out)
        self.assertNotIn("Lines of narrative", self.out)
        self.assertEqual(self.report["released"], [("sample-stale", "2026-09-01"), ("sample-caps", "2026-09-02")])
        self.assertIn("## SAMPLE-CAPS-2026-09-10 — A capitalized claim [patch] — todo\n- **Scope:** five.\n- **Claim released:**", self.out)

    def test_drops_sprint_sections_and_folds_legacy_statuses(self):
        self.assertNotIn("Closed in this sprint", self.out)
        self.assertEqual(self.report["dropped"], ["Closed in this sprint (Closure I phase)"])
        done = self.out.split("# Done", 1)[1]
        self.assertIn("- **SAMPLE-LEGACY-2026-09-03** — A legacy item without a status [patch]. Landed as PR #12: the outcome. continued on the next line.", done)
        self.assertIn("- **SAMPLE-ODD-2026-09-05** — An off-set status [minor] (loses to batched as-built). The reading.", done)
        self.assertIn("- **SAMPLE-REJECTED-2026-09-06** — A rejected trial [patch] [perf] (rejected). The trial and its reading.", done)
        self.assertIn("## SAMPLE-BLOCKED-2026-09-08 — A blocked item [arch] — blocked\n- **Blocked:** no AVX-512 hardware.\n- **Scope:** four.", self.out)
        self.assertIn("## SAMPLE-SPACED-2026-09-09 — An old spelling [patch] — in-progress\n", self.out)

    def test_done_items_are_one_line_in_the_closing_section(self):
        done = self.out.split("# Done", 1)[1]
        self.assertIn('<a id="sample-done"></a>- **SAMPLE-DONE-2026-09-04** — A done item [patch]. [#13](https://example.invalid/pull/13): the outcome.', done)
        self.assertNotIn("More narrative", self.out)
        self.assertNotIn("## SAMPLE-DONE", self.out)

    def test_open_items_keep_their_records_within_the_limit(self):
        self.assertIn("## SAMPLE-OPEN-2026-09-01 — An open item [patch] — todo\n- **Scope:** one.\n- **Acceptance:** two.\n", self.out)
        self.assertLess(self.report["lines"][1], self.report["lines"][0])

    def test_a_second_pass_changes_nothing(self):
        again, report = compaction.compact(self.out, "2026-09-16", "2026-09-14")
        self.assertEqual(again, self.out)
        self.assertEqual(report["anchors"][2], [])
        self.assertEqual(report["normalized"], [])


DONE_ENTRY = re.compile(r'(?m)^<a id="([^"]+)"></a>- \*\*.*$')
TODAY = "2026-09-17"


def done_anchors(board: str) -> list[str]:
    return [match.group(1) for match in DONE_ENTRY.finditer(board)]


def open_item(ident: str) -> str:
    anchor = ident.lower()
    return f'<a id="{anchor}"></a>\n## {ident} — Item {anchor} [patch] — todo\n- **Scope:** {anchor}.\n\n'


def recorded(board: str, ident: str) -> str:
    """`board` with `ident`'s open item marked done and compacted, as a record commit leaves it."""
    anchor = ident.lower()
    heading = f"## {ident} — Item {anchor} [patch] — todo\n- **Scope:** {anchor}."
    assert heading in board, ident
    board = board.replace(
        heading,
        f"## {ident} — Item {anchor} [patch] — done\n- [#1](https://example.invalid/pull/1): {anchor} landed.",
    )
    return compaction.compact(board, TODAY, TODAY)[0]


def merge(base: str, ours: str, theirs: str) -> tuple[str, int]:
    """`git merge-file` over three texts: the merged text and its conflict count."""
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for name, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            path = Path(directory) / name
            path.write_text(text, encoding="utf-8", newline="\n")
            paths.append(str(path))
        result = subprocess.run(["git", "merge-file", "-p", *paths], capture_output=True, check=False)
    return result.stdout.decode("utf-8"), result.returncode


class DoneOrder(unittest.TestCase):
    """Two records made on separate branches merge without a hand resolution."""

    LANDED = [f"LANDED-{n:02d}" for n in range(40)]
    FIRST, SECOND = "RECORD-A", "RECORD-C"
    # An item still open between the two records. Two branches deleting
    # adjacent open blocks conflict whatever the done order is; that form is
    # outside this ordering's reach and is left to the per-item files the
    # board item names.
    BETWEEN = "STILL-OPEN-B"

    def setUp(self):
        board = "# Backlog — sample\n\n" + "".join(
            open_item(i) for i in self.LANDED + [self.FIRST, self.BETWEEN, self.SECOND]
        )
        for ident in self.LANDED:
            board = recorded(board, ident)
        self.base = board

    def test_done_section_is_in_anchor_hash_order(self):
        anchors = done_anchors(self.base)
        self.assertEqual(len(anchors), len(self.LANDED))
        keys = [hashlib.sha256(anchor.encode()).hexdigest() for anchor in anchors]
        self.assertEqual(keys, sorted(keys))

    def test_the_fixture_pair_lands_apart(self):
        # The merge test below relies on this: two records inserted at
        # adjacent places still conflict, and that is the remaining 1%.
        order = done_anchors(recorded(recorded(self.base, self.FIRST), self.SECOND))
        gap = abs(order.index(self.FIRST.lower()) - order.index(self.SECOND.lower()))
        self.assertGreater(gap, 1)

    @unittest.skipIf(shutil.which("git") is None, "git is not installed")
    def test_concurrent_records_merge_cleanly_where_appending_conflicts(self):
        ours = recorded(self.base, self.FIRST)
        theirs = recorded(self.base, self.SECOND)
        merged, conflicts = merge(self.base, ours, theirs)
        self.assertEqual(conflicts, 0, merged)
        self.assertEqual(merged, recorded(recorded(self.base, self.FIRST), self.SECOND))

        # The instrument bites: the same two records written the old way,
        # appended after the last entry, conflict.
        def appended(board: str, ident: str) -> str:
            line = next(e.group(0) for e in DONE_ENTRY.finditer(board) if e.group(1) == ident.lower())
            rest = [e.group(0) for e in DONE_ENTRY.finditer(board) if e.group(1) != ident.lower()]
            return board.split("# Done")[0] + "# Done\n" + "\n".join(rest + [line]) + "\n"

        def appended_base() -> str:
            return self.base.split("# Done")[0] + "# Done\n" + "\n".join(
                e.group(0) for e in DONE_ENTRY.finditer(self.base)
            ) + "\n"

        _, old_conflicts = merge(appended_base(), appended(ours, self.FIRST), appended(theirs, self.SECOND))
        self.assertGreater(old_conflicts, 0)


class LinkFaults(unittest.TestCase):
    def test_reports_a_duplicate_anchor_and_a_dangling_link(self):
        board = (
            '# Backlog — sample\n\n<a id="one"></a>\n## ONE-2026-09-01 — One [patch] — todo\n'
            "- **Scope:** see [two](#two) and [missing](#missing).\n\n"
            '<a id="two"></a>\n## TWO-2026-09-02 — Two [patch] — todo\n- **Scope:** two <a id="one"></a>.\n'
        )
        _, report = compaction.compact(board, TODAY, TODAY)
        self.assertEqual(report["duplicates"], ["one"])
        self.assertEqual(report["dangling"], ["missing"])

    def test_a_clean_board_reports_none(self):
        _, report = compaction.compact(BOARD, "2026-09-15", "2026-09-14")
        self.assertEqual((report["duplicates"], report["dangling"]), ([], []))


if __name__ == "__main__":
    unittest.main()
