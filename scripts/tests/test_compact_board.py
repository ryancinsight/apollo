"""The board compaction on a handcrafted board: anchors kept and derived,
statuses normalized, stale claims released, sections dropped, budgets held."""

import importlib.util
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

## SAMPLE-ODD-2026-09-05 — An off-set status [minor] — measured
- The reading.
"""


class Compaction(unittest.TestCase):
    def setUp(self):
        self.out, self.report = compaction.compact(BOARD, "2026-09-15", "2026-09-14")

    def test_keeps_and_derives_anchors_without_losing_any(self):
        self.assertEqual(self.report["anchors"][2], [])
        for anchor in ("sample-open", "sample-stale", "sample-legacy", "sample-done", "sample-odd"):
            self.assertIn(f'<a id="{anchor}"></a>', self.out)

    def test_releases_a_stale_claim_and_drops_its_narrative(self):
        self.assertIn("## SAMPLE-STALE-2026-09-02 — A stale claim [patch] — todo", self.out)
        self.assertIn("**Claim released:** 2026-09-15, the last update 2026-09-01", self.out)
        self.assertNotIn("Integrator:** peer/x", self.out)
        self.assertNotIn("Lines of narrative", self.out)
        self.assertEqual(self.report["released"], [("sample-stale", "2026-09-01")])

    def test_drops_sprint_sections_and_normalizes_statuses(self):
        self.assertNotIn("Closed in this sprint", self.out)
        self.assertEqual(self.report["dropped"], ["Closed in this sprint (Closure I phase)"])
        self.assertIn("- **SAMPLE-LEGACY-2026-09-03** — A legacy item without a status [patch]. Landed as PR #12: the outcome. continued on the next line.", self.out)
        self.assertIn("- **SAMPLE-ODD-2026-09-05** — An off-set status [minor]. The reading.", self.out)
        self.assertEqual(sorted(self.report["normalized"]), [("sample-legacy", "none -> done"), ("sample-odd", "measured -> done")])

    def test_done_items_are_one_line_in_the_closing_section(self):
        done = self.out.split("# Done", 1)[1]
        self.assertIn('<a id="sample-done"></a>- **SAMPLE-DONE-2026-09-04** — A done item [patch]. [#13](https://example.invalid/pull/13): the outcome.', done)
        self.assertNotIn("More narrative", self.out)
        self.assertNotIn("## SAMPLE-DONE", self.out)

    def test_open_items_keep_their_records_within_the_limit(self):
        self.assertIn("## SAMPLE-OPEN-2026-09-01 — An open item [patch] — todo\n- **Scope:** one.\n- **Acceptance:** two.\n", self.out)
        self.assertLess(self.report["lines"][1], self.report["lines"][0])


if __name__ == "__main__":
    unittest.main()
