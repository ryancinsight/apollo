"""The risk-artifact compaction on a handcrafted artifact: anchored findings
kept and trimmed at a sentence end, re-open and anchor lines kept, unanchored
sections dropped, the named library kept whole, a second pass idempotent."""

import importlib.util
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "compact_gap_audit", Path(__file__).parents[1] / "compact_gap_audit.py"
)
compaction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compaction)

ARTIFACT = """# Gap audit

## A finding with a long lead (2026-09-01) <a id="long-lead"></a>
The first sentence of the lead.
The second sentence, which ends the thought.
A third line that runs on without
ending, and keeps going.

More narrative that the record does not keep.
Re-open trigger: when the probe reads above 5%.

## Closed Gaps
- [x] a ledger line
- [x] another

## A body-anchored ledger (2026-07-01)
Prose above the anchor.
<a id="ledger-2026-07"></a>
The line beside it.

## Slop patterns recorded 2026-09-01
- pattern one
- pattern two
"""


class Compaction(unittest.TestCase):
    def setUp(self):
        self.out, self.report = compaction.compact(ARTIFACT, 5, ["Slop patterns"])

    def test_anchored_findings_keep_their_lead_to_a_sentence_end_and_their_triggers(self):
        self.assertIn(
            '## A finding with a long lead (2026-09-01) <a id="long-lead"></a>\n'
            "The first sentence of the lead.\n"
            "The second sentence, which ends the thought.\n"
            "Re-open trigger: when the probe reads above 5%.\n",
            self.out,
        )
        self.assertNotIn("More narrative", self.out)
        self.assertNotIn("keeps going", self.out)

    def test_unanchored_sections_drop_and_body_anchors_survive(self):
        self.assertNotIn("Closed Gaps", self.out)
        self.assertIn(
            "## A body-anchored ledger (2026-07-01)\n"
            "Prose above the anchor.\n"
            '<a id="ledger-2026-07"></a>\n',
            self.out,
        )
        self.assertEqual(self.report["anchors"][2], [])
        self.assertEqual(
            (self.report["kept"], self.report["dropped"], self.report["kept_whole"]), (2, 1, 1)
        )

    def test_the_named_library_is_kept_whole(self):
        self.assertIn("## Slop patterns recorded 2026-09-01\n- pattern one\n- pattern two\n", self.out)

    def test_a_second_pass_changes_nothing(self):
        again, report = compaction.compact(self.out, 5, ["Slop patterns"])
        self.assertEqual(again, self.out)
        self.assertEqual(report["dropped"], 0)


if __name__ == "__main__":
    unittest.main()
