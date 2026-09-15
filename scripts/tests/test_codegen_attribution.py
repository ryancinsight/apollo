"""The loop census and critical-path estimate of the codegen attribution
script, on a handcrafted AT&T loop."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("codegen_attribution", Path(__file__).parents[1] / "codegen_attribution.py")
attribution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attribution)

# One symbol: a two-instruction prologue, a loop of nine instructions closed
# by a backward jump (two loads, one spill reload, an in-lane shuffle, a
# cross-lane shuffle, an FMA, a store, the counter and the jump), a call
# after it, and a second symbol the filter must skip.
ASM = """\t.text
_RNvCsabc_6sample4loop:
\tmovq\t%rcx, %rax
\txorl\t%edx, %edx
.LBB0_1:
\tvmovups\t(%rax,%rdx,8), %ymm0
\tvmovups\t32(%rax,%rdx,8), %ymm1
\tvmovaps\t96(%rsp), %ymm2
\tvpermilps\t$177, %ymm1, %ymm3
\tvperm2f128\t$32, %ymm0, %ymm3, %ymm4
\tvfmadd231ps\t%ymm2, %ymm4, %ymm0
\tvmovups\t%ymm0, (%rax,%rdx,8)
\taddq\t$8, %rdx
\tjne\t.LBB0_1
\tcallq\t_RNvCsabc_6sample6helper
\tretq
_RNvCsabc_6sample5other:
\tvaddps\t%ymm0, %ymm1, %ymm2
\tretq
"""


class LoopCensus(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "sample.s"
        self.path.write_text(ASM, encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def loop_instructions(self):
        body = attribution.labeled_bodies(self.path)["_RNvCsabc_6sample4loop"]
        loops = attribution.loops_of(body)
        self.assertEqual(loops, [(2, 10)])
        instructions = [line for line in body if not attribution.LABEL.match(line)]
        return instructions[2:11]

    def test_finds_the_loop_and_counts_its_classes(self):
        tally = attribution.census(self.loop_instructions())
        self.assertEqual(tally["load"], 3)
        self.assertEqual(tally["spill"], 1)
        self.assertEqual(tally["shuffle"], 2)
        self.assertEqual(tally["cross"], 1)
        self.assertEqual(tally["fma"], 1)
        self.assertEqual(tally["store"], 1)
        self.assertNotIn("call", tally, "the call sits after the loop")
        self.assertEqual(tally["vector"], 7)

    def test_critical_path_follows_the_def_use_chain(self):
        # load ymm1 (5) -> in-lane shuffle (1) -> cross-lane shuffle (3) -> FMA (4) -> store
        self.assertEqual(attribution.critical_path(self.loop_instructions()), 13)

    def test_counts_a_call_inside_a_loop_as_out_of_line_work(self):
        tally = attribution.census(["\tcallq\t_RNvCsabc_6sample6helper", "\tvaddps\t%ymm0, %ymm1, %ymm2"])
        self.assertEqual(tally["call"], 1)
        self.assertEqual(tally["addsub"], 1)

    def test_skips_symbols_the_pattern_does_not_match(self):
        bodies = attribution.labeled_bodies(self.path)
        self.assertEqual(set(bodies), {"_RNvCsabc_6sample4loop", "_RNvCsabc_6sample5other"})
        self.assertEqual(attribution.loops_of(bodies["_RNvCsabc_6sample5other"]), [])


if __name__ == "__main__":
    unittest.main()
