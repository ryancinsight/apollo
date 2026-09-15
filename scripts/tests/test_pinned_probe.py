"""The replicated pinned-probe runner's parsing and summary, on recorded rows."""

import importlib.util
import statistics
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("pinned_probe", Path(__file__).parents[1] / "pinned_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

# Two processes of a large-sizes sweep, in the probe's own report shape; the
# minima are the 262144 readings of two quiet runs on the 285K host.
RUN_A = """selected performance cpu=4
LRG cpu=4 (performance)
case,min_ps,median_ps,median_lower_ps,median_upper_ps,median_confidence_ppm,ordered_samples_ps,iterations_per_sample
performance/apollo-f64/262144,1135200000,1150600000,1149000000,1152000000,950000,1135200000;1150600000,1
performance/rustfft-f64/262144,1252200000,1280400000,1278000000,1283000000,950000,1252200000;1280400000,1
performance/phastft-f64/262144,1233700000,1262000000,1260000000,1264000000,950000,1233700000;1262000000,1
performance/apollo-f64/100,1000,1100,1090,1110,950000,1000;1100,4
performance/rustfft-f64/100,2000,2100,2090,2110,950000,2000;2100,4
"""
RUN_B = """selected performance cpu=4
LRG cpu=4 (performance)
case,min_ps,median_ps,median_lower_ps,median_upper_ps,median_confidence_ppm,ordered_samples_ps,iterations_per_sample
performance/apollo-f64/262144,1126800000,1146300000,1145000000,1148000000,950000,1126800000;1146300000,1
performance/rustfft-f64/262144,1062700000,1106800000,1105000000,1108000000,950000,1062700000;1106800000,1
performance/phastft-f64/262144,1246000000,1273000000,1271000000,1275000000,950000,1246000000;1273000000,1
performance/apollo-f64/100,1200,1300,1290,1310,950000,1200;1300,4
performance/rustfft-f64/100,2000,2100,2090,2110,950000,2000;2100,4
"""


class ParseRows(unittest.TestCase):
    def test_reads_each_arm_minimum_and_skips_headers_and_prose(self):
        rows = probe.parse_rows(RUN_A)
        self.assertEqual(rows[("performance", "apollo", "f64", 262144)], 1_135_200_000)
        self.assertEqual(rows[("performance", "rustfft", "f64", 262144)], 1_252_200_000)
        self.assertEqual(rows[("performance", "phastft", "f64", 262144)], 1_233_700_000)
        self.assertEqual(rows[("performance", "apollo", "f64", 100)], 1000)
        self.assertEqual(len(rows), 5)


class Summarize(unittest.TestCase):
    def test_reports_median_minima_spreads_and_ratio_ranges_over_runs(self):
        records = probe.summarize([probe.parse_rows(RUN_A), probe.parse_rows(RUN_B)])
        by_length = {record["length"]: record for record in records}
        large = by_length[262144]
        self.assertEqual(large["runs"], 2)
        self.assertEqual(large["apollo_min_ps_median"],
                         int(statistics.median([1_135_200_000, 1_126_800_000])))
        self.assertAlmostEqual(large["apollo_min_ps_spread"], 1_135_200_000 / 1_126_800_000 - 1.0)
        ratios = [1_135_200_000 / 1_252_200_000, 1_126_800_000 / 1_062_700_000]
        self.assertAlmostEqual(large["rustfft_ratio_median"], statistics.median(ratios))
        self.assertAlmostEqual(large["rustfft_ratio_min"], min(ratios))
        self.assertAlmostEqual(large["rustfft_ratio_max"], max(ratios))
        self.assertAlmostEqual(large["phastft_ratio_median"],
                               statistics.median([1_135_200_000 / 1_233_700_000,
                                                  1_126_800_000 / 1_246_000_000]))
        small = by_length[100]
        self.assertNotIn("phastft_ratio_median", small, "no PhastFT arm at a non-power of two")
        self.assertAlmostEqual(small["rustfft_ratio_median"], statistics.median([0.5, 0.6]))

    def test_refuses_an_empty_campaign(self):
        with self.assertRaises(ValueError):
            probe.summarize([])


class Render(unittest.TestCase):
    def test_writes_one_csv_row_per_case_with_blank_absent_references(self):
        records = probe.summarize([probe.parse_rows(RUN_A), probe.parse_rows(RUN_B)])
        text = probe.render(records)
        lines = text.splitlines()
        self.assertEqual(lines[0].split(",")[:6],
                         ["core", "scalar", "length", "runs", "apollo_min_ps_median", "apollo_min_ps_spread"])
        self.assertEqual(len(lines), 3)
        small = next(line for line in lines if ",100," in line)
        self.assertTrue(small.endswith(",,,"), "the absent PhastFT ratio renders as blank cells")


if __name__ == "__main__":
    unittest.main()
