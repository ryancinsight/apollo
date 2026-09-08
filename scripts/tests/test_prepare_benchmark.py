"""Filesystem regressions for revision-preserving benchmark preparation."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("prepare_benchmark", Path(__file__).parents[1] / "prepare_benchmark.py")
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)

ROOT_MANIFEST = '''[workspace]
members = ["crates/apollo-bench", "crates/apollo-fft"]
[workspace.package]
edition = "2021"
rust-version = "1.97"
[workspace.dependencies]
hermes = { version = "0.5", git = "https://example.invalid/hermes", features = ["binding"] }
'''
INSTRUMENT_MANIFEST = '''[package]
name = "apollo-bench"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
[dependencies]
hermes = { workspace = true, features = ["topology"] }
'''
TRANSFORM_MANIFEST = '''[package]
name = "apollo-fft"
version = "0.2.0"
edition.workspace = true
rust-version.workspace = true
[dependencies]
leto = { version = "0.9", git = "https://example.invalid/leto" }
[dev-dependencies]
apollo-bench = { path = "../apollo-bench" }
[features]
kernel-strategy-bench = []
'''
LOCK = '''version = 4
[[package]]
name = "leto"
version = "0.9.0"
source = "git+https://example.invalid/leto#{revision}"
[[package]]
name = "hermes"
version = "0.5.0"
source = "git+https://example.invalid/hermes#{revision}"
'''


def write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def contents(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.candidate = Path(self.directory.name).resolve() / "candidate"
        self.baseline = Path(self.directory.name).resolve() / "baseline"
        for root, revision, production in (
            (self.candidate, "6" * 40, "use leto::ComplexLayout;"),
            (self.baseline, "a" * 40, "use leto::transpose_complex_matrices;"),
        ):
            write(root, "Cargo.toml", ROOT_MANIFEST)
            write(root, "Cargo.lock", LOCK.format(revision=revision))
            write(root, "crates/apollo-bench/Cargo.toml", INSTRUMENT_MANIFEST)
            manifest = TRANSFORM_MANIFEST + "".join(
                f'[[bench]]\nname = "{name}"\nharness = false\n'
                for name in preparation.BENCHMARKS
            )
            write(root, "crates/apollo-fft/Cargo.toml", manifest)
            write(root, "crates/apollo-fft/src/lib.rs", production)
            write(root, "crates/provider/Cargo.toml", '[package]\nname = "provider"\nversion = "0.1.0"\n')
            write(root, "crates/provider/src/lib.rs", f'pub const REVISION: &str = "{revision}";')
            write(root, "crates/apollo-bench/src/lib.rs", f'pub const SCHEMA: &str = "{revision}";')
            for name in preparation.BENCHMARKS:
                write(root, f"crates/apollo-fft/benches/{name}.rs", f'fn main() {{ println!("{name}:{revision}"); }}')
        write(self.baseline, "crates/apollo-bench/src/obsolete/record.rs", "obsolete source")
        write(self.candidate, "crates/apollo-bench/src/report/schema.rs", "candidate schema")

    def test_transfers_exact_sources_preserving_each_production_graph(self):
        before = contents(self.baseline)
        candidate_before = contents(self.candidate)
        result = preparation.prepare(self.candidate, self.baseline)
        after = contents(self.baseline)
        expected = {path: data for path, data in before.items()
                    if not any(path.is_relative_to(source) for source in preparation.SOURCE_ROOTS)}
        expected.update({path: data for path, data in candidate_before.items()
                         if any(path.is_relative_to(source) for source in preparation.SOURCE_ROOTS)})
        self.assertEqual(after, expected)
        self.assertEqual(contents(self.candidate), candidate_before)
        self.assertNotEqual(after[Path("Cargo.lock")], candidate_before[Path("Cargo.lock")])
        self.assertEqual(result["removed"], ["crates/apollo-bench/src/obsolete/record.rs"])
        self.assertFalse((self.baseline / "crates/apollo-bench/src/obsolete").exists())
        self.assertEqual(result["source_sha256"], preparation.hashes(preparation.source_files(self.candidate)))

    def test_second_transfer_does_not_rewrite_files(self):
        preparation.prepare(self.candidate, self.baseline)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.baseline.rglob("*") if path.is_file()}
        result = preparation.prepare(self.candidate, self.baseline)
        after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
        self.assertEqual(after, before)
        self.assertEqual((result["written"], result["removed"]), ([], []))

    def test_different_production_requirements_remain_revision_owned(self):
        path = self.candidate / "crates/apollo-fft/Cargo.toml"
        path.write_text(path.read_text().replace('version = "0.9"', 'version = "1.0"'), encoding="utf-8")
        before = (self.baseline / "crates/apollo-fft/Cargo.toml").read_bytes()
        preparation.prepare(self.candidate, self.baseline)
        self.assertEqual((self.baseline / "crates/apollo-fft/Cargo.toml").read_bytes(), before)

    def test_incompatible_inherited_dependency_fails_before_mutation(self):
        write(self.candidate, "Cargo.toml", ROOT_MANIFEST.replace('version = "0.5"', 'version = "0.6"'))
        before = contents(self.baseline)
        with self.assertRaisesRegex(ValueError, "incompatible instrument manifest requirements: instrument.dependencies"):
            preparation.prepare(self.candidate, self.baseline)
        self.assertEqual(contents(self.baseline), before)

    def test_malformed_manifest_fails_before_mutation(self):
        write(self.candidate, "crates/apollo-bench/Cargo.toml", "[dependencies")
        before = contents(self.baseline)
        with self.assertRaises(ValueError):
            preparation.prepare(self.candidate, self.baseline)
        self.assertEqual(contents(self.baseline), before)

    def test_incompatible_benchmark_target_fails_before_mutation(self):
        path = self.candidate / "crates/apollo-fft/Cargo.toml"
        path.write_text(path.read_text().replace('name = "prime_compose"', 'name = "prime_compose"\npath = "../../outside.rs"'), encoding="utf-8")
        before = contents(self.baseline)
        with self.assertRaisesRegex(ValueError, "outside its source transfer"):
            preparation.prepare(self.candidate, self.baseline)
        self.assertEqual(contents(self.baseline), before)

    def test_nested_manifest_fails_before_mutation(self):
        write(self.candidate, "crates/apollo-bench/src/nested/Cargo.toml", ROOT_MANIFEST)
        before = contents(self.baseline)
        with self.assertRaisesRegex(ValueError, "manifest inside source transfer"):
            preparation.prepare(self.candidate, self.baseline)
        self.assertEqual(contents(self.baseline), before)

    def test_overlapping_roots_fail_without_writes(self):
        before = contents(self.candidate)
        with self.assertRaisesRegex(ValueError, "disjoint checkout roots"):
            preparation.prepare(self.candidate, self.candidate)
        self.assertEqual(contents(self.candidate), before)

    def test_file_directory_transitions_replace_obsolete_source(self):
        write(self.baseline, "crates/apollo-bench/src/shape", "old file")
        write(self.candidate, "crates/apollo-bench/src/shape/item.rs", "new nested file")
        write(self.baseline, "crates/apollo-bench/src/collapse/item.rs", "old nested file")
        write(self.candidate, "crates/apollo-bench/src/collapse", "new file")
        preparation.prepare(self.candidate, self.baseline)
        self.assertEqual(preparation.source_files(self.baseline), preparation.source_files(self.candidate))

    def test_symlink_escape_rejected_before_mutation(self):
        outside = Path(self.directory.name) / "outside.rs"
        outside.write_text("outside bytes", encoding="utf-8")
        link = self.baseline / "crates/apollo-bench/src/escape.rs"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"host cannot create symlinks: {error}")
        before = contents(self.baseline)
        with self.assertRaisesRegex(ValueError, "linked path"):
            preparation.prepare(self.candidate, self.baseline)
        self.assertEqual(contents(self.baseline), before)
        self.assertEqual(outside.read_text(), "outside bytes")

    @unittest.skipUnless(os.name == "nt", "junctions are a Windows filesystem feature")
    def test_junction_escape_rejected_before_mutation(self):
        outside = Path(self.directory.name).resolve() / "outside"
        write(outside, "untouched.rs", "outside source bytes")
        link = self.baseline / "crates/apollo-bench/src/escape"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                       check=True, capture_output=True, timeout=10)
        before = contents(self.baseline)
        with self.assertRaisesRegex(ValueError, "linked path"):
            preparation.prepare(self.candidate, self.baseline)
        self.assertEqual(contents(self.baseline), before)
        self.assertEqual((outside / "untouched.rs").read_text(), "outside source bytes")


if __name__ == "__main__":
    unittest.main()
