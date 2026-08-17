//! `xtask benchmark` — regenerate `docs/benchmark_results.md`.
//!
//! Runs the `rustfft_comparison` bench binary and renders its CSV into the
//! committed markdown table. The measurement contract, sweep sizes, and runtime
//! budget all live in that binary; this module only orchestrates and formats, so
//! there is one place to change what is measured.
//!
//! ```text
//! cargo run -p xtask -- benchmark               # default bounded sweep
//! cargo run -p xtask -- benchmark --full        # dense 1..=500 sweep, opt-in
//! cargo run -p xtask -- benchmark --skip-run    # re-render from a saved CSV
//! ```
//!
//! `--skip-run` reads the CSV from stdin-equivalent storage (`--csv <path>`)
//! rather than re-measuring, which is how a table is regenerated from a run
//! captured on a quieter machine than the one rendering it.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

/// Bench binary that owns the measurement contract.
const BENCH_NAME: &str = "rustfft_comparison";
/// Rendered artifact.
const OUTPUT: &str = "docs/benchmark_results.md";
/// Environment switch the bench binary reads for the dense sweep.
const FULL_SWEEP_VAR: &str = "APOLLO_BENCH_FULL_SWEEP";

/// One size's four measurements, in nanoseconds.
#[derive(Default, Clone, Copy)]
struct Row {
    apollo_f64: Option<u128>,
    rustfft_f64: Option<u128>,
    apollo_f32: Option<u128>,
    rustfft_f32: Option<u128>,
}

impl Row {
    /// Ratio rendered as `x`, or `-` when either side is missing.
    fn ratio(apollo: Option<u128>, rustfft: Option<u128>) -> String {
        match (apollo, rustfft) {
            (Some(a), Some(r)) if r > 0 => format!("{:.3}x", a as f64 / r as f64),
            _ => "-".to_owned(),
        }
    }

    fn cell(value: Option<u128>) -> String {
        value.map_or_else(|| "-".to_owned(), |v| v.to_string())
    }
}

pub fn run(mut args: impl Iterator<Item = String>) -> Result<()> {
    let mut full = false;
    let mut skip_run = false;
    let mut csv_path: Option<PathBuf> = None;
    let mut root = PathBuf::from(".");

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--full" => full = true,
            "--skip-run" => skip_run = true,
            "--csv" => {
                csv_path = Some(PathBuf::from(args.next().context("--csv requires a path")?));
            }
            "--root" => {
                root = PathBuf::from(args.next().context("--root requires a path")?);
            }
            other => bail!("unknown benchmark option `{other}`"),
        }
    }

    let csv = if skip_run {
        let path = csv_path.context("--skip-run requires --csv <path>")?;
        fs::read_to_string(&path)
            .with_context(|| format!("reading benchmark CSV from {}", path.display()))?
    } else {
        measure(&root, full)?
    };

    let rows = parse(&csv);
    if rows.is_empty() {
        bail!("no `fft_forward_clone_inclusive` rows found in the benchmark CSV");
    }

    let table = render(&rows, full);
    let output = root.join(OUTPUT);
    fs::write(&output, table).with_context(|| format!("writing {}", output.display()))?;
    eprintln!(
        "benchmark: wrote {} ({} sizes)",
        output.display(),
        rows.len()
    );
    Ok(())
}

/// Runs the bench binary in release and returns its CSV on stdout.
fn measure(root: &Path, full: bool) -> Result<String> {
    let mut command = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned()));
    command
        .current_dir(root)
        .args(["bench", "-p", "apollo-fft", "--bench", BENCH_NAME]);
    if full {
        command.env(FULL_SWEEP_VAR, "1");
    }

    eprintln!(
        "benchmark: running {BENCH_NAME} ({} sweep)",
        if full { "full" } else { "default" }
    );
    let output = command
        .output()
        .with_context(|| format!("running the {BENCH_NAME} bench binary"))?;

    // The bench binary writes its budget and progress lines to stderr; surface
    // them so a recorded run is self-describing rather than silent.
    let stderr = String::from_utf8_lossy(&output.stderr);
    for line in stderr.lines().filter(|l| l.contains(BENCH_NAME)) {
        eprintln!("  {line}");
    }

    if !output.status.success() {
        bail!(
            "{BENCH_NAME} failed with {}; stderr:\n{stderr}",
            output.status
        );
    }

    String::from_utf8(output.stdout).context("benchmark CSV was not valid UTF-8")
}

/// Extracts `fft_forward_clone_inclusive` medians keyed by transform length.
///
/// The CSV carries one row per case as
/// `case,min_ns,median_ns,...` where `case` is `group/operation/parameter`.
/// Unrelated groups are ignored so the parser tolerates a combined report.
fn parse(csv: &str) -> BTreeMap<usize, Row> {
    let mut rows: BTreeMap<usize, Row> = BTreeMap::new();

    for line in csv.lines().skip_while(|l| l.starts_with("case,")) {
        let mut fields = line.split(',');
        let (Some(case), Some(_min), Some(median)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };

        let mut parts = case.split('/');
        let (Some(group), Some(operation), Some(parameter)) =
            (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if group != "fft_forward_clone_inclusive" {
            continue;
        }

        let Ok(len) = parameter.parse::<usize>() else {
            continue;
        };
        let Ok(median) = median.parse::<u128>() else {
            continue;
        };

        let row = rows.entry(len).or_default();
        match operation {
            "apollo_f64" => row.apollo_f64 = Some(median),
            "rustfft_f64" => row.rustfft_f64 = Some(median),
            "apollo_f32" => row.apollo_f32 = Some(median),
            "rustfft_f32" => row.rustfft_f32 = Some(median),
            _ => {}
        }
    }

    rows
}

fn render(rows: &BTreeMap<usize, Row>, full: bool) -> String {
    let mut out = String::new();
    out.push_str("# Benchmark Results\n\n");
    out.push_str("Generated by `cargo run -p xtask -- benchmark`.\n\n");
    let _ = write!(
        out,
        "Source: `crates/apollo-fft/benches/{BENCH_NAME}.rs`, which owns the\n\
         measurement contract, sweep sizes, and runtime budget. `--full` selects the\n\
         dense 1..=500 sweep; `--skip-run --csv <path>` re-renders a saved run.\n\n"
    );
    let _ = write!(
        out,
        "Sweep: {}. Benchmark: clone-inclusive 1D forward complex FFT — each timed\n\
         iteration restores its input, so the copy is charged to both engines and\n\
         cancels in the ratio. Plans are built once, outside the timed region.\n\n",
        if full {
            "full (1..=500)"
        } else {
            "default (representative sizes)"
        }
    );
    out.push_str(
        "Values are median point estimates in nanoseconds; lower is better.\n\
         `Apollo/RustFFT < 1.000x` means Apollo is faster.\n\n",
    );
    out.push_str(
        "Engine-selection columns are absent by design: neither Apollo nor RustFFT\n\
         exposes its chosen algorithm through a public API, so they cannot be\n\
         regenerated and are not invented here.\n\n",
    );
    out.push_str(
        "## Reading these numbers\n\n\
         Medians are integer nanoseconds. At the smallest sizes a transform costs\n\
         only a few nanoseconds, so quantization alone is a large fraction of the\n\
         value and a ratio such as `2.000x` on a 1 ns versus 2 ns row carries no\n\
         signal. Treat sizes below roughly 16 as presence checks, not comparisons.\n\n\
         A run is only as quiet as its host. Concurrent builds on the machine that\n\
         produced a table inflate every absolute figure, and not uniformly. Before\n\
         citing a row as a competitive claim, regenerate it on an idle machine and\n\
         confirm the figure reproduces.\n\n",
    );
    out.push_str(
        "| Size | f64 Apollo (ns) | f64 RustFFT (ns) | f64 Apollo/RustFFT | \
         f32 Apollo (ns) | f32 RustFFT (ns) | f32 Apollo/RustFFT |\n",
    );
    out.push_str("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");

    for (len, row) in rows {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} |",
            len,
            Row::cell(row.apollo_f64),
            Row::cell(row.rustfft_f64),
            Row::ratio(row.apollo_f64, row.rustfft_f64),
            Row::cell(row.apollo_f32),
            Row::cell(row.rustfft_f32),
            Row::ratio(row.apollo_f32, row.rustfft_f32),
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "case,min_ns,median_ns,median_lower_ns,median_upper_ns\n\
fft_forward_clone_inclusive/apollo_f64/8,10,12,12,13\n\
fft_forward_clone_inclusive/rustfft_f64/8,20,24,24,25\n\
fft_forward_clone_inclusive/apollo_f32/8,5,6,6,7\n\
fft_forward_clone_inclusive/rustfft_f32/8,10,12,12,13\n\
some_other_group/apollo_f64/8,99,99,99,99\n";

    #[test]
    fn parses_only_the_comparison_group() {
        let rows = parse(SAMPLE);
        assert_eq!(rows.len(), 1, "unrelated groups must be ignored");
        let row = rows[&8];
        assert_eq!(row.apollo_f64, Some(12));
        assert_eq!(row.rustfft_f64, Some(24));
        assert_eq!(row.apollo_f32, Some(6));
        assert_eq!(row.rustfft_f32, Some(12));
    }

    #[test]
    fn ratio_is_apollo_over_rustfft() {
        // Apollo at half the time must read as 0.500x, not 2.000x.
        assert_eq!(Row::ratio(Some(12), Some(24)), "0.500x");
        assert_eq!(Row::ratio(Some(24), Some(12)), "2.000x");
    }

    #[test]
    fn missing_or_zero_measurements_render_as_dash() {
        assert_eq!(Row::ratio(None, Some(24)), "-");
        assert_eq!(Row::ratio(Some(12), None), "-");
        assert_eq!(Row::ratio(Some(12), Some(0)), "-");
        assert_eq!(Row::cell(None), "-");
    }

    #[test]
    fn rendered_table_has_one_row_per_size_in_ascending_order() {
        let rows = parse(SAMPLE);
        let table = render(&rows, false);
        assert!(table.contains("| 8 | 12 | 24 | 0.500x | 6 | 12 | 0.500x |"));
        assert!(
            table.contains("Generated by `cargo run -p xtask -- benchmark`"),
            "header must state the regeneration command"
        );
    }
}
