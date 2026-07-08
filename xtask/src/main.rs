#![allow(missing_docs)]

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;
#[cfg(feature = "bench-runner")]
use std::time::{Duration, Instant};

mod provider_audit;

const F64_GROUP: &str = "apollo_fft_vs_rustfft_f64";
const F32_GROUP: &str = "apollo_fft_vs_rustfft_f32";
const APOLLO_BENCH: &str = "apollo_clone_inclusive";
const RUSTFFT_BENCH: &str = "rustfft_clone_inclusive";
const RADER_HALF_CYCLIC_THRESHOLD: usize = 1024;
const CANONICAL_SIZES: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
    137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
    156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
    175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
    213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
    232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
    251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
    289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
    308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,
    327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
    346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364,
    365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
    384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
    403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421,
    422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440,
    441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
    460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
    479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497,
    498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 10_007, 32_768,
];

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    reexec_optimized_benchmark_runner_if_needed(&args)?;
    let mut args = args.into_iter();
    match args.next().as_deref() {
        None | Some("benchmark") => run_benchmark(BenchmarkArgs::parse(args)?),
        Some("provider-audit") => provider_audit::run(args),
        Some("-h" | "--help" | "help") => {
            print_help();
            Ok(())
        }
        Some(command) => bail!("unknown xtask command `{command}`"),
    }
}

fn reexec_optimized_benchmark_runner_if_needed(args: &[String]) -> Result<()> {
    let benchmark_command = matches!(args.first().map(String::as_str), None | Some("benchmark"));
    let skip_run = args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--skip-run" | "--no-run"));
    if !benchmark_command || skip_run || (cfg!(feature = "bench-runner") && !cfg!(debug_assertions))
    {
        return Ok(());
    }

    let mut command = Command::new("cargo");
    command.env("RUSTFLAGS", "-C target-cpu=native");
    command.args([
        "run",
        "-p",
        "xtask",
        "--profile",
        "bench",
        "--features",
        "bench-runner",
        "--",
    ]);
    command.args(args);
    let status = command
        .status()
        .context("failed to launch optimized xtask benchmark runner")?;
    std::process::exit(status.code().unwrap_or(1));
}

#[derive(Debug)]
struct BenchmarkArgs {
    sizes: Option<RequestedSizes>,
    skip_run: bool,
    output: PathBuf,
    criterion_root: PathBuf,
    profile: BenchmarkProfile,
}

#[derive(Debug)]
struct RequestedSizes {
    values: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
enum BenchmarkProfile {
    Quick,
    Full,
}

impl BenchmarkProfile {
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "quick" => Ok(Self::Quick),
            "full" => Ok(Self::Full),
            _ => bail!("benchmark profile must be `quick` or `full`"),
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Quick => "quick: sample_size=3, measurement_time=30ms, warm_up_time=5ms",
            Self::Full => "full: sample_size=30, measurement_time=2s, warm_up_time=250ms",
        }
    }

    #[cfg(feature = "bench-runner")]
    const fn sample_size(self) -> usize {
        match self {
            Self::Quick => 3,
            Self::Full => 30,
        }
    }

    #[cfg(feature = "bench-runner")]
    const fn measurement_time(self) -> Duration {
        match self {
            Self::Quick => Duration::from_millis(30),
            Self::Full => Duration::from_secs(2),
        }
    }

    #[cfg(feature = "bench-runner")]
    const fn warm_up_time(self) -> Duration {
        match self {
            Self::Quick => Duration::from_millis(5),
            Self::Full => Duration::from_millis(250),
        }
    }
}

impl BenchmarkArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut parsed = Self {
            sizes: None,
            skip_run: false,
            output: PathBuf::from("benchmark_results.md"),
            criterion_root: PathBuf::from("target/criterion"),
            profile: BenchmarkProfile::Quick,
        };
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--sizes" => {
                    let sizes = args
                        .next()
                        .context("--sizes requires a comma-separated value")?;
                    parsed.sizes = Some(parse_requested_sizes(&sizes)?);
                }
                "--all" => parsed.sizes = None,
                "--skip-run" | "--no-run" => parsed.skip_run = true,
                "--output" => {
                    parsed.output =
                        PathBuf::from(args.next().context("--output requires a markdown path")?);
                }
                "--criterion-root" => {
                    parsed.criterion_root = PathBuf::from(
                        args.next()
                            .context("--criterion-root requires a directory")?,
                    );
                }
                "--profile" => {
                    let profile = args
                        .next()
                        .context("--profile requires `quick` or `full`")?;
                    parsed.profile = BenchmarkProfile::parse(profile.trim())?;
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown benchmark option `{other}`"),
            }
        }
        Ok(parsed)
    }
}

fn parse_requested_sizes(raw: &str) -> Result<RequestedSizes> {
    let mut sizes = BTreeSet::new();
    for part in raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let size = part
            .parse::<usize>()
            .with_context(|| format!("invalid benchmark size `{part}`"))?;
        if size == 0 {
            bail!("benchmark sizes must be positive");
        }
        sizes.insert(size);
    }
    if sizes.is_empty() {
        bail!("--sizes must contain at least one positive size");
    }
    let values = sizes.into_iter().collect::<Vec<_>>();
    Ok(RequestedSizes { values })
}

fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    let (f64, f32) = if args.skip_run {
        if let Some(sizes) = args.sizes.as_ref() {
            validate_requested_measurements(&args.criterion_root, &sizes.values, None)?;
        }
        (
            load_pair(&args.criterion_root, F64_GROUP)?,
            load_pair(&args.criterion_root, F32_GROUP)?,
        )
    } else {
        let sizes = args
            .sizes
            .as_ref()
            .map_or_else(|| CANONICAL_SIZES.to_vec(), |sizes| sizes.values.clone());
        #[cfg(feature = "bench-runner")]
        {
            run_xtask_benchmark(&sizes, args.profile)
        }
        #[cfg(not(feature = "bench-runner"))]
        {
            run_xtask_benchmark(&sizes, args.profile)?
        }
    };
    match args.sizes.as_ref() {
        Some(sizes) => write_subset_table(&args.output, &f64, &f32, args.profile, &sizes.values)?,
        None => write_table(&args.output, &f64, &f32, args.profile)?,
    }
    println!("wrote {}", args.output.display());
    Ok(())
}

type BenchmarkRows = (BTreeMap<usize, (f64, f64)>, BTreeMap<usize, (f64, f64)>);

#[cfg(not(feature = "bench-runner"))]
fn run_xtask_benchmark(_sizes: &[usize], _profile: BenchmarkProfile) -> Result<BenchmarkRows> {
    bail!("xtask benchmark runner must be executed with the `bench-runner` feature")
}

#[cfg(feature = "bench-runner")]
fn run_xtask_benchmark(sizes: &[usize], profile: BenchmarkProfile) -> BenchmarkRows {
    let mut double_rows = BTreeMap::new();
    let mut single_rows = BTreeMap::new();
    for &size in sizes {
        println!("benchmarking size {size}");
        double_rows.insert(size, bench_pair::<DoublePrecision>(size, profile));
        single_rows.insert(size, bench_pair::<SinglePrecision>(size, profile));
    }
    (double_rows, single_rows)
}

#[cfg(feature = "bench-runner")]
trait PrecisionBenchmark {
    type Scalar: rustfft::FftNum;
    type Complex: Clone + Copy + 'static;
    type Plan;

    fn signal(len: usize) -> Vec<Self::Complex>;
    fn plan(len: usize) -> Self::Plan;
    fn apollo_forward_planned(plan: &Self::Plan, data: &mut [Self::Complex]);
    fn rustfft_input(input: &[Self::Complex]) -> Vec<rustfft::num_complex::Complex<Self::Scalar>>;
    fn rustfft_zero() -> rustfft::num_complex::Complex<Self::Scalar>;
}

#[cfg(feature = "bench-runner")]
struct DoublePrecision;

#[cfg(feature = "bench-runner")]
struct SinglePrecision;

#[cfg(feature = "bench-runner")]
impl PrecisionBenchmark for DoublePrecision {
    type Scalar = f64;
    type Complex = eunomia::Complex64;
    type Plan = std::sync::Arc<apollo_fft::FftPlan1D<f64>>;

    #[inline]
    fn signal(len: usize) -> Vec<Self::Complex> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                eunomia::Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect()
    }

    #[inline]
    fn plan(len: usize) -> Self::Plan {
        use apollo_fft::PlanCacheProvider;
        <f64 as PlanCacheProvider>::get_1d_plan(apollo_fft::Shape1D::new(len).unwrap())
    }

    #[inline]
    fn apollo_forward_planned(plan: &Self::Plan, data: &mut [Self::Complex]) {
        plan.forward_complex_slice_inplace(data);
    }

    #[inline]
    fn rustfft_input(input: &[Self::Complex]) -> Vec<rustfft::num_complex::Complex<Self::Scalar>> {
        input
            .iter()
            .map(|z| rustfft::num_complex::Complex::new(z.re, z.im))
            .collect()
    }

    #[inline]
    fn rustfft_zero() -> rustfft::num_complex::Complex<Self::Scalar> {
        rustfft::num_complex::Complex::new(0.0, 0.0)
    }
}

#[cfg(feature = "bench-runner")]
impl PrecisionBenchmark for SinglePrecision {
    type Scalar = f32;
    type Complex = eunomia::Complex32;
    type Plan = std::sync::Arc<apollo_fft::FftPlan1D<f32>>;

    #[inline]
    fn signal(len: usize) -> Vec<Self::Complex> {
        (0..len)
            .map(|i| {
                let x = i as f32;
                eunomia::Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect()
    }

    #[inline]
    fn plan(len: usize) -> Self::Plan {
        use apollo_fft::PlanCacheProvider;
        <f32 as PlanCacheProvider>::get_1d_plan(apollo_fft::Shape1D::new(len).unwrap())
    }

    #[inline]
    fn apollo_forward_planned(plan: &Self::Plan, data: &mut [Self::Complex]) {
        plan.forward_complex_slice_inplace(data);
    }

    #[inline]
    fn rustfft_input(input: &[Self::Complex]) -> Vec<rustfft::num_complex::Complex<Self::Scalar>> {
        input
            .iter()
            .map(|z| rustfft::num_complex::Complex::new(z.re, z.im))
            .collect()
    }

    #[inline]
    fn rustfft_zero() -> rustfft::num_complex::Complex<Self::Scalar> {
        rustfft::num_complex::Complex::new(0.0, 0.0)
    }
}

#[cfg(feature = "bench-runner")]
#[inline]
fn bench_pair<P: PrecisionBenchmark>(len: usize, profile: BenchmarkProfile) -> (f64, f64) {
    let input = P::signal(len);

    let mut data = input.clone();
    let apollo_copy = measure_operation(profile, || {
        data.copy_from_slice(&input);
        std::hint::black_box(&data);
    });

    let apollo_plan = P::plan(len);
    let apollo_total = measure_operation(profile, || {
        data.copy_from_slice(&input);
        P::apollo_forward_planned(&apollo_plan, std::hint::black_box(&mut data));
        std::hint::black_box(&data);
    });

    let rustfft_input = P::rustfft_input(&input);
    let mut rust_data = rustfft_input.clone();
    let rust_copy = measure_operation(profile, || {
        rust_data.copy_from_slice(&rustfft_input);
        std::hint::black_box(&rust_data);
    });

    let mut planner = rustfft::FftPlanner::<P::Scalar>::new();
    let rustfft = planner.plan_fft_forward(len);
    let mut rustfft_scratch = vec![P::rustfft_zero(); rustfft.get_inplace_scratch_len()];
    let rust_total = measure_operation(profile, || {
        rust_data.copy_from_slice(&rustfft_input);
        rustfft.process_with_scratch(std::hint::black_box(&mut rust_data), &mut rustfft_scratch);
        std::hint::black_box(&rust_data);
    });

    let apollo = (apollo_total - apollo_copy).max(0.001);
    let rust = (rust_total - rust_copy).max(0.001);

    (apollo, rust)
}

#[cfg(feature = "bench-runner")]
fn measure_operation(profile: BenchmarkProfile, mut operation: impl FnMut()) -> f64 {
    let warmup_start = Instant::now();
    while warmup_start.elapsed() < profile.warm_up_time() {
        operation();
    }

    let sample_target = sample_target(profile);
    let batch_iters = calibrated_batch_iters(sample_target, &mut operation);
    let n = profile.sample_size();
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(elapsed_per_iter_ns(batch_iters, &mut operation));
    }
    // Median point estimate: robust to OS-scheduling / interrupt outliers that
    // inflate the arithmetic mean for sub-microsecond latency measurements.
    // Applied identically to Apollo and RustFFT, so the comparison stays fair;
    // this is the standard estimator for latency microbenchmarks (vs the prior
    // mean, which exhibited large run-to-run variance on small sizes).
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = n / 2;
    if n == 0 {
        0.0
    } else if n % 2 == 0 {
        f64::midpoint(samples[mid - 1], samples[mid])
    } else {
        samples[mid]
    }
}

#[cfg(feature = "bench-runner")]
fn sample_target(profile: BenchmarkProfile) -> Duration {
    let nanos = profile.measurement_time().as_nanos() / profile.sample_size() as u128;
    Duration::from_nanos(nanos.max(1) as u64)
}

#[cfg(feature = "bench-runner")]
fn calibrated_batch_iters(sample_target: Duration, operation: &mut impl FnMut()) -> u64 {
    let mut iters = 1u64;
    loop {
        let elapsed = elapsed_for_iters(iters, operation);
        if elapsed >= sample_target / 4 || iters >= 1 << 24 {
            let elapsed_ns = elapsed.as_nanos().max(1);
            let target_ns = sample_target.as_nanos().max(1);
            let scaled = (iters as u128 * target_ns).div_ceil(elapsed_ns);
            return scaled.clamp(1, 1 << 24) as u64;
        }
        iters *= 2;
    }
}

#[cfg(feature = "bench-runner")]
fn elapsed_per_iter_ns(iters: u64, operation: &mut impl FnMut()) -> f64 {
    elapsed_for_iters(iters, operation).as_secs_f64() * 1.0e9 / iters as f64
}

#[cfg(feature = "bench-runner")]
fn elapsed_for_iters(iters: u64, operation: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        operation();
    }
    start.elapsed()
}

fn load_pair(root: &Path, group: &str) -> Result<BTreeMap<usize, (f64, f64)>> {
    let apollo = load_bench(root, group, APOLLO_BENCH)?;
    let rustfft = load_bench(root, group, RUSTFFT_BENCH)?;
    let mut paired = BTreeMap::new();
    for size in apollo.keys().filter(|size| rustfft.contains_key(size)) {
        paired.insert(*size, (apollo[size], rustfft[size]));
    }
    Ok(paired)
}

fn load_bench(root: &Path, group: &str, bench: &str) -> Result<BTreeMap<usize, f64>> {
    let mut values = BTreeMap::new();
    let dir = root.join(group).join(bench);
    if !dir.exists() {
        return Ok(values);
    }
    for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let size_text = entry.file_name();
        let Some(size_text) = size_text.to_str() else {
            continue;
        };
        let Ok(size) = size_text.parse::<usize>() else {
            continue;
        };
        let estimates = entry.path().join("new").join("estimates.json");
        if let Some(mean) = mean_ns(&estimates)? {
            values.insert(size, mean);
        }
    }
    Ok(values)
}

fn mean_ns(path: &Path) -> Result<Option<f64>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let json: Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    Ok(json
        .get("mean")
        .and_then(|mean| mean.get("point_estimate"))
        .and_then(Value::as_f64))
}

fn validate_requested_measurements(
    root: &Path,
    sizes: &[usize],
    min_modified: Option<SystemTime>,
) -> Result<()> {
    for &size in sizes {
        for (group, bench) in [
            (F64_GROUP, APOLLO_BENCH),
            (F64_GROUP, RUSTFFT_BENCH),
            (F32_GROUP, APOLLO_BENCH),
            (F32_GROUP, RUSTFFT_BENCH),
        ] {
            let path = estimate_path(root, group, bench, size);
            let metadata = fs::metadata(&path).with_context(|| {
                format!("missing requested Criterion estimate {}", path.display())
            })?;
            if let Some(min_modified) = min_modified {
                let modified = metadata
                    .modified()
                    .with_context(|| format!("read modified time for {}", path.display()))?;
                if modified < min_modified {
                    bail!(
                        "stale requested Criterion estimate {}; rerun without --skip-run or remove the stale cache",
                        path.display()
                    );
                }
            }
        }
    }
    Ok(())
}

fn estimate_path(root: &Path, group: &str, bench: &str, size: usize) -> PathBuf {
    root.join(group)
        .join(bench)
        .join(size.to_string())
        .join("new")
        .join("estimates.json")
}

fn write_table(
    output: &Path,
    f64_rows: &BTreeMap<usize, (f64, f64)>,
    f32_rows: &BTreeMap<usize, (f64, f64)>,
    profile: BenchmarkProfile,
) -> Result<()> {
    let sizes = f64_rows
        .keys()
        .chain(f32_rows.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    write_table_for_sizes(output, f64_rows, f32_rows, profile, sizes.iter().copied())
}

fn write_table_for_sizes(
    output: &Path,
    f64_rows: &BTreeMap<usize, (f64, f64)>,
    f32_rows: &BTreeMap<usize, (f64, f64)>,
    profile: BenchmarkProfile,
    sizes: impl IntoIterator<Item = usize>,
) -> Result<()> {
    let mut lines = table_header(profile);

    for size in sizes {
        lines.push(optional_row(size, f64_rows, f32_rows));
    }

    fs::write(output, lines.join("\n") + "\n")
        .with_context(|| format!("write {}", output.display()))
}

fn write_subset_table(
    output: &Path,
    f64_rows: &BTreeMap<usize, (f64, f64)>,
    f32_rows: &BTreeMap<usize, (f64, f64)>,
    profile: BenchmarkProfile,
    sizes: &[usize],
) -> Result<()> {
    let mut replacements = BTreeMap::new();
    for &size in sizes {
        replacements.insert(size, required_row(size, f64_rows, f32_rows)?);
    }

    if output.exists() {
        let existing =
            fs::read_to_string(output).with_context(|| format!("read {}", output.display()))?;
        let merged = merge_subset_rows(&existing, &replacements, profile)?;
        fs::write(output, merged).with_context(|| format!("write {}", output.display()))?;
        return Ok(());
    }

    write_table_for_sizes(output, f64_rows, f32_rows, profile, sizes.iter().copied())
}

fn table_header(profile: BenchmarkProfile) -> Vec<String> {
    vec![
        "# Benchmark Results".to_string(),
        String::new(),
        "Generated by `cargo run -p xtask -- benchmark`.".to_string(),
        "Source: `xtask` bounded adaptive clone-inclusive runner; `--skip-run` merges existing Criterion JSON estimates.".to_string(),
        format!("Benchmark profile: `{}`.", profile.description()),
        "Benchmark: clone-inclusive 1D forward complex FFT. Values are median point estimates in nanoseconds.".to_string(),
        "Lower time is better. `Apollo/RustFFT < 1.000x` means Apollo is faster.".to_string(),
        String::new(),
        "| Size | Apollo Engine | RustFFT Engine | f64 Apollo (ns) | f64 RustFFT (ns) | f64 Apollo/RustFFT | f32 Apollo (ns) | f32 RustFFT (ns) | f32 Apollo/RustFFT | Last Updated |".to_string(),
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |".to_string(),
    ]
}

fn optional_row(
    size: usize,
    f64_rows: &BTreeMap<usize, (f64, f64)>,
    f32_rows: &BTreeMap<usize, (f64, f64)>,
) -> String {
    let (f64_apollo, f64_rustfft) = f64_rows
        .get(&size)
        .map_or((None, None), |(apollo, rustfft)| {
            (Some(*apollo), Some(*rustfft))
        });
    let (f32_apollo, f32_rustfft) = f32_rows
        .get(&size)
        .map_or((None, None), |(apollo, rustfft)| {
            (Some(*apollo), Some(*rustfft))
        });
    format_measurement_row(size, f64_apollo, f64_rustfft, f32_apollo, f32_rustfft)
}

fn required_row(
    size: usize,
    f64_rows: &BTreeMap<usize, (f64, f64)>,
    f32_rows: &BTreeMap<usize, (f64, f64)>,
) -> Result<String> {
    let (f64_apollo, f64_rustfft) = f64_rows
        .get(&size)
        .copied()
        .with_context(|| format!("missing f64 Criterion pair for requested size {size}"))?;
    let (f32_apollo, f32_rustfft) = f32_rows
        .get(&size)
        .copied()
        .with_context(|| format!("missing f32 Criterion pair for requested size {size}"))?;
    Ok(format_measurement_row(
        size,
        Some(f64_apollo),
        Some(f64_rustfft),
        Some(f32_apollo),
        Some(f32_rustfft),
    ))
}

fn format_measurement_row(
    size: usize,
    f64_apollo: Option<f64>,
    f64_rustfft: Option<f64>,
    f32_apollo: Option<f64>,
    f32_rustfft: Option<f64>,
) -> String {
    let apollo_engine = get_engine_name(size);
    let rustfft_engine = get_rustfft_engine_name(size);
    let timestamp = current_utc_timestamp();
    format!(
        "| {size} | {apollo_engine} | {rustfft_engine} | {} | {} | {} | {} | {} | {} | {timestamp} |",
        fmt(f64_apollo),
        fmt(f64_rustfft),
        ratio(f64_apollo, f64_rustfft),
        fmt(f32_apollo),
        fmt(f32_rustfft),
        ratio(f32_apollo, f32_rustfft)
    )
}

fn merge_subset_rows(
    existing: &str,
    replacements: &BTreeMap<usize, String>,
    profile: BenchmarkProfile,
) -> Result<String> {
    let mut lines = existing.lines().map(str::to_string).collect::<Vec<_>>();
    let header = table_header(profile);
    for line in &mut lines {
        if line.starts_with("Source:") {
            line.clone_from(&header[3]);
        }
        if line.starts_with("Benchmark profile:") {
            *line = format!("Benchmark profile: `{}`.", profile.description());
        }
    }

    let separator = lines
        .iter()
        .position(|line| line.trim_start().starts_with("| ---:"))
        .context("benchmark markdown table separator not found")?;
    let data_start = separator + 1;
    let mut data_end = data_start;
    while data_end < lines.len() && parse_row_size(&lines[data_end]).is_some() {
        data_end += 1;
    }

    let mut rows = BTreeMap::new();
    for line in &lines[data_start..data_end] {
        if let Some(size) = parse_row_size(line) {
            rows.insert(size, line.clone());
        }
    }
    for (&size, row) in replacements {
        rows.insert(size, row.clone());
    }

    lines.splice(data_start..data_end, rows.into_values());
    Ok(lines.join("\n") + "\n")
}

fn parse_row_size(line: &str) -> Option<usize> {
    line.split('|').nth(1)?.trim().parse().ok()
}

fn fmt(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.2}"))
}

fn ratio(apollo: Option<f64>, rustfft: Option<f64>) -> String {
    match (apollo, rustfft) {
        (Some(apollo), Some(rustfft)) => format!("{:.3}x", apollo / rustfft),
        _ => "-".to_string(),
    }
}

fn print_help() {
    println!(
        "Usage:\n  cargo run -p xtask -- benchmark [--sizes 33,38,58] [--skip-run]\n  cargo run -p xtask -- provider-audit [--root <path>]\n\nBenchmark options:\n  --sizes <csv>       Benchmark these FFT sizes and merge only those rows into the markdown table.\n  --all               Benchmark the full canonical size set and rewrite the markdown table.\n  --profile <name>    Timing profile: quick or full. Defaults to quick.\n  --skip-run          Update the markdown table from existing Criterion JSON without measuring.\n  --output <path>     Markdown output path. Defaults to benchmark_results.md.\n\nProvider audit options:\n  --root <path>       Workspace root to inspect. Defaults to the current directory."
    );
}

fn get_engine_name(n: usize) -> &'static str {
    if n <= 1 {
        return "Identity";
    }
    if n.is_power_of_two() {
        if n < 32 {
            return "Winograd";
        }
        if n >= 32768 && n.trailing_zeros() % 2 == 0 {
            return "Four-Step";
        }
        return "Stockham";
    }
    if matches!(n, 144 | 176 | 180) {
        return "Cooley-Tukey";
    }
    let is_short_winograd = matches!(
        n,
        3 | 5
            | 6
            | 7
            | 9
            | 10
            | 11
            | 12
            | 13
            | 14
            | 15
            | 17
            | 18
            | 19
            | 20
            | 21
            | 22
            | 23
            | 24
            | 25
            | 26
            | 27
            | 28
            | 29
            | 30
            | 31
            | 33
            | 34
            | 35
            | 36
            | 37
            | 38
            | 39
            | 40
            | 41
            | 42
            | 43
            | 44
            | 45
            | 46
            | 47
            | 48
            | 49
            | 50
            | 51
            | 52
            | 53
            | 54
            | 55
            | 56
            | 58
            | 60
            | 62
            | 63
            | 81
    );
    if is_short_winograd {
        return "Winograd";
    }
    if matches!(n, 72 | 144 | 180 | 484) {
        return "Precision Policy"; // reduced; others now select Composite/GT via fixed selection for perf
    }
    if let Some((n1, n2)) = get_coprime_factors(n) {
        let short_sizes: [usize; 31] = [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 27, 29, 31,
            32, 36, 37, 41, 43, 47, 53,
        ];
        let has_static = (short_sizes.contains(&n1) && short_sizes.contains(&n2) && n <= 200)
            || (n1 == 3 && [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53].contains(&n2))
            || (n2 == 3 && [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53].contains(&n1))
            || n == 84;
        if has_static {
            return "Good-Thomas (Static)";
        }
    }
    if is_prime23_smooth(n) {
        return "Cooley-Tukey";
    }
    if get_coprime_factors(n).is_some() {
        return "Good-Thomas";
    }
    if is_prime(n) {
        if n == 113 {
            return "Rader (Precision Policy)";
        }
        if n > RADER_HALF_CYCLIC_THRESHOLD {
            return "Rader (Precision Policy)";
        }
        return "Rader";
    }
    "Mixed-Radix"
}

fn get_coprime_factors(n: usize) -> Option<(usize, usize)> {
    let mut remaining = n;
    let mut prime_powers = Vec::new();
    let mut p = 2usize;
    while p * p <= remaining {
        if remaining % p == 0 {
            let mut power = 1usize;
            while remaining % p == 0 {
                power *= p;
                remaining /= p;
            }
            prime_powers.push(power);
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if remaining > 1 {
        prime_powers.push(remaining);
    }
    if prime_powers.len() < 2 {
        return None;
    }
    let n1 = prime_powers.pop()?;
    let n2 = prime_powers.into_iter().product();
    Some((n1, n2))
}

fn is_prime23_smooth(n: usize) -> bool {
    let mut remaining = n;
    for &p in &[2, 3, 5, 7, 11, 13, 17, 23] {
        while remaining % p == 0 {
            remaining /= p;
        }
    }
    remaining == 1
}

fn is_prime(n: usize) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

fn get_rustfft_engine_name(n: usize) -> &'static str {
    if n <= 1 {
        return "Identity";
    }

    let is_avx_butterfly = |len: usize| -> bool {
        matches!(
            len,
            0 | 1
                | 2
                | 3
                | 4
                | 5
                | 6
                | 7
                | 8
                | 9
                | 11
                | 12
                | 13
                | 16
                | 17
                | 18
                | 19
                | 23
                | 24
                | 27
                | 29
                | 31
                | 32
                | 36
                | 48
                | 54
                | 64
                | 72
                | 128
                | 256
                | 512
        )
    };

    if is_avx_butterfly(n) {
        return "Butterfly";
    }

    let mut temp = n;
    for &p in &[2, 3, 5, 7, 11] {
        while temp % p == 0 {
            temp /= p;
        }
    }
    let other = temp;

    if other > 1 {
        if is_avx_butterfly(other) {
            return "Mixed-Radix";
        }
        if is_prime(other) {
            let mut inner = other - 1;
            for &p in &[2, 3, 5, 7, 11] {
                while inner % p == 0 {
                    inner /= p;
                }
            }
            if is_avx_butterfly(inner) {
                if n == other {
                    return "Rader";
                }
                return "Mixed-Radix (Rader)";
            }
        }
        if n == other {
            return "Bluestein";
        }
        return "Mixed-Radix (Bluestein)";
    }

    "Mixed-Radix"
}

fn current_utc_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days = secs / 86400;
    let seconds_of_day = secs % 86400;

    let hours = seconds_of_day / 3600;
    let minutes = (seconds_of_day % 3600) / 60;

    let mut year = 1970;
    let mut days_left = days;

    loop {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if is_leap { 366 } else { 365 };
        if days_left < days_in_year {
            break;
        }
        days_left -= days_in_year;
        year += 1;
    }

    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_lengths = if is_leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &length in &month_lengths {
        if days_left < length {
            break;
        }
        days_left -= length;
        month += 1;
    }

    let day = days_left + 1;

    format!("{year:04}-{month:02}-{day:02} {hours:02}:{minutes:02} UTC")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_sizes_are_sorted_and_deduplicated() {
        let sizes = parse_requested_sizes("10, 3,10").expect("valid sizes");
        assert_eq!(sizes.values, vec![3, 10]);
    }

    #[test]
    fn subset_merge_replaces_requested_rows_and_preserves_others() {
        let existing = [
            "# Benchmark Results",
            "",
            "Generated by `cargo run -p xtask -- benchmark`.",
            "Source: `xtask` bounded adaptive clone-inclusive runner; `--skip-run` merges existing Criterion JSON estimates.",
            "Benchmark profile: `old`.",
            "Benchmark: clone-inclusive 1D forward complex FFT. Values are median point estimates in nanoseconds.",
            "Lower time is better. `Apollo/RustFFT < 1.000x` means Apollo is faster.",
            "",
            "| Size | Apollo Engine | RustFFT Engine | f64 Apollo (ns) | f64 RustFFT (ns) | f64 Apollo/RustFFT | f32 Apollo (ns) | f32 RustFFT (ns) | f32 Apollo/RustFFT | Last Updated |",
            "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
            "| 3 | Winograd | Butterfly | 31.00 | 32.00 | 0.969x | 33.00 | 34.00 | 0.971x | 2026-05-19 12:00 UTC |",
            "| 10 | Good-Thomas (Static) | Mixed-Radix | 160.00 | 50.00 | 3.200x | 160.00 | 50.00 | 3.200x | 2026-05-19 12:00 UTC |",
            "| 12 | Good-Thomas (Static) | Mixed-Radix | 37.00 | 38.00 | 0.974x | 39.00 | 40.00 | 0.975x | 2026-05-19 12:00 UTC |",
        ]
        .join("\n");
        let mut replacements = BTreeMap::new();
        replacements.insert(
            10,
            "| 10 | Good-Thomas (Static) | Mixed-Radix | 40.75 | 55.60 | 0.733x | 42.38 | 51.42 | 0.824x | 2026-05-20 09:56 UTC |".to_string(),
        );

        let merged = merge_subset_rows(&existing, &replacements, BenchmarkProfile::Quick).unwrap();

        assert!(merged.contains("| 3 | Winograd | Butterfly | 31.00 | 32.00 | 0.969x | 33.00 | 34.00 | 0.971x | 2026-05-19 12:00 UTC |"));
        assert!(merged.contains("| 10 | Good-Thomas (Static) | Mixed-Radix | 40.75 | 55.60 | 0.733x | 42.38 | 51.42 | 0.824x | 2026-05-20 09:56 UTC |"));
        assert!(merged.contains("| 12 | Good-Thomas (Static) | Mixed-Radix | 37.00 | 38.00 | 0.974x | 39.00 | 40.00 | 0.975x | 2026-05-19 12:00 UTC |"));
        assert!(!merged.contains("| 10 | Good-Thomas (Static) | Mixed-Radix | 160.00 | 50.00 | 3.200x | 160.00 | 50.00 | 3.200x | 2026-05-19 12:00 UTC |"));
        assert!(merged.contains(
            "Source: `xtask` bounded adaptive clone-inclusive runner; `--skip-run` merges existing Criterion JSON estimates."
        ));
        assert!(merged.contains(
            "Benchmark profile: `quick: sample_size=3, measurement_time=30ms, warm_up_time=5ms`."
        ));
    }
}
