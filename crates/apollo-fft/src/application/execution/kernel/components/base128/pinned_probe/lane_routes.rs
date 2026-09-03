//! Routes that leave the plain power-of-two lane: half storage promoting
//! through the f32 kernel, the two-dimensional route, and lengths that are
//! not a power of two.

use crate::application::execution::kernel::measurement_cores;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// Half-storage transforms against their own `f32` execution kernel.
///
/// `Complex<F16>` has no native kernel: the storage promotes to `Complex32`,
/// runs the f32 route, and demotes back (`precision_bridge`). This probe
/// separates the two halves of that cost — the same length through the f32
/// kernel directly is the floor, and the difference is what the promotion and
/// demotion pay.
#[cfg(test)]
fn half_storage_against_its_kernel(suite: &mut BenchmarkSuite, core: &str) {
    use crate::application::execution::kernel::mixed_radix::dispatch::dispatch_inplace;
    use crate::application::execution::kernel::FftPrecision;
    use eunomia::{Complex, Complex32, F16};

    // Both length classes. The power-of-two lengths are where the storage
    // route grew up — the register-resident bases are all powers of two — and
    // a sweep whose members all share that property cannot show a result that
    // depends on it (`gap_audit.md#length-class-split`). 96, 100 and 384 are
    // the composite lengths where the plan and the free dispatcher measured
    // apart.
    for n in [8usize, 16, 32, 64, 96, 100, 128, 256, 384, 512] {
        let source32: Vec<Complex32> = (0..n)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.017_f32 * x).sin(), 0.25 * (0.031_f32 * x).cos())
            })
            .collect();
        let source16: Vec<Complex<F16>> = source32
            .iter()
            .map(|value| Complex::new(F16::from_f32(value.re), F16::from_f32(value.im)))
            .collect();

        let mut work16 = source16.clone();
        suite.run(BenchmarkCase::new(core, "half-storage", n), || {
            work16.copy_from_slice(&source16);
            <Complex<F16> as FftPrecision>::fft_forward(std::hint::black_box(&mut work16));
        });

        // Conversion alone, both directions: what the bridge costs when the
        // transform between them is removed. A per-lane cost scales with n;
        // a call-shaped cost does not.
        let mut convert_dst = source32.clone();
        let mut convert_back = source16.clone();
        suite.run(BenchmarkCase::new(core, "convert-only", n), || {
            <eunomia::Complex<F16> as crate::application::execution::kernel::precision_bridge::Complex32Bridge>::widen_slice(
                std::hint::black_box(&source16),
                std::hint::black_box(&mut convert_dst),
            );
            <eunomia::Complex<F16> as crate::application::execution::kernel::precision_bridge::Complex32Bridge>::narrow_slice(
                std::hint::black_box(&convert_dst),
                std::hint::black_box(&mut convert_back),
            );
        });

        // The same public entry point, one scalar up. `Complex<F16>` and
        // `Complex32` reach `FftPrecision::fft_forward` through the same
        // dispatch and pay the same plan-cache lookup on the lengths that use
        // one, so the difference between this arm and the storage arm is the
        // conversion and nothing else. The `f32-plan` arm below cannot serve
        // that role: it hoists plan construction out of the timed region,
        // which the one-shot storage entry cannot do.
        let mut work_precision = source32.clone();
        suite.run(BenchmarkCase::new(core, "f32-precision", n), || {
            work_precision.copy_from_slice(&source32);
            <Complex32 as FftPrecision>::fft_forward(std::hint::black_box(&mut work_precision));
        });

        let mut work32 = source32.clone();
        suite.run(BenchmarkCase::new(core, "f32-kernel", n), || {
            work32.copy_from_slice(&source32);
            dispatch_inplace::<f32, false, false>(std::hint::black_box(&mut work32), None);
        });

        // The same length through the public plan, which selects the
        // register-resident bases. If this is far below the kernel arm, the
        // storage route is not paying for arithmetic but for reaching a
        // different kernel family.
        let plan = crate::FftPlan1D::<f32>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let mut work_plan = source32.clone();
        suite.run(BenchmarkCase::new(core, "f32-plan", n), || {
            work_plan.copy_from_slice(&source32);
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut work_plan));
        });
    }
}

/// Two-dimensional transforms whose axis lengths the power-of-two twiddle
/// table cannot serve.
///
/// `FftPlan2D` precomputes a twiddle table per axis, but only for
/// power-of-two lengths; every other length falls through to `mixed_radix`'s
/// free functions, which re-derive the twiddles and the radix decomposition.
/// That fallback runs once per line, so its cost is multiplied by the axis
/// length rather than paid once per transform. The power-of-two shape is the
/// control: it takes the precomputed path either way and must not move.
#[cfg(test)]
fn two_dimensional_lane_route(suite: &mut BenchmarkSuite, core: &str) {
    use eunomia::Complex32;
    use leto::Array2;

    for [nx, ny] in [[96usize, 96], [100, 100], [128, 128]] {
        let source: Array2<Complex32> = Array2::from_shape_fn([nx, ny], |[row, column]| {
            let x = (row * ny + column) as f32;
            Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        });
        let plan = crate::FftPlan2D::<f32>::new(
            crate::Shape2D::new(nx, ny).expect("invariant: probe shapes are non-zero"),
        );
        let mut work = source.clone();
        suite.run(
            BenchmarkCase::new(core, format!("fft2d-{nx}x{ny}"), nx * ny),
            || {
                work.assign(&source.view());
                plan.forward_complex_inplace(std::hint::black_box(&mut work));
            },
        );
    }
}

/// One lane of a non-power-of-two axis, through both routes, in one process.
///
/// This is the whole of the two-dimensional change: the lane body used to be
/// `mixed_radix`'s free function and is now the cached 1-D plan. Measuring the
/// substitution directly keeps both arms in one binary — a two-binary
/// before/after at these magnitudes moves its own control by several points
/// under peer build load.
#[cfg(test)]
fn non_power_of_two_lane_route(suite: &mut BenchmarkSuite, core: &str) {
    use crate::application::execution::kernel::mixed_radix::forward_inplace;
    use eunomia::Complex32;

    // Spans the length classes, not one of them: powers of two, composites
    // the static radix table carries (384, 243, 720), composites it does not
    // (100 is carried, 1000 and 250 are not), and a prime (101), which reaches
    // neither table. `gap_audit.md#length-class-split` is why.
    for n in [
        96usize, 100, 101, 128, 176, 180, 243, 250, 256, 384, 385, 512, 720, 1000,
    ] {
        let source: Vec<Complex32> = (0..n)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.017_f32 * x).sin(), 0.25 * (0.031_f32 * x).cos())
            })
            .collect();
        let plan = crate::FftPlan1D::<f32>::new(
            crate::Shape1D::new(n).expect("invariant: probe lengths are non-zero"),
        );

        let mut work = source.clone();
        suite.run(BenchmarkCase::new(core, "lane-free-fn", n), || {
            work.copy_from_slice(&source);
            forward_inplace::<f32>(std::hint::black_box(&mut work));
        });

        let mut work_plan = source.clone();
        suite.run(BenchmarkCase::new(core, "lane-plan", n), || {
            work_plan.copy_from_slice(&source);
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut work_plan));
        });
    }
}

#[test]
#[ignore = "measurement instrument for the half-storage promotion cost"]
fn half_storage_promotion_cost_by_core_type() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());
    for core in selection.cores() {
        let cpu = core.processor().get();
        let _binding = ProcessorBinding::bind(core.processor())
            .expect("measurement processor must be available");
        std::thread::yield_now();
        let landed = ProcessorIndex::current()
            .expect("Windows supports processor queries")
            .get();
        assert_eq!(landed, cpu, "processor binding must remain exact");
        let core = core.label();
        // A freshly linked binary's first execution is not measurable on this
        // host: the whole first process runs several times slower (n = 64
        // read 482 ns against 36 on the very next run of the same binary),
        // which is enough to invent an order-of-magnitude finding. One
        // discarded pass absorbs it; a second run of the same binary
        // reproduces these numbers.
        let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
        half_storage_against_its_kernel(&mut warmup, core);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        half_storage_against_its_kernel(&mut suite, core);
        println!("HALF cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}

#[test]
#[ignore = "measurement instrument for the two-dimensional lane route"]
fn two_dimensional_lane_route_by_core_type() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());
    for core in selection.cores() {
        let cpu = core.processor().get();
        let _binding = ProcessorBinding::bind(core.processor())
            .expect("measurement processor must be available");
        std::thread::yield_now();
        let landed = ProcessorIndex::current()
            .expect("Windows supports processor queries")
            .get();
        assert_eq!(landed, cpu, "processor binding must remain exact");
        let core = core.label();
        let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
        two_dimensional_lane_route(&mut warmup, core);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        two_dimensional_lane_route(&mut suite, core);
        println!("FFT2D cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}

#[test]
#[ignore = "measurement instrument for the non-power-of-two lane route"]
fn non_power_of_two_lane_route_by_core_type() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());
    for core in selection.cores() {
        let cpu = core.processor().get();
        let _binding = ProcessorBinding::bind(core.processor())
            .expect("measurement processor must be available");
        std::thread::yield_now();
        let landed = ProcessorIndex::current()
            .expect("Windows supports processor queries")
            .get();
        assert_eq!(landed, cpu, "processor binding must remain exact");
        let core = core.label();
        let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
        non_power_of_two_lane_route(&mut warmup, core);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        non_power_of_two_lane_route(&mut suite, core);
        println!("LANE cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
