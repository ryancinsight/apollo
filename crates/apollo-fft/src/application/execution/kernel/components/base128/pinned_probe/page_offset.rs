//! Whether the 32-point f64 forward's time depends on where its samples sit
//! in their page.
//!
//! The liveness probe reads this cell in one of two modes per process — about
//! 43.0k and 43.9k ps for the same binary — while its controls hold still
//! (`backlog.md#atlas-apollo-n32-f64-liveness`). Its working buffer starts on a
//! 64-byte boundary, but the page offset of that boundary, and the stack's,
//! are chosen by the heap and the loader per process. A store to one address
//! followed by a load from another whose low twelve bits match is predicted
//! as forwarding and replayed (4K aliasing), and the leaf spills registers to
//! the stack between its phases, so the relative page offset of the stack and
//! the samples could select the mode.
//!
//! This probe holds everything else fixed within one process and moves only
//! the samples: one 64-byte-aligned window of 32 samples at every 64-byte
//! offset of a page, each timed with the same budget, in one pinned run on the
//! efficiency core where the cell is readable. A split by offset inside the
//! run attributes the mode to the samples' page offset; a flat sweep rules the
//! samples out.

use super::small_sizes::sweep_config;
use crate::application::execution::kernel::measurement_cores;
use apollo_bench::{BenchmarkCase, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};

const PAGE: usize = 4096;
const SAMPLE_BYTES: usize = core::mem::size_of::<Complex64>();
const N: usize = 32;

#[test]
#[ignore = "measurement instrument for the n=32 f64 cell's per-process mode"]
fn n32_forward_by_sample_page_offset() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());
    let Some(core) = selection
        .cores()
        .iter()
        .copied()
        .min_by_key(|core| core.class())
    else {
        eprintln!("no measurement core selected");
        return;
    };
    let cpu = core.processor().get();
    let _binding =
        ProcessorBinding::bind(core.processor()).expect("measurement processor must be available");
    std::thread::yield_now();
    let landed = ProcessorIndex::current()
        .expect("Windows supports processor queries")
        .get();
    assert_eq!(landed, cpu, "processor binding must remain exact");

    let src: Vec<Complex64> = (0..N)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect();
    let plan = crate::FftPlan1D::<f64>::new(
        crate::Shape1D::new(N).expect("invariant: shape lengths are non-zero"),
    );

    // Four pages of samples: up to one page to the first boundary, the page
    // the window moves across, and its last window's tail.
    let mut buffer = vec![Complex64::new(0.0, 0.0); 4 * PAGE / SAMPLE_BYTES];
    let page_start = buffer.as_ptr().align_offset(PAGE) + PAGE / SAMPLE_BYTES;
    let stack_marker = 0_u8;
    let stack_offset = core::ptr::addr_of!(stack_marker) as usize % PAGE;
    println!(
        "PGO cpu={landed} ({}) stack page offset {stack_offset}",
        core.label()
    );

    let mut warmup = BenchmarkSuite::new(sweep_config());
    let mut suite = BenchmarkSuite::new(sweep_config());
    for pass in [&mut warmup, &mut suite] {
        for offset in (0..PAGE).step_by(64) {
            let start = page_start + offset / SAMPLE_BYTES;
            let work = &mut buffer[start..start + N];
            pass.run(
                BenchmarkCase::new(core.label(), "apollo-f64-32", offset),
                || {
                    work.copy_from_slice(&src);
                    plan.forward_complex_slice_inplace(std::hint::black_box(&mut *work));
                },
            );
        }
    }
    print!("{}", suite.report());
}
