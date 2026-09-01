//! Pinned phase attribution for the N = 96 Good-Thomas `(3, 32)` codelet.
//!
//! The setup for each block is outside its timed region. Run this ignored
//! instrument with `--nocapture`; it asserts processor placement but no timing
//! threshold.

use crate::application::execution::kernel::components::winograd::{dft32_impl, WinogradScalar};
use crate::application::execution::kernel::mixed_radix::traits::ShortDft;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::{Complex, Complex32, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use std::time::Instant;

const N: usize = 96;
const CALLS_PER_BLOCK: u32 = 4096;
const BLOCKS: usize = 40;

struct Buffers<F> {
    source: [Complex<F>; N],
    data: [Complex<F>; N],
    scratch: [Complex<F>; N],
}

fn best_block<S>(
    state: &mut S,
    mut prepare: impl FnMut(&mut S),
    mut run: impl FnMut(&mut S),
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..BLOCKS {
        prepare(state);
        let start = Instant::now();
        for _ in 0..CALLS_PER_BLOCK {
            run(std::hint::black_box(&mut *state));
        }
        best = best.min(start.elapsed().as_nanos() as f64 / f64::from(CALLS_PER_BLOCK));
    }
    best
}

fn gather<F: Copy>(buffers: &mut Buffers<F>) {
    for i1 in 0..3 {
        let mut source_index = i1 * 32;
        let row_start = i1 * 32;
        for i2 in 0..32 {
            buffers.scratch[row_start + i2] = buffers.data[source_index];
            source_index += 3;
            if source_index >= N {
                source_index -= N;
            }
        }
    }
}

fn incumbent_rows<F>(buffers: &mut Buffers<F>)
where
    F: WinogradScalar,
{
    for row in buffers.scratch.chunks_exact_mut(32) {
        let row = row.try_into().expect("N=96 has three complete rows");
        dft32_impl::<F, false>(row);
    }
}

fn routed_rows<F>(buffers: &mut Buffers<F>)
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortDft<32>,
{
    for row in buffers.scratch.chunks_exact_mut(32) {
        let row = row.try_into().expect("N=96 has three complete rows");
        <F as ShortDft<32>>::dft::<false>(row);
    }
}

fn columns_and_scatter<F>(buffers: &mut Buffers<F>)
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortDft<3>,
{
    for i2 in 0..32 {
        let mut column = [
            buffers.scratch[i2],
            buffers.scratch[32 + i2],
            buffers.scratch[64 + i2],
        ];
        <F as ShortDft<3>>::dft::<false>(&mut column);

        let mut destination_index = (i2 * 3 * 11) % N;
        for value in column {
            buffers.data[destination_index] = value;
            destination_index += 64;
            if destination_index >= N {
                destination_index -= N;
            }
        }
    }
}

/// Executes the mathematically equivalent Good-Thomas `(32, 3)` orientation.
///
/// This is deliberately an instrument-only alternative: it keeps the same
/// 32 DFT-3 and three DFT-32 leaves while reversing their traversal order.
fn swapped_complete<F>(buffers: &mut Buffers<F>)
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortDft<3> + ShortDft<32>,
{
    for i1 in 0..32 {
        let mut source_index = i1 * 3;
        let mut row = core::array::from_fn(|_| {
            let value = buffers.data[source_index];
            source_index += 32;
            if source_index >= N {
                source_index -= N;
            }
            value
        });
        <F as ShortDft<3>>::dft::<false>(&mut row);
        buffers.scratch[i1 * 3..i1 * 3 + 3].copy_from_slice(&row);
    }

    for i2 in 0..3 {
        let mut column = core::array::from_fn(|i1| buffers.scratch[i1 * 3 + i2]);
        <F as ShortDft<32>>::dft::<false>(&mut column);

        let mut destination_index = (i2 * 32 * 2) % N;
        for value in column {
            buffers.data[destination_index] = value;
            destination_index += 33;
            if destination_index >= N {
                destination_index -= N;
            }
        }
    }
}

fn probe<F>(label: &str, source: [Complex<F>; N], epsilon: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + ShortDft<3> + ShortDft<32> + ShortDft<96>,
    f64: From<F>,
{
    let mut buffers = Buffers {
        source,
        data: source,
        scratch: source,
    };
    let gather_ns = best_block(
        &mut buffers,
        |buffers| buffers.data.copy_from_slice(&buffers.source),
        gather,
    );
    let incumbent_rows_ns = best_block(
        &mut buffers,
        |buffers| {
            buffers.data.copy_from_slice(&buffers.source);
            gather(buffers);
        },
        incumbent_rows,
    );
    let routed_rows_ns = best_block(
        &mut buffers,
        |buffers| {
            buffers.data.copy_from_slice(&buffers.source);
            gather(buffers);
        },
        routed_rows,
    );
    let columns_ns = best_block(
        &mut buffers,
        |buffers| {
            buffers.data.copy_from_slice(&buffers.source);
            gather(buffers);
            routed_rows(buffers);
        },
        columns_and_scatter,
    );
    let complete_ns = best_block(
        &mut buffers,
        |buffers| buffers.data.copy_from_slice(&buffers.source),
        |buffers| <F as ShortDft<96>>::dft::<false>(&mut buffers.data),
    );
    let swapped_ns = best_block(
        &mut buffers,
        |buffers| buffers.data.copy_from_slice(&buffers.source),
        swapped_complete,
    );

    let mut incumbent = buffers.source;
    <F as ShortDft<96>>::dft::<false>(&mut incumbent);
    buffers.data.copy_from_slice(&buffers.source);
    swapped_complete(&mut buffers);
    let max_error = incumbent
        .iter()
        .zip(buffers.data)
        .map(|(expected, actual)| {
            let re = f64::from(expected.re) - f64::from(actual.re);
            let im = f64::from(expected.im) - f64::from(actual.im);
            re.hypot(im)
        })
        .fold(0.0_f64, f64::max);
    let input_l1 = buffers
        .source
        .iter()
        .map(|value| f64::from(value.re).hypot(f64::from(value.im)))
        .sum::<f64>();
    // Each two-level factorization is bounded conservatively by
    // `2 * N * epsilon * input_l1`; comparing the two routes sums those bounds.
    let error_bound = 4.0 * N as f64 * epsilon * input_l1.max(1.0);
    assert!(
        max_error <= error_bound,
        "swapped Good-Thomas max_error={max_error:.3e}, bound={error_bound:.3e}"
    );

    println!(
        "N96 {label}: gather={gather_ns:.2}ns incumbent-rows={incumbent_rows_ns:.2}ns routed-rows={routed_rows_ns:.2}ns columns+scatter={columns_ns:.2}ns complete={complete_ns:.2}ns swapped={swapped_ns:.2}ns"
    );
}

#[test]
#[ignore = "measurement instrument for N=96 Good-Thomas phase attribution"]
fn good_thomas_phase_costs_on_performance_core() {
    let _binding = ProcessorBinding::bind(ProcessorIndex::new(2))
        .expect("measurement processor must be available");
    std::thread::yield_now();
    assert_eq!(
        ProcessorIndex::current()
            .expect("Windows supports processor queries")
            .get(),
        2,
        "processor binding must remain exact"
    );

    let input_f32 = std::array::from_fn(|index| {
        let x = index as f32;
        Complex32::new((0.17 * x).sin(), 0.25 * (0.31 * x).cos())
    });
    probe("f32", input_f32, f64::from(f32::EPSILON));

    let input_f64 = std::array::from_fn(|index| {
        let x = index as f64;
        Complex64::new((0.17 * x).sin(), 0.25 * (0.31 * x).cos())
    });
    probe("f64", input_f64, f64::EPSILON);
}
