use super::*;
use eunomia::assert_abs_diff_eq;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATION_COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

// SAFETY: every operation delegates to `System` unchanged. The thread-local
// counter observes successful allocation attempts without altering their
// pointer, layout, lifetime, or deallocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATION_COUNT.try_with(|count| {
            if let Some(current) = count.get() {
                count.set(Some(current + 1));
            }
        });
        // SAFETY: this method forwards the caller's `GlobalAlloc` contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: this method forwards the caller's `GlobalAlloc` contract.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_thread_allocations<R>(f: impl FnOnce() -> R) -> (R, usize) {
    ALLOCATION_COUNT.with(|count| count.set(Some(0)));
    let result = f();
    let allocations = ALLOCATION_COUNT
        .with(|count| count.replace(None))
        .unwrap_or(0);
    (result, allocations)
}

#[test]
fn hermes_mellin_moment_matches_scalar_formula_at_threshold() {
    let len = HERMES_MOMENT_LEN_THRESHOLD;
    let signal_min = 1.0_f64;
    let signal_max = 5.0_f64;
    let exponent = 1.75_f64;
    let step = (signal_max - signal_min) / (len as f64 - 1.0);
    let signal = (0..len)
        .map(|index| {
            let coordinate = signal_min + index as f64 * step;
            coordinate.ln().sin() + 2.0
        })
        .collect::<Vec<_>>();

    let actual = mellin_moment_hermes(&signal, signal_min, exponent, step);
    let expected = mellin_moment_scalar(&signal, signal_min, exponent, step);

    assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-9);
}

#[test]
fn moment_weights_match_trapezoid_formula() {
    let len = HERMES_MOMENT_LEN_THRESHOLD;
    let signal_min = 0.5_f64;
    let step = 0.001_f64;
    let exponent = 2.25_f64;
    let mut weights = vec![0.0; len];

    fill_moment_weights(&mut weights, signal_min, exponent, step);

    for &index in &[0usize, 1, 257, len - 2, len - 1] {
        let coordinate = signal_min + index as f64 * step;
        let trapezoid = if index == 0 || index + 1 == len {
            0.5
        } else {
            1.0
        };
        let expected = trapezoid * coordinate.powf(exponent - 1.0);
        assert_eq!(weights[index].to_bits(), expected.to_bits());
    }
}

#[test]
fn hermes_log_frequency_rows_match_scalar_formulas_at_threshold() {
    let len = PAR_THRESHOLD;
    let log_min = -0.25_f64;
    let log_max = 1.75_f64;
    let du = (log_max - log_min) / (len as f64 - 1.0);
    let factor = -std::f64::consts::TAU / len as f64;
    let samples = (0..len)
        .map(|index| {
            let x = log_min + index as f64 * du;
            x.sin() + (2.0 * x).cos()
        })
        .collect::<Vec<_>>();
    for k in [0usize, 1, 17, 64, 127, 255] {
        let actual = log_frequency_coeff_hermes(&samples, factor, du, k);
        let expected = samples
            .iter()
            .enumerate()
            .map(|(n, sample)| Complex64::from_polar(*sample, factor * k as f64 * n as f64))
            .sum::<Complex64>()
            * du;

        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
    }
}

#[test]
fn hermes_log_frequency_row_first_use_retains_only_weight_lanes() {
    const LEN: usize = PAR_THRESHOLD;
    let thread = std::thread::spawn(|| {
        let factor = -std::f64::consts::TAU / LEN as f64;
        let scale = 0.125_f64;
        let row = 17usize;
        let samples = (0..LEN)
            .map(|index| {
                let x = index as f64;
                (x * 0.03125).sin() + (x * 0.015625).cos()
            })
            .collect::<Vec<_>>();
        let expected = samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                Complex64::from_polar(*sample, factor * row as f64 * index as f64)
            })
            .sum::<Complex64>()
            * scale;

        let (actual, allocations) =
            count_thread_allocations(|| log_frequency_coeff_hermes(&samples, factor, scale, row));
        let (warm_actual, warm_allocations) =
            count_thread_allocations(|| log_frequency_coeff_hermes(&samples, factor, scale, row));
        let retained_lanes =
            LOG_FREQUENCY_WEIGHT_LANE_SCRATCH.with(mnemosyne::scratch::ScratchPool::capacity);

        (
            actual,
            warm_actual,
            expected,
            allocations,
            warm_allocations,
            retained_lanes,
        )
    });
    let (actual, warm_actual, expected, allocations, warm_allocations, retained_lanes) =
        thread.join().expect("allocation census thread panicked");

    assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
    assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
    assert_eq!(warm_actual, actual);
    assert_eq!(
        allocations, 1,
        "the first real-input row must allocate only its interleaved weight buffer"
    );
    assert_eq!(warm_allocations, 0);
    assert_eq!(retained_lanes, LEN * 2);
}

#[test]
fn hermes_inverse_log_frequency_rows_match_scalar_formulas_at_threshold() {
    let len = PAR_THRESHOLD;
    let log_min = -0.25_f64;
    let log_max = 1.75_f64;
    let du = (log_max - log_min) / (len as f64 - 1.0);
    let inv_du = 1.0 / du;
    let factor = std::f64::consts::TAU / len as f64;
    let spectrum = (0..len)
        .map(|index| {
            Complex64::new(
                (index as f64 * 0.03125).sin(),
                (index as f64 * 0.015625).cos(),
            )
        })
        .collect::<Vec<_>>();
    let spectrum_lanes = complex_interleaved_lanes(&spectrum);

    for n in [0usize, 1, 17, 64, 127, 255] {
        let actual =
            inverse_log_frequency_coeff_hermes(spectrum_lanes, factor, inv_du / len as f64, n);
        let expected = spectrum
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let angle = factor * k as f64 * n as f64;
                s.re * angle.cos() - s.im * angle.sin()
            })
            .sum::<f64>()
            * inv_du
            / len as f64;

        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-10);
    }
}

#[test]
fn log_frequency_weight_lanes_match_trigonometric_formula() {
    let len = PAR_THRESHOLD * 2;
    let factor = -std::f64::consts::TAU / PAR_THRESHOLD as f64;
    let row = 17usize;
    let mut lanes = vec![0.0; len];

    fill_log_frequency_weight_lanes(&mut lanes, factor, row);

    for &index in &[0usize, 1, 17, 64, 127, 255] {
        let angle = factor * row as f64 * index as f64;
        assert_eq!(lanes[index * 2].to_bits(), angle.cos().to_bits());
        assert_eq!(lanes[index * 2 + 1].to_bits(), angle.sin().to_bits());
    }
}
