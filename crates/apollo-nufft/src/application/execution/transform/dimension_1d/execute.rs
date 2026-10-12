//! Exact direct and fast 1D NUFFT executions.

use super::{array1_from_vec, NufftComplexStorage, NufftPlan1D, DIRECT_PAR_OP_THRESHOLD};
use crate::domain::metadata::grid::UniformDomain1D;
use crate::infrastructure::kernel::kaiser_bessel::fft_signed_index;
use crate::DEFAULT_NUFFT_OVERSAMPLING;
use eunomia::Complex64;
use leto::Array1;
use mnemosyne::scratch::ScratchPool;
use moirai::ParallelSliceMut;
use std::f64::consts::PI;

thread_local! {
    static DIRECT_WEIGHT_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

pub(crate) fn write_typed_output<T: NufftComplexStorage>(source: &[Complex64], target: &mut [T]) {
    for (slot, value) in target.iter_mut().zip(source.iter().copied()) {
        *slot = T::from_cpu(value);
    }
}

/// Exact direct 1D type-1 NUFFT.
#[must_use]
pub fn nufft_type1_1d(
    positions: &[f64],
    values: &[Complex64],
    domain: UniformDomain1D,
) -> Array1<Complex64> {
    assert_eq!(
        positions.len(),
        values.len(),
        "positions/value length mismatch"
    );
    let two_pi_over_l = 2.0 * PI / domain.length();
    let mut output = vec![Complex64::new(0.0, 0.0); domain.n];
    let work_items = domain.n.saturating_mul(positions.len());
    if work_items >= DIRECT_PAR_OP_THRESHOLD {
        let value_lanes = interleaved_lanes(values);
        output.par_mut().enumerate(|k, value| {
            *value =
                nufft_type1_coefficient_hermes(k, positions, value_lanes, domain.n, two_pi_over_l);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, value)| {
            *value = nufft_type1_coefficient(k, positions, values, domain.n, two_pi_over_l);
        });
    }
    array1_from_vec(output)
}

pub(crate) fn nufft_type1_coefficient(
    k: usize,
    positions: &[f64],
    values: &[Complex64],
    n_out: usize,
    two_pi_over_l: f64,
) -> Complex64 {
    let k_signed = fft_signed_index(k, n_out);
    positions
        .iter()
        .zip(values.iter())
        .fold(Complex64::new(0.0, 0.0), |acc, (&x, &value)| {
            let angle = -two_pi_over_l * k_signed as f64 * x;
            acc + value * Complex64::new(angle.cos(), angle.sin())
        })
}

pub(crate) fn nufft_type1_coefficient_hermes(
    k: usize,
    positions: &[f64],
    value_lanes: &[f64],
    n_out: usize,
    two_pi_over_l: f64,
) -> Complex64 {
    let k_signed = fft_signed_index(k, n_out);
    DIRECT_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(value_lanes.len(), |weight_lanes| {
            fill_type1_weight_lanes(weight_lanes, positions, k_signed, two_pi_over_l);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                value_lanes,
                weight_lanes,
            )
            .expect("NUFFT type-1 Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re, im)
        })
    })
}

pub(crate) fn nufft_type2_sample(
    x: f64,
    fourier_coeffs: &Array1<Complex64>,
    domain: UniformDomain1D,
) -> Complex64 {
    fourier_coeffs
        .iter()
        .enumerate()
        .fold(Complex64::new(0.0, 0.0), |acc, (k, &value)| {
            let k_signed = fft_signed_index(k, domain.n);
            let angle = 2.0 * PI * k_signed as f64 * x / domain.length();
            acc + value * Complex64::new(angle.cos(), angle.sin())
        })
}

/// Exact direct 1D type-2 NUFFT.
#[must_use]
pub fn nufft_type2_1d(
    fourier_coeffs: &Array1<Complex64>,
    positions: &[f64],
    domain: UniformDomain1D,
) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); positions.len()];
    let work_items = positions.len().saturating_mul(fourier_coeffs.size());
    if work_items >= DIRECT_PAR_OP_THRESHOLD {
        let coeffs = fourier_coeffs
            .as_slice()
            .expect("NUFFT direct type-2 Fourier coefficients must be contiguous");
        let coeff_lanes = interleaved_lanes(coeffs);
        output.par_mut().enumerate(|index, value| {
            *value = nufft_type2_sample_hermes(positions[index], coeff_lanes, domain);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(index, value)| {
            *value = nufft_type2_sample(positions[index], fourier_coeffs, domain);
        });
    }
    output
}

pub(crate) fn nufft_type2_sample_hermes(
    x: f64,
    coeff_lanes: &[f64],
    domain: UniformDomain1D,
) -> Complex64 {
    DIRECT_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(coeff_lanes.len(), |weight_lanes| {
            fill_type2_weight_lanes(weight_lanes, x, domain);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                coeff_lanes,
                weight_lanes,
            )
            .expect("NUFFT type-2 Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re, im)
        })
    })
}

#[inline]
pub(crate) fn interleaved_lanes(values: &[Complex64]) -> &[f64] {
    eunomia::layout::cast_slice(values)
}

pub(crate) fn fill_type1_weight_lanes(
    lanes: &mut [f64],
    positions: &[f64],
    k_signed: i64,
    two_pi_over_l: f64,
) {
    for (&x, lane_pair) in positions.iter().zip(lanes.chunks_exact_mut(2)) {
        let angle = -two_pi_over_l * k_signed as f64 * x;
        lane_pair[0] = angle.cos();
        lane_pair[1] = angle.sin();
    }
}

pub(crate) fn fill_type2_weight_lanes(lanes: &mut [f64], x: f64, domain: UniformDomain1D) {
    let two_pi_over_l = 2.0 * PI / domain.length();
    for (k, lane_pair) in lanes.chunks_exact_mut(2).enumerate() {
        let k_signed = fft_signed_index(k, domain.n);
        let angle = two_pi_over_l * k_signed as f64 * x;
        lane_pair[0] = angle.cos();
        lane_pair[1] = angle.sin();
    }
}

/// Fast 1D type-1 NUFFT convenience wrapper.
#[must_use]
pub fn nufft_type1_1d_fast(
    positions: &[f64],
    values: &[Complex64],
    domain: UniformDomain1D,
    kernel_width: usize,
) -> Array1<Complex64> {
    NufftPlan1D::new(domain, DEFAULT_NUFFT_OVERSAMPLING, kernel_width).type1(positions, values)
}

/// Fast 1D type-2 NUFFT convenience wrapper.
#[must_use]
pub fn nufft_type2_1d_fast(
    fourier_coeffs: &Array1<Complex64>,
    positions: &[f64],
    domain: UniformDomain1D,
    kernel_width: usize,
) -> Vec<Complex64> {
    NufftPlan1D::new(domain, DEFAULT_NUFFT_OVERSAMPLING, kernel_width)
        .type2(fourier_coeffs, positions)
}
