//! Mathematical and data-structure helpers for Spherical Harmonic Transforms.

use super::ShtPlan;
use crate::domain::contracts::error::{ShtError, ShtResult};
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use crate::infrastructure::kernel::spherical_harmonic::spherical_harmonic;
use apollo_fft::PrecisionProfile;
use mnemosyne::scratch::ScratchPool;
use leto::Array2;
use eunomia::Complex64;

/// Below this reduction length, scalar accumulation avoids Hermes dispatch and scratch setup.
pub(super) const SHT_HERMES_DOT_LEN_THRESHOLD: usize = 256;

thread_local! {
    pub(super) static SHT_WEIGHT_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(super) static SHT_COEFF_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

pub(super) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> ShtResult<()> {
    if apollo_fft::application::utilities::leto_interop::profile_matches(actual, expected) {
        Ok(())
    } else {
        Err(ShtError::PrecisionMismatch)
    }
}

pub(super) fn validate_sample_array_shape<T>(plan: &ShtPlan, samples: &Array2<T>) -> ShtResult<()> {
    plan.check_sample_shape(samples.shape())
}

pub(super) fn validate_coefficient_array_shape<T>(
    plan: &ShtPlan,
    coefficients: &Array2<T>,
) -> ShtResult<()> {
    if coefficients.shape() == plan.coefficient_shape() {
        Ok(())
    } else {
        Err(ShtError::CoefficientShapeMismatch)
    }
}

pub(super) fn write_complex_array<T: super::typed::ShtComplexStorage>(
    source: &Array2<Complex64>,
    target: &mut Array2<T>,
) {
    for (slot, value) in target.as_slice_mut().expect("contiguous target").iter_mut().zip(source.iter().copied()) {
        *slot = T::from_complex64(value);
    }
}

pub(super) fn sht_forward_mode_sum(
    samples: &[Complex64],
    degree: usize,
    order: isize,
    theta: f64,
    n_lon: usize,
) -> Complex64 {
    samples
        .iter()
        .enumerate()
        .map(|(lon, &sample)| {
            let phi = phi_for_longitude(lon, n_lon);
            sample * spherical_harmonic(degree, order, theta, phi).conj()
        })
        .sum()
}

pub(super) fn sht_forward_mode_sum_hermes(
    sample_lanes: &[f64],
    degree: usize,
    order: isize,
    theta: f64,
    n_lon: usize,
) -> Complex64 {
    SHT_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(sample_lanes.len(), |weight_lanes| {
            fill_forward_weight_lanes(weight_lanes, degree, order, theta, n_lon);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                sample_lanes,
                weight_lanes,
            )
            .expect("SHT forward Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re, im)
        })
    })
}

pub(super) fn sht_inverse_sample(
    coefficients: &SphericalHarmonicCoefficients,
    all_modes: &[(usize, isize)],
    theta: f64,
    phi: f64,
) -> Complex64 {
    all_modes
        .iter()
        .map(|&(degree, order)| {
            coefficients.get(degree, order) * spherical_harmonic(degree, order, theta, phi)
        })
        .sum()
}

pub(super) fn sht_inverse_sample_hermes(
    coefficient_lanes: &[f64],
    all_modes: &[(usize, isize)],
    theta: f64,
    phi: f64,
) -> Complex64 {
    SHT_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(coefficient_lanes.len(), |weight_lanes| {
            fill_inverse_weight_lanes(weight_lanes, all_modes, theta, phi);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                coefficient_lanes,
                weight_lanes,
            )
            .expect("SHT inverse Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re, im)
        })
    })
}

#[inline]
pub(super) fn interleaved_lanes(values: &[Complex64]) -> &[f64] {
    bytemuck::cast_slice(values)
}

#[cfg(test)]
pub(super) fn coefficient_lanes(
    coefficients: &SphericalHarmonicCoefficients,
    all_modes: &[(usize, isize)],
) -> Vec<f64> {
    let mut lanes = Vec::with_capacity(all_modes.len() * 2);
    for &(degree, order) in all_modes {
        let value = coefficients.get(degree, order);
        lanes.push(value.re);
        lanes.push(value.im);
    }
    lanes
}

pub(super) fn fill_forward_weight_lanes(
    lanes: &mut [f64],
    degree: usize,
    order: isize,
    theta: f64,
    n_lon: usize,
) {
    for (lon, lane_pair) in lanes.chunks_exact_mut(2).enumerate() {
        let phi = phi_for_longitude(lon, n_lon);
        let value = spherical_harmonic(degree, order, theta, phi).conj();
        lane_pair[0] = value.re;
        lane_pair[1] = value.im;
    }
}

pub(super) fn fill_inverse_weight_lanes(
    lanes: &mut [f64],
    all_modes: &[(usize, isize)],
    theta: f64,
    phi: f64,
) {
    for (&(degree, order), lane_pair) in all_modes.iter().zip(lanes.chunks_exact_mut(2)) {
        let value = spherical_harmonic(degree, order, theta, phi);
        lane_pair[0] = value.re;
        lane_pair[1] = value.im;
    }
}

pub(super) fn phi_for_longitude(longitude_index: usize, n_lon: usize) -> f64 {
    std::f64::consts::TAU * longitude_index as f64 / n_lon as f64
}

pub(super) fn array2_from_leto_view<T: Copy>(
    view: leto::ArrayView2<'_, T>,
    _shape_error: ShtError,
) -> ShtResult<Array2<T>> {
    // Contiguous views borrow without copy; strided views materialize once into
    // logical row-major order. One canonical Leto entry point covers both.
    Ok(view.to_contiguous())
}

pub(super) fn leto_array2_from_ndarray<T: Copy>(
    array: &Array2<T>,
) -> ShtResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
    if array.as_slice().is_some() {
        apollo_fft::application::utilities::leto_interop::try_dense_from_contiguous(array)
            .ok_or(ShtError::CoefficientShapeMismatch)
    } else {
        let [rows, cols] = array.shape();
        let values = array.iter().copied().collect::<Vec<_>>();
        leto::Array::from_mnemosyne_vec([rows, cols], values)
            .map_err(|_| ShtError::CoefficientShapeMismatch)
    }
}

pub(super) fn coefficients_from_leto_view(
    plan: &ShtPlan,
    coefficients: leto::ArrayView2<'_, Complex64>,
) -> ShtResult<SphericalHarmonicCoefficients> {
    let values = array2_from_leto_view(coefficients, ShtError::CoefficientShapeMismatch)?;
    if values.shape() != plan.coefficient_shape() {
        return Err(ShtError::CoefficientShapeMismatch);
    }
    Ok(SphericalHarmonicCoefficients::from_values(
        plan.grid().max_degree(),
        values,
    ))
}
