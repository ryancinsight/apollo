//! 1D Type-1 and Type-2 NUFFT executions and plans.
//!
//! This module implements a Kaiser-Bessel gridding NUFFT backed by Apollo's
//! in-repo FFT plan rather than external FFT crates.
//!
//! The exact direct transforms serve as the mathematical reference surface,
//! while the fast paths use oversampled grid spreading followed by Apollo FFT
//! execution and deconvolution.
//!
//! # Mathematical contract
//!
//! Type-1 NUFFT maps non-uniform samples `(x_j, c_j)` to uniform Fourier bins
//!
//! `f_k = Σ_j c_j exp(-2πi k x_j / L)`
//!
//! Type-2 NUFFT maps uniform Fourier bins `f_k` to non-uniform positions
//!
//! `g_j = Σ_k f_k exp(2πi k x_j / L)`
//!
//! The fast paths approximate the direct sums by spreading to an oversampled
//! grid with a Kaiser-Bessel kernel and then applying an FFT on the grid.
//!
//! # Complexity
//!
//! Direct transforms are `O(MN)` where `M` is the number of target bins and
//! `N` is the number of samples or coefficients.
//! Fast transforms are `O(Nw + m log m)` where `w` is the kernel width and
//! `m = σn` is the oversampled grid length.
//!
//! # Failure modes
//!
//! - positions/value arrays must have equal length
//! - oversampling factor must satisfy `sigma >= 2`
//! - kernel width must satisfy `kernel_width >= 2`

use apollo_fft::{f16, ApolloError, ApolloResult, FftPlan1D, PrecisionProfile, Shape1D};
use mnemosyne::scratch::ScratchPool;
use moirai::ParallelSliceMut;
use ndarray::Array1;
use num_complex::{Complex32, Complex64};
use std::borrow::Cow;
use std::f64::consts::PI;

thread_local! {
    static COMPLEX_SCRATCH_POOL: ScratchPool<Complex64> = const { ScratchPool::new() };
    static DIRECT_WEIGHT_LANE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Below this operation count, serial loops avoid parallel scheduling overhead.
const DIRECT_PAR_OP_THRESHOLD: usize = 16_384;

use crate::domain::metadata::grid::UniformDomain1D;
use crate::infrastructure::kernel::kaiser_bessel::{
    bucket_count, fft_signed_index, i0, kb_kernel, kb_kernel_ft,
};
use crate::DEFAULT_NUFFT_OVERSAMPLING;

#[derive(Clone, Copy, Debug)]
struct IndexedPoint1D {
    x: f64,
    value: Complex64,
    bucket: usize,
}

fn sort_points_1d(
    positions: &[f64],
    values: &[Complex64],
    domain: UniformDomain1D,
    oversampled_len: usize,
    kernel_width: usize,
) -> Vec<IndexedPoint1D> {
    let buckets = bucket_count(oversampled_len, kernel_width);
    let bucket_scale = buckets as f64 / domain.length();
    let mut indexed: Vec<_> = positions
        .iter()
        .zip(values.iter())
        .map(|(&x, &value)| {
            let x_mod = x.rem_euclid(domain.length());
            let bucket = ((x_mod * bucket_scale).floor() as usize).min(buckets - 1);
            IndexedPoint1D {
                x: x_mod,
                value,
                bucket,
            }
        })
        .collect();
    indexed.sort_unstable_by_key(|lhs| lhs.bucket);
    indexed
}

fn sort_positions_1d(
    positions: &[f64],
    domain: UniformDomain1D,
    oversampled_len: usize,
    kernel_width: usize,
) -> Vec<(usize, f64, usize)> {
    let buckets = bucket_count(oversampled_len, kernel_width);
    let bucket_scale = buckets as f64 / domain.length();
    let mut indexed: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(original_index, &x)| {
            let x_mod = x.rem_euclid(domain.length());
            let bucket = ((x_mod * bucket_scale).floor() as usize).min(buckets - 1);
            (original_index, x_mod, bucket)
        })
        .collect();
    indexed.sort_unstable_by_key(|lhs| lhs.2);
    indexed
}

/// Reusable 1D NUFFT plan using Kaiser-Bessel spreading.
pub struct NufftPlan1D {
    n_out: usize,
    m: usize,
    w: usize,
    beta: f64,
    i0_beta: f64,
    domain: UniformDomain1D,
    deconv: Array1<f64>,
    fft_plan: FftPlan1D<f64>,
}

impl std::fmt::Debug for NufftPlan1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NufftPlan1D")
            .field("n_out", &self.n_out)
            .field("oversampled_len", &self.m)
            .field("kernel_width", &self.w)
            .field("domain", &self.domain)
            .finish()
    }
}

impl NufftPlan1D {
    /// Create a reusable 1D NUFFT plan.
    #[must_use]
    pub fn new(domain: UniformDomain1D, sigma: usize, kernel_width: usize) -> Self {
        assert!(sigma >= 2, "sigma must be >= 2");
        assert!(kernel_width >= 2, "kernel_width must be >= 2");

        let m = sigma * domain.n;
        let beta = PI * (1.0 - 1.0 / (2.0 * sigma as f64)) * (2 * kernel_width) as f64;
        let i0_beta = i0(beta);

        let deconv = Array1::from_shape_fn(domain.n, |k| {
            let k_signed = fft_signed_index(k, domain.n);
            let xi = k_signed as f64 / m as f64;
            1.0 / kb_kernel_ft(xi, kernel_width, beta, i0_beta)
        });

        let fft_plan =
            FftPlan1D::<f64>::new(Shape1D::new(m).expect("NUFFT oversampled length must be valid"));

        Self {
            n_out: domain.n,
            m,
            w: kernel_width,
            beta,
            i0_beta,
            domain,
            deconv,
            fft_plan,
        }
    }

    /// Run type-1 NUFFT, non-uniform samples to uniform Fourier bins.
    #[must_use]
    pub fn type1(&self, positions: &[f64], values: &[Complex64]) -> Array1<Complex64> {
        let mut grid = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![Complex64::new(0.0, 0.0); self.n_out];
        self.type1_into(positions, values, &mut grid, &mut output);
        Array1::from_vec(output)
    }

    /// Run type-1 NUFFT from Leto position and value views.
    ///
    /// Contiguous Leto views are borrowed directly; strided views are copied once
    /// into logical order before reusing the canonical slice NUFFT kernel.
    pub fn type1_leto(
        &self,
        positions: leto::ArrayView1<'_, f64>,
        values: leto::ArrayView1<'_, Complex64>,
    ) -> ApolloResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let positions = leto_view1_cow(positions, "positions")?;
        let values = leto_view1_cow(values, "values")?;
        if positions.len() != values.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: positions.len().to_string(),
                actual: values.len().to_string(),
            });
        }
        let output = self.type1(positions.as_ref(), values.as_ref());
        leto_array1_from_slice(output.as_slice().expect("NUFFT output must be contiguous"))
    }

    /// Run type-1 NUFFT mapping strictly inside zero-allocation bounding limits.
    pub fn type1_into(
        &self,
        positions: &[f64],
        values: &[Complex64],
        scratch_grid: &mut [Complex64],
        output: &mut [Complex64],
    ) {
        assert_eq!(
            positions.len(),
            values.len(),
            "positions/value length mismatch"
        );
        assert_eq!(scratch_grid.len(), self.m, "scratch_grid length mismatch");
        assert_eq!(output.len(), self.n_out, "output length mismatch");

        scratch_grid.fill(Complex64::new(0.0, 0.0));
        let w = self.w as i64;
        let w_f = self.w as f64;
        let sorted_points = sort_points_1d(positions, values, self.domain, self.m, self.w);

        for point in sorted_points {
            let t = self.m as f64 * point.x / self.domain.length();
            let m0 = t.round() as i64;
            let d = t - m0 as f64;
            for p in -w..=w {
                let weight = kb_kernel(p as f64 - d, w_f, self.beta, self.i0_beta);
                if weight != 0.0 {
                    let m_idx = (m0 + p).rem_euclid(self.m as i64) as usize;
                    scratch_grid[m_idx] += point.value * weight;
                }
            }
        }

        self.fft_plan.forward_complex_slice_inplace(scratch_grid);
        let spectrum = &*scratch_grid;

        for k in 0..self.n_out {
            let k_signed = fft_signed_index(k, self.n_out);
            let m_idx = k_signed.rem_euclid(self.m as i64) as usize;
            output[k] = spectrum[m_idx] * self.deconv[k];
        }
    }

    /// Run type-2 NUFFT, interpolating a uniform Fourier grid at non-uniform positions.
    #[must_use]
    pub fn type2(&self, fourier_coeffs: &Array1<Complex64>, positions: &[f64]) -> Vec<Complex64> {
        let mut spread = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![Complex64::new(0.0, 0.0); positions.len()];
        self.type2_into(
            fourier_coeffs.as_slice().unwrap(),
            positions,
            &mut spread,
            &mut output,
        );
        output
    }

    /// Run type-2 NUFFT from Leto coefficient and position views.
    pub fn type2_leto(
        &self,
        fourier_coeffs: leto::ArrayView1<'_, Complex64>,
        positions: leto::ArrayView1<'_, f64>,
    ) -> ApolloResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let fourier_coeffs = leto_view1_cow(fourier_coeffs, "fourier_coeffs")?;
        let positions = leto_view1_cow(positions, "positions")?;
        if fourier_coeffs.len() != self.n_out {
            return Err(ApolloError::ShapeMismatch {
                expected: self.n_out.to_string(),
                actual: fourier_coeffs.len().to_string(),
            });
        }
        let mut spread = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![Complex64::new(0.0, 0.0); positions.len()];
        self.type2_into(
            fourier_coeffs.as_ref(),
            positions.as_ref(),
            &mut spread,
            &mut output,
        );
        leto_array1_from_slice(&output)
    }

    /// Run type-2 NUFFT mapping strictly inside zero-allocation bounding limits.
    pub fn type2_into(
        &self,
        fourier_coeffs: &[Complex64],
        positions: &[f64],
        scratch_spread: &mut [Complex64],
        output: &mut [Complex64],
    ) {
        assert_eq!(
            fourier_coeffs.len(),
            self.n_out,
            "fourier_coeffs length mismatch"
        );
        assert_eq!(
            scratch_spread.len(),
            self.m,
            "scratch_spread length mismatch"
        );
        assert_eq!(output.len(), positions.len(), "output length mismatch");

        scratch_spread.fill(Complex64::new(0.0, 0.0));
        for k in 0..self.n_out {
            let k_signed = fft_signed_index(k, self.n_out);
            let m_idx = k_signed.rem_euclid(self.m as i64) as usize;
            scratch_spread[m_idx] = fourier_coeffs[k] * self.deconv[k];
        }

        self.fft_plan.inverse_complex_slice_inplace(scratch_spread);
        let inverse_scale = self.m as f64;
        scratch_spread
            .iter_mut()
            .for_each(|value| *value *= inverse_scale);
        let spread = &*scratch_spread;

        let w = self.w as i64;
        let w_f = self.w as f64;
        let sorted_points = sort_positions_1d(positions, self.domain, self.m, self.w);

        for (original_index, x_mod, _) in sorted_points {
            let t = self.m as f64 * x_mod / self.domain.length();
            let m0 = t.round() as i64;
            let d = t - m0 as f64;

            let mut value = Complex64::new(0.0, 0.0);
            for p in -w..=w {
                let weight = kb_kernel(p as f64 - d, w_f, self.beta, self.i0_beta);
                if weight != 0.0 {
                    let m_idx = (m0 + p).rem_euclid(self.m as i64) as usize;
                    value += spread[m_idx] * weight;
                }
            }
            output[original_index] = value;
        }
    }

    /// Run type-1 NUFFT for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    ///
    /// The owner path remains the `Complex64` Kaiser-Bessel spread, Apollo FFT,
    /// and deconvolution pipeline. Typed storage converts represented input
    /// into that path and quantizes once into caller-owned output storage.
    pub fn type1_typed_into<T: NufftComplexStorage>(
        &self,
        positions: &[f64],
        values: &[T],
        scratch_grid: &mut [Complex64],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, T::PROFILE)?;
        if positions.len() != values.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: positions.len().to_string(),
                actual: values.len().to_string(),
            });
        }
        if scratch_grid.len() != self.m || output.len() != self.n_out {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("scratch {}, output {}", self.m, self.n_out),
                actual: format!("scratch {}, output {}", scratch_grid.len(), output.len()),
            });
        }
        COMPLEX_SCRATCH_POOL.with(|pool| {
            pool.with_scratch(values.len(), |values64| {
                for (slot, &val) in values64.iter_mut().zip(values.iter()) {
                    *slot = T::to_complex64(val);
                }
                pool.with_scratch(self.n_out, |output64| {
                    self.type1_into(positions, values64, scratch_grid, output64);
                    write_typed_output(output64, output);
                });
            });
        });
        Ok(())
    }

    /// Run typed type-1 NUFFT from Leto position and value views.
    pub fn type1_leto_typed<T: NufftComplexStorage>(
        &self,
        positions: leto::ArrayView1<'_, f64>,
        values: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let positions = leto_view1_cow(positions, "positions")?;
        let values = leto_view1_cow(values, "values")?;
        let mut scratch = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.n_out];
        self.type1_typed_into(
            positions.as_ref(),
            values.as_ref(),
            &mut scratch,
            &mut output,
            profile,
        )?;
        leto_array1_from_slice(&output)
    }

    /// Run type-2 NUFFT for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn type2_typed_into<T: NufftComplexStorage>(
        &self,
        fourier_coeffs: &[T],
        positions: &[f64],
        scratch_spread: &mut [Complex64],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, T::PROFILE)?;
        if fourier_coeffs.len() != self.n_out
            || scratch_spread.len() != self.m
            || output.len() != positions.len()
        {
            return Err(ApolloError::ShapeMismatch {
                expected: format!(
                    "coefficients {}, scratch {}, output {}",
                    self.n_out,
                    self.m,
                    positions.len()
                ),
                actual: format!(
                    "coefficients {}, scratch {}, output {}",
                    fourier_coeffs.len(),
                    scratch_spread.len(),
                    output.len()
                ),
            });
        }
        COMPLEX_SCRATCH_POOL.with(|pool| {
            pool.with_scratch(fourier_coeffs.len(), |coeffs64| {
                for (slot, &val) in coeffs64.iter_mut().zip(fourier_coeffs.iter()) {
                    *slot = T::to_complex64(val);
                }
                pool.with_scratch(positions.len(), |output64| {
                    self.type2_into(coeffs64, positions, scratch_spread, output64);
                    write_typed_output(output64, output);
                });
            });
        });
        Ok(())
    }

    /// Run typed type-2 NUFFT from Leto coefficient and position views.
    pub fn type2_leto_typed<T: NufftComplexStorage>(
        &self,
        fourier_coeffs: leto::ArrayView1<'_, T>,
        positions: leto::ArrayView1<'_, f64>,
        profile: PrecisionProfile,
    ) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let fourier_coeffs = leto_view1_cow(fourier_coeffs, "fourier_coeffs")?;
        let positions = leto_view1_cow(positions, "positions")?;
        let mut scratch = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); positions.len()];
        self.type2_typed_into(
            fourier_coeffs.as_ref(),
            positions.as_ref(),
            &mut scratch,
            &mut output,
            profile,
        )?;
        leto_array1_from_slice(&output)
    }
}

/// Complex storage accepted by typed NUFFT paths.
pub trait NufftComplexStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `Complex64` arithmetic.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic result back to storage.
    fn from_complex64(value: Complex64) -> Self;
}

impl NufftComplexStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }
}

impl NufftComplexStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl NufftComplexStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self[0].to_f32()), f64::from(self[1].to_f32()))
    }

    fn from_complex64(value: Complex64) -> Self {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    }
}

pub(crate) fn validate_profile(
    actual: PrecisionProfile,
    expected: PrecisionProfile,
) -> ApolloResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(ApolloError::validation(
            "precision_profile",
            format!("{actual:?}"),
            format!(
                "storage {:?} with compute {:?}",
                expected.storage, expected.compute
            ),
        ))
    }
}

pub(crate) fn write_typed_output<T: NufftComplexStorage>(source: &[Complex64], target: &mut [T]) {
    for (slot, value) in target.iter_mut().zip(source.iter().copied()) {
        *slot = T::from_complex64(value);
    }
}

fn leto_view1_cow<'a, T: Copy>(
    view: leto::ArrayView1<'a, T>,
    name: &'static str,
) -> ApolloResult<Cow<'a, [T]>> {
    if let Some(slice) = view.as_slice() {
        Ok(Cow::Borrowed(slice))
    } else {
        let mut values = Vec::with_capacity(view.size());
        for index in 0..view.shape()[0] {
            values.push(*view.get([index]).map_err(|_| ApolloError::ShapeMismatch {
                expected: name.to_string(),
                actual: format!("index {index}"),
            })?);
        }
        Ok(Cow::Owned(values))
    }
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        ApolloError::validation("leto_shape", values.len().to_string(), err.to_string())
    })
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
    Array1::from_vec(output)
}

fn nufft_type1_coefficient(
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

fn nufft_type1_coefficient_hermes(
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

fn nufft_type2_sample(
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
    let work_items = positions.len().saturating_mul(fourier_coeffs.len());
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

fn nufft_type2_sample_hermes(x: f64, coeff_lanes: &[f64], domain: UniformDomain1D) -> Complex64 {
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
fn interleaved_lanes(values: &[Complex64]) -> &[f64] {
    // SAFETY: Complex64 is #[repr(C)] and has the same layout and alignment as [f64; 2].
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<f64>(), values.len() * 2) }
}

fn fill_type1_weight_lanes(
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

fn fill_type2_weight_lanes(lanes: &mut [f64], x: f64, domain: UniformDomain1D) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_NUFFT_KERNEL_WIDTH;
    use approx::assert_abs_diff_eq;
    /// DC mode invariant: f_0 = sum(c_j) because exp(-2pi*i * 0 * x_j / L) = 1 for all j.
    ///
    /// With values [1.0, 0.5-0.25i, -0.5+0.75i, 0.25+0.1i]:
    /// sum = (1.0+0.5-0.5+0.25) + (0-0.25+0.75+0.1)i = 1.25 + 0.6i
    #[test]
    fn type1_dc_mode_equals_sum_of_values() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let positions = vec![0.0, 0.125, 0.25, 0.375];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(-0.5, 0.75),
            Complex64::new(0.25, 0.1),
        ];
        let output = nufft_type1_1d(&positions, &values, domain);
        assert_eq!(output.len(), 8);
        // DC mode: sum of all values (analytical)
        let expected_dc = Complex64::new(1.25, 0.6);
        let err = (output[0] - expected_dc).norm();
        assert!(
            err < 1e-12,
            "DC mode err={err}: got {:?}, expected {:?}",
            output[0],
            expected_dc
        );
        // All outputs must be finite
        for (k, v) in output.iter().enumerate() {
            assert!(v.norm().is_finite(), "mode {k} is non-finite: {v:?}");
        }
    }

    #[test]
    fn hermes_type1_coefficients_match_scalar_formula_at_threshold() {
        let domain = UniformDomain1D::new(128, 1.0).unwrap();
        let sample_count = DIRECT_PAR_OP_THRESHOLD / domain.n;
        let positions = (0..sample_count)
            .map(|index| (index as f64 * 0.37).rem_euclid(domain.length()))
            .collect::<Vec<_>>();
        let values = (0..sample_count)
            .map(|index| Complex64::new((index as f64 * 0.125).sin(), (index as f64 * 0.25).cos()))
            .collect::<Vec<_>>();
        let value_lanes = interleaved_lanes(&values);
        let two_pi_over_l = 2.0 * PI / domain.length();

        for k in [0usize, 1, 17, 64, 96, 127] {
            let actual =
                nufft_type1_coefficient_hermes(k, &positions, value_lanes, domain.n, two_pi_over_l);
            let expected = nufft_type1_coefficient(k, &positions, &values, domain.n, two_pi_over_l);

            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn hermes_type2_samples_match_scalar_formula_at_threshold() {
        let domain = UniformDomain1D::new(128, 1.0).unwrap();
        let coeffs = Array1::from_shape_fn(domain.n, |index| {
            Complex64::new((index as f64 * 0.125).sin(), (index as f64 * 0.25).cos())
        });
        let coeff_lanes = interleaved_lanes(coeffs.as_slice().expect("contiguous coeffs"));

        for x in [0.0_f64, 0.125, 0.37, 0.5, 0.875, 1.25] {
            let actual = nufft_type2_sample_hermes(x, coeff_lanes, domain);
            let expected = nufft_type2_sample(x, &coeffs, domain);

            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn direct_weight_lanes_match_type1_and_type2_phasors() {
        let domain = UniformDomain1D::new(16, 0.5).unwrap();
        let positions = [0.0_f64, 0.125, 0.25, 0.375];
        let mut lanes = vec![0.0; positions.len() * 2];
        let two_pi_over_l = 2.0 * PI / domain.length();
        let k_signed = fft_signed_index(13, domain.n);

        fill_type1_weight_lanes(&mut lanes, &positions, k_signed, two_pi_over_l);
        for (index, x) in positions.iter().enumerate() {
            let angle = -two_pi_over_l * k_signed as f64 * x;
            assert_eq!(lanes[index * 2].to_bits(), angle.cos().to_bits());
            assert_eq!(lanes[index * 2 + 1].to_bits(), angle.sin().to_bits());
        }

        fill_type2_weight_lanes(&mut lanes, 0.25, domain);
        for index in 0..positions.len() {
            let signed = fft_signed_index(index, domain.n);
            let angle = two_pi_over_l * signed as f64 * 0.25;
            assert_eq!(lanes[index * 2].to_bits(), angle.cos().to_bits());
            assert_eq!(lanes[index * 2 + 1].to_bits(), angle.sin().to_bits());
        }
    }

    /// Fast vs exact: fixed non-uniform positions and values.
    /// Tolerance 1e-5 for kernel_width=6, sigma=2.
    #[test]
    fn type1_fast_vs_exact_agreement_fixed_inputs() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let positions = vec![0.0, 0.125, 0.25, 0.375];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(-0.5, 0.75),
            Complex64::new(0.25, 0.1),
        ];
        let exact = nufft_type1_1d(&positions, &values, domain);
        let fast = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
        let scale = exact
            .iter()
            .map(|v| v.norm())
            .fold(1.0_f64, f64::max)
            .max(1e-30);
        for (k, (e, f)) in exact.iter().zip(fast.iter()).enumerate() {
            let err = (e - f).norm() / scale;
            assert!(err < 1e-5, "k={k}: exact={e:?}, fast={f:?}, rel_err={err}");
        }
    }

    #[test]
    fn fast_1d_tracks_exact() {
        let domain = UniformDomain1D::new(32, 0.05).unwrap();
        let positions: Vec<f64> = (0..20)
            .map(|i| ((i as f64 * 0.137) % domain.length()).abs())
            .collect();
        let values: Vec<Complex64> = (0..20)
            .map(|i| Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.2).sin()))
            .collect();
        let exact = nufft_type1_1d(&positions, &values, domain);
        let fast = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
        let scale = exact.iter().map(|value| value.norm()).fold(1.0, f64::max);
        let max_err = exact
            .iter()
            .zip(fast.iter())
            .map(|(lhs, rhs)| (lhs - rhs).norm())
            .fold(0.0, f64::max);
        assert!(max_err / scale < 1e-6);
    }

    #[test]
    fn typed_1d_paths_support_complex64_complex32_and_mixed_f16_storage() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let plan = NufftPlan1D::new(
            domain,
            DEFAULT_NUFFT_OVERSAMPLING,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );
        let positions = vec![0.0, 0.09, 0.21, 0.47];
        let values64 = vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 0.75),
            Complex64::new(0.25, -0.1),
            Complex64::new(0.125, 0.5),
        ];
        let expected = plan.type1(&positions, &values64);
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.m];
        let mut output64 = vec![Complex64::new(0.0, 0.0); plan.n_out];
        plan.type1_typed_into(
            &positions,
            &values64,
            &mut scratch,
            &mut output64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed complex64 type1");
        for (actual, expected) in output64.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 1.0e-12);
        }

        let values32: Vec<Complex32> = values64
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();
        let represented32: Vec<Complex64> = values32
            .iter()
            .copied()
            .map(Complex32::to_complex64)
            .collect();
        let expected32 = plan.type1(&positions, &represented32);
        let mut output32 = vec![Complex32::new(0.0, 0.0); plan.n_out];
        plan.type1_typed_into(
            &positions,
            &values32,
            &mut scratch,
            &mut output32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 type1");
        for (actual, expected) in output32.iter().zip(expected32.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let mut recovered32 = vec![Complex32::new(0.0, 0.0); positions.len()];
        plan.type2_typed_into(
            &output32,
            &positions,
            &mut scratch,
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 type2");
        let expected_type2 = plan.type2(&Array1::from_vec(expected32.to_vec()), &positions);
        for (actual, expected) in recovered32.iter().zip(expected_type2.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-3);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-3);
        }

        let values16: Vec<[f16; 2]> = values64
            .iter()
            .map(|value| {
                [
                    f16::from_f32(value.re as f32),
                    f16::from_f32(value.im as f32),
                ]
            })
            .collect();
        let represented16: Vec<Complex64> = values16
            .iter()
            .copied()
            .map(<[f16; 2]>::to_complex64)
            .collect();
        let expected16 = plan.type1(&positions, &represented16);
        let mut output16 = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; plan.n_out];
        plan.type1_typed_into(
            &positions,
            &values16,
            &mut scratch,
            &mut output16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 type1");
        for (actual, expected) in output16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn leto_type1_matches_slice_reference() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let plan = NufftPlan1D::new(
            domain,
            DEFAULT_NUFFT_OVERSAMPLING,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );
        let positions = vec![0.0, 0.09, 0.21, 0.47];
        let values = vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 0.75),
            Complex64::new(0.25, -0.1),
            Complex64::new(0.125, 0.5),
        ];
        let leto_positions =
            leto::Array1::from_shape_vec([positions.len()], positions.clone()).unwrap();
        let leto_values = leto::Array1::from_shape_vec([values.len()], values.clone()).unwrap();
        let expected = plan.type1(&positions, &values);

        let actual = plan
            .type1_leto(leto_positions.view(), leto_values.view())
            .expect("leto type1");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_type1_matches_slice_reference() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let plan = NufftPlan1D::new(
            domain,
            DEFAULT_NUFFT_OVERSAMPLING,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );
        let positions = vec![0.0, 0.09, 0.21, 0.47];
        let values = vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 0.75),
            Complex64::new(0.25, -0.1),
            Complex64::new(0.125, 0.5),
        ];
        let mut position_backing = Vec::with_capacity(positions.len() * 2);
        let mut value_backing = Vec::with_capacity(values.len() * 2);
        for (position, value) in positions.iter().copied().zip(values.iter().copied()) {
            position_backing.push(position);
            position_backing.push(99.0);
            value_backing.push(value);
            value_backing.push(Complex64::new(99.0, 99.0));
        }
        let leto_positions =
            leto::Array1::from_shape_vec([position_backing.len()], position_backing).unwrap();
        let leto_values =
            leto::Array1::from_shape_vec([value_backing.len()], value_backing).unwrap();
        let strided_positions = leto_positions
            .view()
            .slice(&[(0, positions.len() * 2, 2)])
            .expect("strided positions");
        let strided_values = leto_values
            .view()
            .slice(&[(0, values.len() * 2, 2)])
            .expect("strided values");
        let expected = plan.type1(&positions, &values);

        let actual = plan
            .type1_leto(strided_positions, strided_values)
            .expect("leto type1");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn leto_type2_matches_slice_reference() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let plan = NufftPlan1D::new(
            domain,
            DEFAULT_NUFFT_OVERSAMPLING,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );
        let positions = vec![0.0, 0.09, 0.21, 0.47];
        let coefficients = Array1::from_vec(
            (0..plan.n_out)
                .map(|i| Complex64::new((i as f64 * 0.2).cos(), (i as f64 * 0.3).sin()))
                .collect(),
        );
        let leto_positions =
            leto::Array1::from_shape_vec([positions.len()], positions.clone()).unwrap();
        let leto_coefficients =
            leto::Array1::from_shape_vec([coefficients.len()], coefficients.to_vec()).unwrap();
        let expected = plan.type2(&coefficients, &positions);

        let actual = plan
            .type2_leto(leto_coefficients.view(), leto_positions.view())
            .expect("leto type2");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn typed_leto_type1_and_type2_match_slice_reference() {
        let domain = UniformDomain1D::new(8, 0.125).unwrap();
        let plan = NufftPlan1D::new(
            domain,
            DEFAULT_NUFFT_OVERSAMPLING,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );
        let positions = vec![0.0, 0.09, 0.21, 0.47];
        let values = vec![
            Complex32::new(1.0, 0.25),
            Complex32::new(-0.5, 0.75),
            Complex32::new(0.25, -0.1),
            Complex32::new(0.125, 0.5),
        ];
        let leto_positions =
            leto::Array1::from_shape_vec([positions.len()], positions.clone()).unwrap();
        let leto_values = leto::Array1::from_shape_vec([values.len()], values.clone()).unwrap();
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.m];
        let mut expected_type1 = vec![Complex32::new(0.0, 0.0); plan.n_out];
        plan.type1_typed_into(
            &positions,
            &values,
            &mut scratch,
            &mut expected_type1,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed slice type1");

        let actual_type1 = plan
            .type1_leto_typed(
                leto_positions.view(),
                leto_values.view(),
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed leto type1");
        let actual_type1_view = actual_type1.view();
        let actual_type1 = actual_type1_view
            .as_slice()
            .expect("contiguous leto output");
        for (actual, expected) in actual_type1.iter().zip(expected_type1.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }

        let leto_coefficients =
            leto::Array1::from_shape_vec([expected_type1.len()], expected_type1.clone()).unwrap();
        let mut expected_type2 = vec![Complex32::new(0.0, 0.0); positions.len()];
        plan.type2_typed_into(
            &expected_type1,
            &positions,
            &mut scratch,
            &mut expected_type2,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed slice type2");
        let actual_type2 = plan
            .type2_leto_typed(
                leto_coefficients.view(),
                leto_positions.view(),
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed leto type2");
        let actual_type2_view = actual_type2.view();
        let actual_type2 = actual_type2_view
            .as_slice()
            .expect("contiguous leto output");
        for (actual, expected) in actual_type2.iter().zip(expected_type2.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }

    #[test]
    fn typed_1d_paths_reject_profile_and_shape_mismatch() {
        let domain = UniformDomain1D::new(4, 0.25).unwrap();
        let plan = NufftPlan1D::new(
            domain,
            DEFAULT_NUFFT_OVERSAMPLING,
            DEFAULT_NUFFT_KERNEL_WIDTH,
        );
        let positions = vec![0.0, 0.25];
        let values = vec![Complex32::new(1.0, 0.0), Complex32::new(0.5, 0.25)];
        let mut scratch = vec![Complex64::new(0.0, 0.0); plan.m];
        let mut output = vec![Complex32::new(0.0, 0.0); plan.n_out];
        let err = plan
            .type1_typed_into(
                &positions,
                &values,
                &mut scratch,
                &mut output,
                PrecisionProfile::HIGH_ACCURACY_F64,
            )
            .expect_err("profile mismatch");
        assert!(matches!(
            err,
            ApolloError::Validation { field, .. } if field == "precision_profile"
        ));

        let mut short_output = vec![Complex32::new(0.0, 0.0); plan.n_out - 1];
        let err = plan
            .type1_typed_into(
                &positions,
                &values,
                &mut scratch,
                &mut short_output,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect_err("shape mismatch");
        assert!(matches!(err, ApolloError::ShapeMismatch { .. }));
    }
}

#[cfg(test)]
mod proptest_suite {
    use super::*;
    use crate::domain::metadata::grid::UniformDomain1D;
    use crate::DEFAULT_NUFFT_KERNEL_WIDTH;
    use proptest::prelude::*;

    proptest! {
        /// DC mode invariant: output[0] = Σ_j c_j because exp(-2πi·0·x_j / L) = 1 for all j.
        ///
        /// Formal contract: ∀ n ∈ [4,8], ∀ (x_j, c_j) with x_j ∈ [0,L):
        ///   |f_0 - Σ c_j| < 1e-10
        #[test]
        fn dc_mode_equals_sum_for_arbitrary_positions_and_values(
            n in 4_usize..=8_usize,
            pos_parts in prop::collection::vec(0.0_f64..1.0_f64, 4),
            re_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
            im_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
        ) {
            let domain = UniformDomain1D::new(n, 1.0 / n as f64).unwrap();
            // positions in [0, 0.99) ⊂ [0, length=1.0)
            let positions: Vec<f64> = pos_parts.iter().map(|&p| p * 0.99).collect();
            let values: Vec<Complex64> = re_parts
                .iter()
                .zip(im_parts.iter())
                .map(|(&re, &im)| Complex64::new(re, im))
                .collect();
            let output = nufft_type1_1d(&positions, &values, domain);
            let expected_dc: Complex64 = values.iter().copied().sum();
            let err = (output[0] - expected_dc).norm();
            prop_assert!(
                err < 1e-10,
                "DC mode err={}: got {:?}, expected {:?}",
                err, output[0], expected_dc
            );
        }

        /// Fast path tracks exact direct summation within relative tolerance 1e-3.
        ///
        /// Formal contract: ∀ n ∈ [8,16], ∀ (x_j, c_j):
        ///   ‖fast_k - exact_k‖ / max_k(‖exact_k‖) < 1e-3
        #[test]
        fn fast_tracks_exact_for_arbitrary_inputs(
            n in 8_usize..=16_usize,
            pos_parts in prop::collection::vec(0.0_f64..1.0_f64, 4),
            re_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
            im_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
        ) {
            let domain = UniformDomain1D::new(n, 1.0 / n as f64).unwrap();
            let positions: Vec<f64> = pos_parts.iter().map(|&p| p * 0.99).collect();
            let values: Vec<Complex64> = re_parts
                .iter()
                .zip(im_parts.iter())
                .map(|(&re, &im)| Complex64::new(re, im))
                .collect();
            let exact = nufft_type1_1d(&positions, &values, domain);
            let fast  = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
            let scale = exact
                .iter()
                .map(|v| v.norm())
                .fold(1.0_f64, f64::max)
                .max(1e-30);
            for (k, (e, f)) in exact.iter().zip(fast.iter()).enumerate() {
                let err = (e - f).norm() / scale;
                prop_assert!(
                    err < 1e-3,
                    "k={}: exact={:?}, fast={:?}, rel_err={}",
                    k, e, f, err
                );
            }
        }

        /// Type-1 NUFFT is linear in the source values: T1(s·c) = s·T1(c).
        ///
        /// Formal contract: ∀ n ∈ [4,8], ∀ s ∈ ℤ ∩ [-3,3], ∀ (x_j, c_j):
        ///   ‖T1(s·c)_k - s·T1(c)_k‖ / max_k(‖s·T1(c)_k‖) < 1e-9
        #[test]
        fn type1_is_linear_in_values(
            n      in 4_usize..=8_usize,
            scalar in -3_i32..3_i32,
            pos_parts in prop::collection::vec(0.0_f64..1.0_f64, 4),
            re_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
            im_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
        ) {
            let domain = UniformDomain1D::new(n, 1.0 / n as f64).unwrap();
            let positions: Vec<f64> = pos_parts.iter().map(|&p| p * 0.99).collect();
            let values: Vec<Complex64> = re_parts
                .iter()
                .zip(im_parts.iter())
                .map(|(&re, &im)| Complex64::new(re, im))
                .collect();
            let s = Complex64::new(f64::from(scalar), 0.0);
            let scaled_values: Vec<Complex64> = values.iter().map(|&v| s * v).collect();
            // lhs = T1(s·c);  rhs = s·T1(c)
            let lhs      = nufft_type1_1d(&positions, &scaled_values, domain);
            let rhs_base = nufft_type1_1d(&positions, &values,        domain);
            let rhs: Vec<Complex64> = rhs_base.iter().map(|&v| s * v).collect();
            let scale = rhs
                .iter()
                .map(|v| v.norm())
                .fold(1.0_f64, f64::max)
                .max(1e-30);
            for (k, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
                let err = (l - r).norm() / scale;
                prop_assert!(
                    err < 1e-9,
                    "k={}: lhs={:?}, rhs={:?}, rel_err={}",
                    k, l, r, err
                );
            }
        }
    }
}
