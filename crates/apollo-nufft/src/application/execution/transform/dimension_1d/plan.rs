//! 1D NUFFT plan construction and gridded type-1/type-2 evaluation.

use super::bucket::{sort_points_1d, sort_positions_1d};
use super::{array1_from_vec, array1_value, write_typed_output, NufftComplexStorage};
use crate::domain::metadata::grid::UniformDomain1D;
use crate::infrastructure::kernel::kaiser_bessel::{fft_signed_index, i0, kb_kernel, kb_kernel_ft};
use apollo_fft::{ApolloError, ApolloResult, CpuStorage, FftPlan1D, PrecisionProfile, Shape1D};
use eunomia::Complex64;
use leto::Array1;
use mnemosyne::scratch::ScratchPool;
use std::f64::consts::PI;

thread_local! {
    static COMPLEX_SCRATCH_POOL: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Reusable 1D NUFFT plan using Kaiser-Bessel spreading.
pub struct NufftPlan1D {
    pub(crate) n_out: usize,
    pub(crate) m: usize,
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

        let deconv = Array1::from_shape_fn([domain.n], |[k]| {
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
        array1_from_vec(output)
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
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        if positions.len() != values.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: positions.len().to_string(),
                actual: values.len().to_string(),
            });
        }
        let output = self.type1(positions.as_ref(), values.as_ref());
        apollo_leto_interop::try_array1_from_slice(
            output.as_slice().expect("NUFFT output must be contiguous"),
        )
        .ok_or_else(|| {
            ApolloError::validation(
                "leto_shape",
                output.size().to_string(),
                "Mnemosyne-backed 1D array construction",
            )
        })
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
            output[k] = spectrum[m_idx] * array1_value(&self.deconv, k);
        }
    }

    /// Run type-2 NUFFT, interpolating a uniform Fourier grid at non-uniform positions.
    #[must_use]
    pub fn type2(&self, fourier_coeffs: &Array1<Complex64>, positions: &[f64]) -> Vec<Complex64> {
        let mut spread = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![Complex64::new(0.0, 0.0); positions.len()];
        let fourier_coeffs = apollo_leto_interop::view_cow(&fourier_coeffs.view());
        self.type2_into(fourier_coeffs.as_ref(), positions, &mut spread, &mut output);
        output
    }

    /// Run type-2 NUFFT from Leto coefficient and position views.
    pub fn type2_leto(
        &self,
        fourier_coeffs: leto::ArrayView1<'_, Complex64>,
        positions: leto::ArrayView1<'_, f64>,
    ) -> ApolloResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let fourier_coeffs = apollo_leto_interop::view_cow(&fourier_coeffs);
        let positions = apollo_leto_interop::view_cow(&positions);
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
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
            ApolloError::validation(
                "leto_shape",
                output.len().to_string(),
                "Mnemosyne-backed 1D array construction",
            )
        })
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
            scratch_spread[m_idx] = fourier_coeffs[k] * array1_value(&self.deconv, k);
        }

        // The Type-2 sum is unnormalized, so the inverse is too.
        self.fft_plan
            .inverse_complex_slice_unnorm_inplace(scratch_spread);
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

    /// Run type-1 NUFFT for `Complex64`, `Complex32`, or mixed `[F16; 2]` storage.
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
        profile.ensure_matches(T::PROFILE, || {
            ApolloError::precision_profile_mismatch(profile, T::PROFILE)
        })?;
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
                    *slot = CpuStorage::to_cpu(val);
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
        let positions = apollo_leto_interop::view_cow(&positions);
        let values = apollo_leto_interop::view_cow(&values);
        let mut scratch = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); self.n_out];
        self.type1_typed_into(
            positions.as_ref(),
            values.as_ref(),
            &mut scratch,
            &mut output,
            profile,
        )?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
            ApolloError::validation(
                "leto_shape",
                output.len().to_string(),
                "Mnemosyne-backed 1D array construction",
            )
        })
    }

    /// Run type-2 NUFFT for `Complex64`, `Complex32`, or mixed `[F16; 2]` storage.
    pub fn type2_typed_into<T: NufftComplexStorage>(
        &self,
        fourier_coeffs: &[T],
        positions: &[f64],
        scratch_spread: &mut [Complex64],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        profile.ensure_matches(T::PROFILE, || {
            ApolloError::precision_profile_mismatch(profile, T::PROFILE)
        })?;
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
                    *slot = CpuStorage::to_cpu(val);
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
        let fourier_coeffs = apollo_leto_interop::view_cow(&fourier_coeffs);
        let positions = apollo_leto_interop::view_cow(&positions);
        let mut scratch = vec![Complex64::new(0.0, 0.0); self.m];
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); positions.len()];
        self.type2_typed_into(
            fourier_coeffs.as_ref(),
            positions.as_ref(),
            &mut scratch,
            &mut output,
            profile,
        )?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
            ApolloError::validation(
                "leto_shape",
                output.len().to_string(),
                "Mnemosyne-backed 1D array construction",
            )
        })
    }
}
