//! Reusable spherical harmonic transform plan.
//!
//! The plan uses Gauss-Legendre latitude nodes and uniform longitude nodes.
//! Forward transforms compute coefficients
//! `a_lm = integral f(theta, phi) conj(Y_lm(theta, phi)) dOmega` by product
//! quadrature. Inverse transforms evaluate
//! `f(theta, phi) = sum_l sum_m a_lm Y_lm(theta, phi)` on the same grid.

mod longitude;
mod quadrature;
#[cfg(test)]
mod tests;
mod typed;

pub use typed::{ShtComplexStorage, ShtRealStorage};

use crate::domain::contracts::error::{ShtError, ShtResult};
use crate::domain::metadata::grid::SphericalGridSpec;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use crate::infrastructure::kernel::spherical_harmonic::gauss_legendre_nodes_weights;
use apollo_fft::PrecisionProfile;
use eunomia::Complex64;
use leto::Array2;
use longitude::{
    harmonic_amplitude, longitude_plan, longitude_spectrum, longitude_synthesis, order_bin,
    SHT_LONGITUDE_SCRATCH,
};
use quadrature::{array2_from_leto_view, coefficients_from_leto_view};

/// Reusable spherical harmonic transform (SHT) plan.
///
/// Pre-computes Gauss-Legendre nodes and weights for the latitude axis and caches the
/// validated [`SphericalGridSpec`]. The same plan can be reused for multiple transforms
/// without recomputing the quadrature rule.
#[derive(Debug, Clone, PartialEq)]
pub struct ShtPlan {
    pub(super) grid: SphericalGridSpec,
    pub(super) cos_theta_nodes: Vec<f64>,
    pub(super) theta_weights: Vec<f64>,
}

impl ShtPlan {
    /// Create a validated SHT plan.
    pub fn new(latitudes: usize, longitudes: usize, max_degree: usize) -> ShtResult<Self> {
        let grid = SphericalGridSpec::new(latitudes, longitudes, max_degree)?;
        let (cos_theta_nodes, theta_weights) = gauss_legendre_nodes_weights(latitudes);
        Ok(Self {
            grid,
            cos_theta_nodes,
            theta_weights,
        })
    }

    /// Return the validated grid specification.
    #[must_use]
    pub const fn grid(&self) -> SphericalGridSpec {
        self.grid
    }

    /// Return colatitude angle for a latitude index.
    #[must_use]
    pub fn theta(&self, latitude_index: usize) -> f64 {
        self.cos_theta_nodes[latitude_index].acos()
    }

    /// Return longitude angle for a longitude index.
    #[must_use]
    pub fn phi(&self, longitude_index: usize) -> f64 {
        std::f64::consts::TAU * longitude_index as f64 / self.grid.longitudes() as f64
    }

    /// Forward SHT for real-valued samples on the plan grid.
    pub fn forward_real(&self, samples: &Array2<f64>) -> ShtResult<SphericalHarmonicCoefficients> {
        self.check_sample_shape(samples.shape())?;
        let complex_samples = samples.mapv(|value| Complex64::new(value, 0.0));
        self.forward_complex(&complex_samples)
    }

    /// Forward SHT for real-valued Leto sample views on the plan grid.
    pub fn forward_real_leto(
        &self,
        samples: leto::ArrayView2<'_, f64>,
    ) -> ShtResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let samples = array2_from_leto_view(samples);
        let coefficients = self.forward_real(&samples)?;
        apollo_leto_interop::try_dense_from_array(coefficients.values())
            .ok_or(ShtError::CoefficientShapeMismatch)
    }

    /// Forward SHT for complex-valued samples on the plan grid.
    pub fn forward_complex(
        &self,
        samples: &Array2<Complex64>,
    ) -> ShtResult<SphericalHarmonicCoefficients> {
        self.check_sample_shape(samples.shape())?;
        let max_degree = self.grid.max_degree();
        let mut coefficients = SphericalHarmonicCoefficients::zeros(max_degree);
        let longitude_weight = std::f64::consts::TAU / self.grid.longitudes() as f64;
        let n_lat = self.grid.latitudes();
        let n_lon = self.grid.longitudes();

        // Pre-collect all (degree, order) mode pairs for deterministic indexing.
        let all_modes: Vec<(usize, isize)> = (0..=max_degree)
            .flat_map(|l| (-(l as isize)..=(l as isize)).map(move |m| (l, m)))
            .collect();

        // Parallelize over latitude rows; each row contributes to all modes independently.
        let num_modes = all_modes.len();
        let mut flat_contributions = vec![Complex64::new(0.0, 0.0); n_lat * num_modes];

        // `samples` is C-contiguous after view materialization, so each latitude
        // row is a contiguous `n_lon`-length window of the backing slice.
        let samples_flat = samples
            .as_slice()
            .expect("samples are contiguous after materialization");
        // The longitude sum is a DFT bin (see the `longitude` module), which
        // removes the `n_lon` factor from the harmonic evaluation count.
        // apollo-fft routes every width, so this is the only path.
        moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
            &mut flat_contributions,
            num_modes,
            |lat, row_contrib| {
                let theta = self.theta(lat);
                let weight = self.theta_weights[lat];
                let row = &samples_flat[lat * n_lon..lat * n_lon + n_lon];
                let quadrature_weight = weight * longitude_weight;
                SHT_LONGITUDE_SCRATCH.with(|pool| {
                    pool.with_scratch(n_lon, |spectrum| {
                        longitude_spectrum(&longitude_plan(n_lon), row, spectrum);
                        for (mode_idx, &(degree, order)) in all_modes.iter().enumerate() {
                            let amplitude = harmonic_amplitude(degree, order, theta).conj();
                            row_contrib[mode_idx] =
                                amplitude * spectrum[order_bin(order, n_lon)] * quadrature_weight;
                        }
                    });
                });
            },
        );

        // Accumulate all latitude contributions into coefficients.
        for lat in 0..n_lat {
            let offset = lat * num_modes;
            for mode_idx in 0..num_modes {
                let (degree, order) = all_modes[mode_idx];
                let coeff = flat_contributions[offset + mode_idx];
                let existing = coefficients.get(degree, order);
                coefficients.set(degree, order, existing + coeff);
            }
        }

        Ok(coefficients)
    }

    /// Forward SHT for complex-valued Leto sample views on the plan grid.
    pub fn forward_complex_leto(
        &self,
        samples: leto::ArrayView2<'_, Complex64>,
    ) -> ShtResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let samples = array2_from_leto_view(samples);
        let coefficients = self.forward_complex(&samples)?;
        apollo_leto_interop::try_dense_from_array(coefficients.values())
            .ok_or(ShtError::CoefficientShapeMismatch)
    }

    /// Inverse SHT evaluating real-valued samples on the plan grid.
    pub fn inverse_real(
        &self,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> ShtResult<Array2<f64>> {
        Ok(self.inverse_complex(coefficients)?.mapv(|value| value.re))
    }

    /// Inverse SHT from Leto coefficient views into real-valued samples.
    pub fn inverse_real_leto(
        &self,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> ShtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let coefficients = coefficients_from_leto_view(self, coefficients)?;
        let samples = self.inverse_real(&coefficients)?;
        apollo_leto_interop::try_dense_from_array(&samples)
            .ok_or(ShtError::CoefficientShapeMismatch)
    }

    /// Inverse SHT evaluating complex-valued samples on the plan grid.
    pub fn inverse_complex(
        &self,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> ShtResult<Array2<Complex64>> {
        self.check_coefficient_shape(coefficients)?;
        let max_degree = self.grid.max_degree();
        let n_lat = self.grid.latitudes();
        let n_lon = self.grid.longitudes();

        // Pre-collect all (degree, order) mode pairs for deterministic iteration.
        let all_modes: Vec<(usize, isize)> = (0..=max_degree)
            .flat_map(|l| (-(l as isize)..=(l as isize)).map(move |m| (l, m)))
            .collect();

        let mut samples = Array2::<Complex64>::zeros([n_lat, n_lon]);

        // Synthesis is the same factorization read backwards: accumulate each
        // mode into the DFT bin its order occupies, then one inverse transform
        // produces the whole latitude row. apollo-fft routes every width, so
        // this is the only path.
        let synthesize = |lat: usize, row: &mut [Complex64]| {
            let theta = self.theta(lat);
            SHT_LONGITUDE_SCRATCH.with(|pool| {
                pool.with_scratch(n_lon, |orders| {
                    orders.fill(Complex64::new(0.0, 0.0));
                    for &(degree, order) in &all_modes {
                        orders[order_bin(order, n_lon)] += coefficients.get(degree, order)
                            * harmonic_amplitude(degree, order, theta);
                    }
                    longitude_synthesis(&longitude_plan(n_lon), orders);
                    row.copy_from_slice(orders);
                });
            });
        };
        if let Some(flat_samples) = samples.as_slice_mut() {
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                flat_samples,
                n_lon,
                synthesize,
            );
        } else {
            let mut row = vec![Complex64::new(0.0, 0.0); n_lon];
            for lat in 0..n_lat {
                synthesize(lat, &mut row);
                for (lon, &value) in row.iter().enumerate() {
                    samples[[lat, lon]] = value;
                }
            }
        }

        Ok(samples)
    }

    /// Inverse SHT from Leto coefficient views into complex-valued samples.
    pub fn inverse_complex_leto(
        &self,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> ShtResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let coefficients = coefficients_from_leto_view(self, coefficients)?;
        let samples = self.inverse_complex(&coefficients)?;
        apollo_leto_interop::try_dense_from_array(&samples)
            .ok_or(ShtError::CoefficientShapeMismatch)
    }

    /// Forward real-sample SHT for `f64`, `f32`, or mixed `f16` sample storage.
    pub fn forward_real_typed_into<T: ShtRealStorage, O: ShtComplexStorage>(
        &self,
        samples: &Array2<T>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        T::forward_real_into(self, samples, output, sample_profile, coefficient_profile)
    }

    /// Forward real-sample SHT from typed Leto sample storage.
    pub fn forward_real_leto_typed<T: ShtRealStorage, O: ShtComplexStorage>(
        &self,
        samples: leto::ArrayView2<'_, T>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<leto::Array<O, leto::MnemosyneStorage<O>, 2>> {
        let samples = array2_from_leto_view(samples);
        let mut output = Array2::<O>::from_elem(
            self.coefficient_shape(),
            O::from_cpu(Complex64::new(0.0, 0.0)),
        );
        self.forward_real_typed_into(&samples, &mut output, sample_profile, coefficient_profile)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or(ShtError::CoefficientShapeMismatch)
    }

    /// Forward complex-sample SHT for `Complex64`, `Complex32`, or mixed `[f16; 2]`.
    pub fn forward_complex_typed_into<T: ShtComplexStorage, O: ShtComplexStorage>(
        &self,
        samples: &Array2<T>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        T::forward_complex_into(self, samples, output, sample_profile, coefficient_profile)
    }

    /// Forward complex-sample SHT from typed Leto sample storage.
    pub fn forward_complex_leto_typed<T: ShtComplexStorage, O: ShtComplexStorage>(
        &self,
        samples: leto::ArrayView2<'_, T>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<leto::Array<O, leto::MnemosyneStorage<O>, 2>> {
        let samples = array2_from_leto_view(samples);
        let mut output = Array2::<O>::from_elem(
            self.coefficient_shape(),
            O::from_cpu(Complex64::new(0.0, 0.0)),
        );
        self.forward_complex_typed_into(
            &samples,
            &mut output,
            sample_profile,
            coefficient_profile,
        )?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or(ShtError::CoefficientShapeMismatch)
    }

    /// Inverse SHT into complex sample storage.
    pub fn inverse_complex_typed_into<T: ShtComplexStorage, O: ShtComplexStorage>(
        &self,
        coefficients: &Array2<T>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        T::inverse_complex_into(
            self,
            coefficients,
            output,
            coefficient_profile,
            sample_profile,
        )
    }

    /// Inverse SHT from typed Leto coefficient storage into complex samples.
    pub fn inverse_complex_leto_typed<T: ShtComplexStorage, O: ShtComplexStorage>(
        &self,
        coefficients: leto::ArrayView2<'_, T>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<leto::Array<O, leto::MnemosyneStorage<O>, 2>> {
        let coefficients = array2_from_leto_view(coefficients);
        let mut output = Array2::<O>::from_elem(
            [self.grid.latitudes(), self.grid.longitudes()],
            O::from_cpu(Complex64::new(0.0, 0.0)),
        );
        self.inverse_complex_typed_into(
            &coefficients,
            &mut output,
            coefficient_profile,
            sample_profile,
        )?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or(ShtError::CoefficientShapeMismatch)
    }

    /// Inverse SHT into real sample storage by taking the synthesized real part.
    pub fn inverse_real_typed_into<T: ShtComplexStorage, O: ShtRealStorage>(
        &self,
        coefficients: &Array2<T>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        T::inverse_real_into(
            self,
            coefficients,
            output,
            coefficient_profile,
            sample_profile,
        )
    }

    /// Inverse SHT from typed Leto coefficient storage into real samples.
    pub fn inverse_real_leto_typed<T: ShtComplexStorage, O: ShtRealStorage>(
        &self,
        coefficients: leto::ArrayView2<'_, T>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<leto::Array<O, leto::MnemosyneStorage<O>, 2>> {
        let coefficients = array2_from_leto_view(coefficients);
        let mut output = Array2::<O>::from_elem(
            [self.grid.latitudes(), self.grid.longitudes()],
            O::from_cpu(0.0),
        );
        self.inverse_real_typed_into(
            &coefficients,
            &mut output,
            coefficient_profile,
            sample_profile,
        )?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or(ShtError::CoefficientShapeMismatch)
    }

    pub(super) fn check_sample_shape(&self, shape: [usize; 2]) -> ShtResult<()> {
        if shape != [self.grid.latitudes(), self.grid.longitudes()] {
            return Err(ShtError::SampleShapeMismatch);
        }
        Ok(())
    }

    pub(super) fn check_coefficient_shape(
        &self,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> ShtResult<()> {
        let expected = [self.grid.max_degree() + 1, 2 * self.grid.max_degree() + 1];
        if coefficients.max_degree() != self.grid.max_degree()
            || coefficients.values().shape() != expected
        {
            return Err(ShtError::CoefficientShapeMismatch);
        }
        Ok(())
    }

    pub(super) fn coefficient_shape(&self) -> [usize; 2] {
        [self.grid.max_degree() + 1, 2 * self.grid.max_degree() + 1]
    }
}
