//! Reusable spherical harmonic transform plan.
//!
//! The plan uses Gauss-Legendre latitude nodes and uniform longitude nodes.
//! Forward transforms compute coefficients
//! `a_lm = integral f(theta, phi) conj(Y_lm(theta, phi)) dOmega` by product
//! quadrature. Inverse transforms evaluate
//! `f(theta, phi) = sum_l sum_m a_lm Y_lm(theta, phi)` on the same grid.

mod helpers;
mod typed;
#[cfg(test)]
mod tests;

pub use typed::{ShtRealStorage, ShtComplexStorage};

use crate::domain::contracts::error::{ShtError, ShtResult};
use crate::domain::metadata::grid::SphericalGridSpec;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use crate::infrastructure::kernel::spherical_harmonic::gauss_legendre_nodes_weights;
use apollo_fft::PrecisionProfile;
use ndarray::Array2;
use num_complex::Complex64;
use helpers::{
    array2_from_leto_view, coefficients_from_leto_view, leto_array2_from_ndarray,
    interleaved_lanes, coefficient_lanes, sht_forward_mode_sum_hermes, sht_forward_mode_sum,
    sht_inverse_sample_hermes, sht_inverse_sample, SHT_HERMES_DOT_LEN_THRESHOLD,
};

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
        self.check_sample_shape(samples.dim())?;
        let complex_samples = samples.mapv(|value| Complex64::new(value, 0.0));
        self.forward_complex(&complex_samples)
    }

    /// Forward SHT for real-valued Leto sample views on the plan grid.
    pub fn forward_real_leto(
        &self,
        samples: leto::ArrayView2<'_, f64>,
    ) -> ShtResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let samples = array2_from_leto_view(samples, ShtError::SampleShapeMismatch)?;
        let coefficients = self.forward_real(&samples)?;
        leto_array2_from_ndarray(coefficients.values())
    }

    /// Forward SHT for complex-valued samples on the plan grid.
    pub fn forward_complex(
        &self,
        samples: &Array2<Complex64>,
    ) -> ShtResult<SphericalHarmonicCoefficients> {
        self.check_sample_shape(samples.dim())?;
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
        let contributions: Vec<Vec<Complex64>> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n_lat, |lat| {
                let theta = self.theta(lat);
                let weight = self.theta_weights[lat];
                let row = samples.row(lat);
                let row_values;
                let row = match row.as_slice() {
                    Some(row) => row,
                    None => {
                        row_values = row.iter().copied().collect::<Vec<_>>();
                        &row_values
                    }
                };
                let sample_lanes =
                    (n_lon >= SHT_HERMES_DOT_LEN_THRESHOLD).then(|| interleaved_lanes(row));
                all_modes
                    .iter()
                    .map(|&(degree, order)| {
                        let lon_sum = match sample_lanes {
                            Some(lanes) => {
                                sht_forward_mode_sum_hermes(lanes, degree, order, theta, n_lon)
                            }
                            None => sht_forward_mode_sum(row, degree, order, theta, n_lon),
                        };
                        lon_sum * (weight * longitude_weight)
                    })
                    .collect()
            });

        // Accumulate all latitude contributions into coefficients.
        for lat_contribs in contributions {
            for (mode_idx, coeff) in lat_contribs.into_iter().enumerate() {
                let (degree, order) = all_modes[mode_idx];
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
        let samples = array2_from_leto_view(samples, ShtError::SampleShapeMismatch)?;
        let coefficients = self.forward_complex(&samples)?;
        leto_array2_from_ndarray(coefficients.values())
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
        leto_array2_from_ndarray(&samples)
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
        let coefficient_lanes = (all_modes.len() >= SHT_HERMES_DOT_LEN_THRESHOLD)
            .then(|| coefficient_lanes(coefficients, &all_modes));

        // Parallelize over latitude rows; each row is computed independently.
        let row_values: Vec<Vec<Complex64>> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n_lat, |lat| {
                let theta = self.theta(lat);
                (0..n_lon)
                    .map(|lon| {
                        let phi = self.phi(lon);
                        match &coefficient_lanes {
                            Some(lanes) => sht_inverse_sample_hermes(lanes, &all_modes, theta, phi),
                            None => sht_inverse_sample(coefficients, &all_modes, theta, phi),
                        }
                    })
                    .collect()
            });

        // Assemble into output array.
        let mut samples = Array2::<Complex64>::zeros((n_lat, n_lon));
        for (lat, row) in row_values.into_iter().enumerate() {
            for (lon, value) in row.into_iter().enumerate() {
                samples[[lat, lon]] = value;
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
        leto_array2_from_ndarray(&samples)
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
        let samples = array2_from_leto_view(samples, ShtError::SampleShapeMismatch)?;
        let mut output = Array2::<O>::from_elem(
            self.coefficient_shape(),
            O::from_complex64(Complex64::new(0.0, 0.0)),
        );
        self.forward_real_typed_into(&samples, &mut output, sample_profile, coefficient_profile)?;
        leto_array2_from_ndarray(&output)
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
        let samples = array2_from_leto_view(samples, ShtError::SampleShapeMismatch)?;
        let mut output = Array2::<O>::from_elem(
            self.coefficient_shape(),
            O::from_complex64(Complex64::new(0.0, 0.0)),
        );
        self.forward_complex_typed_into(
            &samples,
            &mut output,
            sample_profile,
            coefficient_profile,
        )?;
        leto_array2_from_ndarray(&output)
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
        let coefficients = array2_from_leto_view(coefficients, ShtError::CoefficientShapeMismatch)?;
        let mut output = Array2::<O>::from_elem(
            (self.grid.latitudes(), self.grid.longitudes()),
            O::from_complex64(Complex64::new(0.0, 0.0)),
        );
        self.inverse_complex_typed_into(
            &coefficients,
            &mut output,
            coefficient_profile,
            sample_profile,
        )?;
        leto_array2_from_ndarray(&output)
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
        let coefficients = array2_from_leto_view(coefficients, ShtError::CoefficientShapeMismatch)?;
        let mut output = Array2::<O>::from_elem(
            (self.grid.latitudes(), self.grid.longitudes()),
            O::from_f64(0.0),
        );
        self.inverse_real_typed_into(
            &coefficients,
            &mut output,
            coefficient_profile,
            sample_profile,
        )?;
        leto_array2_from_ndarray(&output)
    }

    pub(super) fn check_sample_shape(&self, shape: (usize, usize)) -> ShtResult<()> {
        if shape != (self.grid.latitudes(), self.grid.longitudes()) {
            return Err(ShtError::SampleShapeMismatch);
        }
        Ok(())
    }

    pub(super) fn check_coefficient_shape(
        &self,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> ShtResult<()> {
        let expected = (self.grid.max_degree() + 1, 2 * self.grid.max_degree() + 1);
        if coefficients.max_degree() != self.grid.max_degree()
            || coefficients.values().dim() != expected
        {
            return Err(ShtError::CoefficientShapeMismatch);
        }
        Ok(())
    }

    pub(super) fn coefficient_shape(&self) -> (usize, usize) {
        (self.grid.max_degree() + 1, 2 * self.grid.max_degree() + 1)
    }
}
