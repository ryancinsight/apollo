//! Reusable spherical harmonic transform plan.
//!
//! The plan uses Gauss-Legendre latitude nodes and uniform longitude nodes.
//! Forward transforms compute coefficients
//! `a_lm = integral f(theta, phi) conj(Y_lm(theta, phi)) dOmega` by product
//! quadrature. Inverse transforms evaluate
//! `f(theta, phi) = sum_l sum_m a_lm Y_lm(theta, phi)` on the same grid.

use crate::domain::contracts::error::{ShtError, ShtResult};
use crate::domain::metadata::grid::SphericalGridSpec;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use crate::infrastructure::kernel::spherical_harmonic::{
    gauss_legendre_nodes_weights, spherical_harmonic,
};
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array2;
use num_complex::{Complex32, Complex64};

/// Reusable spherical harmonic transform (SHT) plan.
///
/// Pre-computes Gauss-Legendre nodes and weights for the latitude axis and caches the
/// validated [`SphericalGridSpec`]. The same plan can be reused for multiple transforms
/// without recomputing the quadrature rule.
///
/// # Complexity Theorem
///
/// Let `L = max_degree`, `N_lat = latitudes`, `N_lon = longitudes`, and
/// `M = (L + 1)^2` (total number of spectral modes).
///
/// | Transform | Complexity          | Description                                          |
/// |-----------|---------------------|------------------------------------------------------|
/// | Forward   | O(M · N_lat · N_lon) | Quadrature sum over all grid points for each mode   |
/// | Inverse   | O(N_lat · N_lon · M) | Synthesis sum over all modes for each grid point    |
///
/// Both operations are equivalent to dense matrix–vector products of dimension
/// `(M) × (N_lat · N_lon)`. Moirai parallelism distributes the `N_lat` latitude rows
/// across available threads, giving a practical wall-time factor of `1/P` for `P` cores
/// on the outer loop.
///
/// # Quadrature Exactness
///
/// The Gauss-Legendre nodes guarantee exact integration for products of spherical
/// harmonics of degree `<= L` provided the following grid constraints hold:
///
/// - `N_lat > L`: The `N_lat`-point GL rule is exact for polynomials of degree
///   `<= 2*N_lat - 1 >= 2L`, which covers all products `Y_l^m * conj(Y_{l'}^{m'})`
///   with `l, l' <= L` (degree `<= 2L` in `cos θ`). See Theorem 2 and Theorem 4 in
///   [`crate::infrastructure::kernel::spherical_harmonic`].
/// - `N_lon >= 2L + 1`: The uniform longitude grid recovers all azimuthal modes
///   `|m| <= L` without aliasing (DFT orthogonality identity).
///
/// Under these constraints, `inverse(forward(f)) = f` in exact arithmetic for any
/// field `f` band-limited to degree `<= L` (Theorem 4 in
/// [`crate::infrastructure::kernel::spherical_harmonic`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ShtPlan {
    grid: SphericalGridSpec,
    cos_theta_nodes: Vec<f64>,
    theta_weights: Vec<f64>,
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
                all_modes
                    .iter()
                    .map(|&(degree, order)| {
                        let lon_sum: Complex64 = (0..n_lon)
                            .map(|lon| {
                                let phi = self.phi(lon);
                                samples[[lat, lon]]
                                    * spherical_harmonic(degree, order, theta, phi).conj()
                            })
                            .sum();
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

        // Parallelize over latitude rows; each row is computed independently.
        let row_values: Vec<Vec<Complex64>> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n_lat, |lat| {
                let theta = self.theta(lat);
                (0..n_lon)
                    .map(|lon| {
                        let phi = self.phi(lon);
                        all_modes
                            .iter()
                            .map(|&(degree, order)| {
                                coefficients.get(degree, order)
                                    * spherical_harmonic(degree, order, theta, phi)
                            })
                            .sum()
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

    fn check_sample_shape(&self, shape: (usize, usize)) -> ShtResult<()> {
        if shape != (self.grid.latitudes(), self.grid.longitudes()) {
            return Err(ShtError::SampleShapeMismatch);
        }
        Ok(())
    }

    fn check_coefficient_shape(
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

    fn coefficient_shape(&self) -> (usize, usize) {
        (self.grid.max_degree() + 1, 2 * self.grid.max_degree() + 1)
    }
}

/// Real sample storage accepted by typed SHT paths.
pub trait ShtRealStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `f64` arithmetic.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic back to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute typed forward real SHT.
    fn forward_real_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        samples: &Array2<Self>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(sample_profile, Self::PROFILE)?;
        validate_profile(coefficient_profile, O::PROFILE)?;
        validate_sample_array_shape(plan, samples)?;
        validate_coefficient_array_shape(plan, output)?;
        let samples64 = samples.mapv(Self::to_f64);
        let coefficients = plan.forward_real(&samples64)?;
        write_complex_array(coefficients.values(), output);
        Ok(())
    }
}

impl ShtRealStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }
}

impl ShtRealStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl ShtRealStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

/// Complex sample/coefficient storage accepted by typed SHT paths.
pub trait ShtComplexStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `Complex64` arithmetic.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic back to storage.
    fn from_complex64(value: Complex64) -> Self;

    /// Execute typed forward complex SHT.
    fn forward_complex_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        samples: &Array2<Self>,
        output: &mut Array2<O>,
        sample_profile: PrecisionProfile,
        coefficient_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(sample_profile, Self::PROFILE)?;
        validate_profile(coefficient_profile, O::PROFILE)?;
        validate_sample_array_shape(plan, samples)?;
        validate_coefficient_array_shape(plan, output)?;
        let samples64 = samples.mapv(Self::to_complex64);
        let coefficients = plan.forward_complex(&samples64)?;
        write_complex_array(coefficients.values(), output);
        Ok(())
    }

    /// Execute typed inverse SHT into complex samples.
    fn inverse_complex_into<O: ShtComplexStorage>(
        plan: &ShtPlan,
        coefficients: &Array2<Self>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(coefficient_profile, Self::PROFILE)?;
        validate_profile(sample_profile, O::PROFILE)?;
        validate_coefficient_array_shape(plan, coefficients)?;
        validate_sample_array_shape(plan, output)?;
        let coefficients64 = coefficients.mapv(Self::to_complex64);
        let owner_coefficients =
            SphericalHarmonicCoefficients::from_values(plan.grid.max_degree(), coefficients64);
        let samples = plan.inverse_complex(&owner_coefficients)?;
        write_complex_array(&samples, output);
        Ok(())
    }

    /// Execute typed inverse SHT into real samples.
    fn inverse_real_into<O: ShtRealStorage>(
        plan: &ShtPlan,
        coefficients: &Array2<Self>,
        output: &mut Array2<O>,
        coefficient_profile: PrecisionProfile,
        sample_profile: PrecisionProfile,
    ) -> ShtResult<()> {
        validate_profile(coefficient_profile, Self::PROFILE)?;
        validate_profile(sample_profile, O::PROFILE)?;
        validate_coefficient_array_shape(plan, coefficients)?;
        validate_sample_array_shape(plan, output)?;
        let coefficients64 = coefficients.mapv(Self::to_complex64);
        let owner_coefficients =
            SphericalHarmonicCoefficients::from_values(plan.grid.max_degree(), coefficients64);
        let samples = plan.inverse_real(&owner_coefficients)?;
        for (slot, value) in output.iter_mut().zip(samples.iter().copied()) {
            *slot = O::from_f64(value);
        }
        Ok(())
    }
}

impl ShtComplexStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }
}

impl ShtComplexStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl ShtComplexStorage for [f16; 2] {
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

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> ShtResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(ShtError::PrecisionMismatch)
    }
}

fn validate_sample_array_shape<T>(plan: &ShtPlan, samples: &Array2<T>) -> ShtResult<()> {
    plan.check_sample_shape(samples.dim())
}

fn validate_coefficient_array_shape<T>(plan: &ShtPlan, coefficients: &Array2<T>) -> ShtResult<()> {
    if coefficients.dim() == plan.coefficient_shape() {
        Ok(())
    } else {
        Err(ShtError::CoefficientShapeMismatch)
    }
}

fn write_complex_array<T: ShtComplexStorage>(source: &Array2<Complex64>, target: &mut Array2<T>) {
    for (slot, value) in target.iter_mut().zip(source.iter().copied()) {
        *slot = T::from_complex64(value);
    }
}

fn array2_from_leto_view<T: Copy>(
    view: leto::ArrayView2<'_, T>,
    shape_error: ShtError,
) -> ShtResult<Array2<T>> {
    let [rows, cols] = view.shape();
    let mut values = Vec::with_capacity(view.size());
    for row in 0..rows {
        for col in 0..cols {
            values.push(*view.get([row, col]).map_err(|_| shape_error)?);
        }
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|_| shape_error)
}

fn leto_array2_from_ndarray<T: Copy>(
    array: &Array2<T>,
) -> ShtResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
    let (rows, cols) = array.dim();
    let values = array.iter().copied().collect::<Vec<_>>();
    leto::Array::from_mnemosyne_slice([rows, cols], &values)
        .map_err(|_| ShtError::CoefficientShapeMismatch)
}

fn coefficients_from_leto_view(
    plan: &ShtPlan,
    coefficients: leto::ArrayView2<'_, Complex64>,
) -> ShtResult<SphericalHarmonicCoefficients> {
    let values = array2_from_leto_view(coefficients, ShtError::CoefficientShapeMismatch)?;
    if values.dim() != plan.coefficient_shape() {
        return Err(ShtError::CoefficientShapeMismatch);
    }
    Ok(SphericalHarmonicCoefficients::from_values(
        plan.grid.max_degree(),
        values,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn coefficient_shape(plan: &ShtPlan) -> (usize, usize) {
        (
            plan.grid().max_degree() + 1,
            2 * plan.grid().max_degree() + 1,
        )
    }

    #[test]
    fn typed_real_forward_supports_f64_f32_and_mixed_f16_storage() {
        let plan = ShtPlan::new(6, 13, 2).expect("plan");
        let constant = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
        let samples64 = Array2::from_elem(
            (plan.grid().latitudes(), plan.grid().longitudes()),
            constant,
        );
        let expected = plan.forward_real(&samples64).expect("forward");
        let shape = coefficient_shape(&plan);

        let mut out64 = Array2::<Complex64>::zeros(shape);
        plan.forward_real_typed_into(
            &samples64,
            &mut out64,
            PrecisionProfile::HIGH_ACCURACY_F64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed f64 real forward");
        for (actual, expected) in out64.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let samples32 = samples64.mapv(|value| value as f32);
        let represented32 = samples32.mapv(f64::from);
        let expected32 = plan
            .forward_real(&represented32)
            .expect("represented f32 forward");
        let mut out32 = Array2::<Complex32>::zeros(shape);
        plan.forward_real_typed_into(
            &samples32,
            &mut out32,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed f32 real forward");
        for (actual, expected) in out32.iter().zip(expected32.values().iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let samples16 = samples64.mapv(|value| f16::from_f32(value as f32));
        let represented16 = samples16.mapv(|value| f64::from(value.to_f32()));
        let expected16 = plan
            .forward_real(&represented16)
            .expect("represented f16 forward");
        let mut out16 = Array2::from_elem(shape, [f16::from_f32(0.0), f16::from_f32(0.0)]);
        plan.forward_real_typed_into(
            &samples16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 real forward");
        for (actual, expected) in out16.iter().zip(expected16.values().iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn leto_real_forward_matches_ndarray_reference() {
        let plan = ShtPlan::new(6, 13, 2).expect("plan");
        let constant = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
        let samples = Array2::from_elem(
            (plan.grid().latitudes(), plan.grid().longitudes()),
            constant,
        );
        let input = leto::Array2::from_shape_vec(
            [plan.grid().latitudes(), plan.grid().longitudes()],
            samples.iter().copied().collect(),
        )
        .expect("leto samples");
        let expected = plan.forward_real(&samples).expect("ndarray forward");

        let actual = plan
            .forward_real_leto(input.view())
            .expect("leto real forward");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous coefficients");
        for (actual, expected) in actual.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_real_forward_matches_ndarray_reference() {
        let plan = ShtPlan::new(6, 13, 2).expect("plan");
        let samples = Array2::from_shape_fn(
            (plan.grid().latitudes(), plan.grid().longitudes()),
            |(lat, lon)| (lat as f64 * 0.2).sin() + (lon as f64 * 0.1).cos(),
        );
        let mut backing = Vec::with_capacity(samples.len() * 2);
        for value in samples.iter().copied() {
            backing.push(value);
            backing.push(99.0);
        }
        let input = leto::Array2::from_shape_vec(
            [plan.grid().latitudes(), plan.grid().longitudes() * 2],
            backing,
        )
        .expect("leto samples");
        let strided = input
            .view()
            .slice(&[
                (0, plan.grid().latitudes(), 1),
                (0, plan.grid().longitudes() * 2, 2),
            ])
            .expect("strided samples");
        let expected = plan.forward_real(&samples).expect("ndarray forward");

        let actual = plan.forward_real_leto(strided).expect("leto real forward");
        let actual_view = actual.view();
        let actual = actual_view.as_slice().expect("contiguous coefficients");
        for (actual, expected) in actual.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_complex_forward_and_inverse_match_ndarray_reference() {
        let plan = ShtPlan::new(6, 13, 2).expect("plan");
        let samples = Array2::from_shape_fn(
            (plan.grid().latitudes(), plan.grid().longitudes()),
            |(lat, lon)| spherical_harmonic(1, 1, plan.theta(lat), plan.phi(lon)),
        );
        let input = leto::Array2::from_shape_vec(
            [plan.grid().latitudes(), plan.grid().longitudes()],
            samples.iter().copied().collect(),
        )
        .expect("leto samples");
        let expected_coefficients = plan.forward_complex(&samples).expect("ndarray forward");

        let actual_coefficients = plan
            .forward_complex_leto(input.view())
            .expect("leto complex forward");
        let actual_coefficients_view = actual_coefficients.view();
        let actual_coefficients_slice = actual_coefficients_view
            .as_slice()
            .expect("contiguous coefficients");
        for (actual, expected) in actual_coefficients_slice
            .iter()
            .zip(expected_coefficients.values().iter())
        {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let coefficients = leto::Array2::from_shape_vec(
            [
                plan.grid().max_degree() + 1,
                2 * plan.grid().max_degree() + 1,
            ],
            expected_coefficients.values().iter().copied().collect(),
        )
        .expect("leto coefficients");
        let expected_inverse = plan
            .inverse_complex(&expected_coefficients)
            .expect("ndarray inverse");
        let actual_inverse = plan
            .inverse_complex_leto(coefficients.view())
            .expect("leto inverse");
        let actual_inverse_view = actual_inverse.view();
        let actual_inverse = actual_inverse_view
            .as_slice()
            .expect("contiguous inverse samples");
        for (actual, expected) in actual_inverse.iter().zip(expected_inverse.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_leto_forward_and_inverse_match_ndarray_reference() {
        let plan = ShtPlan::new(6, 13, 2).expect("plan");
        let samples = Array2::from_shape_fn(
            (plan.grid().latitudes(), plan.grid().longitudes()),
            |(lat, lon)| (lat as f32 * 0.2).sin() + (lon as f32 * 0.1).cos(),
        );
        let input = leto::Array2::from_shape_vec(
            [plan.grid().latitudes(), plan.grid().longitudes()],
            samples.iter().copied().collect(),
        )
        .expect("leto samples");
        let shape = coefficient_shape(&plan);
        let mut expected_coefficients = Array2::<Complex32>::zeros(shape);
        plan.forward_real_typed_into(
            &samples,
            &mut expected_coefficients,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed ndarray forward");

        let actual_coefficients = plan
            .forward_real_leto_typed::<f32, Complex32>(
                input.view(),
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed leto forward");
        let actual_coefficients_view = actual_coefficients.view();
        let actual_coefficients_slice = actual_coefficients_view
            .as_slice()
            .expect("contiguous coefficients");
        for (actual, expected) in actual_coefficients_slice
            .iter()
            .zip(expected_coefficients.iter())
        {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }

        let coefficients = leto::Array2::from_shape_vec(
            [shape.0, shape.1],
            expected_coefficients.iter().copied().collect(),
        )
        .expect("leto coefficients");
        let mut expected_samples =
            Array2::<f32>::zeros((plan.grid().latitudes(), plan.grid().longitudes()));
        plan.inverse_real_typed_into(
            &expected_coefficients,
            &mut expected_samples,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed ndarray inverse");
        let actual_samples = plan
            .inverse_real_leto_typed::<Complex32, f32>(
                coefficients.view(),
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed leto inverse");
        let actual_samples_view = actual_samples.view();
        let actual_samples = actual_samples_view.as_slice().expect("contiguous samples");
        for (actual, expected) in actual_samples.iter().zip(expected_samples.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn typed_complex_forward_and_inverse_support_complex32_storage() {
        let plan = ShtPlan::new(6, 13, 2).expect("plan");
        let samples64 = Array2::from_shape_fn(
            (plan.grid().latitudes(), plan.grid().longitudes()),
            |(lat, lon)| spherical_harmonic(1, 1, plan.theta(lat), plan.phi(lon)),
        );
        let samples32 = samples64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let represented32 = samples32.mapv(Complex32::to_complex64);
        let expected = plan.forward_complex(&represented32).expect("forward");
        let shape = coefficient_shape(&plan);

        let mut coefficients32 = Array2::<Complex32>::zeros(shape);
        plan.forward_complex_typed_into(
            &samples32,
            &mut coefficients32,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 forward");
        for (actual, expected) in coefficients32.iter().zip(expected.values().iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let mut recovered32 =
            Array2::<Complex32>::zeros((plan.grid().latitudes(), plan.grid().longitudes()));
        plan.inverse_complex_typed_into(
            &coefficients32,
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 inverse");
        for (actual, expected) in recovered32.iter().zip(samples32.iter()) {
            assert!((actual.re - expected.re).abs() < 1.0e-4);
            assert!((actual.im - expected.im).abs() < 1.0e-4);
        }
    }

    #[test]
    fn typed_real_inverse_and_mismatch_rejections_are_value_semantic() {
        let plan = ShtPlan::new(5, 11, 2).expect("plan");
        let mut coefficients = SphericalHarmonicCoefficients::zeros(plan.grid().max_degree());
        coefficients.set(0, 0, Complex64::new(1.0, 0.0));
        let coefficient_shape = coefficient_shape(&plan);
        let coefficients32 = coefficients
            .values()
            .mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let mut samples32 =
            Array2::<f32>::zeros((plan.grid().latitudes(), plan.grid().longitudes()));

        plan.inverse_real_typed_into(
            &coefficients32,
            &mut samples32,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed real inverse");
        let expected = plan.inverse_real(&coefficients).expect("inverse");
        for (actual, expected) in samples32.iter().zip(expected.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let err = plan
            .inverse_real_typed_into(
                &coefficients32,
                &mut samples32,
                PrecisionProfile::HIGH_ACCURACY_F64,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect_err("profile mismatch");
        assert_eq!(err, ShtError::PrecisionMismatch);

        let bad_coefficients =
            Array2::<Complex32>::zeros((coefficient_shape.0, coefficient_shape.1 + 1));
        let err = plan
            .inverse_real_typed_into(
                &bad_coefficients,
                &mut samples32,
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect_err("shape mismatch");
        assert_eq!(err, ShtError::CoefficientShapeMismatch);
    }
}
