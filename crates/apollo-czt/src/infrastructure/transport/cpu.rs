use crate::application::execution::plan::czt::dimension_1d::CztPlan;
use crate::domain::contracts::error::CztError;
use leto::Array1;
use eunomia::Complex64;

/// Computes the forward CZT using the synchronous standard CPU pipeline.
pub fn czt(
    input: &Array1<Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<Array1<Complex64>, CztError> {
    CztPlan::new(input.size(), output_len, a, w)?.forward(input)
}

/// Computes the forward CZT from a Leto view using the synchronous CPU pipeline.
pub fn czt_leto(
    input: leto::ArrayView1<'_, Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, CztError> {
    CztPlan::new(input.shape()[0], output_len, a, w)?.forward_leto(input)
}

/// Computes the forward direct CZT from a Leto view using the synchronous CPU pipeline.
pub fn czt_direct_leto(
    input: leto::ArrayView1<'_, Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, CztError> {
    CztPlan::new(input.shape()[0], output_len, a, w)?.forward_direct_leto(input)
}

/// Computes the forward CZT using strict direct $O(NM)$ methods.
pub fn czt_direct(
    input: &Array1<Complex64>,
    output_len: usize,
    a: Complex64,
    w: Complex64,
) -> Result<Array1<Complex64>, CztError> {
    CztPlan::new(input.size(), output_len, a, w)?.forward_direct(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use leto::Storage;

    #[test]
    fn czt_leto_matches_ndarray_transport() {
        let input = vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, -0.25),
            Complex64::new(0.5, 0.125),
        ];
        let a = Complex64::from_polar(1.0, 0.125);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0);
        let ndarray_input = Array1::from(input.clone());
        let expected = czt(&ndarray_input, 6, a, w).expect("ndarray transport");
        let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("leto input");

        let actual = czt_leto(leto_input.view(), 6, a, w).expect("leto transport");

        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }
}
