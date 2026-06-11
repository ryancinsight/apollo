use crate::application::execution::plan::fwht::dimension_1d::FwhtPlan;
use crate::domain::contracts::error::FwhtError;
use ndarray::Array1;

/// Forward FWHT convenience wrapper.
///
/// Creates a plan from the signal length and calls forward.
/// Fails if the signal length is zero or not a power of two.
pub fn fwht(signal: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
    FwhtPlan::new(signal.len())?.forward(signal)
}

/// Forward FWHT convenience wrapper for a Leto view.
pub fn fwht_leto(
    signal: leto::ArrayView1<'_, f64>,
) -> Result<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>, FwhtError> {
    FwhtPlan::new(signal.shape()[0])?.forward_leto(signal)
}

/// Forward FWHT convenience wrapper for caller-owned Leto output.
///
/// Creates a plan from the signal length and writes directly into `output`
/// when the view is contiguous.
pub fn fwht_leto_into(
    signal: leto::ArrayView1<'_, f64>,
    output: leto::ArrayViewMut1<'_, f64>,
) -> Result<(), FwhtError> {
    FwhtPlan::new(signal.shape()[0])?.forward_leto_into(signal, output)
}

/// Inverse FWHT convenience wrapper.
///
/// Creates a plan from the spectrum length and calls inverse.
/// Fails if the spectrum length is zero or not a power of two.
pub fn ifwht(spectrum: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
    FwhtPlan::new(spectrum.len())?.inverse(spectrum)
}

/// Inverse FWHT convenience wrapper for a Leto view.
pub fn ifwht_leto(
    spectrum: leto::ArrayView1<'_, f64>,
) -> Result<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>, FwhtError> {
    FwhtPlan::new(spectrum.shape()[0])?.inverse_leto(spectrum)
}

/// Inverse FWHT convenience wrapper for caller-owned Leto output.
///
/// Creates a plan from the spectrum length and writes directly into `output`
/// when the view is contiguous.
pub fn ifwht_leto_into(
    spectrum: leto::ArrayView1<'_, f64>,
    output: leto::ArrayViewMut1<'_, f64>,
) -> Result<(), FwhtError> {
    FwhtPlan::new(spectrum.shape()[0])?.inverse_leto_into(spectrum, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use leto::Storage;

    #[test]
    fn leto_transport_caller_owned_output_matches_allocating_forward() {
        let signal =
            leto::Array1::from_shape_vec([8], vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0])
                .expect("signal");
        let expected = fwht_leto(signal.view()).expect("allocating fwht");
        let mut actual = leto::Array::<f64, leto::MnemosyneStorage<f64>, 1>::zeros_mnemosyne([8]);

        fwht_leto_into(signal.view(), actual.view_mut()).expect("caller-owned fwht");

        for (actual, expected) in actual
            .storage()
            .as_slice()
            .iter()
            .zip(expected.storage().as_slice())
        {
            assert_relative_eq!(actual, expected, epsilon = 0.0);
        }
    }

    #[test]
    fn leto_transport_caller_owned_output_roundtrips_inverse() {
        let signal =
            leto::Array1::from_shape_vec([8], vec![1.0, -2.0, 3.5, 0.25, -1.5, 2.0, 0.0, 4.0])
                .expect("signal");
        let spectrum = fwht_leto(signal.view()).expect("allocating fwht");
        let mut recovered =
            leto::Array::<f64, leto::MnemosyneStorage<f64>, 1>::zeros_mnemosyne([8]);

        ifwht_leto_into(spectrum.view(), recovered.view_mut()).expect("caller-owned ifwht");

        for (actual, expected) in recovered
            .storage()
            .as_slice()
            .iter()
            .zip(signal.storage().as_slice())
        {
            assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_transport_caller_owned_output_rejects_length_mismatch() {
        let signal = leto::Array1::from_shape_vec([4], vec![1.0, 2.0, 3.0, 4.0]).expect("signal");
        let mut output = leto::Array1::from_shape_vec([3], vec![0.0; 3]).expect("output");

        assert!(matches!(
            fwht_leto_into(signal.view(), output.view_mut()),
            Err(FwhtError::LengthMismatch)
        ));
    }
}
