use eunomia::{FloatElement, NumericElement, RealField};

use crate::Normalization;

/// Apply an unnormalized Type-III discrete cosine transform.
///
/// For an input of length `N`, this computes
/// `y[j] = X[0]/2 + Σ(k=1..N-1, X[k] cos((π/N) k (j+1/2)))`.
/// Arithmetic follows the selected [`RealField`] implementation. NaNs
/// propagate through affected sums, infinities follow ordinary IEEE-754
/// operations, and exact zero sums may have a positive sign because
/// accumulation starts at `+0`.
///
/// The output length determines the number of projected positions, preserving
/// the existing Apollo direct-kernel contract. An empty input leaves `output`
/// unchanged.
pub fn dct3<T: RealField>(signal: &[T], output: &mut [T]) {
    let length = signal.len();
    if length == 0 {
        return;
    }
    for (position, slot) in output.iter_mut().enumerate() {
        let mut sum = T::ZERO;
        for (frequency, value) in signal.iter().copied().enumerate() {
            sum +=
                value * coefficient::<T>(Normalization::Unnormalized, length, position, frequency);
        }
        *slot = sum;
    }
}

pub(crate) fn coefficient<T: RealField>(
    normalization: Normalization,
    length: usize,
    position: usize,
    frequency: usize,
) -> T {
    let length = scalar::<T>(length);
    match normalization {
        Normalization::Unnormalized if frequency == 0 => T::from_f64(0.5),
        Normalization::Orthonormal if frequency == 0 => {
            <T as NumericElement>::sqrt(T::ONE / length)
        }
        Normalization::Unnormalized => angle::<T>(length, position, frequency),
        Normalization::Orthonormal => {
            let scale = <T as NumericElement>::sqrt(T::from_f64(2.0) / length);
            scale * angle::<T>(length, position, frequency)
        }
    }
}

fn angle<T: RealField>(length: T, position: usize, frequency: usize) -> T {
    let half = T::from_f64(0.5);
    let factor = T::PI / length;
    let angle = factor * scalar::<T>(frequency) * (scalar::<T>(position) + half);
    <T as FloatElement>::cos(angle)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "transform indices enter the selected floating-point domain to define its grid"
)]
fn scalar<T: RealField>(value: usize) -> T {
    T::from_f64(value as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_point_threshold<T: RealField>() -> T {
        // The largest phase is 3π/4. Its represented π and two products give
        // γ3 phase error. Multiplying by the AC amplitude four propagates that
        // error through cosine's unit Lipschitz bound; eight ulps cover the two
        // output operations and conversion of the independent reference. The
        // assertion verifies the locked cosine provider at these finite phases
        // rather than assuming a universal libm bound.
        let unit_roundoff = T::EPSILON / T::from_f64(2.0);
        let gamma_three =
            T::from_f64(3.0) * unit_roundoff / (T::ONE - T::from_f64(3.0) * unit_roundoff);
        let maximum_phase = T::from_f64(2.356_194_490_192_345) * (T::ONE + unit_roundoff);
        T::from_f64(4.0) * maximum_phase * gamma_three + T::from_f64(8.0) * unit_roundoff
    }

    fn projection<T: RealField>() {
        let signal = [T::from_f64(2.0), T::from_f64(4.0)];
        let mut output = [T::ZERO; 2];
        dct3(&signal, &mut output);
        let bound = two_point_threshold::<T>();
        let first = T::from_f64(3.828_427_124_746_190_3);
        let second = T::from_f64(-1.828_427_124_746_190_3);
        assert!(<T as NumericElement>::abs(output[0] - first) <= bound);
        assert!(<T as NumericElement>::abs(output[1] - second) <= bound);
    }

    fn boundary_values<T: RealField>() {
        let mut output = [T::ZERO; 1];
        dct3(&[T::from_f64(6.0)], &mut output);
        assert_eq!(output, [T::from_f64(3.0)]);

        dct3(&[T::NAN], &mut output);
        assert!(<T as NumericElement>::is_nan(output[0]));

        dct3(&[T::INFINITY], &mut output);
        assert_eq!(output, [T::INFINITY]);

        dct3(&[T::from_f64(-0.0)], &mut output);
        assert!(<T as RealField>::is_sign_positive(output[0]));
    }

    #[test]
    fn unnormalized_projection_uses_each_supported_precision() {
        projection::<f32>();
        projection::<f64>();
    }

    #[test]
    fn length_one_and_special_values_follow_the_scalar_contract() {
        boundary_values::<f32>();
        boundary_values::<f64>();
    }

    #[test]
    fn empty_input_preserves_output() {
        let mut output = [3.0_f64, 5.0];
        dct3(&[], &mut output);
        assert_eq!(output, [3.0, 5.0]);
    }

    #[test]
    fn output_length_controls_projected_positions() {
        let signal = [2.0_f64, 4.0];
        let mut output = [0.0; 1];
        dct3(&signal, &mut output);
        let expected = 3.828_427_124_746_190_3;
        assert!((output[0] - expected).abs() <= two_point_threshold::<f64>());
    }
}
