//! FFT output shift utilities.
//!
//! ## Mathematical contract
//!
//! `fftshift(x)` rearranges a DFT output by moving the zero-frequency component from bin 0
//! to the center of the array. For a length-`n` array:
//!
//! - Shift amount: `s = n / 2` (floor division)
//! - `output[k] = input[(k + n − s) % n]`
//!
//! For even `n`: the zero-frequency ends up at position `n/2`.
//! For odd `n`: zero-frequency ends up at position `(n−1)/2`.
//!
//! `ifftshift(x)` is the inverse: it undoes `fftshift`.
//! - Shift amount: `s = (n + 1) / 2` (ceil division)
//! - `output[k] = input[(k + n − s) % n]`
//!
//! Property: `ifftshift(fftshift(x)) = x` for all lengths `n`.

/// Rearrange DFT output to center zero-frequency component.
///
/// Implements numpy-compatible `fftshift` for any `Copy` element type.
/// For `n = 0` or `n = 1` returns a clone of the input.
#[must_use]
pub fn fftshift<T: Copy>(input: &[T]) -> Vec<T> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    let shift = n / 2;
    shift_left(input, shift)
}

/// Undo `fftshift`: moves the zero-frequency component back to bin 0.
///
/// Implements numpy-compatible `ifftshift` for any `Copy` element type.
/// For `n = 0` or `n = 1` returns a clone of the input.
///
/// Property: `ifftshift(fftshift(x)) = x`.
#[must_use]
pub fn ifftshift<T: Copy>(input: &[T]) -> Vec<T> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    let shift = n.div_ceil(2);
    shift_left(input, shift)
}

fn shift_left<T: Copy>(input: &[T], shift: usize) -> Vec<T> {
    let split = input.len() - shift;
    let mut output = Vec::with_capacity(input.len());
    output.extend_from_slice(&input[split..]);
    output.extend_from_slice(&input[..split]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// fftshift([0,1,2,3,-4,-3,-2,-1]) = [-4,-3,-2,-1,0,1,2,3].
    /// Reference: numpy.fft.fftshift([0,1,2,3,-4,-3,-2,-1]).
    #[test]
    fn fftshift_n8_matches_numpy_reference() {
        let input = [0.0_f64, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0];
        let shifted = fftshift(&input);
        let expected = [-4.0_f64, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        assert_eq!(shifted.len(), 8);
        for (got, exp) in shifted.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = f64::EPSILON);
        }
    }

    /// ifftshift(fftshift(x)) = x for even n=8.
    #[test]
    fn ifftshift_undoes_fftshift_even_n() {
        let input = [0.0_f64, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0];
        let roundtrip = ifftshift(&fftshift(&input));
        for (got, exp) in roundtrip.iter().zip(input.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = f64::EPSILON);
        }
    }

    /// fftshift for odd n=7.
    /// numpy.fft.fftshift([0,1,2,3,-3,-2,-1]) = [-3,-2,-1,0,1,2,3].
    #[test]
    fn fftshift_n7_odd_matches_numpy_reference() {
        let input = [0.0_f64, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0];
        let shifted = fftshift(&input);
        let expected = [-3.0_f64, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        for (got, exp) in shifted.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = f64::EPSILON);
        }
    }

    /// ifftshift(fftshift(x)) = x for odd n=7.
    #[test]
    fn ifftshift_undoes_fftshift_odd_n() {
        let input = [0.0_f64, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0];
        let roundtrip = ifftshift(&fftshift(&input));
        for (got, exp) in roundtrip.iter().zip(input.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = f64::EPSILON);
        }
    }

    /// fftshift(ifftshift(x)) = x for even n=8.
    #[test]
    fn fftshift_undoes_ifftshift_even_n() {
        let input = [-4.0_f64, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let roundtrip = fftshift(&ifftshift(&input));
        for (got, exp) in roundtrip.iter().zip(input.iter()) {
            assert_abs_diff_eq!(got, exp, epsilon = f64::EPSILON);
        }
    }

    /// fftshift on n=1 returns identity.
    #[test]
    fn fftshift_n1_identity() {
        assert_eq!(fftshift(&[42.0_f64]), vec![42.0]);
    }

    /// fftshift on n=2: [a,b] -> [b,a].
    #[test]
    fn fftshift_n2() {
        let shifted = fftshift(&[1.0_f64, 2.0]);
        assert_eq!(shifted, [2.0, 1.0]);
    }

    /// ifftshift on n=2: [a,b] -> [b,a] (same as fftshift for even n).
    #[test]
    fn ifftshift_n2() {
        let shifted = ifftshift(&[1.0_f64, 2.0]);
        assert_eq!(shifted, [2.0, 1.0]);
    }

    /// Works for integer element types (genericity check).
    #[test]
    fn fftshift_integer_type() {
        let input = [0_i32, 1, 2, 3, -4, -3, -2, -1];
        let shifted = fftshift(&input);
        assert_eq!(shifted, [-4, -3, -2, -1, 0, 1, 2, 3]);
    }
}
