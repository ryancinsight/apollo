use crate::application::execution::kernel::components::winograd::traits::WinogradScalar;

/// In-place DFT-3.
///
/// ## Mathematical derivation
///
/// For N=3, W3 = exp(-2πi/3), the DFT matrix rows give:
/// ```text
/// Y[0] = X[0] + X[1] + X[2]
/// Y[1] = X[0] + W3^1*X[1] + W3^2*X[2]   (fwd)
/// Y[2] = X[0] + W3^2*X[1] + W3^1*X[2]   (fwd)
/// ```
/// With W3^1 = -1/2 - i*(sqrt(3)/2) and W3^2 = -1/2 + i*(sqrt(3)/2):
/// ```text
/// Y[1] = (X[0] - (X[1]+X[2])/2) - i*(sqrt(3)/2)*(X[1]-X[2])
/// Y[2] = (X[0] - (X[1]+X[2])/2) + i*(sqrt(3)/2)*(X[1]-X[2])
/// ```
/// Conjugate the imaginary twiddle component for inverse.
///
/// Real multiplications: 4. Complex additions: 6.
#[inline(always)]
pub(crate) fn dft3_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 3);
    let s = F::from_precise(0.8660254037844386);
    let w_r = F::from_precise(-0.5);
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let sum_re = x1.re + x2.re;
    let sum_im = x1.im + x2.im;
    let diff_re = x1.re - x2.re;
    let diff_im = x1.im - x2.im;
    let m0_re = x0.re + sum_re * w_r;
    let m0_im = x0.im + sum_im * w_r;
    let (m1_re, m1_im) = if INVERSE {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    data[0] = num_complex::Complex::new(x0.re + sum_re, x0.im + sum_im);
    data[1] = num_complex::Complex::new(m0_re + m1_re, m0_im + m1_im);
    data[2] = num_complex::Complex::new(m0_re - m1_re, m0_im - m1_im);
}

#[inline(always)]
pub(crate) fn dft3_values<F: WinogradScalar, const INVERSE: bool>(
    x0: num_complex::Complex<F>,
    x1: num_complex::Complex<F>,
    x2: num_complex::Complex<F>,
) -> (
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
) {
    let s = F::from_precise(0.8660254037844386);
    let w_r = F::from_precise(-0.5);
    let sum_re = x1.re + x2.re;
    let sum_im = x1.im + x2.im;
    let diff_re = x1.re - x2.re;
    let diff_im = x1.im - x2.im;
    let m0_re = x0.re + sum_re * w_r;
    let m0_im = x0.im + sum_im * w_r;
    let (m1_re, m1_im) = if INVERSE {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    (
        num_complex::Complex::new(x0.re + sum_re, x0.im + sum_im),
        num_complex::Complex::new(m0_re + m1_re, m0_im + m1_im),
        num_complex::Complex::new(m0_re - m1_re, m0_im - m1_im),
    )
}
