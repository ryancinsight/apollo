use super::trait_def::MixedRadixScalar;

pub(super) fn build_rader_spectrum_vec<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    n: usize,
    inverse: bool,
    generator_inverse: usize,
) -> Vec<F::Complex> {
    let l = n - 1;
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut kernel = vec![F::complex(0.0, 0.0); l];
    let mut curr_inv = 1usize;
    for value in kernel.iter_mut().take(l) {
        let angle = sign * std::f64::consts::TAU * (curr_inv as f64) / (n as f64);
        *value = F::complex(angle.cos(), angle.sin());
        curr_inv = (curr_inv * generator_inverse) % n;
    }
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut kernel);
    kernel
}

/// Build cyclic and negacyclic kernel spectra for the Nussbaumer decomposition
/// of the Rader convolution of length M = N-1.
///
/// Returns `(cyclic_spectrum, negacyclic_spectrum)` each of length m = M/2.
///
/// ## Algorithm
///
/// Given the length-M Rader kernel in the time domain, split into lower/upper halves:
/// - `kernel_cyc[j] = kernel[j] + kernel[j+m]` (modulo x^m - 1)
/// - `kernel_neg[j] = kernel[j] - kernel[j+m]` (modulo x^m + 1)
///
/// Weight the negacyclic kernel by the twist twiddle `e^{i*pi*j/m}`, then FFT both.
///
/// The implementation streams the two halves directly into their CRT residues.
/// It does not materialize the full length-M kernel, reducing peak build memory
/// from `4m` complex values to `2m` complex values.
pub(super) fn build_rader_negacyclic_spectra<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>>,
>(
    n: usize,
    inverse: bool,
    generator_inverse: usize,
) -> (Vec<F::Complex>, Vec<F::Complex>) {
    let m = (n - 1) / 2;
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };

    let mut kernel_cyc = Vec::with_capacity(m);
    let mut curr_inv = 1usize;
    for _ in 0..m {
        let angle = sign * std::f64::consts::TAU * (curr_inv as f64) / (n as f64);
        kernel_cyc.push(F::complex(angle.cos(), angle.sin()));
        curr_inv = (curr_inv * generator_inverse) % n;
    }

    let mut kernel_neg = Vec::with_capacity(m);
    let pi_over_m = std::f64::consts::PI / (m as f64);
    for j in 0..m {
        let angle = sign * std::f64::consts::TAU * (curr_inv as f64) / (n as f64);
        let upper = F::complex(angle.cos(), angle.sin());
        let lower = kernel_cyc[j];
        kernel_cyc[j] = lower + upper;
        let twist_angle = pi_over_m * (j as f64);
        let twist = F::complex(twist_angle.cos(), twist_angle.sin());
        kernel_neg.push((lower - upper) * twist);
        curr_inv = (curr_inv * generator_inverse) % n;
    }

    // FFT both kernels to get the spectra.
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut kernel_cyc);
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(&mut kernel_neg);

    (kernel_cyc, kernel_neg)
}

/// Build twist twiddles `e^{i*pi*j/m}` for the negacyclic convolution path.
///
/// Returns a vector of length m containing the twist factors used for weighting
/// before the forward FFT and unweighting after the inverse FFT.
pub(super) fn build_rader_negacyclic_twiddles<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>>,
>(
    m: usize,
) -> Vec<F::Complex> {
    let pi_over_m = std::f64::consts::PI / (m as f64);
    let mut twiddles = vec![F::complex(0.0, 0.0); m];
    for j in 0..m {
        let angle = pi_over_m * (j as f64);
        twiddles[j] = F::complex(angle.cos(), angle.sin());
    }
    twiddles
}
