use super::trait_def::MixedRadixScalar;

pub(super) fn build_rader_spectrum_vec<F: MixedRadixScalar>(
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
