//! Bluestein's chirp-z transform: a DFT of *any* length.
//!
//! Every other strategy in this crate needs `n` to have a shape — a power of
//! two, a factorization over the supported radices, a coprime split, or
//! primality for Rader. Bluestein needs nothing: it re-expresses the transform
//! as a convolution, and the convolution runs at a padded power-of-two length
//! that is always available. That makes it the correct terminal route for the
//! lengths every shaped strategy declines.
//!
//! It is the terminal route, not a preferred one. It costs three
//! power-of-two transforms at `p >= 2n - 1` plus two chirp multiplies, so any
//! strategy that applies is faster and is selected first.
//!
//! # Derivation
//!
//! With `w = exp(s * 2*pi*i / n)` (`s = -1` forward, `s = +1` inverse), the
//! transform is `X[k] = sum_j x[j] * w^(j*k)`. Substituting the identity
//! `j*k = (j^2 + k^2 - (k - j)^2) / 2` splits the kernel into three factors
//! that each depend on one index:
//!
//! ```text
//!   X[k] = c(k) * sum_j (x[j] * c(j)) * conj(c(k - j)),
//!   where c(m) = exp(s * pi * i * m^2 / n)
//! ```
//!
//! The sum is a convolution of `a[j] = x[j] * c(j)` with `conj(c)`. Evaluating
//! it as a cyclic convolution at length `p = next_pow2(2n - 1)` leaves the
//! wrap-around region zero, so the cyclic result equals the linear one over
//! the `n` outputs that are read.
//!
//! # Why the chirp is reduced modulo `2n`
//!
//! `c(m)` depends on `m^2` only through `m^2 mod 2n`, because raising the
//! exponent by `2n` adds `2*pi` to the angle. Reducing first keeps the
//! argument to `sin`/`cos` inside a couple of turns. Passing `m^2` unreduced
//! would hand the trig functions arguments growing as `n^2`, where the
//! absolute error of argument reduction grows with the magnitude and the phase
//! loses significance precisely at the large `n` this route exists to serve.
//!
//! # Tables
//!
//! The chirp, the output chirp and the transform of the kernel depend on
//! the length and direction only. [`ChirpTables`] holds them; a plan owns
//! one per direction ([`BluesteinState`]) so a warm call runs two
//! transforms at `p` and allocates nothing (the padded work buffer is the
//! thread's Bluestein scratch role), while the free-function route
//! ([`bluestein_fft`]) builds them per call.

use eunomia::Complex;

use crate::application::execution::kernel::mixed_radix::dispatch::dispatch_inplace;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// The chirp-z tables for one length and direction.
pub(crate) struct ChirpTables<C> {
    /// `c(m)`, `m < n`.
    chirp: Box<[C]>,
    /// `c(m) / p`: the output chirp with the inverse transform's `1/p`.
    chirp_out: Box<[C]>,
    /// `c(m) / (p n)`: the normalized inverse's output chirp; empty forward.
    chirp_out_normalized: Box<[C]>,
    /// The unnormalized forward transform of `conj(c)` padded to `p`.
    kernel_spectrum: Box<[C]>,
}

impl<F: MixedRadixScalar<Complex = Complex<F>>> ChirpTables<Complex<F>> {
    /// The tables for `n > 1` in direction `INVERSE`.
    pub(crate) fn new<const INVERSE: bool>(n: usize) -> Self {
        debug_assert!(n > 1);
        let p = (2 * n - 1).next_power_of_two();
        // The convolution is evaluated at `p`, so `p` must be a length the
        // power-of-two path serves directly. If it were not,
        // `dispatch_inplace` could route back here and recurse.
        debug_assert!(p.is_power_of_two() && p >= 2 * n - 1);
        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        // `1/p` undoes the unnormalized inverse transform; `1/n` is the
        // caller's normalization. Folding both into the output chirp keeps
        // the whole pass multiplicative, with no separate scaling sweep.
        let scale = 1.0 / p as f64;
        let scale_normalized = 1.0 / (p as f64 * n as f64);

        let mut chirp = Vec::with_capacity(n);
        let mut chirp_out = Vec::with_capacity(n);
        let mut chirp_out_normalized = Vec::with_capacity(if INVERSE { n } else { 0 });
        let mut kernel = vec![F::complex(0.0, 0.0); p];
        let two_n = 2 * n;
        for m in 0..n {
            // `m * m` reaches ~n^2 and would overflow `usize` for large n on
            // a 32-bit target; the widened product costs one multiply per
            // element, once per table.
            let residue = ((m as u128 * m as u128) % two_n as u128) as usize;
            let angle = sign * std::f64::consts::PI * residue as f64 / n as f64;
            let (sin, cos) = angle.sin_cos();
            chirp.push(F::complex(cos, sin));
            chirp_out.push(F::complex(cos * scale, sin * scale));
            if INVERSE {
                chirp_out_normalized
                    .push(F::complex(cos * scale_normalized, sin * scale_normalized));
            }
            // The kernel is `conj(c)`, which is even in `m`, so one value
            // fills both ends. `p >= 2n - 1` keeps `p - m > m` for every
            // `m < n`, so these never collide.
            let conj = F::complex(cos, -sin);
            kernel[m] = conj;
            if m > 0 {
                kernel[p - m] = conj;
            }
        }
        dispatch_inplace::<F, false, false>(&mut kernel, None);
        Self {
            chirp: chirp.into_boxed_slice(),
            chirp_out: chirp_out.into_boxed_slice(),
            chirp_out_normalized: chirp_out_normalized.into_boxed_slice(),
            kernel_spectrum: kernel.into_boxed_slice(),
        }
    }
}

/// A plan's chirp-z tables: forward at construction, inverse on the first
/// inverse execution, as the plan's inverse twiddles are.
pub(crate) struct BluesteinState<C> {
    forward: ChirpTables<C>,
    inverse: std::sync::OnceLock<ChirpTables<C>>,
}

impl<F: MixedRadixScalar<Complex = Complex<F>>> BluesteinState<Complex<F>> {
    /// The state for `n > 1`, forward tables built.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            forward: ChirpTables::new::<false>(n),
            inverse: std::sync::OnceLock::new(),
        }
    }

    /// The tables for direction `INVERSE`, the inverse built on first use.
    pub(crate) fn tables<const INVERSE: bool>(&self, n: usize) -> &ChirpTables<Complex<F>> {
        if INVERSE {
            self.inverse.get_or_init(|| ChirpTables::new::<true>(n))
        } else {
            &self.forward
        }
    }
}

/// Transform `data` in place at any length, via chirp-z, building the
/// tables for this call.
///
/// `NORMALIZE` divides by `n`, and like every other kernel in this module it
/// applies only on the inverse.
pub(crate) fn bluestein_fft<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
) {
    if data.len() <= 1 {
        return;
    }
    let tables = ChirpTables::<Complex<F>>::new::<INVERSE>(data.len());
    bluestein_with::<F, INVERSE, NORMALIZE>(data, &tables);
}

/// Transform `data` in place through prepared `tables` of its length and
/// direction: two power-of-two transforms at `p` over the thread's
/// Bluestein scratch, no allocation once that role is warm.
pub(crate) fn bluestein_with<
    F: MixedRadixScalar<Complex = Complex<F>>,
    const INVERSE: bool,
    const NORMALIZE: bool,
>(
    data: &mut [F::Complex],
    tables: &ChirpTables<F::Complex>,
) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let p = tables.kernel_spectrum.len();
    assert!(
        tables.chirp.len() == n && p >= 2 * n - 1,
        "invariant: the chirp-z tables are built for this length"
    );
    let chirp_out: &[F::Complex] = if INVERSE && NORMALIZE {
        &tables.chirp_out_normalized
    } else {
        &tables.chirp_out
    };
    F::with_bluestein_scratch(p, |work| {
        let work = &mut work[..p];
        work[..n].copy_from_slice(data);
        work[n..].fill(F::complex(0.0, 0.0));
        F::pointwise_mul(&mut work[..n], &tables.chirp);
        dispatch_inplace::<F, false, false>(work, None);
        F::pointwise_mul(work, &tables.kernel_spectrum);
        dispatch_inplace::<F, true, false>(work, None);
        F::pointwise_mul(&mut work[..n], chirp_out);
        data.copy_from_slice(&work[..n]);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::Complex64;

    /// The defining sum, shared with no part of the implementation.
    fn naive(x: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let n = x.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                let mut acc = Complex64::new(0.0, 0.0);
                for (j, xj) in x.iter().enumerate() {
                    let angle = sign * std::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                    let (s, c) = angle.sin_cos();
                    acc = Complex64::new(
                        acc.re + xj.re * c - xj.im * s,
                        acc.im + xj.re * s + xj.im * c,
                    );
                }
                acc
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|j| {
                let t = j as f64 / n as f64;
                Complex64::new((7.0 * t).sin() - 0.25 * t, (23.0 * t).cos())
            })
            .collect()
    }

    /// Lengths with no shaped route: `19^2`, `29^2`, `31^2`, a prime, and a
    /// smooth length that other strategies would claim — the last confirms
    /// Bluestein is correct on its own, not merely correct where it is used.
    #[test]
    fn matches_the_defining_sum() {
        for n in [361usize, 841, 961, 1153, 2, 3, 60, 512] {
            let x = signal(n);
            let mut got = x.clone();
            bluestein_fft::<f64, false, false>(&mut got);
            let expect = naive(&x, false);
            // Bluestein's error grows with the padded length, not `n`; the
            // three transforms at `p < 4n` bound it near `p * eps`, which at
            // these sizes is well under 1e-9.
            let worst = got
                .iter()
                .zip(&expect)
                .map(|(g, e)| (g.re - e.re).abs().max((g.im - e.im).abs()))
                .fold(0.0f64, f64::max);
            assert!(worst < 1e-9, "n = {n}: forward disagrees by {worst:e}");
        }
    }

    #[test]
    fn normalized_inverse_undoes_the_forward() {
        for n in [361usize, 841, 961, 1153] {
            let x = signal(n);
            let mut round = x.clone();
            bluestein_fft::<f64, false, false>(&mut round);
            bluestein_fft::<f64, true, true>(&mut round);
            let worst = round
                .iter()
                .zip(&x)
                .map(|(g, e)| (g.re - e.re).abs().max((g.im - e.im).abs()))
                .fold(0.0f64, f64::max);
            assert!(worst < 1e-9, "n = {n}: roundtrip drifts by {worst:e}");
        }
    }

    /// A plan runs the same tables through the same transforms as the free
    /// route, so the two agree bit for bit in every direction.
    #[test]
    fn plan_matches_the_free_route_bit_for_bit() {
        for n in [361usize, 961] {
            let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D::new(n).expect("non-zero"));
            let x = signal(n);
            let mut planned = x.clone();
            let mut free = x.clone();
            plan.forward_complex_slice_inplace(&mut planned);
            bluestein_fft::<f64, false, false>(&mut free);
            assert_eq!(planned, free, "n = {n}: forward");
            plan.inverse_complex_slice_inplace(&mut planned);
            bluestein_fft::<f64, true, true>(&mut free);
            assert_eq!(planned, free, "n = {n}: normalized inverse");
        }
    }
}
