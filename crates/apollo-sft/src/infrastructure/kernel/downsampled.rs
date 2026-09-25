//! Sparse recovery by downsampling and syndrome decoding (ADR 0064).
//!
//! # Mathematical contract
//!
//! For `x ∈ ℂ^N` with spectrum `X`, the length-`B` DFT of the shifted
//! downsampled signal `x[dn + ℓ]`, `d = N / B`, aliases the spectrum onto `B`
//! buckets:
//!
//! ```text
//! d · x̂_{d,ℓ}[b] = Σ_{s ≡ b (mod B)} X[s] · z_s^ℓ,   z_s = exp(2πi s / N).
//! ```
//!
//! A bucket holding `a` tones therefore yields syndromes `m_ℓ = Σ_j p_j z_j^ℓ`
//! whose `z_j` are the roots of the Prony polynomial
//! `P(z) = z^a + c_{a-1} z^{a-1} + … + c_0` with `m_{a+i} + Σ_j c_j m_{i+j} = 0`
//! for `i < a`; the frequencies are `s_j = N · arg(z_j) / 2π` and the
//! amplitudes solve the Vandermonde system `m_ℓ = Σ_j p_j z_j^ℓ`
//! (Hsieh, Lu, Pei, arXiv:1407.8315, Section III).
//!
//! Each round takes `2 · MAX_COLLISIONS + 1` shifts, decodes every non-empty
//! bucket at the smallest tone count whose decode reproduces all syndromes,
//! subtracts the resolved tones from the next round's aliased spectra, and
//! doubles the bucket count for the buckets left unresolved, until none
//! remain or `B` reaches `N`. Every step is exact in exact arithmetic, so an
//! exactly `K`-sparse input is recovered whatever its support; the cost is
//! `O(K log K)` while no bucket at `B = 4K` holds more than `MAX_COLLISIONS`
//! tones (Theorem 1 of the paper bounds that event for uniform supports).

use eunomia::Complex64;
use leto::Array1;
use std::f64::consts::TAU;

/// The most tones one bucket is decoded for, the paper's default: the Prony
/// system stays four by four and the polynomial's roots are found to full
/// precision by the simultaneous iteration below.
pub(crate) const MAX_COLLISIONS: usize = 4;

/// Shifts per round: the syndromes a `MAX_COLLISIONS`-tone decode consumes,
/// plus one it did not fit, so even the widest decode is verified.
const SHIFTS: usize = 2 * MAX_COLLISIONS + 1;

/// `2c` of the `O(log N · u)` forward-error bound of the aliased transforms,
/// the constant the apollo-fft oracles use.
const TOLERANCE_FACTOR: f64 = 16.0;

/// Iterations of the simultaneous root iteration; a degree-four polynomial
/// with separated unit-circle roots converges in well under half of them.
const ROOT_ITERATIONS: usize = 96;

/// The rounds ran out before every bucket resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Unresolved {
    /// Buckets still colliding at the last bucket count tried.
    pub buckets: usize,
    /// The last bucket count tried.
    pub bucket_count: usize,
}

/// Recovers the frequency support of `signal` by aliasing it onto `buckets`
/// buckets and decoding each, doubling the count on a failed round up to
/// `rounds` times.
///
/// Returns the resolved `(frequency, coefficient)` pairs in no particular
/// order; the caller ranks them. `buckets` must divide `signal.len()` and
/// every doubling up to the length must too, which the plan configuration
/// guarantees.
///
/// # Errors
///
/// [`Unresolved`] when buckets still collide after `rounds` rounds.
pub(crate) fn recover(
    signal: &[Complex64],
    buckets: usize,
    rounds: usize,
) -> Result<Vec<(usize, Complex64)>, Unresolved> {
    let n = signal.len();
    debug_assert!(buckets > 0 && n.is_multiple_of(buckets));
    let mut resolved: Vec<(usize, Complex64)> = Vec::new();
    let mut bucket_count = buckets;
    let mut samples = Array1::from(vec![Complex64::default(); bucket_count]);
    let mut aliased: Vec<Array1<Complex64>> = (0..SHIFTS)
        .map(|_| Array1::from(vec![Complex64::default(); bucket_count]))
        .collect();
    let mut unresolved = 0;
    for _ in 0..rounds {
        let stride = n / bucket_count;
        if samples.size() != bucket_count {
            samples = Array1::from(vec![Complex64::default(); bucket_count]);
            for spectrum in &mut aliased {
                *spectrum = Array1::from(vec![Complex64::default(); bucket_count]);
            }
        }
        let mut tolerance = 0.0_f64;
        for (shift, spectrum) in aliased.iter_mut().enumerate() {
            // The shifted signal is `x[(d j + ℓ) mod N]`: the aliasing
            // identity reads `x` periodically, and shifts past the stride
            // wrap around the end.
            let mut magnitude_sum = 0.0;
            for (j, slot) in samples.iter_mut().enumerate() {
                let value = signal[(stride * j + shift) % n];
                *slot = value;
                magnitude_sum += value.norm();
            }
            apollo_fft::fft_1d_complex_into(&samples, spectrum);
            let scale = stride as f64;
            for bin in spectrum.iter_mut() {
                *bin *= scale;
            }
            tolerance = tolerance.max(bound(bucket_count, scale * magnitude_sum));
        }
        subtract_resolved(&mut aliased, &resolved, n, bucket_count);

        unresolved = 0;
        let mut syndromes = [Complex64::default(); SHIFTS];
        for bucket in 0..bucket_count {
            for (slot, spectrum) in syndromes.iter_mut().zip(&aliased) {
                *slot = spectrum[[bucket]];
            }
            if syndromes.iter().all(|m| m.norm() <= tolerance) {
                continue;
            }
            match decode(&syndromes, n, bucket_count, bucket, tolerance) {
                Some(tones) => resolved.extend(tones),
                None => unresolved += 1,
            }
        }
        if unresolved == 0 {
            return Ok(resolved);
        }
        if bucket_count >= n {
            break;
        }
        bucket_count *= 2;
    }
    Err(Unresolved {
        buckets: unresolved,
        bucket_count,
    })
}

/// The per-syndrome error bound of a length-`bucket_count` transform of
/// samples whose scaled magnitudes sum to `scaled_l1`, in `f64` arithmetic.
fn bound(bucket_count: usize, scaled_l1: f64) -> f64 {
    let stages = (bucket_count as f64).log2().max(1.0);
    TOLERANCE_FACTOR * stages * (f64::EPSILON / 2.0) * scaled_l1
}

/// Removes the tones already resolved from the aliased spectra of the
/// current round: tone `s` with coefficient `p` contributes `p · z_s^ℓ` to
/// bucket `s mod B` of shift `ℓ`.
fn subtract_resolved(
    aliased: &mut [Array1<Complex64>],
    resolved: &[(usize, Complex64)],
    n: usize,
    bucket_count: usize,
) {
    for &(frequency, coefficient) in resolved {
        let bucket = frequency % bucket_count;
        let root = unit_root(frequency, n);
        let mut power = Complex64::new(1.0, 0.0);
        for spectrum in aliased.iter_mut() {
            spectrum[[bucket]] -= coefficient * power;
            power *= root;
        }
    }
}

/// `exp(2πi s / n)`.
fn unit_root(frequency: usize, n: usize) -> Complex64 {
    let angle = TAU * frequency as f64 / n as f64;
    Complex64::new(angle.cos(), angle.sin())
}

/// `z^exponent` by repeated squaring.
fn power(z: Complex64, exponent: usize) -> Complex64 {
    let mut base = z;
    let mut result = Complex64::new(1.0, 0.0);
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result *= base;
        }
        base *= base;
        remaining >>= 1;
    }
    result
}

/// Decodes one bucket's syndromes at the smallest tone count whose decode
/// reproduces every syndrome within `tolerance`, or `None` when no count up
/// to [`MAX_COLLISIONS`] does.
fn decode(
    syndromes: &[Complex64; SHIFTS],
    n: usize,
    bucket_count: usize,
    bucket: usize,
    tolerance: f64,
) -> Option<Vec<(usize, Complex64)>> {
    (1..=MAX_COLLISIONS)
        .find_map(|tones| decode_as(syndromes, tones, n, bucket_count, bucket, tolerance))
}

/// Decodes the syndromes as exactly `tones` tones: the Prony polynomial's
/// roots snapped to the bucket's residue class, the amplitudes solved from
/// the snapped roots, and every syndrome reproduced.
fn decode_as(
    syndromes: &[Complex64; SHIFTS],
    tones: usize,
    n: usize,
    bucket_count: usize,
    bucket: usize,
    tolerance: f64,
) -> Option<Vec<(usize, Complex64)>> {
    let coefficients = prony_coefficients(syndromes, tones)?;
    let roots = polynomial_roots(&coefficients);
    let band = tolerance / syndromes[0].norm().max(tolerance);
    let mut frequencies = Vec::with_capacity(tones);
    for root in roots {
        if (root.norm() - 1.0).abs() > band.max(1e-6) {
            return None;
        }
        let turns = root.arg() / TAU;
        let frequency = (turns * n as f64).round().rem_euclid(n as f64) as usize;
        if frequency % bucket_count != bucket || frequencies.contains(&frequency) {
            return None;
        }
        frequencies.push(frequency);
    }
    let exact_roots: Vec<Complex64> = frequencies.iter().map(|&f| unit_root(f, n)).collect();
    let amplitudes = vandermonde_solve(&exact_roots, &syndromes[..tones])?;
    for (shift, &measured) in syndromes.iter().enumerate() {
        let predicted: Complex64 = exact_roots
            .iter()
            .zip(&amplitudes)
            .map(|(root, amplitude)| amplitude * power(*root, shift))
            .sum();
        if (predicted - measured).norm() > tolerance * tones as f64 {
            return None;
        }
    }
    Some(frequencies.into_iter().zip(amplitudes).collect())
}

/// The coefficients `c_0..c_{a-1}` of the monic Prony polynomial from the
/// Hankel system `m_{a+i} + Σ_j c_j m_{i+j} = 0`, or `None` when the system
/// is singular (fewer than `a` tones present).
fn prony_coefficients(syndromes: &[Complex64; SHIFTS], tones: usize) -> Option<Vec<Complex64>> {
    // One flat row-major buffer; a row per tone was a heap row per tone.
    let mut matrix = vec![Complex64::default(); tones * tones];
    for (i, row) in matrix.chunks_exact_mut(tones).enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = syndromes[i + j];
        }
    }
    let mut rhs: Vec<Complex64> = (0..tones).map(|i| -syndromes[tones + i]).collect();
    solve_in_place(&mut matrix, tones, &mut rhs)
}

/// Solves the Vandermonde system `Σ_j p_j z_j^ℓ = m_ℓ`, `ℓ < a`, for the
/// amplitudes `p`.
fn vandermonde_solve(roots: &[Complex64], syndromes: &[Complex64]) -> Option<Vec<Complex64>> {
    let size = roots.len();
    let mut matrix = vec![Complex64::default(); size * size];
    for (row, slot_row) in matrix.chunks_exact_mut(size).enumerate() {
        for (slot, &z) in slot_row.iter_mut().zip(roots) {
            *slot = power(z, row);
        }
    }
    let mut rhs = syndromes.to_vec();
    solve_in_place(&mut matrix, size, &mut rhs)
}

/// Gaussian elimination with partial pivoting over a small square complex
/// system held as one flat row-major buffer — row `r` is
/// `matrix[r * size..(r + 1) * size]`. `None` when a pivot vanishes
/// relative to the largest entry.
fn solve_in_place(
    matrix: &mut [Complex64],
    size: usize,
    rhs: &mut [Complex64],
) -> Option<Vec<Complex64>> {
    let scale = matrix.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    for column in 0..size {
        let pivot_row = (column..size).max_by(|&a, &b| {
            matrix[a * size + column]
                .norm()
                .total_cmp(&matrix[b * size + column].norm())
        })?;
        if matrix[pivot_row * size + column].norm() <= scale * 1e-12 {
            return None;
        }
        if column != pivot_row {
            let (low, high) = if column < pivot_row {
                (column, pivot_row)
            } else {
                (pivot_row, column)
            };
            let (head, tail) = matrix.split_at_mut(high * size);
            head[low * size..(low + 1) * size].swap_with_slice(&mut tail[..size]);
            rhs.swap(column, pivot_row);
        }
        let pivot = matrix[column * size + column];
        for row in column + 1..size {
            let factor = matrix[row * size + column] / pivot;
            if factor.norm() == 0.0 {
                continue;
            }
            let (head, tail) = matrix.split_at_mut(row * size);
            let target = &mut tail[column..size];
            let source = &head[column * size + column..(column + 1) * size];
            for (slot, &above) in target.iter_mut().zip(source) {
                *slot -= factor * above;
            }
            let above = rhs[column];
            rhs[row] -= factor * above;
        }
    }
    let mut solution = vec![Complex64::default(); size];
    for row in (0..size).rev() {
        let matrix_row = &matrix[row * size..(row + 1) * size];
        let mut sum = rhs[row];
        for (k, &coefficient) in matrix_row.iter().enumerate().skip(row + 1) {
            sum -= coefficient * solution[k];
        }
        solution[row] = sum / matrix_row[row];
    }
    Some(solution)
}

/// The roots of the monic polynomial `z^a + c_{a-1} z^{a-1} + … + c_0` by
/// the Weierstrass simultaneous iteration, each polished by Newton steps.
fn polynomial_roots(coefficients: &[Complex64]) -> Vec<Complex64> {
    let degree = coefficients.len();
    if degree == 1 {
        return vec![-coefficients[0]];
    }
    let evaluate = |z: Complex64| -> Complex64 {
        coefficients
            .iter()
            .rev()
            .fold(Complex64::new(1.0, 0.0), |acc, &c| acc * z + c)
    };
    let derivative = |z: Complex64| -> Complex64 {
        let mut acc = Complex64::new(degree as f64, 0.0);
        for (power, &c) in coefficients.iter().enumerate().skip(1).rev() {
            acc = acc * z + c * power as f64;
        }
        acc
    };
    // Distinct starting points off every axis of symmetry.
    let seed = Complex64::new(0.4, 0.9);
    let mut roots: Vec<Complex64> = (0..degree).map(|k| power(seed, k + 1)).collect();
    for _ in 0..ROOT_ITERATIONS {
        let mut moved = 0.0_f64;
        for i in 0..degree {
            let mut denominator = Complex64::new(1.0, 0.0);
            for j in 0..degree {
                if j != i {
                    denominator *= roots[i] - roots[j];
                }
            }
            if denominator.norm() == 0.0 {
                continue;
            }
            let step = evaluate(roots[i]) / denominator;
            roots[i] -= step;
            moved = moved.max(step.norm());
        }
        if moved <= 1e-15 {
            break;
        }
    }
    for root in &mut roots {
        for _ in 0..3 {
            let slope = derivative(*root);
            if slope.norm() == 0.0 {
                break;
            }
            *root -= evaluate(*root) / slope;
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::{polynomial_roots, power, prony_coefficients, recover, unit_root, SHIFTS};
    use eunomia::Complex64;

    /// The Prony coefficients of synthetic syndromes recover their roots.
    #[test]
    fn prony_roots_are_the_tones() {
        let n = 64;
        let tones = [
            (3usize, Complex64::new(1.0, 0.5)),
            (19, Complex64::new(-2.0, 0.25)),
            (35, Complex64::new(0.0, 3.0)),
        ];
        let mut syndromes = [Complex64::default(); SHIFTS];
        for (shift, slot) in syndromes.iter_mut().enumerate() {
            *slot = tones
                .iter()
                .map(|&(f, p)| p * power(unit_root(f, n), shift))
                .sum();
        }
        let coefficients =
            prony_coefficients(&syndromes, 3).expect("three tones give a regular system");
        let mut roots: Vec<usize> = polynomial_roots(&coefficients)
            .iter()
            .map(|z| {
                (z.arg() / std::f64::consts::TAU * n as f64)
                    .round()
                    .rem_euclid(n as f64) as usize
            })
            .collect();
        roots.sort_unstable();
        assert_eq!(roots, vec![3, 19, 35]);
    }

    /// A forced collision beyond the decoder's width resolves once the bucket
    /// count has doubled past the tones' spacing.
    #[test]
    fn colliding_tones_resolve_on_a_later_round() {
        let n = 256;
        let buckets = 4;
        // Six tones in one residue class mod 4, more than MAX_COLLISIONS.
        let support = [1usize, 5, 9, 13, 17, 21];
        let signal: Vec<Complex64> = (0..n)
            .map(|t| {
                support
                    .iter()
                    .map(|&f| unit_root(f * t % n, n) * Complex64::new(1.0 + f as f64, 0.0))
                    .sum::<Complex64>()
            })
            .collect();
        let mut recovered =
            recover(&signal, buckets, 12).expect("doubling reaches a resolving count");
        recovered.sort_by_key(|&(f, _)| f);
        let frequencies: Vec<usize> = recovered.iter().map(|&(f, _)| f).collect();
        assert_eq!(frequencies, support);
        for &(f, p) in &recovered {
            // The spectrum of Σ a_f e^{2πi f t / n} is n · a_f at bin f.
            let want = n as f64 * (1.0 + f as f64);
            assert!(
                (p.re - want).abs() <= 1e-9 * want && p.im.abs() <= 1e-9 * want,
                "bin {f}: {p}"
            );
        }
    }
}
