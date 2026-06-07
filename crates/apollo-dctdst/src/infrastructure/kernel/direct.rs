use moirai::ParallelSliceMut;

/// Parallel dispatch threshold. Below this length, serial iteration avoids rayon spawn overhead.
const PAR_THRESHOLD: usize = 256;

/// Direct O(N²) analytical kernel for the Type-II Discrete Cosine Transform (DCT-II).
///
/// # Theorem: DCT-II Direct Projection
///
/// For a real sequence x ∈ ℝᴺ, the unnormalized DCT-II is defined by the
/// closed-form projection:
///
/// \\[ X_k = \sum_{n=0}^{N-1} x_n \cos\left[ \frac{\pi}{N} \left(n + \frac{1}{2}\right) k \right], \quad k = 0, \ldots, N-1 \\]
///
/// The basis functions `cos(πk(n + ½)/N)` are orthogonal over the half-sample-shifted
/// grid `{n + ½}_{n=0}^{N-1}`. This property follows from the finite trigonometric sum
/// identity:
///
/// \\[ \sum_{n=0}^{N-1} \cos\left[\frac{\pi k (2n+1)}{2N}\right] \cos\left[\frac{\pi l (2n+1)}{2N}\right] = \frac{N}{2} \delta_{kl} \quad (k, l \geq 1) \\]
///
/// confirming that DCT-III is the inverse of DCT-II up to the factor N/2.
///
/// # Complexity
///
/// Time: O(N²) — each of N output coefficients requires a length-N inner product.
/// Space: O(1) auxiliary — the caller owns both the input and output buffers.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 2.1, pp. 24–26.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dct2(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * k as f64).cos();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * k as f64).cos();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-III Discrete Cosine Transform (DCT-III).
///
/// # Theorem: DCT-III Direct Projection (Inverse of DCT-II)
///
/// For a real coefficient sequence X ∈ ℝᴺ, the unnormalized DCT-III is defined by:
///
/// \\[ x_k = \frac{1}{2} X_0 + \sum_{n=1}^{N-1} X_n \cos\left[ \frac{\pi}{N} n \left(k + \frac{1}{2}\right) \right], \quad k = 0, \ldots, N-1 \\]
///
/// The leading `X_0 / 2` term arises from the n = 0 basis function evaluating to
/// `cos(0) = 1` while sharing the same orthogonality weight as the n ≥ 1 terms under
/// the standard half-sample-shifted inner product.
///
/// # Inverse relationship with DCT-II
///
/// The DCT-III is the exact inverse of DCT-II up to a scaling factor:
///
/// \\[ \text{DCT-III}(\text{DCT-II}(x)) = \frac{N}{2} \cdot x \\]
///
/// Proof: The cosine basis functions satisfy the finite orthogonality relation
/// `Σ_{n=0}^{N-1} cos(πk(2n+1)/(2N)) cos(πl(2n+1)/(2N)) = (N/2) δ_{kl}` for k, l ≥ 1.
/// The k = 0 row has norm N, and the factor of ½ in the definition compensates to yield
/// a uniform N/2 diagonal.
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 2.3, Inverse DCT, pp. 30–33.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dct3(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;
    let x0 = signal[0] * 0.5;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let mut sum = x0;
            for n_idx in 1..n {
                sum += signal[n_idx] * (factor * n_idx as f64 * (k as f64 + 0.5)).cos();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = x0;
            for n_idx in 1..n {
                sum += signal[n_idx] * (factor * n_idx as f64 * (k as f64 + 0.5)).cos();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-II Discrete Sine Transform (DST-II).
///
/// # Theorem: DST-II Direct Projection
///
/// For a real sequence x ∈ ℝᴺ, the unnormalized DST-II is defined by:
///
/// \\[ X_k = \sum_{n=0}^{N-1} x_n \sin\left[ \frac{\pi}{N} \left(n + \frac{1}{2}\right) (k + 1) \right], \quad k = 0, \ldots, N-1 \\]
///
/// The basis functions `sin(π(k+1)(n+½)/N)` encode antisymmetric (odd) boundary
/// conditions on the half-sample-shifted grid `{n + ½}_{n=0}^{N-1}`. Their orthogonality
/// follows from the finite sine sum identity:
///
/// \\[ \sum_{n=0}^{N-1} \sin\left[\frac{\pi (k+1)(2n+1)}{2N}\right] \sin\left[\frac{\pi (l+1)(2n+1)}{2N}\right] = \frac{N}{2} \delta_{kl} \\]
///
/// confirming that DST-III is the inverse of DST-II up to the factor N/2.
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 3.1, DST-II, pp. 162–165.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dst2(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 1.0)).sin();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 1.0)).sin();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-III Discrete Sine Transform (DST-III).
///
/// # Theorem: DST-III Direct Projection (Inverse of DST-II)
///
/// For a real coefficient sequence X ∈ ℝᴺ, the unnormalized DST-III is defined by:
///
/// \\[ x_k = \frac{(-1)^k}{2} X_{N-1} + \sum_{n=0}^{N-2} X_n \sin\left[ \frac{\pi}{N} (n + 1) \left(k + \frac{1}{2}\right) \right], \quad k = 0, \ldots, N-1 \\]
///
/// The boundary term `(-1)^k X_{N-1} / 2` corresponds to the n = N-1 basis function
/// `sin(πN(2k+1)/(2N)) = sin(π(2k+1)/2) = (-1)^k`, which oscillates with period 2
/// at the Nyquist frequency. The factor ½ is required for the same orthogonality
/// normalization reason as in DCT-III.
///
/// # Inverse relationship with DST-II
///
/// The DST-III is the exact inverse of DST-II up to a scaling factor:
///
/// \\[ \text{DST-III}(\text{DST-II}(x)) = \frac{N}{2} \cdot x \\]
///
/// Proof: The sine basis functions satisfy the finite orthogonality relation
/// `Σ_{n=0}^{N-1} sin(π(k+1)(2n+1)/(2N)) sin(π(l+1)(2n+1)/(2N)) = (N/2) δ_{kl}`.
/// The n = N-1 boundary term contributes (-1)^k per output, and the factor ½ in the
/// definition yields a uniform N/2 diagonal under the full N×N product.
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// For N ≥ 16 the O(N log N) fast kernel in `infrastructure::kernel::fast` is preferred.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 3.3, DST-III, Inverse DST, pp. 166–169.)
/// - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
///   Acoust. Speech Signal Process.*, 28(1), 27–34.
pub fn dst3(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;
    let xn = signal[n - 1] * 0.5;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            let mut sum = sign * xn;
            for n_idx in 0..(n - 1) {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 1.0) * (k as f64 + 0.5)).sin();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            let mut sum = sign * xn;
            for n_idx in 0..(n - 1) {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 1.0) * (k as f64 + 0.5)).sin();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-I Discrete Cosine Transform (DCT-I).
///
/// # Theorem: DCT-I Direct Projection
///
/// For a real sequence x ∈ ℝᴺ with N ≥ 2, the unnormalized DCT-I is defined by:
///
/// \\[ X_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\!\left(\frac{\pi n k}{N-1}\right), \quad k = 0, \ldots, N-1 \\]
///
/// The boundary samples x₀ and x_{N-1} receive half the cosine weight relative to the
/// interior samples. This corresponds to a symmetric extension of x about both endpoints,
/// placing implicit even-symmetric mirror copies at n = −1 and n = N with weight 1
/// (full-sample boundary, FFTW REDFT00).
///
/// # Self-Inverse Property
///
/// The DCT-I matrix C₁ is symmetric (C₁ᵀ = C₁). The finite orthogonality relation:
///
/// \\[ \sum_{n=0}^{N-1} w_n \cos\!\left(\frac{\pi n k}{N-1}\right) \cos\!\left(\frac{\pi n j}{N-1}\right) = \frac{N-1}{2} \delta_{kj} \quad (k, j \neq 0, N-1) \\]
///
/// where w₀ = w_{N-1} = 1/2, wₙ = 1 otherwise, yields C₁ · C₁ = 2(N−1) · I.
/// Therefore DCT-I is its own inverse up to a factor of 2(N−1):
///
/// \\[ \text{DCT-I}(\text{DCT-I}(x)) = 2(N-1) \cdot x \\]
///
/// Proof by hand for N = 3, x = [1, 2, 3]:
/// - Forward: X = [8, −2, 0]  (boundary formula, interior cos(πnk/2))
/// - Inverse: Y = DCT-I([8,−2,0]) = [4, 8, 12] = 4·x = 2(N−1)·x ✓
///
/// # Validity Constraint
///
/// Requires N ≥ 2. When N < 2 this function zeroes the output and returns immediately
/// because the formula references x_{N-1} and the divisor N−1 = 0 is undefined.
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 2.4, DCT-I, pp. 26–28.)
/// - Martucci, S. A. (1994). Symmetric convolution and the discrete sine and cosine
///   transforms. *IEEE Trans. Signal Process.*, 42(5), 1038–1051.
pub fn dct1(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n < 2 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let x0 = signal[0];
    let xn = signal[n - 1];
    let factor = std::f64::consts::PI / (n - 1) as f64;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let sign = if k % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            let mut sum = x0 + sign * xn;
            for n_idx in 1..=(n - 2) {
                sum += 2.0 * signal[n_idx] * (factor * n_idx as f64 * k as f64).cos();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let sign = if k % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            let mut sum = x0 + sign * xn;
            for n_idx in 1..=(n - 2) {
                sum += 2.0 * signal[n_idx] * (factor * n_idx as f64 * k as f64).cos();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-IV Discrete Cosine Transform (DCT-IV).
///
/// # Theorem: DCT-IV Direct Projection
///
/// For a real sequence x ∈ ℝᴺ, the unnormalized DCT-IV is defined by:
///
/// \\[ X_k = \sum_{n=0}^{N-1} x_n \cos\!\left(\frac{\pi (n + \tfrac{1}{2})(k + \tfrac{1}{2})}{N}\right), \quad k = 0, \ldots, N-1 \\]
///
/// The half-sample shifts on both the spatial index n and the frequency index k
/// produce a basis with no privileged DC or Nyquist component; every row and column
/// of the DCT-IV matrix has identical L₂ norm.
///
/// # Self-Inverse Property
///
/// The DCT-IV matrix C₄ is symmetric (C₄ᵀ = C₄). The finite orthogonality relation:
///
/// \\[ \sum_{n=0}^{N-1} \cos\!\left(\frac{\pi (n+\tfrac{1}{2})(k+\tfrac{1}{2})}{N}\right)
///     \cos\!\left(\frac{\pi (n+\tfrac{1}{2})(j+\tfrac{1}{2})}{N}\right) = \frac{N}{2} \delta_{kj} \\]
///
/// yields C₄ · C₄ = (N/2) · I. Therefore:
///
/// \\[ \text{DCT-IV}(\text{DCT-IV}(x)) = \frac{N}{2} \cdot x \\]
///
/// Proof by hand for N = 2, x = [1, 3]:
/// - Forward: X = [cos(π/8) + 3cos(3π/8), cos(3π/8) − 3cos(π/8)] ≈ [2.072, −2.389]
/// - Inverse: DCT-IV(X) ≈ [1, 3] = (N/2)·x with N/2 = 1 ✓
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 2.6, DCT-IV, pp. 31–33.)
/// - Martucci, S. A. (1994). Symmetric convolution and the discrete sine and cosine
///   transforms. *IEEE Trans. Signal Process.*, 42(5), 1038–1051.
pub fn dct4(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 0.5)).cos();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 0.5)).cos();
            }
            *out = sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-I Discrete Sine Transform (DST-I).
///
/// # Theorem: DST-I Direct Projection
///
/// For a real sequence x ∈ ℝᴺ, the unnormalized DST-I is defined by:
///
/// \\[ X_k = 2 \sum_{n=0}^{N-1} x_n \sin\!\left(\frac{\pi (n+1)(k+1)}{N+1}\right), \quad k = 0, \ldots, N-1 \\]
///
/// The denominator N+1 embeds implicit odd-symmetric zero boundary conditions at
/// n = −1 and n = N (FFTW RODFT00). No boundary sample is included explicitly; all
/// N samples are strictly interior.
///
/// # Self-Inverse Property
///
/// The DST-I matrix S₁ is symmetric (S₁ᵀ = S₁). The finite orthogonality relation:
///
/// \\[ \sum_{n=0}^{N-1} \sin\!\left(\frac{\pi (n+1)(k+1)}{N+1}\right)
///     \sin\!\left(\frac{\pi (n+1)(j+1)}{N+1}\right) = \frac{N+1}{2} \delta_{kj} \\]
///
/// combined with the outer factor of 2 yields S₁ · S₁ = 2(N+1) · I. Therefore:
///
/// \\[ \text{DST-I}(\text{DST-I}(x)) = 2(N+1) \cdot x \\]
///
/// Proof by hand for N = 2, x = [1, 3]:
/// - Forward: X = [2(sin(π/3) + 3sin(2π/3)), 2(sin(2π/3) − 3sin(2π/3))]
///   = [4√3, −2√3]
/// - Inverse: DST-I([4√3, −2√3]) = [6, 18] = 6·x = 2(N+1)·x with N+1=3 ✓
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 3.4, DST-I, pp. 163–165.)
/// - Martucci, S. A. (1994). Symmetric convolution and the discrete sine and cosine
///   transforms. *IEEE Trans. Signal Process.*, 42(5), 1038–1051.
pub fn dst1(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / (n + 1) as f64;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 1.0) * (k as f64 + 1.0)).sin();
            }
            *out = 2.0 * sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 1.0) * (k as f64 + 1.0)).sin();
            }
            *out = 2.0 * sum;
        });
    }
}

/// Direct O(N²) analytical kernel for the Type-IV Discrete Sine Transform (DST-IV).
///
/// # Theorem: DST-IV Direct Projection
///
/// For a real sequence x ∈ ℝᴺ, the unnormalized DST-IV is defined by:
///
/// \\[ X_k = \sum_{n=0}^{N-1} x_n \sin\!\left(\frac{\pi (n + \tfrac{1}{2})(k + \tfrac{1}{2})}{N}\right), \quad k = 0, \ldots, N-1 \\]
///
/// The half-sample shifts on both spatial and frequency indices produce a purely
/// antisymmetric basis (sine rather than cosine) with no zero-frequency or
/// Nyquist component; the structure is the odd counterpart of DCT-IV.
///
/// # Self-Inverse Property
///
/// The DST-IV matrix S₄ is symmetric (S₄ᵀ = S₄). The finite orthogonality relation:
///
/// \\[ \sum_{n=0}^{N-1} \sin\!\left(\frac{\pi (n+\tfrac{1}{2})(k+\tfrac{1}{2})}{N}\right)
///     \sin\!\left(\frac{\pi (n+\tfrac{1}{2})(j+\tfrac{1}{2})}{N}\right) = \frac{N}{2} \delta_{kj} \\]
///
/// yields S₄ · S₄ = (N/2) · I. Therefore:
///
/// \\[ \text{DST-IV}(\text{DST-IV}(x)) = \frac{N}{2} \cdot x \\]
///
/// Proof by hand for N = 2, x = [1, 3]:
/// - Forward: X ≈ [3.1544, −0.2242]
/// - Inverse: DST-IV(X) ≈ [1, 3] = (N/2)·x with N/2 = 1 ✓
///
/// # Complexity
///
/// Time: O(N²). Space: O(1) auxiliary.
///
/// # References
///
/// - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
///   Applications*. Academic Press. (Definition 3.6, DST-IV, pp. 166–168.)
/// - Martucci, S. A. (1994). Symmetric convolution and the discrete sine and cosine
///   transforms. *IEEE Trans. Signal Process.*, 42(5), 1038–1051.
pub fn dst4(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }
    let factor = std::f64::consts::PI / n as f64;

    if n >= PAR_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 0.5)).sin();
            }
            *out = sum;
        });
    } else {
        output.iter_mut().enumerate().for_each(|(k, out)| {
            let mut sum = 0.0;
            for n_idx in 0..n {
                sum += signal[n_idx] * (factor * (n_idx as f64 + 0.5) * (k as f64 + 0.5)).sin();
            }
            *out = sum;
        });
    }
}
