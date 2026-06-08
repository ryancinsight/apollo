//! Fast O(N log N) DCT-II, DCT-III, DST-II, and DST-III kernels via 2N-point complex FFT.
//!
//! # Theorem: 2N-Point FFT Derivation of DCT-II/III and DST-II/III
//!
//! All four Type-II and Type-III real discrete trigonometric transforms of length N
//! reduce to a single 2N-point complex DFT (or its Hermitian inverse) via twiddle-factor
//! extraction. The `apollo-fft` forward kernel computes the unnormalized DFT:
//!
//! ```text
//! F[k] = Σ_{n=0}^{N-1} x[n] exp(-2πikn/N)
//! ```
//!
//! The inverse kernel computes the normalized IDFT dividing by the array length:
//!
//! ```text
//! y[n] = (1/N) Σ_{k=0}^{N-1} X[k] exp(2πikn/N)
//! ```
//!
//! ## Sub-theorem 1: DCT-II via 2N-point forward DFT
//!
//! Let x ∈ ℝᴺ. Let x̃ ∈ ℂ²ᴺ be x zero-padded to length 2N:
//!
//! ```text
//! x̃[n] = x[n] for n < N, x̃[n] = 0 for n ≥ N.
//! ```
//!
//! Let F = DFT_{2N}(x̃). Then:
//!
//! ```text
//! DCT-II[k] = Re(exp(-iπk/(2N)) · F[k])   for k = 0, ..., N-1
//! ```
//!
//! *Proof*: By definition,
//!
//! ```text
//! F[k] = Σ_{n=0}^{N-1} x[n] exp(-2πikn/(2N)).
//! ```
//!
//! Let W_k = exp(-iπk/(2N)). Then:
//!
//! ```text
//! W_k · F[k] = Σ_{n=0}^{N-1} x[n] exp(-iπk/(2N)) exp(-2πikn/(2N))
//!            = Σ_{n=0}^{N-1} x[n] exp(-iπk(2n+1)/(2N)).
//! ```
//!
//! Taking the real part:
//!
//! ```text
//! Re(W_k · F[k]) = Σ_{n=0}^{N-1} x[n] cos(πk(2n+1)/(2N)) = DCT-II[k]. ∎
//! ```
//!
//! ## Sub-theorem 2: DST-II via the same 2N-point forward DFT
//!
//! Using F = DFT_{2N}(x̃) from Sub-theorem 1 (no second FFT required):
//!
//! ```text
//! DST-II[k] = -Im(exp(-iπ(k+1)/(2N)) · F[k+1])   for k = 0, ..., N-1
//! ```
//!
//! *Proof*: Let W_{k+1} = exp(-iπ(k+1)/(2N)). Then:
//!
//! ```text
//! W_{k+1} · F[k+1] = Σ_{n=0}^{N-1} x[n] exp(-iπ(k+1)(2n+1)/(2N)).
//! ```
//!
//! Taking the negative imaginary part:
//!
//! ```text
//! -Im(W_{k+1} · F[k+1]) = Σ_{n=0}^{N-1} x[n] sin(π(k+1)(2n+1)/(2N)) = DST-II[k]. ∎
//! ```
//!
//! Note: `F[k+1]` for k ∈ {0,...,N-1} indexes positions 1..=N in a length-2N array; all
//! indices are in bounds.
//!
//! ## Sub-theorem 3: DCT-III via 2N-point Hermitian IDFT
//!
//! Construct G ∈ ℂ²ᴺ as the Hermitian-symmetric spectrum:
//!
//! ```text
//! G[0]    = X[0]                           (real DC term)
//! G[k]    = X[k] · exp(iπk/(2N))          for k = 1, ..., N-1
//! G[N]    = 0                              (Nyquist term)
//! G[2N-k] = conj(G[k])                    for k = 1, ..., N-1
//! ```
//!
//! Let y = IDFT_{2N}(G) (normalized: `y[n] = (1/(2N)) Σ G[k] exp(2πikn/(2N))`). Then:
//!
//! ```text
//! DCT-III[n] = N · Re(y[n])   for n = 0, ..., N-1
//! ```
//!
//! *Proof*: By Hermitian symmetry of G, y is real up to floating-point noise. Expanding:
//!
//! ```text
//! (2N) · y[n] = G[0] + Σ_{k=1}^{N-1} G[k] exp(2πikn/(2N))
//!                     + G[N] exp(πin)
//!                     + Σ_{k=1}^{N-1} G[2N-k] exp(2πi(2N-k)n/(2N))
//! ```
//!
//! Since `G[N] = 0` and `G[2N-k] = conj(G[k])`:
//!
//! ```text
//! (2N) · y[n] = X[0] + Σ_{k=1}^{N-1} G[k] exp(2πikn/(2N))
//!                     + Σ_{k=1}^{N-1} conj(G[k]) exp(-2πikn/(2N))
//!             = X[0] + 2 · Re(Σ_{k=1}^{N-1} G[k] exp(2πikn/(2N)))
//!             = X[0] + 2 · Σ_{k=1}^{N-1} X[k] cos(πk(2n+1)/(2N)).
//! ```
//!
//! Therefore: `N · y[n] = X[0]/2 + Σ_{k=1}^{N-1} X[k] cos(πk(2n+1)/(2N)) = DCT-III[n]`. ∎
//!
//! ## Sub-theorem 4: DST-III via 2N-point forward DFT with complex input
//!
//! Let X' ∈ ℝᴺ be defined by `X'[n] = X[n]` for n < N-1, `X'[N-1] = X[N-1]/2` (half boundary term).
//! Form V ∈ ℂ²ᴺ zero-padded:
//!
//! ```text
//! V[n] = X'[n] · exp(-iπn/(2N))   for n = 0, ..., N-1
//! V[n] = 0                         for n = N, ..., 2N-1
//! ```
//!
//! Let G = DFT_{2N}(V) (unnormalized forward FFT). Then:
//!
//! ```text
//! DST-III[k] = Im(exp(iπ(2k+1)/(2N)) · conj(G[k]))   for k = 0, ..., N-1
//! ```
//!
//! *Proof*: Since `X'[n]` ∈ ℝ, `conj(V[n]) = X'[n] · exp(+iπn/(2N))`. Therefore:
//!
//! ```text
//! conj(G[k]) = Σ_{n=0}^{N-1} conj(V[n]) exp(2πikn/(2N))
//!            = Σ_{n=0}^{N-1} X'[n] exp(iπn/(2N)) exp(2πikn/(2N))
//!            = Σ_{n=0}^{N-1} X'[n] exp(iπn(2k+1)/(2N)).
//! ```
//!
//! Multiplying by exp(iπ(2k+1)/(2N)):
//!
//! ```text
//! exp(iπ(2k+1)/(2N)) · conj(G[k]) = Σ_{n=0}^{N-1} X'[n] exp(iπ(n+1)(2k+1)/(2N)).
//! ```
//!
//! Taking the imaginary part:
//!
//! ```text
//! Im[...] = Σ_{n=0}^{N-1} X'[n] sin(π(n+1)(2k+1)/(2N)).
//! ```
//!
//! At n = N-1: `X'[N-1] · sin(πN(2k+1)/(2N)) = X[N-1]/2 · sin(π(2k+1)/2) = X[N-1]/2 · (-1)^k`.
//! For n = 0,...,N-2: `X[n] · sin(π(n+1)(2k+1)/(2N))`.
//!
//! Thus: `Im[...] = (-1)^k · X[N-1]/2 + Σ_{n=0}^{N-2} X[n] sin(π(n+1)(2k+1)/(2N)) = DST-III[k]`. ∎
//!
//! # References
//!
//! - Bracewell, R. N. (1984). Discrete Hartley transform. *J. Opt. Soc. Am.*, 73(12), 1832–1835.
//! - Rao, K. R. & Yip, P. (1990). *Discrete Cosine Transform: Algorithms, Advantages,
//!   Applications*. Academic Press.
//! - Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans.
//!   Acoust. Speech Signal Process.*, 28(1), 27–34.

use apollo_fft::{Complex64, PlanCacheProvider, Shape1D};
use mnemosyne::scratch::ScratchPool;
use std::f64::consts::PI;

/// Crossover threshold: for N ≥ `FAST_THRESHOLD`, the 2N-point FFT path (O(N log N))
/// is faster than the direct O(N²) kernel.
///
/// Verification: N = 16 → 2N · log₂(2N) = 32 · 5 = 160 < N² = 256. ✓
/// N = 8  → 2N · log₂(2N) = 16 · 4  = 64  < N² = 64. (breakeven; use 16 to be conservative.)
pub const FAST_THRESHOLD: usize = 16;

thread_local! {
    static COMPLEX_SCRATCH_POOL: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Shared 2N-point forward DFT kernel for DCT-II and DST-II.
///
/// Computes one unnormalized 2N-point forward FFT of the zero-padded real input `signal`
/// and fills `dct_output` and `dst_output` simultaneously, avoiding a redundant FFT call
/// when both transforms are needed.
///
/// # Mathematical contract
///
/// Given F = DFT_{2N}(x̃) where x̃ is `signal` zero-padded to length 2N:
/// - `dct_output[k] = Re(exp(-iπk/(2N)) · F[k])`       for k = 0,...,N-1  (Sub-theorem 1)
/// - `dst_output[k] = -Im(exp(-iπ(k+1)/(2N)) · F[k+1])` for k = 0,...,N-1  (Sub-theorem 2)
///
/// # Panics
///
/// Only in debug builds when slice lengths are inconsistent with `signal.len()`.
pub fn dct2_dst2_fast(signal: &[f64], dct_output: &mut [f64], dst_output: &mut [f64]) {
    let n = signal.len();
    debug_assert_eq!(
        dct_output.len(),
        n,
        "dct2_dst2_fast: dct_output length mismatch"
    );
    debug_assert_eq!(
        dst_output.len(),
        n,
        "dct2_dst2_fast: dst_output length mismatch"
    );

    let two_n = 2 * n;
    let half_cycle = PI / two_n as f64;

    COMPLEX_SCRATCH_POOL.with(|pool| {
        pool.with_scratch(two_n, |buf| {
            for (i, &x) in signal.iter().enumerate() {
                buf[i] = Complex64::new(x, 0.0);
            }
            for i in n..two_n {
                buf[i] = Complex64::new(0.0, 0.0);
            }

            let plan = f64::get_1d_plan(Shape1D::new(two_n).expect("Shape1D"));
            plan.forward_complex_slice_inplace(buf);

            // fill_dct2_from_fft
            for k in 0..n {
                let angle = -(half_cycle * k as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                let w = Complex64::new(cos_a, sin_a);
                dct_output[k] = (w * buf[k]).re;
            }

            // fill_dst2_from_fft
            for k in 0..n {
                let angle = -(half_cycle * (k as f64 + 1.0));
                let (sin_a, cos_a) = angle.sin_cos();
                let w = Complex64::new(cos_a, sin_a);
                dst_output[k] = -(w * buf[k + 1]).im;
            }
        });
    });
}

/// Fast DCT-II via 2N-point forward FFT. Complexity: O(N log N).
///
/// Computes the DCT projection from one shared 2N-point FFT without allocating
/// the DST output used by [`dct2_dst2_fast`].
/// Suitable for N ≥ [`FAST_THRESHOLD`]; use the direct O(N²) kernel for smaller N.
///
/// # Mathematical contract
///
/// `output[k] = Σ_{n=0}^{N-1} signal[n] · cos(πk(2n+1)/(2N))` for k = 0,...,N-1.
///
/// Derived via Sub-theorem 1: `output[k] = Re(exp(-iπk/(2N)) · DFT_{2N}(x̃)[k])`.
pub fn dct2_fast(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    debug_assert_eq!(
        output.len(),
        n,
        "dct2_fast: output length mismatch"
    );
    let two_n = 2 * n;
    let half_cycle = PI / two_n as f64;

    COMPLEX_SCRATCH_POOL.with(|pool| {
        pool.with_scratch(two_n, |buf| {
            for (i, &x) in signal.iter().enumerate() {
                buf[i] = Complex64::new(x, 0.0);
            }
            for i in n..two_n {
                buf[i] = Complex64::new(0.0, 0.0);
            }

            let plan = f64::get_1d_plan(Shape1D::new(two_n).expect("Shape1D"));
            plan.forward_complex_slice_inplace(buf);

            // fill_dct2_from_fft
            for k in 0..n {
                let angle = -(half_cycle * k as f64);
                let (sin_a, cos_a) = angle.sin_cos();
                let w = Complex64::new(cos_a, sin_a);
                output[k] = (w * buf[k]).re;
            }
        });
    });
}

/// Fast DST-II via 2N-point forward FFT. Complexity: O(N log N).
///
/// Computes the DST projection from one shared 2N-point FFT without allocating
/// the DCT output used by [`dct2_dst2_fast`].
/// Suitable for N ≥ [`FAST_THRESHOLD`]; use the direct O(N²) kernel for smaller N.
///
/// # Mathematical contract
///
/// `output[k] = Σ_{n=0}^{N-1} signal[n] · sin(π(k+1)(2n+1)/(2N))` for k = 0,...,N-1.
///
/// Derived via Sub-theorem 2: `output[k] = -Im(exp(-iπ(k+1)/(2N)) · DFT_{2N}(x̃)[k+1])`.
pub fn dst2_fast(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    debug_assert_eq!(
        output.len(),
        n,
        "dst2_fast: output length mismatch"
    );
    let two_n = 2 * n;
    let half_cycle = PI / two_n as f64;

    COMPLEX_SCRATCH_POOL.with(|pool| {
        pool.with_scratch(two_n, |buf| {
            for (i, &x) in signal.iter().enumerate() {
                buf[i] = Complex64::new(x, 0.0);
            }
            for i in n..two_n {
                buf[i] = Complex64::new(0.0, 0.0);
            }

            let plan = f64::get_1d_plan(Shape1D::new(two_n).expect("Shape1D"));
            plan.forward_complex_slice_inplace(buf);

            // fill_dst2_from_fft
            for k in 0..n {
                let angle = -(half_cycle * (k as f64 + 1.0));
                let (sin_a, cos_a) = angle.sin_cos();
                let w = Complex64::new(cos_a, sin_a);
                output[k] = -(w * buf[k + 1]).im;
            }
        });
    });
}

/// Fast DCT-III via 2N-point Hermitian IDFT. Complexity: O(N log N).
///
/// Constructs a Hermitian-symmetric spectrum G, applies the normalized IDFT (which
/// divides by 2N), then scales the real part by N to recover the unnormalized DCT-III.
///
/// # Mathematical contract
///
/// `output[n] = X[0]/2 + Σ_{k=1}^{N-1} X[k] · cos(πk(2n+1)/(2N))` for n = 0,...,N-1.
///
/// Derived via Sub-theorem 3: `output[n] = N · Re(IDFT_{2N}(G)[n])`.
///
/// Suitable for N ≥ [`FAST_THRESHOLD`]; use the direct O(N²) kernel for smaller N.
pub fn dct3_fast(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    let two_n = 2 * n;
    // π / (2N): fundamental angular step.
    let half_cycle = PI / two_n as f64;

    COMPLEX_SCRATCH_POOL.with(|pool| {
        pool.with_scratch(two_n, |g| {
            // Build Hermitian-symmetric spectrum G of length 2N.
            //   G[0]    = X[0]                       (real DC)
            //   G[k]    = X[k] · exp(+iπk/(2N))     for k = 1..N-1
            //   G[N]    = 0                           (Nyquist)
            //   G[2N-k] = conj(G[k])                 for k = 1..N-1
            g[0] = Complex64::new(signal[0], 0.0);

            for k in 1..n {
                let angle = half_cycle * k as f64; // +πk/(2N)
                let (sin_a, cos_a) = angle.sin_cos();
                let twiddle = Complex64::new(cos_a, sin_a); // exp(+iπk/(2N))
                g[k] = Complex64::new(signal[k], 0.0) * twiddle;
            }
            g[n] = Complex64::new(0.0, 0.0);
            // Hermitian conjugate half: G[2N-k] = conj(G[k]) for k = 1..N-1.
            for k in 1..n {
                g[two_n - k] = g[k].conj();
            }

            // y = IDFT_{2N}(G): normalized in-place.
            let plan = f64::get_1d_plan(Shape1D::new(two_n).expect("Shape1D"));
            plan.inverse_complex_slice_inplace(g);

            // DCT-III[n] = N · Re(y[n]).
            let n_f = n as f64;
            for i in 0..n {
                output[i] = n_f * g[i].re;
            }
        });
    });
}

/// Fast DST-III via 2N-point forward FFT with complex input. Complexity: O(N log N).
///
/// Constructs a complex pre-twiddled input V (with the boundary half-term X'[N-1] = X[N-1]/2),
/// applies the 2N-point forward DFT, then extracts DST-III values via conjugate and
/// post-twiddle.
///
/// # Mathematical contract
///
/// `output[k] = (-1)^k · X[N-1]/2 + Σ_{n=0}^{N-2} X[n] · sin(π(n+1)(2k+1)/(2N))`.
///
/// Derived via Sub-theorem 4: `output[k] = Im(exp(iπ(2k+1)/(2N)) · conj(DFT_{2N}(V)[k]))`.
///
/// Suitable for N ≥ [`FAST_THRESHOLD`]; use the direct O(N²) kernel for smaller N.
pub fn dst3_fast(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    let two_n = 2 * n;
    // π / (2N): fundamental angular step.
    let half_cycle = PI / two_n as f64;

    COMPLEX_SCRATCH_POOL.with(|pool| {
        pool.with_scratch(two_n, |v| {
            // Build complex V of length 2N:
            //   V[i] = X'[i] · exp(-iπi/(2N))   for i = 0,...,N-1
            //   V[i] = 0                          for i = N,...,2N-1
            // where X'[i] = X[i] for i < N-1, X'[N-1] = X[N-1]/2 (half boundary term).
            for i in 0..n {
                let x_prime = if i < n - 1 {
                    signal[i]
                } else {
                    signal[i] * 0.5
                };
                let angle = -(half_cycle * i as f64); // -πi/(2N)
                let (sin_a, cos_a) = angle.sin_cos();
                let twiddle = Complex64::new(cos_a, sin_a); // exp(-iπi/(2N))
                v[i] = Complex64::new(x_prime, 0.0) * twiddle;
            }
            for i in n..two_n {
                v[i] = Complex64::new(0.0, 0.0);
            }

            // G = DFT_{2N}(V): in-place.
            let plan = f64::get_1d_plan(Shape1D::new(two_n).expect("Shape1D"));
            plan.forward_complex_slice_inplace(v);

            // DST-III[k] = Im(exp(iπ(2k+1)/(2N)) · conj(G[k]))
            for k in 0..n {
                let angle = half_cycle * (2.0 * k as f64 + 1.0); // π(2k+1)/(2N)
                let (sin_a, cos_a) = angle.sin_cos();
                let twiddle = Complex64::new(cos_a, sin_a); // exp(iπ(2k+1)/(2N))
                output[k] = (twiddle * v[k].conj()).im;
            }
        });
    });
}
