//! Twiddle-table construction and real-FFT half-complex split routines.
//!
//! ## Current role
//!
//! This module no longer contains a DIT execution kernel. The radix-2 iterative
//! DIT butterfly engine was retired in favour of the Stockham autosort kernel
//! (`mixed_radix.rs`) which requires no bit-reversal permutation pass and
//! delivers better throughput via cache-friendly ping-pong buffering.
//!
//! The functions remaining here are:
//!
//! 1. **Twiddle-table builders** (`build_forward_twiddle_table_{32,64}`,
//!    `build_inverse_twiddle_table_{32,64}`): construct contiguous per-stage
//!    twiddle tables used by the Stockham kernel and by the 2-D / 3-D plan
//!    axes. All four delegate to the SSOT in `twiddle_table.rs`.
//!
//! 2. **Real-FFT pack/unpack** (`forward_real_inplace_64`,
//!    `inverse_real_inplace_64`, `forward_real_inplace_32`,
//!    `inverse_real_inplace_32`): implement the Cooley-Tukey real-FFT trick.
//!    An N-point real DFT is computed by packing the real signal into an
//!    (N/2)-point complex signal, running the Stockham complex FFT, then
//!    extracting the half-spectrum via a frequency-domain butterfly.
//!
//! ## Twiddle-table mathematical contract
//!
//! Theorem (Unified Twiddle Table): A single (N-1)-entry contiguous table
//! with per-stage layout suffices for all log2(N) Stockham stages.
//!
//! Layout invariant: for stage s with sub-transform length L = 2^s,
//! table[base..base+L/2] holds W_L^j = exp(-2*pi*i*j/L) for j = 0..L/2-1,
//! where base = L/2 - 1 (sum of all shorter stage lengths). This lets
//! the Stockham kernel read twiddles sequentially with no stride. QED.
//!
//! ## Real-FFT contract
//!
//! Theorem (Real-FFT): For real input x[n] of length N (N even and power of two),
//! the N-point DFT can be evaluated by:
//!
//! 1. Pack z[k] = x[2k] + i*x[2k+1]  (M = N/2 complex samples).
//! 2. Compute the M-point forward complex FFT: Z = FFT_M(z).
//! 3. Extract X[k] for k = 0..M via the Cooley-Tukey split formula:
//!    X[k] = (Z[k] + Z[M-k]^*)/2 + i*W_N^k * (Z[k] - Z[M-k]^*)/2i,
//!    where W_N^k = exp(-2*pi*i*k/N).
//!
//! This halves the arithmetic cost vs. a direct N-point complex FFT. QED.
//!
//! ## Failure modes
//!
//! - Empty slice: returns immediately (N=0).
//! - N=1: returns immediately (trivial transform).
//! - N not a power of 2: triggers `debug_assert!` in debug builds.

use super::mixed_radix::{forward_inplace_64_with_twiddles, inverse_inplace_64_with_twiddles};
use num_complex::{Complex32, Complex64};

// ── twiddle table helpers ─────────────────────────────────────────────────────

/// Build a contiguous per-stage forward twiddle table for an N-point DFT.
///
/// # Layout
///
/// The table stores twiddles for log₂N butterfly stages in order from stage 1
/// (len=2) through stage K (len=N). Stage s with group length `len = 2^s` requires
/// `half = len/2` entries `W_{len}^j = exp(-2πi·j/len)` for j = 0…half-1.
///
/// These are laid out contiguously: `[stage1_entries | stage2_entries | … | stageK_entries]`.
/// Stage s has `2^(s-1)` entries. Total length = Σ 2^(s-1) for s=1..K = N-1.
///
/// # Cache benefit
///
/// The original strided-table approach reads `T_fwd[j * stride]` where
/// `stride = N / len = N / 2^s`. For stage 1 (len=2, stride=N/2), each of the N/2
/// independent butterfly groups reads a different element strided by N/2, causing
/// L1/L2 cache misses for N ≥ 256. The contiguous layout ensures each stage reads
/// a sequential sub-slice, maximising cache-line utilization.
///
/// # Correctness
///
/// `W_{len}^j = exp(-2πi·j/len)`. With `len = 2^s` and `j < 2^(s-1)`, the entry at
/// position `base + j` (where `base = 2^(s-1) - 1`) is `exp(-2πi·j / 2^s)`.
pub fn build_forward_twiddle_table_64(n: usize) -> Vec<Complex64> {
    super::twiddle_table::build_twiddle_table(n, -1.0)
}

/// Build a contiguous per-stage inverse twiddle table (positive exponent sign).
pub fn build_inverse_twiddle_table_64(n: usize) -> Vec<Complex64> {
    super::twiddle_table::build_twiddle_table(n, 1.0)
}

/// Build a contiguous per-stage forward twiddle table for f32.
///
/// Twiddles computed in f64 for accuracy, then cast to f32.
pub fn build_forward_twiddle_table_32(n: usize) -> Vec<Complex32> {
    super::twiddle_table::build_twiddle_table(n, -1.0)
}

/// Build post-processing twiddle table for real-input forward FFT.
///
/// Returns N/2 + 1 entries: `post[k] = exp(-2πi·k/N)` for k = 0..=N/2.
///
/// These are the standard DFT twiddle factors W_N^k. During the unpack step of
/// `forward_real_inplace_64`, each complex bin `X[k]` for k = 1..N/2-1 requires
/// multiplication by W_N^k to separate the even and odd sub-DFTs. The DC (k=0)
/// and Nyquist (k=N/2) bins use simple real sums and require no multiplication.
pub fn build_real_fwd_post_twiddles_64(n: usize) -> Vec<Complex64> {
    debug_assert!(n.is_power_of_two() && n >= 4);
    let m = n >> 1;
    let len = m + 1;
    let mut twiddles = Vec::with_capacity(len);
    #[allow(clippy::uninit_vec)]
    unsafe {
        twiddles.set_len(len)
    };
    for (k, slot) in twiddles.iter_mut().enumerate() {
        let a = -std::f64::consts::TAU * k as f64 / n as f64;
        *slot = Complex64::new(a.cos(), a.sin());
    }
    twiddles
}

/// Real-input forward FFT via half-length complex packing.
///
/// # Algorithm
///
/// Theorem (split-radix real DFT): For real x[n] of length N = 2M, define
/// the complex signal z[k] = x[2k] + i·x[2k+1] for k = 0..M-1. Let
/// Z = FFT_M(z), A[l] = DFT of even sub-sequence, B[l] = DFT of odd sub-sequence.
/// Then Z[l] = A[l] + i·B[l] and Z[M-l]* = A[l] - i·B[l], giving
/// A[l] = (Z[l] + Z[M-l]*)/2 and B[l] = -i·(Z[l] - Z[M-l]*)/2.
/// The N-point DFT is X[l] = A[l] + W_N^l·B[l] for l = 0..N-1, where W_N^l = exp(-2πi·l/N).
/// Together: X[l] = (Z[l] + Z[M-l]*)/2 - i·W_N^l·(Z[l] - Z[M-l]*)/2.
/// Conjugate symmetry X[N-l] = X[l]* (since x is real) yields all N bins.
///
/// # Complexity
///
/// O((N/2)·log₂(N/2)) arithmetic operations for the inner FFT plus O(N) for
/// packing and unpacking — approximately half the work of a full N-point complex FFT.
///
/// # Twiddle reuse
///
/// The N/2-point FFT uses `fft_twiddles[0..N/2-1]`. By the contiguous per-stage
/// layout invariant, stages 1..log₂(N/2) of the N-point table occupy exactly
/// the first N/2-1 entries. The post-processing twiddles W_N^k are distinct from
/// the FFT twiddles and are supplied separately as `post_twiddles`.
///
/// # In-place unpack correctness
///
/// After the inner FFT, `output[0..M]` holds Z[0..M-1]. The unpack writes
/// X[l] to `output[l]` and X[N-l] = X[l]* to `output[N-l]`. For each pair
/// (l, M-l) with l < M/2, both Z[l] and Z[M-l] are read before either is
/// overwritten — since `output[M-l]` (index > M/2) has not been touched by
/// any earlier pair (which only wrote to indices ≤ pair_l < M/2).
///
/// # Preconditions
///
/// - `n = input.len() = output.len()`, a power of two, ≥ 4.
/// - `fft_twiddles.len()` ≥ n/2 - 1.
/// - `post_twiddles.len()` = n/2 + 1.
pub fn forward_real_inplace_64(
    input: &[f64],
    output: &mut [Complex64],
    fft_twiddles: &[Complex64],
    post_twiddles: &[Complex64],
) {
    let n = input.len();
    debug_assert!(
        n.is_power_of_two() && n >= 4,
        "real FFT requires PoT length >= 4"
    );
    let m = n >> 1;
    debug_assert_eq!(output.len(), n);
    debug_assert!(fft_twiddles.len() >= m - 1);
    debug_assert_eq!(post_twiddles.len(), m + 1);

    // Pack: z[k] = x[2k] + i·x[2k+1] into output[0..m]
    for k in 0..m {
        output[k] = Complex64::new(input[2 * k], input[2 * k + 1]);
    }

    // N/2-point forward FFT using the first m-1 twiddle entries.
    // Correct by the contiguous-layout invariant: stages 1..log₂M occupy
    // positions 0..M-1 of the N-point forward twiddle table.
    forward_inplace_64_with_twiddles(&mut output[..m], Some(&fft_twiddles[..m - 1]));

    // In-place unpack. Save Z[0] before overwriting output[0].
    let z0 = output[0];

    // Process symmetric pairs l, m-l for l = 1..ceil(m/2).
    // For l < m/2: Z[l] is at output[l] and Z[m-l] is at output[m-l] (m-l > m/2,
    // so not yet written), avoiding read-after-write aliasing within the loop.
    //
    // Twiddle symmetry: post_twiddles[m-l] = exp(-2πi·(N/2-l)/N)
    //   = exp(-πi)·exp(2πi·l/N) = -conj(post_twiddles[l]).
    // One twiddle read per pair halves post-twiddle cache pressure for large N
    // (N=65536 saves 256 KB of twiddle reads in this loop alone).
    let pair_end = (m + 1) / 2;
    for l in 1..pair_end {
        let ml = m - l;
        let zl = output[l]; // Z[l]   — not yet overwritten
        let zml = output[ml]; // Z[m-l] — not yet overwritten (ml ≥ pair_end > l)
        let a = (zl + zml.conj()) * 0.5;
        let b = (zl - zml.conj()) * Complex64::new(0.0, -0.5);
        let a2 = (zml + zl.conj()) * 0.5;
        let b2 = (zml - zl.conj()) * Complex64::new(0.0, -0.5);
        let wl = post_twiddles[l]; // post_twiddles[ml] = -conj(wl) by symmetry above
        let xl = a + wl * b; // X[l]
        let xml = a2 - wl.conj() * b2; // X[m-l] = a2 + (-conj(wl))·b2
        output[l] = xl;
        output[ml] = xml;
        output[n - l] = xl.conj(); // conjugate symmetry: X[N-l] = X[l]*
        output[n - ml] = xml.conj(); // X[N-(m-l)] = X[m-l]*
    }

    // Middle element at l = m/2 (exists iff m is even; for all PoT N ≥ 4, m = N/2 is even).
    // post_twiddles[m/2] = exp(-2πi·(N/4)/N) = exp(-πi/2) = -i.
    // Analytically: a = zmid.re, b = zmid.im (both real); -i·zmid.im = -i·b.
    // xmid = zmid.re + (-i·zmid.im) = zmid.re - i·zmid.im = conj(zmid). No multiply needed.
    if m % 2 == 0 {
        let mid = m / 2;
        let zmid = output[mid];
        output[mid] = zmid.conj(); // X[m/2]  = conj(Z[m/2])
        output[n - mid] = zmid; // X[3m/2] = conj(X[m/2])* = Z[m/2]
    }

    // DC bin: X[0] = Z[0].re + Z[0].im  (W_N^0 = 1, A[0] = Z[0].re, B[0] = Z[0].im)
    output[0] = Complex64::new(z0.re + z0.im, 0.0);
    // Nyquist bin: X[m] = Z[0].re - Z[0].im  (W_N^m = -1)
    output[m] = Complex64::new(z0.re - z0.im, 0.0);
}

/// Inverse real FFT via half-length complex packing (conjugate of the forward trick).
///
/// # Algorithm
///
/// Given a full N-point Hermitian spectrum X (X[N-k] = X[k]* for all k), recover
/// the real signal x of length N. The algorithm inverts `forward_real_inplace_64`:
///
/// **Pre-process** (solve for Z from X using the forward unpack formula):
///
/// From the forward unpack identity, letting a = Z[k] and b = conj(Z[M-k]):
///
///   X[k] = a*(1-i·W_k)/2 + b*(1+i·W_k)/2
///   conj(X[M-k]) = a*(1+i·W_k)/2 + b*(1-i·W_k)/2
///
/// where W_k = post_twiddles[k] = exp(-2πi·k/N) and M = N/2.
///
/// Adding and subtracting:
///   a + b = X[k] + conj(X[M-k])
///   a - b = i·conj(W_k)·(X[k] - conj(X[M-k]))
///
/// Solving: Z[k] = (X[k] + conj(X[M-k]))/2 + i·conj(W_k)·(X[k] - conj(X[M-k]))/2
///
/// Note: i·conj(W_k) = i·(W_k.re - i·W_k.im) = (W_k.im, W_k.re) as Complex64.
///
/// **Special cases:**
/// - k = 0: W_0 = 1, X[0] and X[M] are real, so Z[0] = (X[0]+X[M])/2 + i·(X[0]-X[M])/2.
///
/// **M-point normalized IFFT:** `inverse_inplace_64_with_twiddles` on scratch[0..M]
/// divides by M, giving z[k] = x[2k] + i·x[2k+1].
///
/// **Unpack:** x[2k] = z[k].re, x[2k+1] = z[k].im.
///
/// # Twiddle reuse
///
/// `fft_twiddles` is the N-point inverse table. Stages 1..log₂M occupy
/// positions 0..M-1, identical to the M-point inverse table.
/// `post_twiddles` is the same table as `build_real_fwd_post_twiddles_64(N)`,
/// reused from the plan's forward field.
///
/// # Complexity
///
/// O((N/2)·log₂(N/2)) for the inner IFFT plus O(N) pre-process and unpack.
/// Eliminates the N-point complex IFFT and the per-call N-element allocation of
/// the naive path.
///
/// # Memory
///
/// Requires a caller-supplied scratch of M = N/2 Complex64 entries (plan-owned
/// via `real_inv_scratch`).
///
/// # Preconditions
///
/// - `input.len() = output.len() = N`, power of two, ≥ 4.
/// - `scratch.len()` = N/2.
/// - `fft_twiddles.len()` ≥ N/2 - 1.
/// - `post_twiddles.len()` = N/2 + 1.
/// - `input` has Hermitian symmetry: input[N-k] = input[k]* for all k.
pub fn inverse_real_inplace_64(
    input: &[Complex64],
    output: &mut [f64],
    scratch: &mut [Complex64],
    fft_twiddles: &[Complex64],
    post_twiddles: &[Complex64],
) {
    let n = input.len();
    debug_assert!(
        n.is_power_of_two() && n >= 4,
        "iRFFT requires PoT length >= 4"
    );
    let m = n >> 1;
    debug_assert_eq!(output.len(), n);
    debug_assert_eq!(scratch.len(), m);
    debug_assert!(fft_twiddles.len() >= m - 1);
    debug_assert_eq!(post_twiddles.len(), m + 1);

    // k = 0: W_0 = 1, conj(W_0) = 1, i*conj(W_0) = i.
    // X[0] and X[M] are real (Hermitian spectrum at DC and Nyquist).
    scratch[0] = Complex64::new(
        (input[0].re + input[m].re) * 0.5,
        (input[0].re - input[m].re) * 0.5,
    );

    // Process pairs (k, m-k) for k = 1..m/2 with one twiddle read per pair.
    // Twiddle symmetry: post_twiddles[m-k] = -conj(post_twiddles[k]).
    // Let wk = post_twiddles[k]; wmk = -conj(wk).
    //   i·conj(wk)  = (wk.im, +wk.re)  [since i·(wk.re - i·wk.im) = wk.im + i·wk.re]
    //   i·conj(wmk) = i·(-wk)           = (wk.im, -wk.re)
    // Both factors share wk.im and differ only in sign of wk.re.
    // One twiddle read per pair halves post-twiddle bandwidth for large N.
    let half_m = m / 2; // m = N/2 is always even for PoT N ≥ 4
    for k in 1..half_m {
        let mk = m - k;
        let xk = input[k];
        let xmk = input[mk];
        let xmk_conj = xmk.conj();
        let xk_conj = xk.conj();
        let sum_k = xk + xmk_conj;
        let diff_k = xk - xmk_conj;
        let sum_mk = xmk + xk_conj;
        let diff_mk = xmk - xk_conj;
        let wk = post_twiddles[k];
        let i_conj_wk = Complex64::new(wk.im, wk.re);
        let i_conj_wmk = Complex64::new(wk.im, -wk.re);
        scratch[k] = (sum_k + i_conj_wk * diff_k) * 0.5;
        scratch[mk] = (sum_mk + i_conj_wmk * diff_mk) * 0.5;
    }
    // k = half_m (Nyquist, self-paired: m - half_m = half_m).
    // post_twiddles[half_m] = -i; i·conj(-i) = -1.
    // Closed form: scratch[half_m] = (xk + xk.conj() + (-1)·(xk - xk.conj())) * 0.5
    //   = (2·xk.conj()) * 0.5 = xk.conj(). No twiddle read or multiply.
    scratch[half_m] = input[half_m].conj();

    // M-point normalized IFFT on scratch[0..M].
    // `inverse_inplace_64_with_twiddles` applies the 1/M scale, giving z[k] = x[2k]+i*x[2k+1].
    // Correctness: stages 1..log₂M occupy the first M-1 positions of the N-point inverse
    // twiddle table (same contiguous-layout invariant as the forward path).
    inverse_inplace_64_with_twiddles(scratch, Some(&fft_twiddles[..m - 1]));

    // Unpack: z[k] = x[2k] + i*x[2k+1]
    for k in 0..m {
        output[2 * k] = scratch[k].re;
        output[2 * k + 1] = scratch[k].im;
    }
}

/// Build a contiguous per-stage inverse twiddle table for f32.
pub fn build_inverse_twiddle_table_32(n: usize) -> Vec<Complex32> {
    super::twiddle_table::build_twiddle_table(n, 1.0)
}
