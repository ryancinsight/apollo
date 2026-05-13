//! Radix-2 Stockham autosort FFT entry points.
//!
//! ## Mathematical contract
//!
//! For N = 2^k, the forward transform computes:
//!
//! `X[k] = Σ_{n=0}^{N-1} x[n] · exp(-2πi · k·n / N)` (unnormalized)
//!
//! The unnormalized inverse computes:
//!
//! `y[n] = Σ_{k=0}^{N-1} X[k] · exp(+2πi · k·n / N)` (unnormalized, no 1/N factor)
//!
//! The normalized inverse divides the unnormalized result by N.
//!
//! ## Algorithm
//!
//! Theorem (Stockham autosort): The N-point DFT with N = 2^k decomposes into
//! log₂ N breadth-first radix-2 stages that alternate source and destination
//! buffers. Each stage maps the current digit order into the next digit order,
//! so the final stage is naturally ordered and no separate permutation pass is
//! part of the public transform contract.
//!
//! ## Twiddle table construction
//!
//! Theorem (Unified Twiddle Table): A single N/2-entry table
//! T[j] = exp(-2πi·j/N), j = 0…N/2-1, suffices for all log₂ N stages.
//!
//! Proof: In stage s the butterfly half-length is `half = 2^(s-1)`. The twiddle
//! for butterfly position j within each group of `len = 2·half` elements is
//! W_{len}^j = exp(-2πi·j/len). Substituting len = N/stride where stride = N/len
//! gives W_{len}^j = exp(-2πi·j·stride/N) = T[j·stride]. Hence every stage's
//! twiddle sequence is a stride-sub-sample of T with stride = N/len, which equals
//! N >> len.trailing_zeros(). The sub-sample reads T[j·stride] for j = 0…half-1,
//! accessing exactly the same values as the per-stage allocation without any
//! heap allocation. □
//!
//! Corollary: The full twiddle table is computed once before the butterfly loop
//! (O(N/2) trigonometric calls, amortized over all stages), eliminating all k
//! per-stage Vec allocations for N = 2^k.
//!
//! ## Energy invariance (Parseval / Plancherel)
//!
//! Theorem (Parseval): For any x ∈ ℂⁿ with DFT X computed by this kernel,
//! `Σ_n |x[n]|² = (1/N) · Σ_k |X[k]|²`.
//!
//! Proof sketch: The N×N DFT matrix F satisfies F·F* = N·I (unitary up to √N).
//! Therefore ‖Fx‖² = xᴴF*Fx = N·‖x‖², giving ‖X‖² = N·‖x‖². Dividing by N
//! yields the normalized Parseval identity. This kernel computes the unnormalized
//! Fx, so callers must divide the spectral energy sum by N to recover time-domain
//! energy — consistent with the `energy_frequency_domain / N` normalization in
//! the plan-level `parseval_identity_holds` test. □
//!
//! ## Failure modes
//!
//! - Empty slice: returns immediately (N=0).
//! - N=1: returns immediately (trivial transform).
//! - N not a power of 2: triggers `debug_assert!` in debug builds.

#![allow(clippy::uninit_vec)]

use crate::application::execution::kernel::stockham::StockhamKernel;
use num_complex::{Complex32, Complex64};
use std::cell::RefCell;

thread_local! {
    static TL_STOCKHAM_64: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    static TL_STOCKHAM_32: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn with_stockham_scratch_64(n: usize, f: impl FnOnce(&mut [Complex64])) {
    TL_STOCKHAM_64.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < n {
            // Grow without zero-init: Stockham kernel writes every element before reading.
            let cur = scratch.len();
            scratch.reserve(n - cur);
            // SAFETY: Complex64 = [f64; 2]; no validity invariant beyond bit pattern.
            // Stockham kernel writes before reading every slot.
            unsafe { scratch.set_len(n) };
        }
        f(&mut scratch[..n]);
    });
}

#[inline]
fn with_stockham_scratch_32(n: usize, f: impl FnOnce(&mut [Complex32])) {
    TL_STOCKHAM_32.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < n {
            let cur = scratch.len();
            scratch.reserve(n - cur);
            unsafe { scratch.set_len(n) };
        }
        f(&mut scratch[..n]);
    });
}

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
/// Single trig loop shared by all four twiddle-table builders.
///
/// `sign`: −1.0 for forward (exp(−2πi·j/len)), +1.0 for inverse.
/// `make_complex`: converts (cos, sin) → the target `Complex` type.
/// Monomorphized by the compiler — identical machine code to the
/// former hand-written specializations.
#[inline]
fn build_twiddle_table_inner<C>(
    n: usize,
    sign: f64,
    make_complex: impl Fn(f64, f64) -> C,
) -> Vec<C> {
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return Vec::new();
    }
    let log_n = n.trailing_zeros() as usize;
    let mut table = Vec::with_capacity(n - 1);
    // SAFETY: We initialize exactly `n - 1` elements in the loop below.
    // The total entries across log₂n stages is Σ_{s=1}^{log₂n} 2^{s-1} = n - 1.
    unsafe { table.set_len(n - 1) };
    let mut len = 2usize;
    let mut idx = 0;
    for _ in 0..log_n {
        let half = len >> 1;
        // W_base = exp(sign * 2πi / len): one trig call per stage.
        // Recurrence: tw[j] = tw[j-1] * W_base — one cmul vs one sin+cos per entry.
        // This matches the recurrence used in radix_composite and bluestein.
        let base_angle = sign * std::f64::consts::TAU / len as f64;
        let w_re = base_angle.cos();
        let w_im = base_angle.sin();
        // tw[0] = 1.0
        let mut tw_re = 1.0f64;
        let mut tw_im = 0.0f64;
        for _ in 0..half {
            unsafe { *table.get_unchecked_mut(idx) = make_complex(tw_re, tw_im) };
            idx += 1;
            // tw = tw * w_base (complex multiply in f64)
            let new_re = tw_re * w_re - tw_im * w_im;
            let new_im = tw_re * w_im + tw_im * w_re;
            tw_re = new_re;
            tw_im = new_im;
        }
        len <<= 1;
    }
    table
}

/// Build a contiguous per-stage forward twiddle table for f64.
///
/// Entry at `base + j` (where `base = 2^(s-1) − 1`) is `exp(−2πi·j / 2^s)`.
pub fn build_forward_twiddle_table_64(n: usize) -> Vec<Complex64> {
    build_twiddle_table_inner(n, -1.0, Complex64::new)
}

/// Build a contiguous per-stage inverse twiddle table (positive exponent sign).
pub fn build_inverse_twiddle_table_64(n: usize) -> Vec<Complex64> {
    build_twiddle_table_inner(n, 1.0, Complex64::new)
}

/// Build a contiguous per-stage forward twiddle table for f32.
///
/// Twiddles computed in f64 for accuracy, then cast to f32.
pub fn build_forward_twiddle_table_32(n: usize) -> Vec<Complex32> {
    build_twiddle_table_inner(n, -1.0, |re, im| Complex32::new(re as f32, im as f32))
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
    (0..=m)
        .map(|k| {
            let a = -std::f64::consts::TAU * k as f64 / n as f64;
            Complex64::new(a.cos(), a.sin())
        })
        .collect()
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
    forward_inplace_64_with_twiddles(&mut output[..m], &fft_twiddles[..m - 1]);

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
    inverse_inplace_64_with_twiddles(scratch, &fft_twiddles[..m - 1]);

    // Unpack: z[k] = x[2k] + i*x[2k+1]
    for k in 0..m {
        output[2 * k] = scratch[k].re;
        output[2 * k + 1] = scratch[k].im;
    }
}

/// Build a contiguous per-stage inverse twiddle table for f32.
pub fn build_inverse_twiddle_table_32(n: usize) -> Vec<Complex32> {
    build_twiddle_table_inner(n, 1.0, |re, im| Complex32::new(re as f32, im as f32))
}

// ── public API ────────────────────────────────────────────────────────────────────────────────

/// Radix-2 Stockham autosort forward FFT (unnormalized, f64).
///
/// `N` must be a power of 2.
///
/// ## Twiddle accuracy
///
/// Twiddles are evaluated directly from the N/2 unified table (see module doc,
/// Unified Twiddle Table theorem), bounding phase error to O(ε_mach) per stage.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    // Use on-stack table for small N to avoid heap allocation; precomputed contiguous
    // layout otherwise. The fallback unified-stride path is preserved for correctness
    // when called outside a plan context.
    let table = build_forward_twiddle_table_64(n);
    forward_inplace_64_with_twiddles(data, &table);
}

/// Forward FFT using a precomputed contiguous per-stage twiddle table.
///
/// ## Twiddle layout
///
/// `twiddles` must be the output of `build_forward_twiddle_table_64(n)`.
/// Stage s (group length `len = 2^s`) reads `half = len/2` entries from
/// `twiddles[base..]` where `base = 2^(s-1) - 1` (= total entries for stages 1..s-1).
/// Sequential access eliminates strided cache misses at all stage boundaries.
///
/// ## Correctness
///
/// This entry point uses the same DFT sign convention as `forward_inplace_64`
/// and delegates ordering to the Stockham autosort stage recurrence.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    with_stockham_scratch_64(n, |scratch| {
        f64::forward_with_scratch(data, scratch, twiddles);
    });
}

/// Inverse FFT using a precomputed contiguous per-stage inverse twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_64(n)`.
/// Result is unnormalized (no 1/N factor).
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    with_stockham_scratch_64(n, |scratch| {
        f64::inverse_unnorm_with_scratch(data, scratch, twiddles);
    });
}

/// Normalized inverse FFT using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_64(n)`.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    with_stockham_scratch_64(n, |scratch| {
        f64::inverse_with_scratch(data, scratch, twiddles);
    });
}

/// Forward FFT (f32) using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_forward_twiddle_table_32(n)`.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    with_stockham_scratch_32(n, |scratch| {
        f32::forward_with_scratch(data, scratch, twiddles);
    });
}

/// Inverse FFT (f32, unnormalized) using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_32(n)`.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    with_stockham_scratch_32(n, |scratch| {
        f32::inverse_unnorm_with_scratch(data, scratch, twiddles);
    });
}

/// Normalized inverse FFT using a precomputed contiguous per-stage twiddle table.
///
/// `twiddles` must be the output of `build_inverse_twiddle_table_32(n)`.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    with_stockham_scratch_32(n, |scratch| {
        f32::inverse_with_scratch(data, scratch, twiddles);
    });
}

/// Radix-2 Stockham autosort inverse FFT (unnormalized, f64).
///
/// Produces `Σ X[k] exp(+2πi·k·n/N)` without the 1/N normalization.
/// `N` must be a power of 2. Delegates to the precomputed-twiddle path for
/// zero per-call heap allocation.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_64(n);
    inverse_inplace_unnorm_64_with_twiddles(data, &table);
}

/// Radix-2 Stockham autosort inverse FFT (normalized by 1/N, f64).
///
/// `N` must be a power of 2.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_64(n);
    // Use fused-final-stage normalization to avoid a separate O(N) scale pass.
    inverse_inplace_64_with_twiddles(data, &table);
}

/// Radix-2 Stockham autosort forward FFT (unnormalized, f32).
///
/// Twiddle factors are evaluated in f64 then cast to f32 to minimize phase error.
/// Delegates to the contiguous per-stage twiddle path for cache efficiency.
/// `N` must be a power of 2.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_forward_twiddle_table_32(n);
    forward_inplace_32_with_twiddles(data, &table);
}

/// Radix-2 Stockham autosort inverse FFT (unnormalized, f32).
///
/// `N` must be a power of 2. Delegates to the contiguous per-stage twiddle path.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_32(n);
    inverse_inplace_unnorm_32_with_twiddles(data, &table);
}

/// Radix-2 Stockham autosort inverse FFT (normalized by 1/N, f32).
///
/// `N` must be a power of 2.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "radix-2 requires power-of-2 length");
    let table = build_inverse_twiddle_table_32(n);
    // Use fused-final-stage normalization to avoid a separate O(N) scale pass.
    inverse_inplace_32_with_twiddles(data, &table);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err_64(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f64, f64::max)
    }

    fn max_abs_err_32(a: &[Complex32], b: &[Complex32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f32, f32::max)
    }

    /// X[0] = x[0]+x[1] = 3, X[1] = x[0]-x[1] = -1.
    #[test]
    fn two_point_forward_matches_analytical() {
        let mut data = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        forward_inplace_64(&mut data);
        assert!((data[0] - Complex64::new(3.0, 0.0)).norm() < 1e-14);
        assert!((data[1] - Complex64::new(-1.0, 0.0)).norm() < 1e-14);
    }

    /// Normalized inverse of [3,-1] with N=2 → [1, 2].
    #[test]
    fn two_point_inverse_normalized_recovers_input() {
        let mut data = vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)];
        inverse_inplace_64(&mut data);
        assert!((data[0] - Complex64::new(1.0, 0.0)).norm() < 1e-14);
        assert!((data[1] - Complex64::new(2.0, 0.0)).norm() < 1e-14);
    }

    /// Unnormalized inverse should be N times the normalized inverse.
    #[test]
    fn unnorm_inverse_differs_from_norm_by_factor_n() {
        let input = vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.5, 1.0),
            Complex64::new(0.25, -0.25),
            Complex64::new(0.75, 0.0),
        ];
        let n = input.len() as f64;
        let mut unnorm = input.clone();
        inverse_inplace_unnorm_64(&mut unnorm);
        let mut norm = input.clone();
        inverse_inplace_64(&mut norm);
        let err = max_abs_err_64(&unnorm, &norm.iter().map(|x| x * n).collect::<Vec<_>>());
        assert!(err < 1e-13, "unnorm / norm ratio failed: err={err}");
    }

    /// Forward then normalized inverse roundtrip for sizes 2, 4, 8, 16, 32, 1024.
    #[test]
    fn roundtrip_holds_for_power_of_two_sizes() {
        for &n in &[2usize, 4, 8, 16, 32, 1024] {
            let input: Vec<Complex64> = (0..n)
                .map(|k| {
                    let t = k as f64 / n as f64;
                    Complex64::new(
                        (std::f64::consts::TAU * 3.0 * t).sin()
                            + 0.5 * (std::f64::consts::TAU * 7.0 * t).cos(),
                        0.25 * (std::f64::consts::TAU * 2.0 * t).sin(),
                    )
                })
                .collect();
            let mut spectrum = input.clone();
            forward_inplace_64(&mut spectrum);
            inverse_inplace_64(&mut spectrum);
            let err = max_abs_err_64(&spectrum, &input);
            assert!(
                err < 1e-10,
                "roundtrip failed for n={n}: ℓ∞ error={err:.2e}"
            );
        }
    }

    /// N=1 is a no-op.
    #[test]
    fn single_element_is_identity() {
        let mut data = vec![Complex64::new(3.7, -1.2)];
        let original = data[0];
        forward_inplace_64(&mut data);
        assert_eq!(data[0], original);
        inverse_inplace_64(&mut data);
        assert_eq!(data[0], original);
    }

    /// f32 forward matches f64 forward within f32 precision.
    #[test]
    fn forward_32_matches_64_within_f32_precision() {
        let n = 16usize;
        let input64: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.31).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let input32: Vec<Complex32> = input64
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let mut out64 = input64.clone();
        forward_inplace_64(&mut out64);
        let mut out32 = input32.clone();
        forward_inplace_32(&mut out32);

        let err = max_abs_err_32(
            &out32,
            &out64
                .iter()
                .map(|c| Complex32::new(c.re as f32, c.im as f32))
                .collect::<Vec<_>>(),
        );
        assert!(
            err < 1e-4f32,
            "f32 vs f64 forward error too large: {err:.2e}"
        );
    }

    /// f32 roundtrip.
    #[test]
    fn roundtrip_32_holds_for_n_16() {
        let n = 16usize;
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.31).sin(), (k as f32 * 0.17).cos()))
            .collect();
        let mut data = input.clone();
        forward_inplace_32(&mut data);
        inverse_inplace_32(&mut data);
        let err = max_abs_err_32(&data, &input);
        assert!(err < 1e-5f32, "f32 roundtrip error={err:.2e}");
    }
}
