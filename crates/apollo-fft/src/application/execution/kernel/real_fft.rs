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
//! 2. **Real-FFT pack/unpack** (`RealFft::forward_real_inplace`,
//!    `RealFft::inverse_real_inplace`): implement the Cooley-Tukey real-FFT trick.
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

use super::mixed_radix::dispatch_inplace;

pub(crate) trait RealFft:
    crate::application::execution::kernel::mixed_radix::MixedRadixScalar
{
    fn build_forward_twiddle_table(n: usize) -> Vec<Self::Complex>;
    fn build_inverse_twiddle_table(n: usize) -> Vec<Self::Complex>;
    fn build_real_fwd_post_twiddles(n: usize) -> Vec<Self::Complex>;
    fn forward_real_inplace(
        input: &[Self],
        output: &mut [Self::Complex],
        fft_twiddles: &[Self::Complex],
        post_twiddles: &[Self::Complex],
    );
    fn inverse_real_inplace(
        input: &[Self::Complex],
        output: &mut [Self],
        scratch: &mut [Self::Complex],
        fft_twiddles: &[Self::Complex],
        post_twiddles: &[Self::Complex],
    );
}

impl RealFft for f64 {
    #[inline]
    fn build_forward_twiddle_table(n: usize) -> Vec<num_complex::Complex64> {
        super::twiddle_table::build_twiddle_table(n, -1.0)
    }

    #[inline]
    fn build_inverse_twiddle_table(n: usize) -> Vec<num_complex::Complex64> {
        super::twiddle_table::build_twiddle_table(n, 1.0)
    }

    fn build_real_fwd_post_twiddles(n: usize) -> Vec<num_complex::Complex64> {
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
            *slot = num_complex::Complex64::new(a.cos(), a.sin());
        }
        twiddles
    }

    fn forward_real_inplace(
        input: &[f64],
        output: &mut [num_complex::Complex64],
        fft_twiddles: &[num_complex::Complex64],
        post_twiddles: &[num_complex::Complex64],
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

        for k in 0..m {
            output[k] = num_complex::Complex64::new(input[2 * k], input[2 * k + 1]);
        }

        dispatch_inplace::<f64, false, false>(&mut output[..m], Some(&fft_twiddles[..m - 1]));

        let z0 = output[0];

        let pair_end = (m + 1) / 2;
        for l in 1..pair_end {
            let ml = m - l;
            let zl = output[l];
            let zml = output[ml];
            let a = (zl + zml.conj()) * 0.5;
            let b = (zl - zml.conj()) * num_complex::Complex64::new(0.0, -0.5);
            let a2 = (zml + zl.conj()) * 0.5;
            let b2 = (zml - zl.conj()) * num_complex::Complex64::new(0.0, -0.5);
            let wl = post_twiddles[l];
            let xl = a + wl * b;
            let xml = a2 - wl.conj() * b2;
            output[l] = xl;
            output[ml] = xml;
            output[n - l] = xl.conj();
            output[n - ml] = xml.conj();
        }

        if m % 2 == 0 {
            let mid = m / 2;
            let zmid = output[mid];
            output[mid] = zmid.conj();
            output[n - mid] = zmid;
        }

        output[0] = num_complex::Complex64::new(z0.re + z0.im, 0.0);
        output[m] = num_complex::Complex64::new(z0.re - z0.im, 0.0);
    }

    fn inverse_real_inplace(
        input: &[num_complex::Complex64],
        output: &mut [f64],
        scratch: &mut [num_complex::Complex64],
        fft_twiddles: &[num_complex::Complex64],
        post_twiddles: &[num_complex::Complex64],
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

        scratch[0] = num_complex::Complex64::new(
            (input[0].re + input[m].re) * 0.5,
            (input[0].re - input[m].re) * 0.5,
        );

        let half_m = m / 2;
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
            let i_conj_wk = num_complex::Complex64::new(wk.im, wk.re);
            let i_conj_wmk = num_complex::Complex64::new(wk.im, -wk.re);
            scratch[k] = (sum_k + i_conj_wk * diff_k) * 0.5;
            scratch[mk] = (sum_mk + i_conj_wmk * diff_mk) * 0.5;
        }
        scratch[half_m] = input[half_m].conj();

        dispatch_inplace::<f64, true, true>(scratch, Some(&fft_twiddles[..m - 1]));

        for k in 0..m {
            output[2 * k] = scratch[k].re;
            output[2 * k + 1] = scratch[k].im;
        }
    }
}

impl RealFft for f32 {
    #[inline]
    fn build_forward_twiddle_table(n: usize) -> Vec<num_complex::Complex32> {
        super::twiddle_table::build_twiddle_table(n, -1.0)
    }

    #[inline]
    fn build_inverse_twiddle_table(n: usize) -> Vec<num_complex::Complex32> {
        super::twiddle_table::build_twiddle_table(n, 1.0)
    }

    fn build_real_fwd_post_twiddles(n: usize) -> Vec<num_complex::Complex32> {
        debug_assert!(n.is_power_of_two() && n >= 4);
        let m = n >> 1;
        let len = m + 1;
        let mut twiddles = Vec::with_capacity(len);
        #[allow(clippy::uninit_vec)]
        unsafe {
            twiddles.set_len(len)
        };
        for (k, slot) in twiddles.iter_mut().enumerate() {
            let a = -std::f32::consts::TAU * k as f32 / n as f32;
            *slot = num_complex::Complex32::new(a.cos(), a.sin());
        }
        twiddles
    }

    fn forward_real_inplace(
        input: &[f32],
        output: &mut [num_complex::Complex32],
        fft_twiddles: &[num_complex::Complex32],
        post_twiddles: &[num_complex::Complex32],
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

        for k in 0..m {
            output[k] = num_complex::Complex32::new(input[2 * k], input[2 * k + 1]);
        }

        dispatch_inplace::<f32, false, false>(&mut output[..m], Some(&fft_twiddles[..m - 1]));

        let z0 = output[0];

        let pair_end = (m + 1) / 2;
        for l in 1..pair_end {
            let ml = m - l;
            let zl = output[l];
            let zml = output[ml];
            let a = (zl + zml.conj()) * 0.5;
            let b = (zl - zml.conj()) * num_complex::Complex32::new(0.0, -0.5);
            let a2 = (zml + zl.conj()) * 0.5;
            let b2 = (zml - zl.conj()) * num_complex::Complex32::new(0.0, -0.5);
            let wl = post_twiddles[l];
            let xl = a + wl * b;
            let xml = a2 - wl.conj() * b2;
            output[l] = xl;
            output[ml] = xml;
            output[n - l] = xl.conj();
            output[n - ml] = xml.conj();
        }

        if m % 2 == 0 {
            let mid = m / 2;
            let zmid = output[mid];
            output[mid] = zmid.conj();
            output[n - mid] = zmid;
        }

        output[0] = num_complex::Complex32::new(z0.re + z0.im, 0.0);
        output[m] = num_complex::Complex32::new(z0.re - z0.im, 0.0);
    }

    fn inverse_real_inplace(
        input: &[num_complex::Complex32],
        output: &mut [f32],
        scratch: &mut [num_complex::Complex32],
        fft_twiddles: &[num_complex::Complex32],
        post_twiddles: &[num_complex::Complex32],
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

        scratch[0] = num_complex::Complex32::new(
            (input[0].re + input[m].re) * 0.5,
            (input[0].re - input[m].re) * 0.5,
        );

        let half_m = m / 2;
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
            let i_conj_wk = num_complex::Complex32::new(wk.im, wk.re);
            let i_conj_wmk = num_complex::Complex32::new(wk.im, -wk.re);
            scratch[k] = (sum_k + i_conj_wk * diff_k) * 0.5;
            scratch[mk] = (sum_mk + i_conj_wmk * diff_mk) * 0.5;
        }
        scratch[half_m] = input[half_m].conj();

        dispatch_inplace::<f32, true, true>(scratch, Some(&fft_twiddles[..m - 1]));

        for k in 0..m {
            output[2 * k] = scratch[k].re;
            output[2 * k + 1] = scratch[k].im;
        }
    }
}
