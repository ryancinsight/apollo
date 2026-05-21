use super::FftPlan3D;
use crate::application::execution::kernel::mixed_radix::{
    forward_inplace_64_with_twiddles, inverse_inplace_64_with_twiddles,
};
use crate::application::execution::kernel::{fft_forward, fft_inverse};
use num_complex::Complex64;

impl FftPlan3D {
    /// Z-axis r2c forward for a single row: pack + sub-FFT + Cooley-Tukey extraction.
    ///
    /// `in_row` has `nz` f64 values; `out_row` receives `nz_c = m+1` Complex64 values.
    pub(super) fn r2c_z_forward_row_64(&self, in_row: &[f64], out_row: &mut [Complex64], m: usize) {
        // Stage 1: pack pairs into the caller-owned output row prefix.
        for k in 0..m {
            out_row[k] = Complex64::new(in_row[2 * k], in_row[2 * k + 1]);
        }

        // Stage 2: length-m complex FFT in-place.
        match &self.twiddle_zh_fwd_64 {
            Some(tw) => forward_inplace_64_with_twiddles(&mut out_row[..m], Some(tw.as_ref())),
            None => fft_forward(&mut out_row[..m]),
        }

        // Stage 3: Cooley-Tukey extraction - X[k] for k = 0..m (inclusive).
        // X[k] = (H_k + H_mk*) / 2 - j*W_k*(H_k - H_mk*) / 2
        // where H_k = h[k % m] and H_mk = conj(h[(m-k) % m]).
        // Process pairs simultaneously to avoid read-after-write aliasing.
        let j = Complex64::new(0.0, 1.0);

        let h0 = out_row[0];

        // k = 0: H_0 = h[0], H_m0 = conj(h[0]).
        {
            let hm0_conj = h0.conj(); // conj(h[(m-0)%m]) = conj(h[0])
            let wk = self.r2c_twiddles_64[0]; // W_0 = 1
            let sum = h0 + hm0_conj;
            let diff = h0 - hm0_conj;
            out_row[0] = sum * 0.5 - j * wk * diff * 0.5;
        }

        // k = m: H_km = h[0] (same slot), H_mk = conj(h[0]).
        {
            let hm_conj = h0.conj();
            let wk = self.r2c_twiddles_64[m]; // W_m = exp(-pii) = -1
            let sum = h0 + hm_conj;
            let diff = h0 - hm_conj;
            out_row[m] = sum * 0.5 - j * wk * diff * 0.5;
        }

        // k = 1..m/2 and symmetric counterpart m-k (processed as pairs).
        let k_max = m / 2 + usize::from(m % 2 == 1);
        for k in 1..k_max {
            let mk = m - k;
            let hk = out_row[k];
            let hmk = out_row[mk]; // h[(m-k)]
            let wk = self.r2c_twiddles_64[k];
            let wmk = self.r2c_twiddles_64[mk];

            // X[k]
            let hmk_conj = hmk.conj();
            let sum_k = hk + hmk_conj;
            let diff_k = hk - hmk_conj;
            out_row[k] = sum_k * 0.5 - j * wk * diff_k * 0.5;

            if k != mk {
                // X[m-k] - distinct slot, process as pair.
                let hk_conj = hk.conj();
                let sum_mk = hmk + hk_conj;
                let diff_mk = hmk - hk_conj;
                out_row[mk] = sum_mk * 0.5 - j * wmk * diff_mk * 0.5;
            }
        }

        // When m is even, k = m/2 is its own symmetric partner.
        if m.is_multiple_of(2) && m >= 2 {
            let k = m / 2;
            let hk = out_row[k];
            let wk = self.r2c_twiddles_64[k];
            let hk_conj = hk.conj();
            let sum_k = hk + hk_conj;
            let diff_k = hk - hk_conj;
            out_row[k] = sum_k * 0.5 - j * wk * diff_k * 0.5;
        }
    }

    /// Z-axis c2r inverse for a single row: inverse split + sub-IFFT + unpack.
    ///
    /// `in_row` has `nz_c = m+1` Complex64 values (the half-spectrum after x/y IFFTs).
    /// `out_row` receives `nz` f64 values.
    ///
    /// Normalization: each output value is divided by `nz` so that combined with
    /// the x-axis (/nx) and y-axis (/ny) IFFT normalizations, the total is `/(nx*ny*nz)`.
    pub(super) fn r2c_z_inverse_row_64(
        &self,
        in_row: &mut [Complex64],
        out_row: &mut [f64],
        m: usize,
    ) {
        // Stage 1: Recover H[k] for k = 0..m-1 from X[0..m] using the inverse split.
        //
        // Theorem (inverse split): from X[k] = (H_k + H_mk*)/2 - j*W_k*(H_k - H_mk*)/2
        // and X[m-k] = (H_mk + H_k*)/2 + j*conj(W_k)*(H_mk - H_k*)/2, solving gives:
        //   H[k] = (X[k] + conj(X[m-k]) + j*conj(W_k)*(X[k] - conj(X[m-k]))) / 2
        // for k = 0..m-1, where conj(W_k) = exp(+2pii*k/nz) and X[m-k] is at index
        // m-k in the nz_c = m+1 half-spectrum (k=0 uses X[m], the Nyquist bin). QED

        let j = Complex64::new(0.0, 1.0);

        // k = 0: conj(X[m-0]) = conj(X[m]).
        //
        // X[m] is the Nyquist bin stored at index m in the nz_c = m+1 half-spectrum.
        // For real input X[m] is real, so conj(X[m]) = X[m].
        // conj(W_0) = conj(1) = 1.
        //
        // Derivation: X[0] = Re{H[0]} + Im{H[0]}, X[m] = Re{H[0]} - Im{H[0]}.
        // H[0] = (X[0] + X[m])/2 + j*(X[0] - X[m])/2, which matches the general
        // formula with xmk_conj = conj(X[m]).
        {
            let xk = in_row[0];
            let xmk_conj = in_row[m].conj(); // X[m] is at index m in the half-spectrum
            let w_conj = self.r2c_twiddles_64[0].conj(); // = 1
            in_row[0] = (xk + xmk_conj + j * w_conj * (xk - xmk_conj)) * 0.5;
        }

        // k = 1..m-1: symmetric pairs.
        let k_max = m / 2 + usize::from(m % 2 == 1);
        for k in 1..k_max {
            let mk = m - k;
            let xk = in_row[k];
            let xmk_conj = in_row[mk].conj();
            let w_conj = self.r2c_twiddles_64[k].conj(); // exp(+2pii*k/nz)

            let h_k = (xk + xmk_conj + j * w_conj * (xk - xmk_conj)) * 0.5;

            if k != mk {
                let xmk = in_row[mk];
                let xk_conj = in_row[k].conj();
                let wm_conj = self.r2c_twiddles_64[mk].conj();
                in_row[mk] = (xmk + xk_conj + j * wm_conj * (xmk - xk_conj)) * 0.5;
            }
            in_row[k] = h_k;
        }

        // k = m/2 singleton (even m).
        if m.is_multiple_of(2) && m >= 2 {
            let k = m / 2;
            let xk = in_row[k];
            let xmk_conj = in_row[k].conj(); // self-conjugate slot
            let w_conj = self.r2c_twiddles_64[k].conj();
            in_row[k] = (xk + xmk_conj + j * w_conj * (xk - xmk_conj)) * 0.5;
        }

        // Stage 2: normalized IFFT of length m in-place.
        match &self.twiddle_zh_inv_64 {
            Some(tw) => inverse_inplace_64_with_twiddles(&mut in_row[..m], Some(tw.as_ref())),
            None => fft_inverse(&mut in_row[..m]),
        }

        // Stage 3: unpack h[k] -> out_row[2k], out_row[2k+1].
        //
        // No additional scaling factor. The normalized IFFT_m satisfies
        //   IFFT_m(FFT_m(h)) = h
        // so stage 2 recovers h exactly. The forward z r2c packs
        // h[n] = x[2n] + j*x[2n+1] and applies an unnormalized FFT_m;
        // stage 1 recovers H_true, stage 2 recovers h_true; unpacking gives
        // x back directly. No residual normalization factor arises here: the
        // z-axis forward/inverse pair is its own identity (H_true recovered via
        // the inverse split, IFFT_m cancels FFT_m). The x- and y-axis normalized
        // IFFTs in the outer c2r caller cancel their respective DFTs identically.
        for k in 0..m {
            out_row[2 * k] = in_row[k].re;
            out_row[2 * k + 1] = in_row[k].im;
        }
    }
}
