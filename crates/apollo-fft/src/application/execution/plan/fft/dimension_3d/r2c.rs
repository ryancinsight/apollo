use super::FftPlan3D;
use super::RAYON_THRESHOLD;
use crate::application::execution::plan::fft::workspace::uninit_copy_vec;
use ndarray::Array3;
use num_complex::Complex64;
use rayon::prelude::*;

impl FftPlan3D {
    /// Forward real-to-complex 3D transform into caller-owned half-spectrum buffer.
    pub fn forward_r2c_into(&self, input: &Array3<f64>, output: &mut Array3<Complex64>) {
        self.check_real_shape(input.dim(), "r2c forward input");
        self.check_half_complex_shape(output.dim(), "r2c forward output");

        let nz = self.nz;
        let nz_c = self.nz_c;
        let m = nz / 2;

        // -- Step 1: Z-axis R2C pass ------------------------------------------
        //
        // nz == 1: trivial - cast each real sample to complex with zero imaginary
        // part (DC-only spectrum). No sub-FFT or Cooley-Tukey extraction needed.
        // Falls through to x/y FFT passes below so the full 3D transform is
        // computed even when nz == 1 (e.g. purely 1D or 2D grids).
        //
        // nz > 1: split-radix r2c - processes each (i,j) row independently via
        // rayon par_chunks zip. Pack m complex values, FFT of length m,
        // Cooley-Tukey extraction.
        if nz == 1 {
            ndarray::Zip::from(&mut *output)
                .and(input)
                .for_each(|out, &v| {
                    *out = Complex64::new(v, 0.0);
                });
        } else {
            let in_sl = input
                .as_slice_memory_order()
                .expect("r2c input must be contiguous");
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("r2c output must be contiguous");
            let n_rows = self.nx * self.ny;
            let large = n_rows > RAYON_THRESHOLD / nz_c.max(1);
            if large {
                in_sl
                    .par_chunks(nz)
                    .zip(out_sl.par_chunks_mut(nz_c))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_forward_row_64(in_row, out_row, m);
                    });
            } else {
                in_sl
                    .chunks(nz)
                    .zip(out_sl.chunks_mut(nz_c))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_forward_row_64(in_row, out_row, m);
                    });
            }
        }

        // -- Step 2: Y-axis complex FFT on (nx, ny, nz_c) data ---------------
        // r2c_axis1_pass_64 returns trivially when ny == 1.
        {
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("r2c output must be contiguous");
            self.r2c_axis1_pass_64(out_sl, true);
        }

        // -- Step 3: X-axis complex FFT on (nx, ny, nz_c) data ---------------
        // r2c_axis0_pass_64 returns trivially when nx == 1.
        {
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("r2c output must be contiguous");
            self.r2c_axis0_pass_64(out_sl, true);
        }
    }

    /// Inverse complex-to-real 3D transform into caller-owned real buffer.
    ///
    /// `scratch` must have shape `(nx, ny, nz_c)` and is overwritten.
    pub fn inverse_c2r_into(
        &self,
        input: &Array3<Complex64>,
        output: &mut Array3<f64>,
        scratch: &mut Array3<Complex64>,
    ) {
        self.check_half_complex_shape(input.dim(), "c2r inverse input");
        self.check_real_shape(output.dim(), "c2r inverse output");
        self.check_half_complex_shape(scratch.dim(), "c2r inverse scratch");
        scratch.assign(input);
        self.inverse_c2r_into_with_scratch(scratch, output);
    }

    /// Inner c2r using a pre-filled mutable scratch buffer.
    pub(super) fn inverse_c2r_into_with_scratch(
        &self,
        scratch: &mut Array3<Complex64>,
        output: &mut Array3<f64>,
    ) {
        let nz = self.nz;
        let nz_c = self.nz_c;
        let m = nz / 2;

        // -- Step 1: X-axis complex IFFT on (nx, ny, nz_c) data --------------
        // r2c_axis0_pass_64 returns trivially when nx == 1.
        // Applies normalization 1/nx.
        {
            let sc_sl = scratch
                .as_slice_memory_order_mut()
                .expect("c2r scratch must be contiguous");
            self.r2c_axis0_pass_64(sc_sl, false);
        }

        // -- Step 2: Y-axis complex IFFT on (nx, ny, nz_c) data --------------
        // r2c_axis1_pass_64 returns trivially when ny == 1.
        // Applies normalization 1/ny.
        {
            let sc_sl = scratch
                .as_slice_memory_order_mut()
                .expect("c2r scratch must be contiguous");
            self.r2c_axis1_pass_64(sc_sl, false);
        }

        // -- Step 3: Z-axis C2R pass ------------------------------------------
        //
        // nz == 1: trivial - extract the real part of each DC-bin complex sample.
        // Steps 1 and 2 above already applied 1/nx and 1/ny normalization, so no
        // additional factor is needed here (total = 1/(nx*ny*1) = 1/(nx*ny*nz)).
        //
        // nz > 1: split-radix c2r - inverse Cooley-Tukey extraction + length-m
        // IFFT (applies 1/(2m) = 1/nz normalization) -> total 1/(nx*ny*nz). QED
        if nz == 1 {
            ndarray::Zip::from(output)
                .and(&*scratch)
                .for_each(|out, &v| {
                    *out = v.re;
                });
        } else {
            let sc_sl = scratch
                .as_slice_memory_order_mut()
                .expect("c2r scratch must be contiguous");
            let out_sl = output
                .as_slice_memory_order_mut()
                .expect("c2r output must be contiguous");
            let n_rows = self.nx * self.ny;
            let large = n_rows > RAYON_THRESHOLD / nz_c.max(1);
            if large {
                sc_sl
                    .par_chunks_mut(nz_c)
                    .zip(out_sl.par_chunks_mut(nz))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_inverse_row_64(in_row, out_row, m);
                    });
            } else {
                sc_sl
                    .chunks_mut(nz_c)
                    .zip(out_sl.chunks_mut(nz))
                    .for_each(|(in_row, out_row)| {
                        self.r2c_z_inverse_row_64(in_row, out_row, m);
                    });
            }
        }
    }

    /// Forward real-to-complex 3D transform.
    ///
    /// # Mathematical Contract
    ///
    /// For real `x in R^{nx x ny x nz}`, computes the unique half-spectrum
    /// `X in C^{nx x ny x (nz/2+1)}`. The omitted conjugate-symmetric modes
    /// satisfy `X[kx,ky,kz] = X*[(-kx)%nx, (-ky)%ny, nz-kz]` for `kz > nz/2`.
    ///
    /// ## Algorithm - Separable R2C via Cooley-Tukey Split (Sorensen et al. 1987)
    ///
    /// **Z-axis (real -> half-complex)**: For each (i,j) row:
    /// 1. Pack real pairs: `h[k] = x[2k] + j*x[2k+1]` for `k = 0..m-1`, `m = nz/2`.
    /// 2. Apply length-m complex FFT: `H = DFT_m(h)`.
    /// 3. Extract half-spectrum via the split identity (Theorem below):
    ///    `X[k] = (H_k + H_mk*)/2 - j*W_k*(H_k - H_mk*)/2`
    ///    where `H_k = H[k mod m]`, `H_mk = conj(H[(m-k) mod m])`,
    ///    and `W_k = exp(-2pii*k/nz)` (precomputed in `r2c_twiddles_64`).
    ///
    /// **Y-axis and X-axis**: Standard complex FFT passes on the `(nx,ny,nz_c)` data.
    ///
    /// ## Theorem: Cooley-Tukey R2C Split
    ///
    /// For real `x[n]`, the N-point DFT splits as `X[k] = E[k] + W_N^k * O[k]`
    /// where `E[k]` and `O[k]` are M = N/2 point DFTs of even/odd samples.
    /// Forming `h[k] = x[2k] + j*x[2k+1]` gives `H[k] = E[k] + j*O[k]`.
    /// Hermitian symmetry of `E` and `O` (both DFTs of real sequences) gives
    /// `E[M-k] = E[k]*` and `O[M-k] = O[k]*`, hence `H[(M-k)%M] = E[k]* + j*O[k]*`.
    /// Therefore `E[k] = (H[k] + H[(M-k)%M]*)/2` and
    /// `O[k] = (H[k] - H[(M-k)%M]*)/(2j)`, yielding the split formula above. QED
    ///
    /// ## Normalization
    ///
    /// Forward: no normalization (unnormalized DFT). Inverse `inverse_c2r_into`
    /// normalizes by `1/(nx*ny*nz)`, matching FFTW convention.
    ///
    /// ## Correctness Invariant
    ///
    /// `inverse_c2r_into(forward_r2c_into(x), out, scratch)` recovers `x` with
    /// absolute error `< 1e-10` for f64 on 64³ grids.
    #[must_use]
    pub fn forward_r2c(&self, input: &Array3<f64>) -> Array3<Complex64> {
        let mut out = Array3::<Complex64>::from_shape_vec(
            (self.nx, self.ny, self.nz_c),
            uninit_copy_vec(self.nx * self.ny * self.nz_c),
        )
        .expect("uninit Complex64 r2c 3D buffer length must match plan shape");
        self.forward_r2c_into(input, &mut out);
        out
    }

    /// Inverse complex-to-real 3D transform.
    ///
    /// # Mathematical Contract
    ///
    /// For `X in C^{nx x ny x (nz/2+1)}` (the r2c half-spectrum), recovers
    /// `x in R^{nx x ny x nz}` via the conjugate-symmetric IDFT.
    ///
    /// ## Algorithm - Inverse Cooley-Tukey Split
    ///
    /// **X-axis and Y-axis**: Standard complex IFFT passes on `(nx,ny,nz_c)`.
    ///
    /// **Z-axis (half-complex -> real)**: For each (i,j) row:
    /// 1. Recover H[k] from X[0..m] using the inverse split formula:
    ///    `H[k] = (X[k] + conj(X[(m-k)%m]) + j*W_k* * (X[k] - conj(X[(m-k)%m]))) / 2`
    ///    where `W_k* = conj(exp(-2pii*k/nz)) = exp(+2pii*k/nz)`.
    /// 2. Apply normalized length-m IFFT: `h = IFFT_m(H)` (divides by m).
    /// 3. Extract: `x[2k] = Re(h[k])`, `x[2k+1] = Im(h[k])` for k=0..m-1.
    ///    The inverse split recovers the packed length-m spectrum exactly; the
    ///    normalized length-m IFFT then recovers the packed real-pair sequence.
    ///
    /// Combined with the x- and y-axis IFFT normalizations by `1/nx` and `1/ny`,
    /// the total normalization is `1/(nx*ny*nz)`. QED
    #[must_use]
    pub fn inverse_c2r(&self, input: &Array3<Complex64>) -> Array3<f64> {
        let mut out = Array3::<f64>::from_shape_vec(
            (self.nx, self.ny, self.nz),
            uninit_copy_vec(self.nx * self.ny * self.nz),
        )
        .expect("uninit f64 c2r 3D buffer length must match plan shape");
        let mut scratch = input.clone();
        self.inverse_c2r_into_with_scratch(&mut scratch, &mut out);
        out
    }
}
