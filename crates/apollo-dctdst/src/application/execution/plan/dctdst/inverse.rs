use super::helpers::{leto_array1_from_slice, leto_view1_cow};
use super::DctDstPlan;
use crate::domain::contracts::error::{DctDstError, DctDstResult};
use crate::domain::metadata::kind::RealTransformKind;
use crate::infrastructure::kernel::direct::{dct1, dct2, dct3, dct4, dst1, dst2, dst3, dst4};
use crate::infrastructure::kernel::fast::{
    dct2_fast, dct3_fast, dst2_fast, dst3_fast, FAST_THRESHOLD,
};
use leto::{Array, Array2, Array3, MnemosyneStorage, Storage, StorageMut};

impl DctDstPlan {
    /// Compute the inverse of the given forward transform.
    ///
    /// DCT-III is the inverse of DCT-II (up to a 2/N scaling factor).
    /// DST-III is the inverse of DST-II (up to a 2/N scaling factor).
    /// DCT-I, DCT-IV, DST-I, and DST-IV are self-inverse; each is scaled by
    /// `1/(2(N−1))`, `2/N`, `1/(2(N+1))`, and `2/N` respectively.
    /// The result is scaled to recover the original signal.
    ///
    /// Returns `LengthMismatch` when the input slice length differs from the
    /// plan length.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn inverse(&self, signal: &[f64]) -> DctDstResult<Vec<f64>> {
        let mut output = vec![0.0_f64; self.len()];
        self.inverse_into(signal, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse transform over a Leto real-valued 1D view.
    pub fn inverse_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![0.0_f64; self.len()];
        self.inverse_into(&signal, &mut output)?;
        Ok(leto_array1_from_slice(&output))
    }

    /// Execute a separable 2D inverse transform over a square `N x N` field.
    ///
    /// Returns `LengthMismatch` unless `input.shape() == (N, N)`.
    pub fn inverse_2d(&self, input: &Array2<f64>) -> DctDstResult<Array2<f64>> {
        let [rows, cols] = input.shape();
        if rows != self.len() || cols != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array2::<f64>::zeros([rows, cols]);
        self.inverse_2d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 2D inverse transform over a Leto `N x N` view.
    pub fn inverse_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let input = input.to_contiguous();
        let n = self.len();
        let mut output = Array::<f64, MnemosyneStorage<f64>, 2>::zeros_mnemosyne([n, n]);
        self.inverse_2d_into(&input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 2D inverse transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are square
    /// `N x N` arrays matching the plan length `N`.
    pub fn inverse_2d_into<S: Storage<f64>, SO: StorageMut<f64>>(
        &self,
        input: &Array<f64, S, 2>,
        output: &mut Array<f64, SO, 2>,
    ) -> DctDstResult<()> {
        let n = self.len();
        if input.shape() != [n, n] || output.shape() != [n, n] {
            return Err(DctDstError::LengthMismatch);
        }

        let mut stage = Array2::<f64>::zeros([n, n]);
        let mut line_in = vec![0.0_f64; n];
        let mut line_out = vec![0.0_f64; n];

        for i in 0..n {
            for j in 0..n {
                line_in[j] = input[[i, j]];
            }
            self.inverse_into(&line_in, &mut line_out)?;
            for j in 0..n {
                stage[[i, j]] = line_out[j];
            }
        }

        for j in 0..n {
            for i in 0..n {
                line_in[i] = stage[[i, j]];
            }
            self.inverse_into(&line_in, &mut line_out)?;
            for i in 0..n {
                output[[i, j]] = line_out[i];
            }
        }

        Ok(())
    }

    /// Execute a separable 3D inverse transform over a cubic `N x N x N` field.
    ///
    /// Returns `LengthMismatch` unless `input.shape() == (N, N, N)`.
    pub fn inverse_3d(&self, input: &Array3<f64>) -> DctDstResult<Array3<f64>> {
        let dims = input.shape();
        if dims[0] != self.len() || dims[1] != self.len() || dims[2] != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array3::<f64>::zeros(dims);
        self.inverse_3d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 3D inverse transform over a Leto `N x N x N` view.
    pub fn inverse_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let input = input.to_contiguous();
        let n = self.len();
        let mut output = Array::<f64, MnemosyneStorage<f64>, 3>::zeros_mnemosyne([n, n, n]);
        self.inverse_3d_into(&input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 3D inverse transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are cubic
    /// `N x N x N` arrays matching the plan length `N`.
    pub fn inverse_3d_into<S: Storage<f64>, SO: StorageMut<f64>>(
        &self,
        input: &Array<f64, S, 3>,
        output: &mut Array<f64, SO, 3>,
    ) -> DctDstResult<()> {
        let n = self.len();
        if input.shape() != [n, n, n] || output.shape() != [n, n, n] {
            return Err(DctDstError::LengthMismatch);
        }

        let mut stage1 = Array3::<f64>::zeros([n, n, n]);
        let mut stage2 = Array3::<f64>::zeros([n, n, n]);
        let mut line_in = vec![0.0_f64; n];
        let mut line_out = vec![0.0_f64; n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    line_in[k] = input[[i, j, k]];
                }
                self.inverse_into(&line_in, &mut line_out)?;
                for k in 0..n {
                    stage1[[i, j, k]] = line_out[k];
                }
            }
        }

        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    line_in[j] = stage1[[i, j, k]];
                }
                self.inverse_into(&line_in, &mut line_out)?;
                for j in 0..n {
                    stage2[[i, j, k]] = line_out[j];
                }
            }
        }

        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    line_in[i] = stage2[[i, j, k]];
                }
                self.inverse_into(&line_in, &mut line_out)?;
                for i in 0..n {
                    output[[i, j, k]] = line_out[i];
                }
            }
        }

        Ok(())
    }

    /// Compute the inverse of the configured transform into caller-owned output.
    ///
    /// DCT-III is the inverse of DCT-II (up to a 2/N scaling factor).
    /// DST-III is the inverse of DST-II (up to a 2/N scaling factor).
    /// DCT-I, DCT-IV, DST-I, and DST-IV are self-inverse; each is scaled by
    /// `1/(2(N−1))`, `2/N`, `1/(2(N+1))`, and `2/N` respectively.
    /// The result is scaled to recover the original signal.
    ///
    /// Returns `LengthMismatch` when either slice length differs from the plan
    /// length.
    ///
    /// Dispatches to the O(N log N) FFT fast path for N ≥ 16, and to the
    /// direct O(N²) analytical kernel for N < 16.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn inverse_into(&self, signal: &[f64], output: &mut [f64]) -> DctDstResult<()> {
        if signal.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        if output.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let n = self.len();
        let mut raw = vec![0.0_f64; n];

        if n >= FAST_THRESHOLD {
            match self.kind() {
                RealTransformKind::DctII => dct3_fast(signal, &mut raw),
                RealTransformKind::DstII => dst3_fast(signal, &mut raw),
                RealTransformKind::DctIII => dct2_fast(signal, &mut raw),
                RealTransformKind::DstIII => dst2_fast(signal, &mut raw),
                RealTransformKind::DctI => dct1(signal, &mut raw),
                RealTransformKind::DctIV => dct4(signal, &mut raw),
                RealTransformKind::DstI => dst1(signal, &mut raw),
                RealTransformKind::DstIV => dst4(signal, &mut raw),
            }
        } else {
            match self.kind() {
                RealTransformKind::DctII => dct3(signal, &mut raw),
                RealTransformKind::DstII => dst3(signal, &mut raw),
                RealTransformKind::DctIII => dct2(signal, &mut raw),
                RealTransformKind::DstIII => dst2(signal, &mut raw),
                RealTransformKind::DctI => dct1(signal, &mut raw),
                RealTransformKind::DctIV => dct4(signal, &mut raw),
                RealTransformKind::DstI => dst1(signal, &mut raw),
                RealTransformKind::DstIV => dst4(signal, &mut raw),
            }
        }

        // Scale factor derived from the self-inverse identity of each transform kind:
        //   DCT-II/III, DST-II/III, DCT-IV, DST-IV: paired/self-inverse scale = 2/N
        //   DCT-I: C₁·C₁ = 2(N−1)·I  →  scale = 1/(2(N−1))
        //   DST-I: S₁·S₁ = 2(N+1)·I  →  scale = 1/(2(N+1))
        let scale = match self.kind() {
            RealTransformKind::DctI => 1.0 / (2.0 * (n - 1) as f64),
            RealTransformKind::DstI => 1.0 / (2.0 * (n + 1) as f64),
            _ => 2.0 / n as f64,
        };
        for (slot, value) in output.iter_mut().zip(raw.into_iter()) {
            *slot = value * scale;
        }
        Ok(())
    }
}
