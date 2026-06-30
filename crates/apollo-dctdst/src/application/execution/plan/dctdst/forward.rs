use super::helpers::{
    array2_from_leto_view, array3_from_leto_view, leto_array1_from_slice, leto_array2_from_dense,
    leto_array3_from_dense, leto_view1_cow,
};
use super::DctDstPlan;
use crate::domain::contracts::error::{DctDstError, DctDstResult};
use crate::domain::metadata::kind::RealTransformKind;
use crate::infrastructure::kernel::direct::{dct1, dct2, dct3, dct4, dst1, dst2, dst3, dst4};
use crate::infrastructure::kernel::fast::{
    dct2_fast, dct3_fast, dst2_fast, dst3_fast, FAST_THRESHOLD,
};
use leto::{Array2, Array3};

impl DctDstPlan {
    /// Execute the forward transform and return allocated coefficients.
    ///
    /// Returns `LengthMismatch` when the input slice length differs from the
    /// plan length.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn forward(&self, signal: &[f64]) -> DctDstResult<Vec<f64>> {
        let mut output = vec![0.0_f64; self.len()];
        self.forward_into(signal, &mut output)?;
        Ok(output)
    }

    /// Execute the forward transform over a Leto real-valued 1D view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical
    /// order before entering the canonical slice execution path.
    pub fn forward_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![0.0_f64; self.len()];
        self.forward_into(&signal, &mut output)?;
        Ok(leto_array1_from_slice(&output))
    }

    /// Execute a separable 2D forward transform over a square `N x N` field.
    ///
    /// The plan length `N` is applied to both axes; each row is transformed
    /// first, then each column.
    ///
    /// Returns `LengthMismatch` unless `input.shape() == (N, N)`.
    pub fn forward_2d(&self, input: &Array2<f64>) -> DctDstResult<Array2<f64>> {
        let [rows, cols] = input.shape();
        if rows != self.len() || cols != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array2::<f64>::zeros([rows, cols]);
        self.forward_2d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 2D forward transform over a Leto `N x N` view.
    pub fn forward_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let input = array2_from_leto_view(input);
        let output = self.forward_2d(&input)?;
        Ok(leto_array2_from_dense(&output))
    }

    /// Execute a separable 2D forward transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are square
    /// `N x N` arrays matching the plan length `N`.
    pub fn forward_2d_into(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
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
            self.forward_into(&line_in, &mut line_out)?;
            for j in 0..n {
                stage[[i, j]] = line_out[j];
            }
        }

        for j in 0..n {
            for i in 0..n {
                line_in[i] = stage[[i, j]];
            }
            self.forward_into(&line_in, &mut line_out)?;
            for i in 0..n {
                output[[i, j]] = line_out[i];
            }
        }

        Ok(())
    }

    /// Execute a separable 3D forward transform over a cubic `N x N x N` field.
    ///
    /// The plan length `N` is applied to all three axes in z, then y, then x
    /// order.
    ///
    /// Returns `LengthMismatch` unless `input.shape() == (N, N, N)`.
    pub fn forward_3d(&self, input: &Array3<f64>) -> DctDstResult<Array3<f64>> {
        let dims = input.shape();
        if dims[0] != self.len() || dims[1] != self.len() || dims[2] != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array3::<f64>::zeros(dims);
        self.forward_3d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 3D forward transform over a Leto `N x N x N` view.
    pub fn forward_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let input = array3_from_leto_view(input);
        let output = self.forward_3d(&input)?;
        Ok(leto_array3_from_dense(&output))
    }

    /// Execute a separable 3D forward transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are cubic
    /// `N x N x N` arrays matching the plan length `N`.
    pub fn forward_3d_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
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
                self.forward_into(&line_in, &mut line_out)?;
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
                self.forward_into(&line_in, &mut line_out)?;
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
                self.forward_into(&line_in, &mut line_out)?;
                for i in 0..n {
                    output[[i, j, k]] = line_out[i];
                }
            }
        }

        Ok(())
    }

    /// Execute the forward transform into a caller-supplied buffer.
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
    pub fn forward_into(&self, signal: &[f64], output: &mut [f64]) -> DctDstResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }

        let n = self.len();
        if n >= FAST_THRESHOLD {
            match self.kind() {
                RealTransformKind::DctII => dct2_fast(signal, output),
                RealTransformKind::DctIII => dct3_fast(signal, output),
                RealTransformKind::DstII => dst2_fast(signal, output),
                RealTransformKind::DstIII => dst3_fast(signal, output),
                RealTransformKind::DctI => dct1(signal, output),
                RealTransformKind::DctIV => dct4(signal, output),
                RealTransformKind::DstI => dst1(signal, output),
                RealTransformKind::DstIV => dst4(signal, output),
            }
        } else {
            match self.kind() {
                RealTransformKind::DctII => dct2(signal, output),
                RealTransformKind::DctIII => dct3(signal, output),
                RealTransformKind::DstII => dst2(signal, output),
                RealTransformKind::DstIII => dst3(signal, output),
                RealTransformKind::DctI => dct1(signal, output),
                RealTransformKind::DctIV => dct4(signal, output),
                RealTransformKind::DstI => dst1(signal, output),
                RealTransformKind::DstIV => dst4(signal, output),
            }
        }

        Ok(())
    }
}
