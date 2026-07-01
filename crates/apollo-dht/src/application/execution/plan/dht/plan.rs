//! Reusable Discrete Hartley Transform plan.

use super::helpers::{
    FAST_SCRATCH, LANE_IN_SCRATCH, LANE_OUT_SCRATCH,
};
use super::typed::HartleyStorage;
use crate::domain::contracts::error::{DhtError, DhtResult};
use crate::domain::metadata::length::HartleyLength;
use crate::domain::spectrum::coefficients::HartleySpectrum;
use crate::infrastructure::kernel::direct::transform_real;
use crate::infrastructure::kernel::fast::dht_fast_with_scratch;
use apollo_fft::PrecisionProfile;
use leto::{Array, Array2, Array3, MnemosyneStorage, Storage, StorageMut};

const FAST_KERNEL_THRESHOLD: usize = 512;

/// Reusable 1D real-to-real DHT plan.
#[derive(Debug)]
pub struct DhtPlan {
    length: HartleyLength,
}

impl DhtPlan {
    fn forward_2d_impl<S: Storage<f64>, SO: StorageMut<f64>>(&self, input: &Array<f64, S, 2>, output: &mut Array<f64, SO, 2>) -> DhtResult<()> {
        let n = self.len();
        let [rows, cols] = input.shape();
        let [out_rows, out_cols] = output.shape();
        if rows != n || cols != n {
            return Err(DhtError::ShapeMismatch2d {
                expected: n,
                rows,
                cols,
            });
        }
        if out_rows != n || out_cols != n {
            return Err(DhtError::ShapeMismatch2d {
                expected: n,
                rows: out_rows,
                cols: out_cols,
            });
        }

        LANE_IN_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |lane_in| {
                LANE_OUT_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |lane_out| {
                        for r in 0..n {
                            for c in 0..n {
                                lane_in[c] = input[[r, c]];
                            }
                            self.forward_into(lane_in, lane_out)?;
                            for c in 0..n {
                                output[[r, c]] = lane_out[c];
                            }
                        }

                        for c in 0..n {
                            for r in 0..n {
                                lane_in[r] = output[[r, c]];
                            }
                            self.forward_into(lane_in, lane_out)?;
                            for r in 0..n {
                                output[[r, c]] = lane_out[r];
                            }
                        }

                        Ok(())
                    })
                })
            })
        })
    }

    fn forward_3d_impl<S: Storage<f64>, SO: StorageMut<f64>>(&self, input: &Array<f64, S, 3>, output: &mut Array<f64, SO, 3>) -> DhtResult<()> {
        let n = self.len();
        let [d0, d1, d2] = input.shape();
        let [o0, o1, o2] = output.shape();
        if d0 != n || d1 != n || d2 != n {
            return Err(DhtError::ShapeMismatch3d {
                expected: n,
                d0,
                d1,
                d2,
            });
        }
        if o0 != n || o1 != n || o2 != n {
            return Err(DhtError::ShapeMismatch3d {
                expected: n,
                d0: o0,
                d1: o1,
                d2: o2,
            });
        }

        LANE_IN_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |lane_in| {
                LANE_OUT_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |lane_out| {
                        for j in 0..n {
                            for k in 0..n {
                                for i in 0..n {
                                    lane_in[i] = input[[i, j, k]];
                                }
                                self.forward_into(lane_in, lane_out)?;
                                for i in 0..n {
                                    output[[i, j, k]] = lane_out[i];
                                }
                            }
                        }

                        for i in 0..n {
                            for k in 0..n {
                                for j in 0..n {
                                    lane_in[j] = output[[i, j, k]];
                                }
                                self.forward_into(lane_in, lane_out)?;
                                for j in 0..n {
                                    output[[i, j, k]] = lane_out[j];
                                }
                            }
                        }

                        for i in 0..n {
                            for j in 0..n {
                                for k in 0..n {
                                    lane_in[k] = output[[i, j, k]];
                                }
                                self.forward_into(lane_in, lane_out)?;
                                for k in 0..n {
                                    output[[i, j, k]] = lane_out[k];
                                }
                            }
                        }

                        Ok(())
                    })
                })
            })
        })
    }

    /// Create a DHT plan for a non-empty signal length.
    pub fn new(len: usize) -> DhtResult<Self> {
        let length = HartleyLength::new(len)?;
        Ok(Self { length })
    }

    /// Return validated transform length.
    #[must_use]
    pub const fn length(&self) -> HartleyLength {
        self.length
    }

    /// Return transform length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length.get()
    }

    /// Return true when transform length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length.is_empty()
    }

    /// Execute the unnormalized forward DHT.
    pub fn forward(&self, signal: &[f64]) -> DhtResult<HartleySpectrum> {
        let mut output = vec![0.0; self.len()];
        self.forward_into(signal, &mut output)?;
        Ok(HartleySpectrum::new(output))
    }

    /// Execute the unnormalized forward DHT into a zero-allocation buffer.
    pub fn forward_into(&self, signal: &[f64], output: &mut [f64]) -> DhtResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(DhtError::LengthMismatch);
        }
        if self.len() >= FAST_KERNEL_THRESHOLD {
            FAST_SCRATCH.with(|pool| {
                pool.with_scratch(self.len(), |scratch| {
                    dht_fast_with_scratch(signal, output, scratch);
                });
            });
            Ok(())
        } else {
            transform_real(signal, output)
        }
    }

    /// Execute the inverse DHT by reusing the forward kernel and applying `1 / N`.
    pub fn inverse(&self, spectrum: &HartleySpectrum) -> DhtResult<Vec<f64>> {
        let mut output = vec![0.0; self.len()];
        self.inverse_into(spectrum.values(), &mut output)?;
        Ok(output)
    }

    /// Execute the inverse DHT securely into a zero-allocation buffer.
    pub fn inverse_into(&self, spectrum: &[f64], output: &mut [f64]) -> DhtResult<()> {
        if spectrum.len() != self.len() || output.len() != self.len() {
            return Err(DhtError::LengthMismatch);
        }
        self.forward_into(spectrum, output)?;
        let scale = 1.0 / self.len() as f64;
        output.iter_mut().for_each(|value| *value *= scale);
        Ok(())
    }

    /// Apply one raw unnormalized DHT pass.
    pub fn transform_unscaled(&self, input: &[f64]) -> DhtResult<Vec<f64>> {
        let mut output = vec![0.0; self.len()];
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the unnormalized separable 2D forward DHT on an N×N array.
    pub fn forward_2d(&self, input: &Array2<f64>) -> DhtResult<Array2<f64>> {
        let n = self.len();
        let mut result = Array2::<f64>::zeros([n, n]);
        self.forward_2d_impl(input, &mut result)?;
        Ok(result)
    }

    /// Execute the unnormalized separable 2D forward DHT on a Leto N×N view.
    pub fn forward_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let n = self.len();
        let input = input.as_array();
        let mut output = Array::<f64, MnemosyneStorage<f64>, 2>::zeros_mnemosyne([n, n]);
        self.forward_2d_impl(&input, &mut output)?;
        Ok(output)
    }

    /// Execute the unnormalized separable 2D forward DHT into a caller-owned buffer.
    pub fn forward_2d_into(&self, input: &Array2<f64>, output: &mut Array2<f64>) -> DhtResult<()> {
        self.forward_2d_impl(input, output)
    }

    /// Execute the normalized separable 2D inverse DHT on an N×N spectrum.
    pub fn inverse_2d(&self, input: &Array2<f64>) -> DhtResult<Array2<f64>> {
        let n = self.len();
        let mut result = Array2::<f64>::zeros([n, n]);
        self.forward_2d_impl(input, &mut result)?;
        let scale = 1.0 / (n * n) as f64;
        result.mapv_inplace(|v| v * scale);
        Ok(result)
    }

    /// Execute the normalized separable 2D inverse DHT on a Leto N×N spectrum.
    pub fn inverse_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let n = self.len();
        let input = input.as_array();
        let mut output = Array::<f64, MnemosyneStorage<f64>, 2>::zeros_mnemosyne([n, n]);
        self.forward_2d_impl(&input, &mut output)?;
        let scale = 1.0 / (n * n) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(output)
    }

    /// Execute the normalized separable 2D inverse DHT into a caller-owned buffer.
    pub fn inverse_2d_into(&self, input: &Array2<f64>, output: &mut Array2<f64>) -> DhtResult<()> {
        self.forward_2d_impl(input, output)?;
        let scale = 1.0 / (self.len() * self.len()) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(())
    }

    /// Execute the unnormalized separable 3D forward DHT on an N×N×N array.
    pub fn forward_3d(&self, input: &Array3<f64>) -> DhtResult<Array3<f64>> {
        let n = self.len();
        let mut result = Array3::<f64>::zeros([n, n, n]);
        self.forward_3d_impl(input, &mut result)?;
        Ok(result)
    }

    /// Execute the unnormalized separable 3D forward DHT on a Leto N×N×N view.
    pub fn forward_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let n = self.len();
        let input = input.as_array();
        let mut output = Array::<f64, MnemosyneStorage<f64>, 3>::zeros_mnemosyne([n, n, n]);
        self.forward_3d_impl(&input, &mut output)?;
        Ok(output)
    }

    /// Execute the unnormalized separable 3D forward DHT into a caller-owned buffer.
    pub fn forward_3d_into(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> DhtResult<()> {
        self.forward_3d_impl(input, output)
    }

    /// Execute the normalized separable 3D inverse DHT on an N×N×N spectrum.
    pub fn inverse_3d(&self, input: &Array3<f64>) -> DhtResult<Array3<f64>> {
        let n = self.len();
        let mut result = Array3::<f64>::zeros([n, n, n]);
        self.forward_3d_impl(input, &mut result)?;
        let scale = 1.0 / (n * n * n) as f64;
        result.mapv_inplace(|v| v * scale);
        Ok(result)
    }

    /// Execute the normalized separable 3D inverse DHT on a Leto N×N×N spectrum.
    pub fn inverse_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let n = self.len();
        let input = input.as_array();
        let mut output = Array::<f64, MnemosyneStorage<f64>, 3>::zeros_mnemosyne([n, n, n]);
        self.forward_3d_impl(&input, &mut output)?;
        let scale = 1.0 / (n * n * n) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(output)
    }

    /// Execute the normalized separable 3D inverse DHT into a caller-owned buffer.
    pub fn inverse_3d_into(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> DhtResult<()> {
        self.forward_3d_impl(input, output)?;
        let n = self.len();
        let scale = 1.0 / (n * n * n) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(())
    }

    /// Execute the unnormalized DHT for `f64`, `f32`, or mixed `f16` storage.
    pub fn forward_typed_into<T: HartleyStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        T::forward_into(self, signal, output, profile)
    }

    /// Execute the normalized inverse DHT for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: HartleyStorage>(
        &self,
        spectrum: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        T::inverse_into(self, spectrum, output, profile)
    }
}
