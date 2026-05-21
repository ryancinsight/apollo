//! 3D Type-1 NUFFT executions and plans.
//!
//! This module implements a Kaiser-Bessel gridding NUFFT backed by Apollo's
//! in-repo FFT plan rather than external FFT crates.
//!
//! The exact direct transform serves as the mathematical reference surface,
//! while the fast path uses oversampled grid spreading followed by Apollo FFT
//! execution and deconvolution.
//!
//! # Mathematical contract
//!
//! Type-1 NUFFT maps non-uniform samples `(x_j, y_j, z_j, c_j)` to uniform
//! Fourier bins
//!
//! `f_{k_x,k_y,k_z} = Σ_j c_j exp(-2πi (k_x x_j / L_x + k_y y_j / L_y + k_z z_j / L_z))`
//!
//! The fast path approximates the direct sum by spreading to an oversampled
//! grid with a Kaiser-Bessel kernel and then applying separable Apollo FFT
//! passes on each axis.
//!
//! # Complexity
//!
//! The direct transform is `O(MN)` where `M = n_x n_y n_z` and `N` is the number
//! of samples. The fast transform is `O(N w^3 + m_x m_y m_z log(m_x m_y m_z))`
//! where `w` is the kernel width and `(m_x, m_y, m_z)` is the oversampled grid.
//!
//! # Failure modes
//!
//! - positions/value arrays must have equal length
//! - oversampling factor must satisfy `sigma >= 2`
//! - kernel width must satisfy `kernel_width >= 2`

use apollo_fft::{ApolloError, ApolloResult, FftPlan1D, PrecisionProfile, Shape1D};
use ndarray::{Array1, Array3};
use num_complex::Complex64;
use std::cmp::Ordering;
use std::f64::consts::PI;

use crate::application::execution::transform::dimension_1d::{
    validate_profile, write_typed_output, NufftComplexStorage,
};
use crate::domain::metadata::grid::UniformGrid3D;
use crate::infrastructure::kernel::kaiser_bessel::{
    axis_deconv, bucket_count, fft_signed_index, i0, kb_kernel,
};
use crate::DEFAULT_NUFFT_OVERSAMPLING;

fn supported_radix2_oversampled_len(n: usize, sigma: usize, kernel_width: usize) -> usize {
    let lower = n
        .checked_mul(sigma)
        .expect("NUFFT oversampled length overflow")
        .max(2 * kernel_width + 1);
    lower.next_power_of_two()
}

#[derive(Clone, Copy)]
struct IndexedPoint3D {
    x: f64,
    y: f64,
    z: f64,
    value: Complex64,
    bucket: usize,
}

#[derive(Clone, Copy)]
struct IndexedPosition3D {
    index: usize,
    x: f64,
    y: f64,
    z: f64,
    bucket: usize,
}

fn sort_points_3d(
    positions: &[(f64, f64, f64)],
    values: &[Complex64],
    grid: UniformGrid3D,
    oversampled_shape: (usize, usize, usize),
    kernel_width: usize,
) -> Vec<IndexedPoint3D> {
    let (lx, ly, lz) = grid.lengths();
    let bx = bucket_count(oversampled_shape.0, kernel_width);
    let by = bucket_count(oversampled_shape.1, kernel_width);
    let bz = bucket_count(oversampled_shape.2, kernel_width);
    let sx = bx as f64 / lx;
    let sy = by as f64 / ly;
    let sz = bz as f64 / lz;

    let mut indexed: Vec<_> = positions
        .iter()
        .zip(values.iter())
        .map(|(&(x, y, z), &value)| {
            let x_mod = x.rem_euclid(lx);
            let y_mod = y.rem_euclid(ly);
            let z_mod = z.rem_euclid(lz);
            let ix = ((x_mod * sx).floor() as usize).min(bx - 1);
            let iy = ((y_mod * sy).floor() as usize).min(by - 1);
            let iz = ((z_mod * sz).floor() as usize).min(bz - 1);
            IndexedPoint3D {
                x: x_mod,
                y: y_mod,
                z: z_mod,
                value,
                bucket: (ix * by + iy) * bz + iz,
            }
        })
        .collect();

    indexed.sort_unstable_by(|lhs, rhs| {
        lhs.bucket
            .cmp(&rhs.bucket)
            .then_with(|| lhs.x.partial_cmp(&rhs.x).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.y.partial_cmp(&rhs.y).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.z.partial_cmp(&rhs.z).unwrap_or(Ordering::Equal))
    });
    indexed
}

fn sort_positions_3d(
    positions: &[(f64, f64, f64)],
    grid: UniformGrid3D,
    oversampled_shape: (usize, usize, usize),
    kernel_width: usize,
) -> Vec<IndexedPosition3D> {
    let (lx, ly, lz) = grid.lengths();
    let bx = bucket_count(oversampled_shape.0, kernel_width);
    let by = bucket_count(oversampled_shape.1, kernel_width);
    let bz = bucket_count(oversampled_shape.2, kernel_width);
    let sx = bx as f64 / lx;
    let sy = by as f64 / ly;
    let sz = bz as f64 / lz;

    let mut indexed: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(index, &(x, y, z))| {
            let x_mod = x.rem_euclid(lx);
            let y_mod = y.rem_euclid(ly);
            let z_mod = z.rem_euclid(lz);
            let ix = ((x_mod * sx).floor() as usize).min(bx - 1);
            let iy = ((y_mod * sy).floor() as usize).min(by - 1);
            let iz = ((z_mod * sz).floor() as usize).min(bz - 1);
            IndexedPosition3D {
                index,
                x: x_mod,
                y: y_mod,
                z: z_mod,
                bucket: (ix * by + iy) * bz + iz,
            }
        })
        .collect();

    indexed.sort_unstable_by(|lhs, rhs| {
        lhs.bucket
            .cmp(&rhs.bucket)
            .then_with(|| lhs.x.partial_cmp(&rhs.x).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.y.partial_cmp(&rhs.y).unwrap_or(Ordering::Equal))
            .then_with(|| lhs.z.partial_cmp(&rhs.z).unwrap_or(Ordering::Equal))
    });
    indexed
}

/// Reusable 3D type-1 NUFFT plan using separable Kaiser-Bessel spreading.
pub struct NufftPlan3D {
    grid: UniformGrid3D,
    mx: usize,
    my: usize,
    mz: usize,
    w: usize,
    beta: f64,
    i0_beta: f64,
    deconv_x: ndarray::Array1<f64>,
    deconv_y: ndarray::Array1<f64>,
    deconv_z: ndarray::Array1<f64>,
    fft_x: FftPlan1D<f64>,
    fft_y: FftPlan1D<f64>,
    fft_z: FftPlan1D<f64>,
}

impl std::fmt::Debug for NufftPlan3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NufftPlan3D")
            .field("grid", &self.grid)
            .field("oversampled_shape", &(self.mx, self.my, self.mz))
            .field("kernel_width", &self.w)
            .finish()
    }
}

impl NufftPlan3D {
    /// Create a reusable 3D NUFFT plan.
    #[must_use]
    pub fn new(grid: UniformGrid3D, sigma: usize, kernel_width: usize) -> Self {
        assert!(sigma >= 2, "sigma must be >= 2");
        assert!(kernel_width >= 2, "kernel_width must be >= 2");

        let mx = supported_radix2_oversampled_len(grid.nx, sigma, kernel_width);
        let my = supported_radix2_oversampled_len(grid.ny, sigma, kernel_width);
        let mz = supported_radix2_oversampled_len(grid.nz, sigma, kernel_width);
        let beta = PI * (1.0 - 1.0 / (2.0 * sigma as f64)) * (2 * kernel_width) as f64;
        let i0_beta = i0(beta);

        let deconv_x = axis_deconv(grid.nx, mx, kernel_width, beta, i0_beta);
        let deconv_y = axis_deconv(grid.ny, my, kernel_width, beta, i0_beta);
        let deconv_z = axis_deconv(grid.nz, mz, kernel_width, beta, i0_beta);

        let fft_x = FftPlan1D::<f64>::new(
            Shape1D::new(mx).expect("NUFFT oversampled length must be valid"),
        );
        let fft_y = FftPlan1D::<f64>::new(
            Shape1D::new(my).expect("NUFFT oversampled length must be valid"),
        );
        let fft_z = FftPlan1D::<f64>::new(
            Shape1D::new(mz).expect("NUFFT oversampled length must be valid"),
        );

        Self {
            grid,
            mx,
            my,
            mz,
            w: kernel_width,
            beta,
            i0_beta,
            deconv_x,
            deconv_y,
            deconv_z,
            fft_x,
            fft_y,
            fft_z,
        }
    }

    /// Run type-1 3D NUFFT.
    #[must_use]
    pub fn type1(&self, positions: &[(f64, f64, f64)], values: &[Complex64]) -> Array3<Complex64> {
        let mut grid = Array3::<Complex64>::zeros((self.mx, self.my, self.mz));
        let mut wx = vec![0.0_f64; 2 * self.w + 1];
        let mut wy = vec![0.0_f64; 2 * self.w + 1];
        let mut wz = vec![0.0_f64; 2 * self.w + 1];
        let mut output = Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        self.type1_into(
            positions,
            values,
            &mut grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut output,
        );
        output
    }

    /// Run type-1 3D NUFFT into caller-owned work buffers and output storage.
    pub fn type1_into(
        &self,
        positions: &[(f64, f64, f64)],
        values: &[Complex64],
        scratch_grid: &mut Array3<Complex64>,
        scratch_wx: &mut [f64],
        scratch_wy: &mut [f64],
        scratch_wz: &mut [f64],
        output: &mut Array3<Complex64>,
    ) {
        assert_eq!(
            positions.len(),
            values.len(),
            "positions/value length mismatch"
        );
        assert_eq!(scratch_grid.dim(), (self.mx, self.my, self.mz));
        assert_eq!(output.dim(), (self.grid.nx, self.grid.ny, self.grid.nz));
        assert_eq!(scratch_wx.len(), 2 * self.w + 1);
        assert_eq!(scratch_wy.len(), 2 * self.w + 1);
        assert_eq!(scratch_wz.len(), 2 * self.w + 1);

        scratch_grid.fill(Complex64::new(0.0, 0.0));
        let (lx, ly, lz) = self.grid.lengths();
        let w = self.w as i64;
        let w_f = self.w as f64;

        let sorted_points = sort_points_3d(
            positions,
            values,
            self.grid,
            (self.mx, self.my, self.mz),
            self.w,
        );

        for point in sorted_points {
            let tx = self.mx as f64 * point.x / lx;
            let ty = self.my as f64 * point.y / ly;
            let tz = self.mz as f64 * point.z / lz;
            let m0x = tx.round() as i64;
            let m0y = ty.round() as i64;
            let m0z = tz.round() as i64;
            let dx = tx - m0x as f64;
            let dy = ty - m0y as f64;
            let dz = tz - m0z as f64;

            for (offset, p) in (-w..=w).enumerate() {
                scratch_wx[offset] = kb_kernel(p as f64 - dx, w_f, self.beta, self.i0_beta);
                scratch_wy[offset] = kb_kernel(p as f64 - dy, w_f, self.beta, self.i0_beta);
                scratch_wz[offset] = kb_kernel(p as f64 - dz, w_f, self.beta, self.i0_beta);
            }

            for (px, &wxv) in scratch_wx.iter().enumerate() {
                if wxv == 0.0 {
                    continue;
                }
                let ix = (m0x + px as i64 - w).rem_euclid(self.mx as i64) as usize;
                for (py, &wyv) in scratch_wy.iter().enumerate() {
                    if wyv == 0.0 {
                        continue;
                    }
                    let iy = (m0y + py as i64 - w).rem_euclid(self.my as i64) as usize;
                    let wxy = wxv * wyv;
                    for (pz, &wzv) in scratch_wz.iter().enumerate() {
                        if wzv == 0.0 {
                            continue;
                        }
                        let iz = (m0z + pz as i64 - w).rem_euclid(self.mz as i64) as usize;
                        scratch_grid[[ix, iy, iz]] += point.value * (wxy * wzv);
                    }
                }
            }
        }

        self.forward_oversampled_grid_into(scratch_grid, output);
    }

    /// Run type-1 3D NUFFT for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn type1_typed_into<T: NufftComplexStorage>(
        &self,
        positions: &[(f64, f64, f64)],
        values: &[T],
        scratch_grid: &mut Array3<Complex64>,
        scratch_wx: &mut [f64],
        scratch_wy: &mut [f64],
        scratch_wz: &mut [f64],
        output: &mut Array3<T>,
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, T::PROFILE)?;
        if positions.len() != values.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: positions.len().to_string(),
                actual: values.len().to_string(),
            });
        }
        if scratch_grid.dim() != (self.mx, self.my, self.mz)
            || output.dim() != (self.grid.nx, self.grid.ny, self.grid.nz)
            || scratch_wx.len() != 2 * self.w + 1
            || scratch_wy.len() != 2 * self.w + 1
            || scratch_wz.len() != 2 * self.w + 1
        {
            return Err(ApolloError::ShapeMismatch {
                expected: format!(
                    "scratch {:?}, weights {}, output {:?}",
                    (self.mx, self.my, self.mz),
                    2 * self.w + 1,
                    (self.grid.nx, self.grid.ny, self.grid.nz)
                ),
                actual: format!(
                    "scratch {:?}, weights ({}, {}, {}), output {:?}",
                    scratch_grid.dim(),
                    scratch_wx.len(),
                    scratch_wy.len(),
                    scratch_wz.len(),
                    output.dim()
                ),
            });
        }
        let values64: Vec<Complex64> = values.iter().copied().map(T::to_complex64).collect();
        let mut output64 = Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        self.type1_into(
            positions,
            &values64,
            scratch_grid,
            scratch_wx,
            scratch_wy,
            scratch_wz,
            &mut output64,
        );
        for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
            *slot = T::from_complex64(value);
        }
        Ok(())
    }

    fn forward_oversampled_grid_into(
        &self,
        grid: &mut Array3<Complex64>,
        output: &mut Array3<Complex64>,
    ) {
        self.fft_z_pass(grid);
        self.fft_y_pass(grid);
        self.fft_x_pass(grid);

        for kx in 0..self.grid.nx {
            for ky in 0..self.grid.ny {
                for kz in 0..self.grid.nz {
                    let kx_idx =
                        fft_signed_index(kx, self.grid.nx).rem_euclid(self.mx as i64) as usize;
                    let ky_idx =
                        fft_signed_index(ky, self.grid.ny).rem_euclid(self.my as i64) as usize;
                    let kz_idx =
                        fft_signed_index(kz, self.grid.nz).rem_euclid(self.mz as i64) as usize;
                    output[[kx, ky, kz]] = grid[[kx_idx, ky_idx, kz_idx]]
                        * (self.deconv_x[kx] * self.deconv_y[ky] * self.deconv_z[kz]);
                }
            }
        }
    }

    fn ifft_z_pass(&self, grid: &mut Array3<Complex64>) {
        let mut lane = Array1::<Complex64>::zeros(self.mz);
        for ix in 0..self.mx {
            for iy in 0..self.my {
                for iz in 0..self.mz {
                    lane[iz] = grid[[ix, iy, iz]];
                }
                self.fft_z.inverse_complex_inplace(&mut lane);
                for iz in 0..self.mz {
                    grid[[ix, iy, iz]] = lane[iz];
                }
            }
        }
    }

    fn ifft_y_pass(&self, grid: &mut Array3<Complex64>) {
        let mut lane = Array1::<Complex64>::zeros(self.my);
        for ix in 0..self.mx {
            for iz in 0..self.mz {
                for iy in 0..self.my {
                    lane[iy] = grid[[ix, iy, iz]];
                }
                self.fft_y.inverse_complex_inplace(&mut lane);
                for iy in 0..self.my {
                    grid[[ix, iy, iz]] = lane[iy];
                }
            }
        }
    }

    fn ifft_x_pass(&self, grid: &mut Array3<Complex64>) {
        let mut lane = Array1::<Complex64>::zeros(self.mx);
        for iy in 0..self.my {
            for iz in 0..self.mz {
                for ix in 0..self.mx {
                    lane[ix] = grid[[ix, iy, iz]];
                }
                self.fft_x.inverse_complex_inplace(&mut lane);
                for ix in 0..self.mx {
                    grid[[ix, iy, iz]] = lane[ix];
                }
            }
        }
    }

    /// Type-2 3D NUFFT: interpolate from uniform Fourier coefficients to non-uniform points.
    ///
    /// Given mode array  of shape , computes spatial values
    ///
    ///
    ///
    /// at each non-uniform point  in .
    ///
    /// # Complexity
    ///
    /// O(P * w^3 + mx*my*mz * log(mx*my*mz)) where P = positions.len() and w = kernel width.
    #[must_use]
    pub fn type2(
        &self,
        positions: &[(f64, f64, f64)],
        modes: &Array3<Complex64>,
    ) -> Vec<Complex64> {
        let mut scratch_grid = Array3::<Complex64>::zeros((self.mx, self.my, self.mz));
        let mut wx = vec![0.0_f64; 2 * self.w + 1];
        let mut wy = vec![0.0_f64; 2 * self.w + 1];
        let mut wz = vec![0.0_f64; 2 * self.w + 1];
        let mut output = vec![Complex64::default(); positions.len()];
        self.type2_into(
            positions,
            modes,
            &mut scratch_grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut output,
        );
        output
    }

    /// Run type-2 3D NUFFT into caller-owned work buffers and output storage.
    pub fn type2_into(
        &self,
        positions: &[(f64, f64, f64)],
        modes: &Array3<Complex64>,
        scratch_grid: &mut Array3<Complex64>,
        scratch_wx: &mut [f64],
        scratch_wy: &mut [f64],
        scratch_wz: &mut [f64],
        output: &mut [Complex64],
    ) {
        assert_eq!(
            modes.dim(),
            (self.grid.nx, self.grid.ny, self.grid.nz),
            "modes shape must match plan grid dimensions"
        );
        assert_eq!(
            scratch_grid.dim(),
            (self.mx, self.my, self.mz),
            "scratch_grid shape must match oversampled grid dimensions"
        );
        assert_eq!(
            scratch_wx.len(),
            2 * self.w + 1,
            "scratch_wx length mismatch"
        );
        assert_eq!(
            scratch_wy.len(),
            2 * self.w + 1,
            "scratch_wy length mismatch"
        );
        assert_eq!(
            scratch_wz.len(),
            2 * self.w + 1,
            "scratch_wz length mismatch"
        );
        assert_eq!(output.len(), positions.len(), "output length mismatch");

        // 1. Place deconvolved modes on oversampled grid (zero-padded)
        scratch_grid.fill(Complex64::new(0.0, 0.0));
        for kx in 0..self.grid.nx {
            let kx_idx = fft_signed_index(kx, self.grid.nx).rem_euclid(self.mx as i64) as usize;
            for ky in 0..self.grid.ny {
                let ky_idx = fft_signed_index(ky, self.grid.ny).rem_euclid(self.my as i64) as usize;
                for kz in 0..self.grid.nz {
                    let kz_idx =
                        fft_signed_index(kz, self.grid.nz).rem_euclid(self.mz as i64) as usize;
                    scratch_grid[[kx_idx, ky_idx, kz_idx]] = modes[[kx, ky, kz]]
                        * (self.deconv_x[kx] * self.deconv_y[ky] * self.deconv_z[kz]);
                }
            }
        }

        // 2. Separable inverse FFT on oversampled grid
        self.ifft_x_pass(scratch_grid);
        self.ifft_y_pass(scratch_grid);
        self.ifft_z_pass(scratch_grid);

        // 3. Interpolate at each non-uniform point using KB kernel
        let (lx, ly, lz) = self.grid.lengths();
        let w = self.w as i64;
        let w_f = self.w as f64;

        let sorted = sort_positions_3d(positions, self.grid, (self.mx, self.my, self.mz), self.w);

        for point in sorted {
            let tx = self.mx as f64 * point.x / lx;
            let ty = self.my as f64 * point.y / ly;
            let tz = self.mz as f64 * point.z / lz;

            let m0x = tx.round() as i64;
            let m0y = ty.round() as i64;
            let m0z = tz.round() as i64;

            let dx = tx - m0x as f64;
            let dy = ty - m0y as f64;
            let dz = tz - m0z as f64;

            for (off, p) in (-w..=w).enumerate() {
                scratch_wx[off] = kb_kernel(p as f64 - dx, w_f, self.beta, self.i0_beta);
                scratch_wy[off] = kb_kernel(p as f64 - dy, w_f, self.beta, self.i0_beta);
                scratch_wz[off] = kb_kernel(p as f64 - dz, w_f, self.beta, self.i0_beta);
            }

            let mut value = Complex64::new(0.0, 0.0);
            for (px, &wxv) in scratch_wx.iter().enumerate() {
                if wxv == 0.0 {
                    continue;
                }
                let ix = (m0x + px as i64 - w).rem_euclid(self.mx as i64) as usize;
                for (py, &wyv) in scratch_wy.iter().enumerate() {
                    if wyv == 0.0 {
                        continue;
                    }
                    let iy = (m0y + py as i64 - w).rem_euclid(self.my as i64) as usize;
                    let wxy = wxv * wyv;
                    for (pz, &wzv) in scratch_wz.iter().enumerate() {
                        if wzv == 0.0 {
                            continue;
                        }
                        let iz = (m0z + pz as i64 - w).rem_euclid(self.mz as i64) as usize;
                        value += scratch_grid[[ix, iy, iz]] * (wxy * wzv);
                    }
                }
            }
            output[point.index] = value;
        }
    }

    /// Run type-2 3D NUFFT for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn type2_typed_into<T: NufftComplexStorage>(
        &self,
        positions: &[(f64, f64, f64)],
        modes: &Array3<T>,
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, T::PROFILE)?;
        if modes.dim() != (self.grid.nx, self.grid.ny, self.grid.nz)
            || output.len() != positions.len()
        {
            return Err(ApolloError::ShapeMismatch {
                expected: format!(
                    "modes {:?}, output {}",
                    (self.grid.nx, self.grid.ny, self.grid.nz),
                    positions.len()
                ),
                actual: format!("modes {:?}, output {}", modes.dim(), output.len()),
            });
        }
        let modes64 = modes.mapv(T::to_complex64);
        let output64 = self.type2(positions, &modes64);
        write_typed_output(&output64, output);
        Ok(())
    }

    fn fft_z_pass(&self, grid: &mut Array3<Complex64>) {
        let mut lane = Array1::<Complex64>::zeros(self.mz);
        for ix in 0..self.mx {
            for iy in 0..self.my {
                for iz in 0..self.mz {
                    lane[iz] = grid[[ix, iy, iz]];
                }
                self.fft_z.forward_complex_inplace(&mut lane);
                for iz in 0..self.mz {
                    grid[[ix, iy, iz]] = lane[iz];
                }
            }
        }
    }

    fn fft_y_pass(&self, grid: &mut Array3<Complex64>) {
        let mut lane = Array1::<Complex64>::zeros(self.my);
        for ix in 0..self.mx {
            for iz in 0..self.mz {
                for iy in 0..self.my {
                    lane[iy] = grid[[ix, iy, iz]];
                }
                self.fft_y.forward_complex_inplace(&mut lane);
                for iy in 0..self.my {
                    grid[[ix, iy, iz]] = lane[iy];
                }
            }
        }
    }

    fn fft_x_pass(&self, grid: &mut Array3<Complex64>) {
        let mut lane = Array1::<Complex64>::zeros(self.mx);
        for iy in 0..self.my {
            for iz in 0..self.mz {
                for ix in 0..self.mx {
                    lane[ix] = grid[[ix, iy, iz]];
                }
                self.fft_x.forward_complex_inplace(&mut lane);
                for ix in 0..self.mx {
                    grid[[ix, iy, iz]] = lane[ix];
                }
            }
        }
    }
}

/// Exact direct 3D type-1 NUFFT.
#[must_use]
pub fn nufft_type1_3d(
    positions: &[(f64, f64, f64)],
    values: &[Complex64],
    grid: UniformGrid3D,
) -> Array3<Complex64> {
    assert_eq!(
        positions.len(),
        values.len(),
        "positions/value length mismatch"
    );
    let (lx, ly, lz) = grid.lengths();
    let two_pi_lx = 2.0 * PI / lx;
    let two_pi_ly = 2.0 * PI / ly;
    let two_pi_lz = 2.0 * PI / lz;

    Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
        let kx_signed = fft_signed_index(kx, grid.nx);
        let ky_signed = fft_signed_index(ky, grid.ny);
        let kz_signed = fft_signed_index(kz, grid.nz);
        positions.iter().zip(values.iter()).fold(
            Complex64::new(0.0, 0.0),
            |acc, (&(x, y, z), &value)| {
                let angle = -(two_pi_lx * kx_signed as f64 * x
                    + two_pi_ly * ky_signed as f64 * y
                    + two_pi_lz * kz_signed as f64 * z);
                acc + value * Complex64::new(angle.cos(), angle.sin())
            },
        )
    })
}

/// Exact direct 3D type-2 NUFFT.
#[must_use]
pub fn nufft_type2_3d(
    positions: &[(f64, f64, f64)],
    modes: &Array3<Complex64>,
    grid: UniformGrid3D,
) -> Vec<Complex64> {
    assert_eq!(
        modes.dim(),
        (grid.nx, grid.ny, grid.nz),
        "modes shape must match grid dimensions"
    );
    let (lx, ly, lz) = grid.lengths();
    let two_pi_lx = 2.0 * PI / lx;
    let two_pi_ly = 2.0 * PI / ly;
    let two_pi_lz = 2.0 * PI / lz;

    positions
        .iter()
        .map(|&(x, y, z)| {
            modes
                .indexed_iter()
                .fold(Complex64::new(0.0, 0.0), |acc, ((kx, ky, kz), &value)| {
                    let angle = two_pi_lx * fft_signed_index(kx, grid.nx) as f64 * x
                        + two_pi_ly * fft_signed_index(ky, grid.ny) as f64 * y
                        + two_pi_lz * fft_signed_index(kz, grid.nz) as f64 * z;
                    acc + value * Complex64::new(angle.cos(), angle.sin())
                })
        })
        .collect()
}

/// Fast 3D type-1 NUFFT convenience wrapper.
#[must_use]
pub fn nufft_type1_3d_fast(
    positions: &[(f64, f64, f64)],
    values: &[Complex64],
    grid: UniformGrid3D,
    kernel_width: usize,
) -> Array3<Complex64> {
    NufftPlan3D::new(grid, DEFAULT_NUFFT_OVERSAMPLING, kernel_width).type1(positions, values)
}

/// Fast 3D type-2 NUFFT convenience wrapper.
///
/// Interpolates uniform Fourier coefficients at non-uniform positions using
/// Kaiser-Bessel spreading of the deconvolved oversampled grid followed by
/// a separable inverse Apollo FFT on each axis.
#[must_use]
pub fn nufft_type2_3d_fast(
    positions: &[(f64, f64, f64)],
    modes: &Array3<Complex64>,
    grid: UniformGrid3D,
    kernel_width: usize,
) -> Vec<Complex64> {
    NufftPlan3D::new(grid, DEFAULT_NUFFT_OVERSAMPLING, kernel_width).type2(positions, modes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_NUFFT_KERNEL_WIDTH;
    use crate::DEFAULT_NUFFT_OVERSAMPLING;

    /// DC mode invariant: f_{0,0,0} = sum(c_j) because exp(0) = 1 for all points.
    ///
    /// With values [1.0, 0.5-0.25i, -0.2+0.75i]: sum = 1.3 + 0.5i
    #[test]
    fn type1_3d_dc_mode_equals_sum_of_values() {
        let grid = UniformGrid3D::new(4, 4, 4, 0.1, 0.1, 0.1).unwrap();
        let positions = vec![(0.0, 0.0, 0.0), (0.1, 0.2, 0.3), (0.25, 0.15, 0.05)];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(-0.2, 0.75),
        ];
        let output = nufft_type1_3d(&positions, &values, grid);
        assert_eq!(output.dim(), (4, 4, 4));
        // DC mode: sum of all sample values (analytical)
        let expected_dc = Complex64::new(1.3, 0.5);
        let err = (output[[0, 0, 0]] - expected_dc).norm();
        assert!(
            err < 1e-12,
            "DC mode err={err}: got {:?}, expected {:?}",
            output[[0, 0, 0]],
            expected_dc
        );
        // Verify all outputs are finite
        for v in output.iter() {
            assert!(v.norm().is_finite(), "non-finite output: {v:?}");
        }
    }

    /// Type-2 zero-position invariant:
    /// g(0,0,0) = sum_{kx,ky,kz} f_{kx,ky,kz} because every phase is exp(0).
    #[test]
    fn type2_3d_origin_equals_sum_of_modes() {
        let grid = UniformGrid3D::new(3, 2, 2, 0.5, 0.75, 1.0).unwrap();
        let modes = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(kx, ky, kz)| {
            Complex64::new(
                0.25 + 0.1 * kx as f64 - 0.05 * ky as f64 + 0.03 * kz as f64,
                -0.4 + 0.07 * kx as f64 + 0.11 * ky as f64 - 0.02 * kz as f64,
            )
        });
        let output = nufft_type2_3d(&[(0.0, 0.0, 0.0)][..], &modes, grid);
        let expected: Complex64 = modes.iter().copied().sum();
        assert_eq!(output.len(), 1);
        let err = (output[0] - expected).norm();
        assert!(
            err < 1e-12,
            "origin type-2 err={err}: got {:?}, expected {expected:?}",
            output[0]
        );
    }

    #[test]
    fn fast_3d_tracks_exact() {
        let grid = UniformGrid3D::new(4, 4, 4, 0.1, 0.1, 0.1).unwrap();
        let positions = vec![(0.0, 0.0, 0.0), (0.1, 0.2, 0.3), (0.25, 0.15, 0.05)];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(-0.2, 0.75),
        ];
        let exact = nufft_type1_3d(&positions, &values, grid);
        let fast = nufft_type1_3d_fast(&positions, &values, grid, DEFAULT_NUFFT_KERNEL_WIDTH);
        let scale = exact.iter().map(|value| value.norm()).fold(1.0, f64::max);
        let max_err = exact
            .iter()
            .zip(fast.iter())
            .map(|(lhs, rhs)| (lhs - rhs).norm())
            .fold(0.0, f64::max);
        assert!(max_err / scale < 1e-6);
    }

    #[test]
    fn typed_3d_paths_support_complex32_storage() {
        let grid = UniformGrid3D::new(4, 4, 4, 0.1, 0.1, 0.1).unwrap();
        let plan = NufftPlan3D::new(grid, DEFAULT_NUFFT_OVERSAMPLING, DEFAULT_NUFFT_KERNEL_WIDTH);
        let positions = vec![(0.0, 0.0, 0.0), (0.1, 0.2, 0.3), (0.25, 0.15, 0.05)];
        let values64 = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.25),
            Complex64::new(-0.2, 0.75),
        ];
        let values32: Vec<num_complex::Complex32> = values64
            .iter()
            .map(|value| num_complex::Complex32::new(value.re as f32, value.im as f32))
            .collect();
        let represented32: Vec<Complex64> = values32
            .iter()
            .copied()
            .map(num_complex::Complex32::to_complex64)
            .collect();
        let expected = plan.type1(&positions, &represented32);

        let mut scratch_grid = Array3::<Complex64>::zeros((plan.mx, plan.my, plan.mz));
        let mut wx = vec![0.0_f64; 2 * plan.w + 1];
        let mut wy = vec![0.0_f64; 2 * plan.w + 1];
        let mut wz = vec![0.0_f64; 2 * plan.w + 1];
        let mut output = Array3::<num_complex::Complex32>::zeros((grid.nx, grid.ny, grid.nz));
        plan.type1_typed_into(
            &positions,
            &values32,
            &mut scratch_grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut output,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed 3d type1");
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let mut type2_output = vec![num_complex::Complex32::new(0.0, 0.0); positions.len()];
        plan.type2_typed_into(
            &positions,
            &output,
            &mut type2_output,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed 3d type2");
        let expected_type2 = plan.type2(&positions, &expected);
        for (actual, expected) in type2_output.iter().zip(expected_type2.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-3);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-3);
        }
    }

    #[test]
    fn typed_3d_paths_reject_profile_mismatch() {
        let grid = UniformGrid3D::new(2, 2, 2, 0.25, 0.25, 0.25).unwrap();
        let plan = NufftPlan3D::new(grid, DEFAULT_NUFFT_OVERSAMPLING, DEFAULT_NUFFT_KERNEL_WIDTH);
        let positions = vec![(0.0, 0.0, 0.0)];
        let values = vec![num_complex::Complex32::new(1.0, 0.0)];
        let mut scratch_grid = Array3::<Complex64>::zeros((plan.mx, plan.my, plan.mz));
        let mut wx = vec![0.0_f64; 2 * plan.w + 1];
        let mut wy = vec![0.0_f64; 2 * plan.w + 1];
        let mut wz = vec![0.0_f64; 2 * plan.w + 1];
        let mut output = Array3::<num_complex::Complex32>::zeros((grid.nx, grid.ny, grid.nz));
        let err = plan
            .type1_typed_into(
                &positions,
                &values,
                &mut scratch_grid,
                &mut wx,
                &mut wy,
                &mut wz,
                &mut output,
                PrecisionProfile::HIGH_ACCURACY_F64,
            )
            .expect_err("profile mismatch");
        assert!(matches!(
            err,
            ApolloError::Validation { field, .. } if field == "precision_profile"
        ));
    }

    #[test]
    fn plan_3d_type2_into_matches_type2_allocating() {
        let grid = UniformGrid3D::new(4, 4, 4, 0.1, 0.1, 0.1).unwrap();
        let plan = NufftPlan3D::new(grid, DEFAULT_NUFFT_OVERSAMPLING, DEFAULT_NUFFT_KERNEL_WIDTH);
        let modes = Array3::from_shape_fn((4, 4, 4), |(kx, ky, kz)| {
            Complex64::new(
                0.25 + 0.1 * kx as f64 - 0.05 * ky as f64 + 0.03 * kz as f64,
                -0.4 + 0.07 * kx as f64 + 0.11 * ky as f64 - 0.02 * kz as f64,
            )
        });
        let positions = vec![
            (0.01, 0.02, 0.03),
            (0.05, 0.06, 0.07),
            (0.025, 0.015, 0.005),
        ];
        let expected = plan.type2(&positions, &modes);
        let mut scratch_grid = Array3::<Complex64>::zeros((plan.mx, plan.my, plan.mz));
        let mut wx = vec![0.0; 2 * plan.w + 1];
        let mut wy = vec![0.0; 2 * plan.w + 1];
        let mut wz = vec![0.0; 2 * plan.w + 1];
        let mut output = vec![Complex64::new(0.0, 0.0); positions.len()];
        plan.type2_into(
            &positions,
            &modes,
            &mut scratch_grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut output,
        );
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).norm() < 1e-14,
                "type2_into mismatch: got {actual:?}, expected {expected:?}"
            );
        }
    }
}
