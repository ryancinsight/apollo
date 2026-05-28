//! Direct discrete Radon projection and adjoint backprojection kernels.

use crate::domain::geometry::parallel_beam::ParallelBeamGeometry;
use ndarray::{Array2, ArrayViewMut1, Axis};
use rayon::prelude::*;

/// Execute the forward discrete Radon projection.
///
/// Embarrassingly parallel over the angle dimension: each angle writes to a
/// disjoint sinogram row, so axis_iter_mut(Axis(0)).into_par_iter() provides
/// data-race-free mutable access without synchronisation.
#[must_use]
pub fn forward_project(image: &Array2<f64>, geometry: &ParallelBeamGeometry) -> Array2<f64> {
    let mut sinogram = Array2::zeros((geometry.angle_count(), geometry.detector_count()));
    forward_project_into(image, geometry, &mut sinogram);
    sinogram
}

/// Execute the forward discrete Radon projection into caller-owned storage.
///
/// The output is cleared before accumulation so callers can safely reuse a
/// sinogram buffer across repeated projections.
pub fn forward_project_into(
    image: &Array2<f64>,
    geometry: &ParallelBeamGeometry,
    sinogram: &mut Array2<f64>,
) {
    sinogram.fill(0.0);
    sinogram
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(angle_index, mut row)| {
            let (sin_theta, cos_theta) = geometry.angles()[angle_index].sin_cos();
            for r in 0..geometry.rows() {
                for c in 0..geometry.cols() {
                    let det_coord = geometry.x(c) * cos_theta + geometry.y(r) * sin_theta;
                    deposit(geometry.detector_index(det_coord), image[(r, c)], &mut row);
                }
            }
        });
}

/// Execute the adjoint of the discrete Radon projection.
///
/// Embarrassingly parallel over the image row dimension: each output row reads
/// the sinogram immutably (`Array2<f64>`: Sync) and writes only to its own row
/// of the output image.
#[must_use]
pub fn adjoint_backproject(sinogram: &Array2<f64>, geometry: &ParallelBeamGeometry) -> Array2<f64> {
    let cols = geometry.cols();
    let mut image = Array2::zeros((geometry.rows(), cols));
    adjoint_backproject_into(sinogram, geometry, &mut image);
    image
}

/// Execute the adjoint projection into caller-owned image storage.
pub fn adjoint_backproject_into(
    sinogram: &Array2<f64>,
    geometry: &ParallelBeamGeometry,
    image: &mut Array2<f64>,
) {
    let cols = geometry.cols();
    image
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(r, mut row)| {
            let y = geometry.y(r);
            for c in 0..cols {
                let x = geometry.x(c);
                let mut value = 0.0_f64;
                for (angle_index, angle) in geometry.angles().iter().enumerate() {
                    let (sin_theta, cos_theta) = angle.sin_cos();
                    let det_coord = x * cos_theta + y * sin_theta;
                    value +=
                        sample_linear(geometry.detector_index(det_coord), sinogram, angle_index);
                }
                row[c] = value;
            }
        });
}

/// Deposit mass at fractional detector index into a single sinogram row
/// using linear (nearest-two-bin) weighting.
fn deposit(index: f64, mass: f64, row: &mut ArrayViewMut1<'_, f64>) {
    let ncols = row.len();
    if ncols == 0 || index < 0.0 || index > (ncols - 1) as f64 {
        return;
    }
    let left = index.floor() as usize;
    let right_weight = index - left as f64;
    let left_weight = 1.0 - right_weight;
    row[left] += mass * left_weight;
    if right_weight > 0.0 && left + 1 < ncols {
        row[left + 1] += mass * right_weight;
    }
}

fn sample_linear(index: f64, sinogram: &Array2<f64>, angle_index: usize) -> f64 {
    if index < 0.0 || index > (sinogram.ncols() - 1) as f64 {
        return 0.0;
    }
    let left = index.floor() as usize;
    let right_weight = index - left as f64;
    let left_weight = 1.0 - right_weight;
    let mut value = sinogram[(angle_index, left)] * left_weight;
    if right_weight > 0.0 && left + 1 < sinogram.ncols() {
        value += sinogram[(angle_index, left + 1)] * right_weight;
    }
    value
}
