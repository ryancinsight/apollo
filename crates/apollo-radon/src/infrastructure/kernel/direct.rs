//! Direct discrete Radon projection and adjoint backprojection kernels.

use crate::domain::geometry::parallel_beam::ParallelBeamGeometry;
use mnemosyne::scratch::ScratchPool;
use leto::Array2;

/// Below this angle count, scalar accumulation avoids Hermes dispatch and scratch setup.
const RADON_HERMES_BACKPROJECT_ANGLE_THRESHOLD: usize = 256;

thread_local! {
    static BACKPROJECT_SAMPLE_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static BACKPROJECT_WEIGHT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Execute the forward discrete Radon projection.
///
/// Embarrassingly parallel over the angle dimension: each angle writes to a
/// disjoint, contiguous sinogram row, so chunking the row-major buffer by
/// `detector_count` provides data-race-free mutable access without synchronisation.
#[must_use]
pub fn forward_project(image: &Array2<f64>, geometry: &ParallelBeamGeometry) -> Array2<f64> {
    let mut sinogram = Array2::zeros([geometry.angle_count(), geometry.detector_count()]);
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
    let ncols = sinogram.shape()[1];
    let flat = sinogram
        .as_slice_mut()
        .expect("sinogram must be contiguous (standard layout)");
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        flat,
        ncols,
        |angle_index, row| {
            let (sin_theta, cos_theta) = geometry.angles()[angle_index].sin_cos();
            for r in 0..geometry.rows() {
                for c in 0..geometry.cols() {
                    let det_coord = geometry.x(c) * cos_theta + geometry.y(r) * sin_theta;
                    deposit(geometry.detector_index(det_coord), image[[r, c]], row);
                }
            }
        },
    );
}

/// Execute the adjoint of the discrete Radon projection.
///
/// Embarrassingly parallel over the image row dimension: each output row reads
/// the sinogram immutably (`Array2<f64>`: Sync) and writes only to its own row
/// of the output image.
#[must_use]
pub fn adjoint_backproject(sinogram: &Array2<f64>, geometry: &ParallelBeamGeometry) -> Array2<f64> {
    let cols = geometry.cols();
    let mut image = Array2::zeros([geometry.rows(), cols]);
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
    let ncols = image.shape()[1];
    let flat = image
        .as_slice_mut()
        .expect("image must be contiguous (standard layout)");
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(flat, ncols, |r, row| {
        let y = geometry.y(r);
        for c in 0..cols {
            let x = geometry.x(c);
            row[c] = backproject_pixel(x, y, sinogram, geometry);
        }
    });
}

fn backproject_pixel(
    x: f64,
    y: f64,
    sinogram: &Array2<f64>,
    geometry: &ParallelBeamGeometry,
) -> f64 {
    if geometry.angle_count() >= RADON_HERMES_BACKPROJECT_ANGLE_THRESHOLD {
        backproject_pixel_hermes(x, y, sinogram, geometry)
    } else {
        backproject_pixel_scalar(x, y, sinogram, geometry)
    }
}

fn backproject_pixel_scalar(
    x: f64,
    y: f64,
    sinogram: &Array2<f64>,
    geometry: &ParallelBeamGeometry,
) -> f64 {
    geometry
        .angles()
        .iter()
        .enumerate()
        .map(|(angle_index, angle)| {
            let (sin_theta, cos_theta) = angle.sin_cos();
            let det_coord = x * cos_theta + y * sin_theta;
            sample_linear(geometry.detector_index(det_coord), sinogram, angle_index)
        })
        .sum()
}

fn backproject_pixel_hermes(
    x: f64,
    y: f64,
    sinogram: &Array2<f64>,
    geometry: &ParallelBeamGeometry,
) -> f64 {
    let lane_len = geometry.angle_count() * 2;
    BACKPROJECT_SAMPLE_SCRATCH.with(|sample_pool| {
        sample_pool.with_scratch(lane_len, |samples| {
            BACKPROJECT_WEIGHT_SCRATCH.with(|weight_pool| {
                weight_pool.with_scratch(lane_len, |weights| {
                    fill_backproject_lanes(samples, weights, x, y, sinogram, geometry);
                    hermes_simd::dot::<f64>(samples, weights)
                        .expect("Radon backprojection Hermes dot uses equal-length angle lanes")
                })
            })
        })
    })
}

fn fill_backproject_lanes(
    samples: &mut [f64],
    weights: &mut [f64],
    x: f64,
    y: f64,
    sinogram: &Array2<f64>,
    geometry: &ParallelBeamGeometry,
) {
    for (angle_index, (&angle, (sample_pair, weight_pair))) in geometry
        .angles()
        .iter()
        .zip(samples.chunks_exact_mut(2).zip(weights.chunks_exact_mut(2)))
        .enumerate()
    {
        let (sin_theta, cos_theta) = angle.sin_cos();
        let det_coord = x * cos_theta + y * sin_theta;
        fill_linear_sample_weight(
            geometry.detector_index(det_coord),
            sinogram,
            angle_index,
            sample_pair,
            weight_pair,
        );
    }
}

/// Deposit mass at fractional detector index into a single sinogram row
/// using linear (nearest-two-bin) weighting.
fn deposit(index: f64, mass: f64, row: &mut [f64]) {
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
    if index < 0.0 || index > (sinogram.shape()[1] - 1) as f64 {
        return 0.0;
    }
    let left = index.floor() as usize;
    let right_weight = index - left as f64;
    let left_weight = 1.0 - right_weight;
    let mut value = sinogram[[angle_index, left]] * left_weight;
    if right_weight > 0.0 && left + 1 < sinogram.shape()[1] {
        value += sinogram[[angle_index, left + 1]] * right_weight;
    }
    value
}

fn fill_linear_sample_weight(
    index: f64,
    sinogram: &Array2<f64>,
    angle_index: usize,
    samples: &mut [f64],
    weights: &mut [f64],
) {
    debug_assert_eq!(samples.len(), 2);
    debug_assert_eq!(weights.len(), 2);
    if index < 0.0 || index > (sinogram.shape()[1] - 1) as f64 {
        samples.fill(0.0);
        weights.fill(0.0);
        return;
    }
    let left = index.floor() as usize;
    let right_weight = index - left as f64;
    let left_weight = 1.0 - right_weight;
    samples[0] = sinogram[[angle_index, left]];
    weights[0] = left_weight;
    if right_weight > 0.0 && left + 1 < sinogram.shape()[1] {
        samples[1] = sinogram[[angle_index, left + 1]];
        weights[1] = right_weight;
    } else {
        samples[1] = 0.0;
        weights[1] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hermes_backproject_pixel_matches_scalar_formula_at_threshold() {
        let angles = (0..RADON_HERMES_BACKPROJECT_ANGLE_THRESHOLD)
            .map(|index| std::f64::consts::PI * index as f64 / 511.0)
            .collect::<Vec<_>>();
        let geometry = ParallelBeamGeometry::new(3, 3, angles, 7, 0.75).expect("valid geometry");
        let sinogram = Array2::from_shape_fn(
            [geometry.angle_count(), geometry.detector_count()],
            |[angle, detector]| (angle as f64 * 0.017).sin() + (detector as f64 * 0.031).cos(),
        );

        for (x, y) in [(-0.5, -0.25), (0.0, 0.0), (0.75, 0.5)] {
            let expected = backproject_pixel_scalar(x, y, &sinogram, &geometry);
            let actual = backproject_pixel_hermes(x, y, &sinogram, &geometry);
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-11);
        }
    }

    #[test]
    fn backproject_lanes_match_linear_sample_formula() {
        let angles = (0..RADON_HERMES_BACKPROJECT_ANGLE_THRESHOLD)
            .map(|index| std::f64::consts::PI * index as f64 / 257.0)
            .collect::<Vec<_>>();
        let geometry = ParallelBeamGeometry::new(3, 3, angles, 5, 1.0).expect("valid geometry");
        let sinogram = Array2::from_shape_fn(
            [geometry.angle_count(), geometry.detector_count()],
            |[angle, detector]| angle as f64 * 0.25 - detector as f64 * 0.5,
        );
        let mut samples = vec![f64::NAN; geometry.angle_count() * 2];
        let mut weights = vec![f64::NAN; geometry.angle_count() * 2];

        fill_backproject_lanes(
            &mut samples,
            &mut weights,
            0.25,
            -0.75,
            &sinogram,
            &geometry,
        );

        for angle_index in 0..geometry.angle_count() {
            let angle = geometry.angles()[angle_index];
            let (sin_theta, cos_theta) = angle.sin_cos();
            let det_coord = 0.25 * cos_theta - 0.75 * sin_theta;
            let expected =
                sample_linear(geometry.detector_index(det_coord), &sinogram, angle_index);
            let lane = angle_index * 2;
            let actual = samples[lane] * weights[lane] + samples[lane + 1] * weights[lane + 1];
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }
}
