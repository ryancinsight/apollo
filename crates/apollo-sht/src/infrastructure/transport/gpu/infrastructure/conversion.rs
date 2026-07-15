//! Validated SHT host conversion, grid construction, and Leto interop.

use std::borrow::Cow;

use apollo_fft::{application::utilities::leto_interop, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Array2;

use crate::{
    infrastructure::{
        kernel::spherical_harmonic::gauss_legendre_nodes_weights,
        transport::gpu::{
            application::plan::ShtWgpuPlan,
            domain::{
                error::{WgpuError, WgpuResult},
                storage::ShtGpuStorage,
            },
            infrastructure::kernel::GridPod,
        },
    },
    SphericalGridSpec, SphericalHarmonicCoefficients,
};

pub(super) fn validate_plan(plan: &ShtWgpuPlan) -> WgpuResult<()> {
    let degree_count = plan
        .max_degree()
        .checked_add(1)
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "maximum degree plus one overflows usize".to_owned(),
        })?;
    let minimum_longitudes = plan
        .max_degree()
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "twice the maximum degree plus one overflows usize".to_owned(),
        })?;
    let sample_count = plan
        .latitudes()
        .checked_mul(plan.longitudes())
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "sample count overflows usize".to_owned(),
        })?;
    let mode_count =
        degree_count
            .checked_mul(degree_count)
            .ok_or_else(|| WgpuError::InvalidPlan {
                message: "mode count overflows usize".to_owned(),
            })?;
    if minimum_longitudes > plan.longitudes() {
        return Err(WgpuError::InvalidPlan {
            message: format!(
                "invalid plan maximum degree {} requires at least {minimum_longitudes} longitudes, got {}",
                plan.max_degree(),
                plan.longitudes()
            ),
        });
    }
    if SphericalGridSpec::new(plan.latitudes(), plan.longitudes(), plan.max_degree()).is_err() {
        return Err(WgpuError::InvalidPlan {
            message: format!(
                "invalid plan latitudes={}, longitudes={}, max_degree={}: grid must be non-empty and satisfy max_degree < latitudes and 2*max_degree+1 <= longitudes",
                plan.latitudes(),
                plan.longitudes(),
                plan.max_degree()
            ),
        });
    }
    if u32::try_from(sample_count).is_err() || u32::try_from(mode_count).is_err() {
        return Err(WgpuError::InvalidPlan {
            message: format!(
                "invalid plan latitudes={}, longitudes={}, max_degree={}: dispatch dimensions must fit in u32",
                plan.latitudes(),
                plan.longitudes(),
                plan.max_degree()
            ),
        });
    }
    if isize::try_from(plan.max_degree()).is_err() {
        return Err(WgpuError::InvalidPlan {
            message: format!(
                "invalid plan maximum degree {} exceeds the host mode-index range",
                plan.max_degree()
            ),
        });
    }
    mode_count
        .checked_mul(sample_count)
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "basis storage length overflows usize".to_owned(),
        })?;
    Ok(())
}

pub(super) fn validate_forward(plan: &ShtWgpuPlan, input_len: usize) -> WgpuResult<()> {
    validate_plan(plan)?;
    validate_length(plan.sample_count(), input_len, "sample")
}

pub(super) fn validate_sample_shape(plan: &ShtWgpuPlan, shape: [usize; 2]) -> WgpuResult<()> {
    validate_plan(plan)?;
    let expected = [plan.latitudes(), plan.longitudes()];
    if shape == expected {
        Ok(())
    } else {
        Err(WgpuError::ShapeMismatch {
            message: format!(
                "samples expected {}x{}, got {}x{}",
                expected[0], expected[1], shape[0], shape[1]
            ),
        })
    }
}

pub(super) fn validate_coefficient_shape(
    plan: &ShtWgpuPlan,
    coefficients: &SphericalHarmonicCoefficients,
) -> WgpuResult<()> {
    validate_plan(plan)?;
    if coefficients.max_degree() == plan.max_degree() {
        Ok(())
    } else {
        Err(WgpuError::ShapeMismatch {
            message: format!(
                "coefficient shape mismatch: expected max_degree {}, got {}",
                plan.max_degree(),
                coefficients.max_degree()
            ),
        })
    }
}

pub(super) fn validate_typed_input<T: ShtGpuStorage>(
    plan: &ShtWgpuPlan,
    precision: PrecisionProfile,
    input_len: usize,
) -> WgpuResult<()> {
    if precision != T::PROFILE {
        return Err(WgpuError::InvalidPrecisionProfile);
    }
    validate_forward(plan, input_len)
}

pub(super) fn validate_typed_output<T: ShtGpuStorage>(
    plan: &ShtWgpuPlan,
    precision: PrecisionProfile,
    output_len: usize,
) -> WgpuResult<()> {
    if precision != T::PROFILE {
        return Err(WgpuError::InvalidPrecisionProfile);
    }
    validate_plan(plan)?;
    validate_length(plan.sample_count(), output_len, "output")
}

fn validate_length(expected: usize, actual: usize, name: &str) -> WgpuResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(WgpuError::ShapeMismatch {
            message: format!("{name} length expected {expected}, got {actual}"),
        })
    }
}

pub(super) fn mode_pairs(max_degree: usize) -> impl Iterator<Item = (usize, isize)> {
    (0..=max_degree).flat_map(|degree| {
        let degree_as_order =
            isize::try_from(degree).expect("invariant: accelerator mode degree fits isize");
        (-degree_as_order..=degree_as_order).map(move |order| (degree, order))
    })
}

pub(super) fn grid_samples(plan: &ShtWgpuPlan) -> WgpuResult<Vec<GridPod>> {
    let (cos_theta_nodes, theta_weights) = gauss_legendre_nodes_weights(plan.latitudes());
    let longitude_count = u32::try_from(plan.longitudes()).map_err(|_| WgpuError::InvalidPlan {
        message: format!("longitude count {} exceeds u32", plan.longitudes()),
    })?;
    let longitude_weight = std::f64::consts::TAU / f64::from(longitude_count);
    (0..plan.latitudes())
        .flat_map(|latitude| {
            let cos_theta = cos_theta_nodes[latitude];
            let weight = theta_weights[latitude] * longitude_weight;
            (0..plan.longitudes()).map(move |longitude| (cos_theta, weight, longitude))
        })
        .map(|(cos_theta, weight, longitude)| {
            let longitude = u32::try_from(longitude).map_err(|_| WgpuError::InvalidPlan {
                message: format!("longitude index {longitude} exceeds u32"),
            })?;
            Ok(GridPod {
                cos_theta: quantize_grid_component(cos_theta, "cos(theta)")?,
                phi: quantize_grid_component(
                    std::f64::consts::TAU * f64::from(longitude) / f64::from(longitude_count),
                    "phi",
                )?,
                weight: quantize_grid_component(weight, "quadrature weight")?,
                padding: 0.0,
            })
        })
        .collect()
}

pub(super) fn coefficients_from_modes(
    max_degree: usize,
    raw: &[Complex32],
) -> SphericalHarmonicCoefficients {
    let mut coefficients = SphericalHarmonicCoefficients::zeros(max_degree);
    for ((degree, order), value) in mode_pairs(max_degree).zip(raw.iter().copied()) {
        coefficients.set(
            degree,
            order,
            Complex64::new(f64::from(value.re), f64::from(value.im)),
        );
    }
    coefficients
}

pub(super) fn populate_modes(
    output: &mut [Complex32],
    coefficients: &SphericalHarmonicCoefficients,
) -> WgpuResult<()> {
    for (slot, (degree, order)) in output.iter_mut().zip(mode_pairs(coefficients.max_degree())) {
        let value = coefficients.get(degree, order);
        *slot = Complex32::new(
            exact_accelerator_component(value.re, "real")?,
            exact_accelerator_component(value.im, "imaginary")?,
        );
    }
    Ok(())
}

pub(super) fn exact_accelerator_component(value: f64, component: &'static str) -> WgpuResult<f32> {
    let represented = quantize_accelerator_component(value, component)?;
    if f64::from(represented) == value {
        Ok(represented)
    } else {
        Err(WgpuError::PrecisionLoss { component, value })
    }
}

pub(super) fn quantize_accelerator_component(
    value: f64,
    component: &'static str,
) -> WgpuResult<f32> {
    let represented = value as f32;
    if value.is_finite() && represented.is_finite() {
        Ok(represented)
    } else {
        Err(WgpuError::PrecisionLoss { component, value })
    }
}

fn quantize_grid_component(value: f64, component: &'static str) -> WgpuResult<f32> {
    quantize_accelerator_component(value, component)
}

pub(super) fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    leto_interop::view1_cow(&view)
}

pub(super) fn array2_from_leto_view<T: Copy>(view: leto::ArrayView2<'_, T>) -> Array2<T> {
    view.to_contiguous()
}

pub(super) fn coefficients_from_leto_view(
    plan: &ShtWgpuPlan,
    coefficients: leto::ArrayView2<'_, Complex64>,
) -> WgpuResult<SphericalHarmonicCoefficients> {
    validate_plan(plan)?;
    let values = array2_from_leto_view(coefficients);
    let expected = [plan.max_degree() + 1, 2 * plan.max_degree() + 1];
    if values.shape() != expected {
        let [rows, columns] = values.shape();
        return Err(WgpuError::ShapeMismatch {
            message: format!(
                "coefficient shape mismatch: expected {}x{}, got {}x{}",
                expected[0], expected[1], rows, columns
            ),
        });
    }
    Ok(SphericalHarmonicCoefficients::from_values(
        plan.max_degree(),
        values,
    ))
}

pub(super) fn leto_array2_from_dense<T: Copy>(
    values: &Array2<T>,
    label: &str,
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
    leto_interop::try_dense_from_contiguous(values).ok_or_else(|| WgpuError::InvalidPlan {
        message: format!("failed to allocate Mnemosyne-backed Leto {label}"),
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use eunomia::Complex64;
    use leto::SliceArg;

    use super::{leto_view1_cow, populate_modes};
    use crate::SphericalHarmonicCoefficients;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec([4], vec![1_u32, 2, 3, 4]).expect("input");
        let cow = leto_view1_cow(input.view());
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn leto_view1_cow_materializes_strided_views() {
        let input =
            leto::Array1::from_shape_vec([8], vec![1_u32, 99, 2, 99, 3, 99, 4, 99]).expect("input");
        let view = input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let cow = leto_view1_cow(view);
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn inverse_rejects_nonrepresentable_coefficients_before_provider_dispatch() {
        let mut coefficients = SphericalHarmonicCoefficients::zeros(0);
        coefficients.set(0, 0, Complex64::new(0.1, 0.0));
        let mut modes = [eunomia::Complex32::new(0.0, 0.0)];

        let error = populate_modes(&mut modes, &coefficients)
            .expect_err("nonrepresentable coefficient must be rejected");

        assert!(matches!(
            error,
            crate::infrastructure::transport::gpu::WgpuError::PrecisionLoss {
                component: "real",
                value,
            } if value == 0.1
        ));
    }
}
