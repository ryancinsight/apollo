use apollo_fft::Complex32;
use eunomia::Complex64;

use crate::domain::spectrum::sparse::SparseSpectrum;

use super::{SftWgpuPlan, WgpuError, WgpuResult};

pub(super) fn validate_spectrum(plan: &SftWgpuPlan, spectrum: &SparseSpectrum) -> WgpuResult<()> {
    if spectrum.n != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: spectrum.n,
        });
    }
    if spectrum.frequencies.len() != spectrum.values.len() {
        return Err(WgpuError::InvalidPlan {
            message: format!(
                "sparse spectrum frequency/value lengths differ: {} != {}",
                spectrum.frequencies.len(),
                spectrum.values.len()
            ),
        });
    }
    Ok(())
}

pub(super) fn populate_dense_spectrum(
    dense: &mut [Complex32],
    spectrum: &SparseSpectrum,
    len: usize,
) -> WgpuResult<()> {
    for (&frequency, &value) in spectrum.frequencies.iter().zip(spectrum.values.iter()) {
        let Some(slot) = dense.get_mut(frequency) else {
            return Err(WgpuError::InvalidPlan {
                message: format!("sparse frequency {frequency} is outside transform length {len}"),
            });
        };
        *slot = Complex32::new(
            exact_accelerator_component(value.re, "real")?,
            exact_accelerator_component(value.im, "imaginary")?,
        );
    }
    Ok(())
}

pub(super) fn exact_accelerator_component(value: f64, component: &'static str) -> WgpuResult<f32> {
    let represented = quantize_accelerator_component(value, component)?;
    if value.is_finite() && f64::from(represented) == value {
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

pub(super) fn select_top_k(
    len: usize,
    sparsity: usize,
    dense: &[Complex32],
) -> WgpuResult<SparseSpectrum> {
    let mut ranked: Vec<(usize, Complex32, f32)> = dense
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (index, value, value.norm_sqr()))
        .filter(|(_, _, energy)| *energy > 0.0)
        .collect();
    ranked.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(sparsity);
    ranked.sort_by_key(|(index, _, _)| *index);

    let mut spectrum = SparseSpectrum::new(len);
    for (frequency, value, _) in ranked {
        spectrum
            .insert(
                frequency,
                Complex64::new(f64::from(value.re), f64::from(value.im)),
            )
            .map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan len={len}, sparsity={sparsity}: selected support violates sparse spectrum invariants"
                ),
            })?;
    }
    Ok(spectrum)
}
