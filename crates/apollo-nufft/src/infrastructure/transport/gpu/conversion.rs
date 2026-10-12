//! Host-side conversion and validation helpers for NUFFT WGPU execution.

use crate::NufftComplexStorage;
use apollo_fft::{PrecisionProfile, WgpuError, WgpuResult};
use eunomia::{Complex32, Complex64};
use leto::Array3;
use std::borrow::Cow;

pub(crate) fn validate_pair_lengths(expected: usize, actual: usize) -> WgpuResult<()> {
    if expected != actual {
        return Err(WgpuError::LengthMismatch { expected, actual });
    }
    Ok(())
}

pub(crate) fn validate_typed_profile<T: NufftComplexStorage>(
    actual: PrecisionProfile,
) -> WgpuResult<()> {
    let expected = T::PROFILE;
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(WgpuError::InvalidPrecisionProfile)
    }
}

pub(crate) fn typed_to_complex32<T: NufftComplexStorage>(values: &[T]) -> Cow<'_, [Complex32]> {
    if let Some(slice_c32) = T::as_c32_slice(values) {
        Cow::Borrowed(slice_c32)
    } else {
        let vec: Vec<Complex32> = values
            .iter()
            .copied()
            .map(|value| {
                let represented = value.to_cpu();
                Complex32::new(represented.re as f32, represented.im as f32)
            })
            .collect();
        Cow::Owned(vec)
    }
}

pub(crate) fn write_typed_output<T: NufftComplexStorage>(source: &[Complex64], target: &mut [T]) {
    if let Some(slice_c64) = T::as_c64_slice_mut(target) {
        slice_c64.copy_from_slice(source);
    } else {
        for (slot, value) in target.iter_mut().zip(source.iter().copied()) {
            *slot = T::from_cpu(value);
        }
    }
}

pub(crate) fn write_complex_output(source: &[Complex32], target: &mut [Complex64]) {
    for (slot, value) in target.iter_mut().zip(source.iter().copied()) {
        *slot = Complex64::new(value.re as f64, value.im as f64);
    }
}

pub(crate) fn validate_usize_to_u32(value: usize) -> WgpuResult<()> {
    if value > u32::MAX as usize {
        return Err(WgpuError::InvalidPlan {
            message: format!("WGPU dispatch dimension {value} must fit in u32"),
        });
    }
    Ok(())
}

pub(crate) fn positions3_from_leto_view(
    view: leto::ArrayView2<'_, f32>,
) -> WgpuResult<Vec<(f32, f32, f32)>> {
    let shape = view.shape();
    if shape[1] != 3 {
        return Err(WgpuError::InvalidPlan {
            message: "3D Leto position view must have shape [samples, 3]".to_owned(),
        });
    }
    let mut values = Vec::with_capacity(shape[0]);
    for row in 0..shape[0] {
        values.push((
            *view.get([row, 0]).map_err(|_| WgpuError::InvalidPlan {
                message: "invalid Leto NUFFT-WGPU 3D position view".to_owned(),
            })?,
            *view.get([row, 1]).map_err(|_| WgpuError::InvalidPlan {
                message: "invalid Leto NUFFT-WGPU 3D position view".to_owned(),
            })?,
            *view.get([row, 2]).map_err(|_| WgpuError::InvalidPlan {
                message: "invalid Leto NUFFT-WGPU 3D position view".to_owned(),
            })?,
        ));
    }
    Ok(values)
}

pub(crate) fn array3_from_leto_view<T: Copy>(view: leto::ArrayView3<'_, T>) -> Array3<T> {
    view.to_contiguous()
}

pub(crate) fn host_array_error() -> WgpuError {
    WgpuError::InvalidPlan {
        message: "failed to allocate Mnemosyne-backed Leto NUFFT-WGPU 1D output".to_owned(),
    }
}

pub(crate) fn host_dense_error() -> WgpuError {
    WgpuError::InvalidPlan {
        message: "failed to allocate Mnemosyne-backed Leto NUFFT-WGPU 3D output".to_owned(),
    }
}
