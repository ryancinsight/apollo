use crate::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::domain::error::{NufftWgpuError, NufftWgpuResult};
use apollo_fft::application::utilities::leto_interop;
use apollo_fft::PrecisionProfile;
use apollo_nufft::infrastructure::kernel::kaiser_bessel::{fft_signed_index, i0, kb_kernel_ft};
use apollo_nufft::NufftComplexStorage;
use ndarray::Array3;
use num_complex::{Complex32, Complex64};
use std::borrow::Cow;

pub(crate) struct Fast1DMetadata {
    pub(crate) oversampled_len: usize,
    pub(crate) beta: f64,
    pub(crate) i0_beta: f64,
    pub(crate) deconv: Vec<f32>,
}

pub(crate) struct Fast3DMetadata {
    pub(crate) mx: usize,
    pub(crate) my: usize,
    pub(crate) mz: usize,
    pub(crate) beta: f64,
    pub(crate) i0_beta: f64,
    pub(crate) deconv_xyz: Vec<f32>,
}

pub(crate) fn validate_pair_lengths(expected: usize, actual: usize) -> NufftWgpuResult<()> {
    if expected != actual {
        return Err(NufftWgpuError::InputLengthMismatch { expected, actual });
    }
    Ok(())
}

pub(crate) fn validate_typed_profile<T: NufftComplexStorage>(
    actual: PrecisionProfile,
) -> NufftWgpuResult<()> {
    let expected = T::PROFILE;
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(NufftWgpuError::InvalidPlan {
            message: "precision profile does not match typed NUFFT-WGPU storage",
        })
    }
}

pub(crate) fn typed_to_complex32<T: NufftComplexStorage>(values: &[T]) -> Cow<'_, [Complex32]> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
        // Safety: T is Complex32, so &[T] is layout-compatible with &[Complex32].
        let slice_c32 = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<Complex32>(), values.len())
        };
        Cow::Borrowed(slice_c32)
    } else {
        let vec: Vec<Complex32> = values
            .iter()
            .copied()
            .map(|value| {
                let represented = value.to_complex64();
                Complex32::new(represented.re as f32, represented.im as f32)
            })
            .collect();
        Cow::Owned(vec)
    }
}

pub(crate) fn write_typed_output<T: NufftComplexStorage>(source: &[Complex64], target: &mut [T]) {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        // Safety: T is Complex64, so &mut [T] is layout-compatible with &mut [Complex64].
        let slice_c64 = unsafe {
            std::slice::from_raw_parts_mut(target.as_mut_ptr().cast::<Complex64>(), target.len())
        };
        slice_c64.copy_from_slice(source);
    } else {
        for (slot, value) in target.iter_mut().zip(source.iter().copied()) {
            *slot = T::from_complex64(value);
        }
    }
}

pub(crate) fn validate_usize_to_u32(value: usize) -> NufftWgpuResult<()> {
    if value > u32::MAX as usize {
        return Err(NufftWgpuError::InvalidPlan {
            message: "WGPU dispatch dimension must fit in u32",
        });
    }
    Ok(())
}

pub(crate) fn validate_fast_1d_plan(plan: &NufftWgpuPlan1D) -> NufftWgpuResult<()> {
    if plan.oversampling() < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 1D NUFFT oversampling factor must be >= 2",
        });
    }
    if plan.kernel_width() < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 1D NUFFT kernel width must be >= 2",
        });
    }
    validate_usize_to_u32(plan.domain().n)?;
    let Some(oversampled_len) = plan.domain().n.checked_mul(plan.oversampling()) else {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 1D NUFFT oversampled length overflow",
        });
    };
    validate_usize_to_u32(oversampled_len)
}

pub(crate) fn fast_1d_metadata(plan: &NufftWgpuPlan1D) -> NufftWgpuResult<Fast1DMetadata> {
    validate_fast_1d_plan(plan)?;
    let oversampled_len =
        plan.domain()
            .n
            .checked_mul(plan.oversampling())
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "fast 1D NUFFT oversampled length overflow",
            })?;
    let beta = std::f64::consts::PI
        * (1.0 - 1.0 / (2.0 * plan.oversampling() as f64))
        * (2 * plan.kernel_width()) as f64;
    let i0_beta = i0(beta);
    let deconv = (0..plan.domain().n)
        .map(|k| {
            let xi = fft_signed_index(k, plan.domain().n) as f64 / oversampled_len as f64;
            (1.0 / kb_kernel_ft(xi, plan.kernel_width(), beta, i0_beta)) as f32
        })
        .collect();
    Ok(Fast1DMetadata {
        oversampled_len,
        beta,
        i0_beta,
        deconv,
    })
}

pub(crate) fn fast_3d_metadata(plan: &NufftWgpuPlan3D) -> NufftWgpuResult<Fast3DMetadata> {
    let grid = plan.grid();
    let sigma = plan.oversampling();
    let w = plan.kernel_width();
    if sigma < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT oversampling factor must be >= 2",
        });
    }
    if w < 2 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT kernel width must be >= 2",
        });
    }
    let mx_raw = grid
        .nx
        .checked_mul(sigma)
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mx overflow",
        })?
        .max(2 * w + 1);
    let my_raw = grid
        .ny
        .checked_mul(sigma)
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT my overflow",
        })?
        .max(2 * w + 1);
    let mz_raw = grid
        .nz
        .checked_mul(sigma)
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mz overflow",
        })?
        .max(2 * w + 1);
    let mx = mx_raw
        .checked_next_power_of_two()
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mx radix-2 length overflow",
        })?;
    let my = my_raw
        .checked_next_power_of_two()
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT my radix-2 length overflow",
        })?;
    let mz = mz_raw
        .checked_next_power_of_two()
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT mz radix-2 length overflow",
        })?;
    validate_usize_to_u32(mx)?;
    validate_usize_to_u32(my)?;
    validate_usize_to_u32(mz)?;
    validate_usize_to_u32(
        mx.checked_mul(my)
            .and_then(|v| v.checked_mul(mz))
            .unwrap_or(usize::MAX),
    )?;

    let beta = std::f64::consts::PI * (1.0 - 1.0 / (2.0 * sigma as f64)) * (2 * w) as f64;
    let i0_beta = i0(beta);

    let deconv_x: Vec<f32> = (0..grid.nx)
        .map(|k| {
            let xi = fft_signed_index(k, grid.nx) as f64 / mx as f64;
            (1.0 / kb_kernel_ft(xi, w, beta, i0_beta)) as f32
        })
        .collect();
    let deconv_y: Vec<f32> = (0..grid.ny)
        .map(|k| {
            let xi = fft_signed_index(k, grid.ny) as f64 / my as f64;
            (1.0 / kb_kernel_ft(xi, w, beta, i0_beta)) as f32
        })
        .collect();
    let deconv_z: Vec<f32> = (0..grid.nz)
        .map(|k| {
            let xi = fft_signed_index(k, grid.nz) as f64 / mz as f64;
            (1.0 / kb_kernel_ft(xi, w, beta, i0_beta)) as f32
        })
        .collect();

    let mut deconv_xyz = Vec::with_capacity(grid.nx + grid.ny + grid.nz);
    deconv_xyz.extend_from_slice(&deconv_x);
    deconv_xyz.extend_from_slice(&deconv_y);
    deconv_xyz.extend_from_slice(&deconv_z);

    Ok(Fast3DMetadata {
        mx,
        my,
        mz,
        beta,
        i0_beta,
        deconv_xyz,
    })
}

pub(crate) fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    leto_interop::view1_cow(&view)
}
pub(crate) fn positions3_from_leto_view(
    view: leto::ArrayView2<'_, f32>,
) -> NufftWgpuResult<Vec<(f32, f32, f32)>> {
    let shape = view.shape();
    if shape[1] != 3 {
        return Err(NufftWgpuError::InvalidPlan {
            message: "3D Leto position view must have shape [samples, 3]",
        });
    }
    let mut values = Vec::with_capacity(shape[0]);
    for row in 0..shape[0] {
        values.push((
            *view
                .get([row, 0])
                .map_err(|_| NufftWgpuError::InvalidPlan {
                    message: "invalid Leto NUFFT-WGPU 3D position view",
                })?,
            *view
                .get([row, 1])
                .map_err(|_| NufftWgpuError::InvalidPlan {
                    message: "invalid Leto NUFFT-WGPU 3D position view",
                })?,
            *view
                .get([row, 2])
                .map_err(|_| NufftWgpuError::InvalidPlan {
                    message: "invalid Leto NUFFT-WGPU 3D position view",
                })?,
        ));
    }
    Ok(values)
}

pub(crate) fn array3_from_leto_view<T: Copy>(view: leto::ArrayView3<'_, T>) -> Array3<T> {
    leto_interop::array3_from_view(&view)
}
pub(crate) fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto_interop::try_array1_from_slice(values).ok_or_else(|| NufftWgpuError::BufferMapFailed {
        message: "failed to allocate Mnemosyne-backed Leto NUFFT-WGPU 1D output".to_string(),
    })
}

pub(crate) fn leto_array3_from_ndarray<T: Copy>(
    values: &Array3<T>,
) -> NufftWgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>> {
    leto_interop::try_array3_from_ndarray(values).ok_or_else(|| NufftWgpuError::BufferMapFailed {
        message: "failed to allocate Mnemosyne-backed Leto NUFFT-WGPU 3D output".to_string(),
    })
}
