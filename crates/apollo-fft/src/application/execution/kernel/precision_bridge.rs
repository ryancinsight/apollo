//! Generic storage-precision bridge for kernels executed in `Complex32`.

#![allow(clippy::uninit_vec)]

use eunomia::{Complex, Complex32};
use half::f16;
use mnemosyne::scratch::ScratchPool;

thread_local! {
    static COMPLEX32_BRIDGE_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// Storage element that can be transformed through a `Complex32` execution buffer.
pub(crate) trait Complex32Bridge: Copy {
    /// Load the storage element into `Complex32` compute representation.
    fn to_complex32(self) -> Complex32;

    /// Store a `Complex32` compute result back into the storage representation.
    fn from_complex32(value: Complex32) -> Self;

    /// Widen a whole slice into the compute buffer.
    ///
    /// The promotion is not a detail of a half-precision transform, it is most
    /// of its cost: the kernel runs one `Complex32` transform while the bridge
    /// touches every lane twice. The default is the element-wise loop; a
    /// storage type whose widening has a vectorized form overrides this.
    fn widen_slice(src: &[Self], dst: &mut [Complex32]) {
        for (slot, value) in dst.iter_mut().zip(src) {
            *slot = value.to_complex32();
        }
    }

    /// Narrow the compute buffer back into storage; the inverse of
    /// [`Complex32Bridge::widen_slice`], with the same rationale.
    fn narrow_slice(src: &[Complex32], dst: &mut [Self]) {
        for (slot, value) in dst.iter_mut().zip(src) {
            *slot = Self::from_complex32(*value);
        }
    }
}

impl Complex32Bridge for Complex<f16> {
    #[inline]
    fn to_complex32(self) -> Complex32 {
        Complex32::new(self.re.to_f32(), self.im.to_f32())
    }

    #[inline]
    fn from_complex32(value: Complex32) -> Self {
        Self::new(f16::from_f32(value.re), f16::from_f32(value.im))
    }

    /// One `vcvtph2ps` per eight lanes through eunomia's bulk widen, which
    /// owns the `binary16` conversion vocabulary for the stack and dispatches
    /// F16C at runtime (scalar elsewhere). A complex sample is two adjacent
    /// `binary16` lanes and a `Complex32` two adjacent `f32`, so the
    /// interleaved buffers convert as flat lane arrays with no shuffling.
    #[inline]
    fn widen_slice(src: &[Self], dst: &mut [Complex32]) {
        eunomia::F16::widen_slice(half_lanes(src), float_lanes_mut(dst));
    }

    /// The inverse: one `vcvtps2ph` per eight lanes, rounding to nearest with
    /// ties to even — the same rounding the element-wise `f16::from_f32` path
    /// applies, so the two agree bit for bit.
    #[inline]
    fn narrow_slice(src: &[Complex32], dst: &mut [Self]) {
        eunomia::F16::narrow_slice(float_lanes(src), half_lanes_mut(dst));
    }
}

/// Interleaved `binary16` samples as the flat lane array eunomia converts.
///
/// `Complex<f16>` is `#[repr(C)]` over two `half::f16`, each
/// `#[repr(transparent)]` over `u16`, and `eunomia::F16` is likewise
/// transparent over `u16`; both reinterprets are layout no-ops between
/// plain-old-data types.
#[inline]
fn half_lanes(data: &[Complex<f16>]) -> &[eunomia::F16] {
    eunomia::layout::cast_slice(bytemuck::cast_slice::<_, u16>(data))
}

/// Mutable form of [`half_lanes`].
#[inline]
fn half_lanes_mut(data: &mut [Complex<f16>]) -> &mut [eunomia::F16] {
    eunomia::layout::cast_slice_mut(bytemuck::cast_slice_mut::<_, u16>(data))
}

/// Interleaved `Complex32` samples as their flat `f32` lane array.
#[inline]
fn float_lanes(data: &[Complex32]) -> &[f32] {
    bytemuck::cast_slice(data)
}

/// Mutable form of [`float_lanes`].
#[inline]
fn float_lanes_mut(data: &mut [Complex32]) -> &mut [f32] {
    bytemuck::cast_slice_mut(data)
}

/// Execute `kernel` over a reused `Complex32` scratch view and store results back.
#[inline]
pub(crate) fn run_via_complex32<S, F>(data: &mut [S], kernel: F)
where
    S: Complex32Bridge,
    F: FnOnce(&mut [Complex32]),
{
    let n = data.len();
    COMPLEX32_BRIDGE_SCRATCH.with(|pool| {
        pool.with_scratch(n, |scratch| {
            S::widen_slice(data, scratch);
            kernel(scratch);
            S::narrow_slice(scratch, data);
        });
    });
}

#[cfg(test)]
mod tests {
    use super::{Complex, Complex32, Complex32Bridge};
    use half::f16;

    /// Every `binary16` bit pattern, including the subnormals, infinities and
    /// NaNs the vector path handles in hardware.
    fn every_half_pattern() -> Vec<Complex<f16>> {
        (0..=u16::MAX)
            .step_by(3)
            .map(|bits| Complex::new(f16::from_bits(bits), f16::from_bits(bits.rotate_left(7))))
            .collect()
    }

    /// The bulk widen must agree with the element-wise definition bit for bit:
    /// it is an acceleration of that contract, not a second one. A NaN payload
    /// or a subnormal that the vector path rounded differently would show here.
    #[test]
    fn bulk_widen_matches_the_element_wise_bridge() {
        let source = every_half_pattern();
        let mut bulk = vec![Complex32::new(0.0, 0.0); source.len()];
        <Complex<f16> as Complex32Bridge>::widen_slice(&source, &mut bulk);
        for (index, (widened, value)) in bulk.iter().zip(&source).enumerate() {
            let expected = value.to_complex32();
            assert_eq!(
                (widened.re.to_bits(), widened.im.to_bits()),
                (expected.re.to_bits(), expected.im.to_bits()),
                "lane {index} widened differently in bulk"
            );
        }
    }

    /// The same for the narrowing direction, over f32 values that exercise
    /// rounding ties, overflow to infinity, and flush-to-subnormal.
    #[test]
    fn bulk_narrow_matches_the_element_wise_bridge() {
        let source: Vec<Complex32> = (0..4096)
            .map(|index| {
                let x = index as f32;
                Complex32::new(
                    (x * 0.013).sin() * 65_600.0,
                    (x * 0.001).exp() * f32::from(f16::EPSILON) * 0.5,
                )
            })
            .chain([
                Complex32::new(f32::INFINITY, f32::NEG_INFINITY),
                Complex32::new(f32::NAN, 0.0),
                Complex32::new(1.0 + f32::EPSILON, -0.0),
            ])
            .collect();
        let mut bulk = vec![Complex::new(f16::ZERO, f16::ZERO); source.len()];
        <Complex<f16> as Complex32Bridge>::narrow_slice(&source, &mut bulk);
        for (index, (narrowed, value)) in bulk.iter().zip(&source).enumerate() {
            let expected = <Complex<f16> as Complex32Bridge>::from_complex32(*value);
            assert_eq!(
                (narrowed.re.to_bits(), narrowed.im.to_bits()),
                (expected.re.to_bits(), expected.im.to_bits()),
                "lane {index} narrowed differently in bulk"
            );
        }
    }

    /// A round trip through the bridge preserves every representable value:
    /// widening is exact and narrowing returns the pattern it came from.
    #[test]
    fn widen_then_narrow_is_the_identity_on_storage() {
        let source = every_half_pattern();
        let mut buffer = vec![Complex32::new(0.0, 0.0); source.len()];
        let mut back = vec![Complex::new(f16::ZERO, f16::ZERO); source.len()];
        <Complex<f16> as Complex32Bridge>::widen_slice(&source, &mut buffer);
        <Complex<f16> as Complex32Bridge>::narrow_slice(&buffer, &mut back);
        for (index, (returned, original)) in back.iter().zip(&source).enumerate() {
            if original.re.is_nan() || original.im.is_nan() {
                assert!(returned.re.is_nan() || returned.im.is_nan(), "lane {index}");
                continue;
            }
            assert_eq!(
                (returned.re.to_bits(), returned.im.to_bits()),
                (original.re.to_bits(), original.im.to_bits()),
                "lane {index} did not survive the round trip"
            );
        }
    }
}
