use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

#[cfg(test)]
struct LaneCount;

#[cfg(test)]
impl<T: LaneScalar> LaneKernel<T> for LaneCount {
    type Output = usize;

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> usize {
        <A as SimdStorage<T>>::LANE_COUNT
    }
}

#[cfg(test)]
pub(super) fn exact_lanes_supported<const LANES: usize, T: LaneScalar>() -> bool {
    hermes_simd::vectorize_lanes::<LANES, T, _>(LaneCount) == Some(LANES)
}

struct NativeLaneCount;

impl<T: LaneScalar> LaneKernel<T> for NativeLaneCount {
    type Output = Option<usize>;

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> Self::Output {
        (A::REGISTER_WIDTH_BITS != 0).then_some(<A as SimdStorage<T>>::LANE_COUNT)
    }
}

/// Reports whether a hardware SIMD backend, rather than Hermes' scalar
/// fallback, provides exactly `LANES` elements for `T`.
pub(super) fn native_lanes_supported<const LANES: usize, T: LaneScalar>() -> bool {
    hermes_simd::vectorize_lanes::<LANES, T, _>(NativeLaneCount) == Some(Some(LANES))
}
