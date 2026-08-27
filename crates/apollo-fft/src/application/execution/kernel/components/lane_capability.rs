use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

struct LaneCount;

impl<T: LaneScalar> LaneKernel<T> for LaneCount {
    type Output = usize;

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> usize {
        <A as SimdStorage<T>>::LANE_COUNT
    }
}

pub(super) fn exact_lanes_supported<const LANES: usize, T: LaneScalar>() -> bool {
    hermes_simd::vectorize_lanes::<LANES, T, _>(LaneCount) == Some(LANES)
}
