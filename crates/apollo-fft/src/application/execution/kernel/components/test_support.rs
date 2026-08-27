use eunomia::Complex64;
use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage};

struct NativeLaneCount;

impl LaneKernel<f64> for NativeLaneCount {
    type Output = usize;

    fn call<A: SimdArch + SimdKernel<f64>>(self, _simd: Simd<f64, A>) -> usize {
        <A as SimdStorage<f64>>::LANE_COUNT
    }
}

pub(super) fn executed_or_declined_untouched(
    before: &[Complex64],
    after: &[Complex64],
    executed: bool,
) -> bool {
    let expected = hermes_simd::vectorize_lanes::<4, f64, _>(NativeLaneCount) == Some(4);
    assert_eq!(
        executed, expected,
        "kernel execution must match the independently dispatched four-lane capability"
    );
    if !expected {
        assert_eq!(after.len(), before.len(), "declined kernel changed length");
        for (index, (actual, original)) in after.iter().zip(before).enumerate() {
            assert_eq!(
                (actual.re.to_bits(), actual.im.to_bits()),
                (original.re.to_bits(), original.im.to_bits()),
                "declined kernel changed the representation at index {index}"
            );
        }
    }
    executed
}
