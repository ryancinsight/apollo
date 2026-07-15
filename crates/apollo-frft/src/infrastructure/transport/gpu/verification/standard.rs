use eunomia::Complex32;
use leto::{SliceArg, Storage};

use crate::infrastructure::transport::gpu::FrftWgpuPlan;

use super::support::{
    assert_complex32_close, assert_cpu_differential, backend, cpu_input,
    STANDARD_IDENTITY_TOLERANCE, STANDARD_ROUNDTRIP_TOLERANCE,
};

#[test]
fn forward_at_order_zero_is_identity() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..8)
        .map(|index| Complex32::new(index as f32 * 0.1_f32, -(index as f32) * 0.05_f32))
        .collect::<Vec<_>>();
    let plan = FrftWgpuPlan::new(input.len(), 0.0_f32);
    let output = backend
        .execute_forward(&plan, &input)
        .expect("forward order 0");
    assert_complex32_close(&output, &input, STANDARD_IDENTITY_TOLERANCE, "identity");
}

#[test]
fn forward_matches_cpu_frft_for_existing_orders() {
    let Some(backend) = backend() else {
        return;
    };
    let order_one_input = (0..16)
        .map(|index| Complex32::new((index as f32 * 0.31_f32).sin(), 0.0_f32))
        .collect::<Vec<_>>();
    let general_order_input = (0..8)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.4_f32).cos(),
                (index as f32 * 0.3_f32).sin(),
            )
        })
        .collect::<Vec<_>>();

    for (input, order) in [(&order_one_input, 1.0_f32), (&general_order_input, 0.5_f32)] {
        let plan = FrftWgpuPlan::new(input.len(), order);
        let actual = backend.execute_forward(&plan, input).expect("GPU forward");
        let expected = crate::frft(&cpu_input(input), order as f64).expect("CPU FrFT");
        assert_cpu_differential(&actual, expected.as_slice().expect("contiguous CPU output"));
    }
}

#[test]
fn leto_forward_and_inverse_match_slice_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..8)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.4_f32).cos(),
                (index as f32 * 0.3_f32).sin(),
            )
        })
        .collect::<Vec<_>>();
    let plan = FrftWgpuPlan::new(input.len(), 0.5_f32);
    let expected_forward = backend
        .execute_forward(&plan, &input)
        .expect("slice forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual_forward = backend
        .execute_forward_leto(&plan, leto_input.view())
        .expect("Leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward)
        .expect("slice inverse");
    let leto_spectrum = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto spectrum");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, leto_spectrum.view())
        .expect("Leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn strided_leto_forward_matches_logical_slice() {
    let Some(backend) = backend() else {
        return;
    };
    let logical = (0..8)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.2_f32).sin(),
                (index as f32 * 0.5_f32).cos(),
            )
        })
        .collect::<Vec<_>>();
    let sentinel = Complex32::new(99.0, -99.0);
    let backing = logical
        .iter()
        .copied()
        .flat_map(|value| [value, sentinel])
        .collect::<Vec<_>>();
    let plan = FrftWgpuPlan::new(logical.len(), 0.5_f32);
    let expected = backend
        .execute_forward(&plan, &logical)
        .expect("slice forward");
    let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided view");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("strided Leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn inverse_recovers_input() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..16)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.31_f32).sin(),
                (index as f32 * 0.17_f32).cos(),
            )
        })
        .collect::<Vec<_>>();
    let plan = FrftWgpuPlan::new(input.len(), 1.0_f32);
    let spectrum = backend
        .execute_forward(&plan, &input)
        .expect("forward for roundtrip");
    let recovered = backend
        .execute_inverse(&plan, &spectrum)
        .expect("inverse for roundtrip");
    assert_complex32_close(
        &recovered,
        &input,
        STANDARD_ROUNDTRIP_TOLERANCE,
        "roundtrip",
    );
}
