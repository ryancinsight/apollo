use eunomia::Complex32;
use leto::Storage;

use crate::infrastructure::transport::gpu::UnitaryFrftWgpuPlan;

use super::support::{
    assert_complex32_close, assert_cpu_differential, backend, cpu_input, UNITARY_VALUE_TOLERANCE,
};

#[test]
fn forward_order_zero_is_identity() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..8)
        .map(|index| {
            Complex32::new(
                index as f32 * 0.1_f32 + 0.5_f32,
                -(index as f32 * 0.07_f32) + 0.2_f32,
            )
        })
        .collect::<Vec<_>>();
    let plan = UnitaryFrftWgpuPlan::new(input.len(), 0.0_f32);
    let output = backend
        .execute_unitary_forward(&plan, &input)
        .expect("unitary forward order 0");
    assert_complex32_close(&output, &input, UNITARY_VALUE_TOLERANCE, "identity");
}

#[test]
fn leto_forward_and_inverse_match_slice_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..8)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.23_f32).sin(),
                (index as f32 * 0.31_f32).cos(),
            )
        })
        .collect::<Vec<_>>();
    let plan = UnitaryFrftWgpuPlan::new(input.len(), 0.5_f32);
    let expected_forward = backend
        .execute_unitary_forward(&plan, &input)
        .expect("unitary forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual_forward = backend
        .execute_unitary_forward_leto(&plan, leto_input.view())
        .expect("Leto unitary forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_unitary_inverse(&plan, &expected_forward)
        .expect("unitary inverse");
    let leto_spectrum = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto spectrum");
    let actual_inverse = backend
        .execute_unitary_inverse_leto(&plan, leto_spectrum.view())
        .expect("Leto unitary inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn forward_order_two_is_reversal() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..8)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.31_f32).sin(),
                (index as f32 * 0.17_f32).cos(),
            )
        })
        .collect::<Vec<_>>();
    let plan = UnitaryFrftWgpuPlan::new(input.len(), 2.0_f32);
    let output = backend
        .execute_unitary_forward(&plan, &input)
        .expect("unitary forward order 2");
    let expected = input.iter().copied().rev().collect::<Vec<_>>();
    assert_complex32_close(&output, &expected, UNITARY_VALUE_TOLERANCE, "reversal");
}

#[test]
fn forward_and_inverse_roundtrip_for_existing_orders() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..16)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.23_f32).sin(),
                (index as f32 * 0.31_f32).cos(),
            )
        })
        .collect::<Vec<_>>();
    for order in [0.3_f32, 0.5, 0.7, 1.3, 2.5, 3.1] {
        let plan = UnitaryFrftWgpuPlan::new(input.len(), order);
        let spectrum = backend
            .execute_unitary_forward(&plan, &input)
            .expect("unitary forward for roundtrip");
        let recovered = backend
            .execute_unitary_inverse(&plan, &spectrum)
            .expect("unitary inverse for roundtrip");
        let maximum_error = recovered
            .iter()
            .zip(input.iter())
            .map(|(actual, expected)| (actual - expected).norm())
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_error < 1.0e-4_f32,
            "roundtrip failed at order={order}: max_element_error={maximum_error:.2e}"
        );
    }
}

#[test]
fn forward_preserves_l2_norm_for_existing_orders() {
    let Some(backend) = backend() else {
        return;
    };
    let input = (0..16)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.37_f32).cos(),
                (index as f32 * 0.41_f32).sin(),
            )
        })
        .collect::<Vec<_>>();
    let input_norm_squared: f32 = input.iter().map(|value| value.norm_sqr()).sum();
    for order in [0.3_f32, 0.7, 1.2, 1.8, 2.7] {
        let plan = UnitaryFrftWgpuPlan::new(input.len(), order);
        let output = backend
            .execute_unitary_forward(&plan, &input)
            .expect("unitary forward for norm test");
        let output_norm_squared: f32 = output.iter().map(|value| value.norm_sqr()).sum();
        let relative_error = (output_norm_squared - input_norm_squared).abs() / input_norm_squared;
        assert!(
            relative_error < 5.0e-5_f32,
            "norm not preserved at order={order}: ||output||²={output_norm_squared:.8}, ||input||²={input_norm_squared:.8}, rel_err={relative_error:.2e}"
        );
    }
}

#[test]
fn gpu_matches_cpu_reference() {
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
    let order = 0.5_f32;
    let gpu_plan = UnitaryFrftWgpuPlan::new(input.len(), order);
    let actual = backend
        .execute_unitary_forward(&gpu_plan, &input)
        .expect("GPU unitary forward");
    let cpu_plan =
        crate::UnitaryFrftPlan::new(input.len(), order as f64).expect("CPU unitary plan");
    let expected = cpu_plan
        .forward(&cpu_input(&input))
        .expect("CPU unitary forward");
    assert_cpu_differential(&actual, expected.as_slice().expect("contiguous CPU output"));
}
