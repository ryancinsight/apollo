use crate::{
    infrastructure::transport::gpu::{NttWgpuPlan, WgpuError},
    DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT,
};

use super::support::backend;

#[test]
fn reusable_buffers_match_allocating_forward_and_inverse() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1_u64, 4, 9, 16, 25, 36, 49, 64];
    let plan = backend.plan(input.len());
    let mut buffers = backend.create_buffers(&plan).expect("reusable buffers");

    let expected_forward = backend
        .execute_forward(&plan, &input)
        .expect("allocating forward");
    backend
        .execute_forward_with_buffers(&plan, &input, &mut buffers)
        .expect("buffered forward");
    assert_eq!(backend.buffer_output(&buffers), expected_forward.as_slice());

    let spectrum = backend.buffer_output(&buffers).to_vec();
    let expected_inverse = backend
        .execute_inverse(&plan, &spectrum)
        .expect("allocating inverse");
    backend
        .execute_inverse_with_buffers(&plan, &spectrum, &mut buffers)
        .expect("buffered inverse");
    assert_eq!(backend.buffer_output(&buffers), expected_inverse.as_slice());
    assert_eq!(backend.buffer_output(&buffers), input.as_slice());
}

#[test]
fn quantized_reusable_buffers_match_allocating_quantized_path() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![3_u32, 1, 4, 1, 5, 9, 2, 6];
    let plan = backend.plan(input.len());
    let mut buffers = backend.create_buffers(&plan).expect("reusable buffers");

    let mut expected_forward = vec![0_u32; input.len()];
    backend
        .execute_forward_quantized_into(&plan, &input, &mut expected_forward)
        .expect("allocating quantized forward");
    backend
        .execute_forward_quantized_with_buffers(&plan, &input, &mut buffers)
        .expect("buffered quantized forward");
    let expected_forward_residues = expected_forward
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    assert_eq!(
        backend.buffer_output(&buffers),
        expected_forward_residues.as_slice()
    );

    let mut expected_inverse = vec![0_u32; input.len()];
    backend
        .execute_inverse_quantized_into(&plan, &expected_forward, &mut expected_inverse)
        .expect("allocating quantized inverse");
    backend
        .execute_inverse_quantized_with_buffers(&plan, &expected_forward, &mut buffers)
        .expect("buffered quantized inverse");
    let expected_inverse_residues = expected_inverse
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    assert_eq!(
        backend.buffer_output(&buffers),
        expected_inverse_residues.as_slice()
    );
    assert_eq!(expected_inverse, input);
}

#[test]
fn reusable_buffers_reject_plan_length_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(8);
    let short_plan = backend.plan(4);
    let mut short_buffers = backend.create_buffers(&short_plan).expect("short buffers");
    let buffer_error = backend
        .execute_forward_with_buffers(&plan, &[0; 8], &mut short_buffers)
        .expect_err("buffer length mismatch must produce an error");
    assert!(matches!(
        buffer_error,
        WgpuError::LengthMismatch {
            expected: 8,
            actual: 4,
        }
    ));
}

#[test]
fn plans_reject_invalid_lengths_before_dispatch() {
    let Some(backend) = backend() else {
        return;
    };

    let empty_error = backend
        .execute_forward(
            &NttWgpuPlan::new(0, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
            &[],
        )
        .expect_err("empty plan must fail");
    assert!(matches!(
        empty_error,
        WgpuError::InvalidPlan { ref message } if message.contains("length must be greater than zero")
    ));

    let non_power_of_two_error = backend
        .execute_forward(
            &NttWgpuPlan::new(6, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
            &[0; 6],
        )
        .expect_err("non-power-of-two plan must fail");
    assert!(matches!(
        non_power_of_two_error,
        WgpuError::InvalidPlan { ref message } if message.contains("power of two")
    ));

    let input_error = backend
        .execute_forward(
            &NttWgpuPlan::new(8, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
            &[0; 4],
        )
        .expect_err("input length mismatch must fail");
    assert!(matches!(
        input_error,
        WgpuError::LengthMismatch {
            expected: 8,
            actual: 4,
        }
    ));
}
