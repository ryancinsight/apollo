use leto::Storage;

use crate::infrastructure::transport::gpu::WgpuError;

use super::support::backend;

#[test]
fn quantized_storage_matches_allocating_exact_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1_u32, 1, 2, 3, 5, 8, 13, 21];
    let represented = input.iter().copied().map(u64::from).collect::<Vec<_>>();
    let plan = backend.plan(input.len());
    let expected = backend
        .execute_forward(&plan, &represented)
        .expect("allocating forward");
    let mut actual = vec![0_u32; input.len()];
    backend
        .execute_forward_quantized_into(&plan, &input, &mut actual)
        .expect("quantized forward");
    assert_eq!(
        actual.iter().copied().map(u64::from).collect::<Vec<_>>(),
        expected
    );

    let mut recovered = vec![0_u32; input.len()];
    backend
        .execute_inverse_quantized_into(&plan, &actual, &mut recovered)
        .expect("quantized inverse");
    assert_eq!(recovered, input);
}

#[test]
fn quantized_leto_forward_and_inverse_match_quantized_slice() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![3_u32, 1, 4, 1, 5, 9, 2, 6];
    let plan = backend.plan(input.len());
    let mut expected_forward = vec![0_u32; input.len()];
    backend
        .execute_forward_quantized_into(&plan, &input, &mut expected_forward)
        .expect("quantized forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual_forward = backend
        .execute_forward_quantized_leto(&plan, leto_input.view())
        .expect("Leto quantized forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let mut expected_inverse = vec![0_u32; expected_forward.len()];
    backend
        .execute_inverse_quantized_into(&plan, &expected_forward, &mut expected_inverse)
        .expect("quantized inverse");
    let leto_spectrum = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto spectrum");
    let actual_inverse = backend
        .execute_inverse_quantized_leto(&plan, leto_spectrum.view())
        .expect("Leto quantized inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn quantized_storage_rejects_output_length_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(8);
    let mut output = vec![0_u32; 4];
    let error = backend
        .execute_forward_quantized_into(&plan, &[0; 8], &mut output)
        .expect_err("output length mismatch must produce an error");
    assert!(matches!(
        error,
        WgpuError::LengthMismatch {
            expected: 8,
            actual: 4,
        }
    ));
}
