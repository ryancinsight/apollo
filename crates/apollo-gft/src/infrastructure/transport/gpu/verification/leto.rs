//! Value-semantic GFT GPU Leto host-boundary contracts.

use leto::{SliceArg, Storage};

use super::support::{backend, path4_plan_and_basis};

#[test]
fn leto_forward_and_inverse_match_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (_cpu_plan, basis, signal) = path4_plan_and_basis();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let expected_forward = backend
        .execute_forward(&plan, &signal, &basis)
        .expect("slice forward");
    let signal_leto = leto::Array1::from_shape_vec([signal.len()], signal).expect("signal");
    let basis_leto = leto::Array1::from_shape_vec([basis.len()], basis).expect("basis");
    let actual_forward = backend
        .execute_forward_leto(&plan, signal_leto.view(), basis_leto.view())
        .expect("Leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward, basis_leto.storage().as_slice())
        .expect("slice inverse");
    let spectrum =
        leto::Array1::from_shape_vec([expected_forward.len()], expected_forward).expect("spectrum");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, spectrum.view(), basis_leto.view())
        .expect("Leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn leto_strided_forward_matches_logical_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (_cpu_plan, basis, signal) = path4_plan_and_basis();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let interleaved = leto::Array1::from_shape_vec(
        [signal.len() * 2],
        signal
            .iter()
            .flat_map(|value| [*value, 99.0_f32])
            .collect::<Vec<_>>(),
    )
    .expect("interleaved signal");
    let signal_view = interleaved
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided signal");
    let basis_leto = leto::Array1::from_shape_vec([basis.len()], basis).expect("basis");
    let expected = backend
        .execute_forward(&plan, &signal, basis_leto.storage().as_slice())
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, signal_view, basis_leto.view())
        .expect("strided Leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}
