//! Device-present verification for typed dense FFT stream dispatch.

use hephaestus_core::{CommandStream, ComputeDevice, HephaestusError, KernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::GpuFft3d;

fn device_or_skip(application_name: &str) -> Option<WgpuDevice> {
    match WgpuDevice::try_default(application_name) {
        Ok(device) => Some(device),
        Err(HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => {
            panic!("typed FFT device-present regression requires a working provider: {error}");
        }
    }
}

#[test]
fn typed_external_buffers_preserve_delta_roundtrip_when_device_exists() {
    let Some(device) = device_or_skip("apollo-fft-typed-stream-test") else {
        return;
    };
    let plan = GpuFft3d::new(device.clone(), 2, 2, 2)
        .expect("2x2x2 typed FFT plan must fit the acquired device");
    let input = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let real = device
        .upload(&input)
        .expect("typed upload of the analytical delta field");
    let imaginary = device
        .alloc_zeroed(input.len())
        .expect("typed allocation of the imaginary component");

    let mut forward_stream = device.stream().expect("typed command stream");
    plan.encode_forward_split(&mut forward_stream, &real, &imaginary)
        .expect("typed forward encoding");
    forward_stream.submit().expect("typed forward submission");

    let mut forward_real = [0.0_f32; 8];
    let mut forward_imaginary = [0.0_f32; 8];
    device
        .download(&real, &mut forward_real)
        .expect("typed forward real readback");
    device
        .download(&imaginary, &mut forward_imaginary)
        .expect("typed forward imaginary readback");
    assert_eq!(forward_real, [1.0; 8]);
    assert_eq!(forward_imaginary, [0.0; 8]);

    let mut inverse_stream = device.stream().expect("typed command stream");
    plan.encode_inverse_split(&mut inverse_stream, &real, &imaginary)
        .expect("typed inverse encoding");
    inverse_stream.submit().expect("typed inverse submission");

    let mut reconstructed_real = [0.0_f32; 8];
    let mut reconstructed_imaginary = [0.0_f32; 8];
    device
        .download(&real, &mut reconstructed_real)
        .expect("typed inverse real readback");
    device
        .download(&imaginary, &mut reconstructed_imaginary)
        .expect("typed inverse imaginary readback");
    assert_eq!(reconstructed_real, input);
    assert_eq!(reconstructed_imaginary, [0.0; 8]);
}

#[test]
fn typed_external_bluestein_delta_matches_dft_and_roundtrips_when_device_exists() {
    let Some(device) = device_or_skip("apollo-fft-typed-bluestein-test") else {
        return;
    };
    let plan = GpuFft3d::new(device.clone(), 2, 3, 2)
        .expect("2x3x2 typed FFT plan must fit the acquired device");
    let input = [
        1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let real = device
        .upload(&input)
        .expect("typed upload of the analytical delta field");
    let imaginary = device
        .alloc_zeroed(input.len())
        .expect("typed allocation of the imaginary component");

    let mut forward_stream = device.stream().expect("typed command stream");
    plan.encode_forward_split(&mut forward_stream, &real, &imaginary)
        .expect("typed Bluestein forward encoding");
    forward_stream
        .submit()
        .expect("typed Bluestein forward submission");

    let mut forward_real = [0.0_f32; 12];
    let mut forward_imaginary = [0.0_f32; 12];
    device
        .download(&real, &mut forward_real)
        .expect("typed Bluestein real readback");
    device
        .download(&imaginary, &mut forward_imaginary)
        .expect("typed Bluestein imaginary readback");

    // The 3-point axis uses Bluestein with M=8. A transformed component
    // traverses fewer than 256 f32 rounding operations (two radix-2 axes,
    // chirp pre/post factors, two 8-point FFTs, point multiplication, and
    // normalization), so gamma_256 bounds the delta DFT error by gamma_256
    // times the input l1 norm. The inverse consumes 12 unit-magnitude
    // coefficients, yielding gamma_256 * (1 + 12) for the roundtrip.
    let unit_roundoff = f32::EPSILON / 2.0;
    let gamma_256 = 256.0 * unit_roundoff / (1.0 - 256.0 * unit_roundoff);
    let forward_bound = gamma_256;
    let roundtrip_bound = gamma_256 * 13.0;
    for value in forward_real {
        assert!((value - 1.0).abs() <= forward_bound);
    }
    for value in forward_imaginary {
        assert!(value.abs() <= forward_bound);
    }

    let mut inverse_stream = device.stream().expect("typed command stream");
    plan.encode_inverse_split(&mut inverse_stream, &real, &imaginary)
        .expect("typed Bluestein inverse encoding");
    inverse_stream
        .submit()
        .expect("typed Bluestein inverse submission");

    let mut reconstructed_real = [0.0_f32; 12];
    let mut reconstructed_imaginary = [0.0_f32; 12];
    device
        .download(&real, &mut reconstructed_real)
        .expect("typed Bluestein inverse real readback");
    device
        .download(&imaginary, &mut reconstructed_imaginary)
        .expect("typed Bluestein inverse imaginary readback");
    for (actual, expected) in reconstructed_real.into_iter().zip(input) {
        assert!((actual - expected).abs() <= roundtrip_bound);
    }
    for value in reconstructed_imaginary {
        assert!(value.abs() <= roundtrip_bound);
    }
}
