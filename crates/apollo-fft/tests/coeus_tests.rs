//! Tests for Coeus tensor and autograd FFT integration.
#![cfg(feature = "coeus")]

use apollo_fft::coeus::{fft_1d, fft_1d_var, ifft_1d, ifft_1d_var};
use coeus_core::{Complex, MoiraiBackend};
use coeus_tensor::Tensor;
use coeus_autograd::var::Var;

#[test]
fn test_coeus_tensor_fft_parity() {
    let data = vec![1.0f64, 2.0, 1.0, -1.0, 1.5, 3.0, 2.2, 0.5];
    let signal = Tensor::<f64, MoiraiBackend>::from_vec(data.clone());

    // Forward FFT
    let spectrum = fft_1d(&signal);
    assert_eq!(spectrum.shape(), &[8]);

    // Inverse FFT
    let reconstructed = ifft_1d(&spectrum);

    // Assert reconstruction parity (original signal is recovered)
    let original = signal.as_slice();
    let recon = reconstructed.as_slice();
    for i in 0..8 {
        let diff = (original[i] - recon[i]).abs();
        assert!(
            diff < 1e-7,
            "Reconstruction mismatch at index {i}: original={}, reconstructed={}",
            original[i],
            recon[i]
        );
    }
}

#[test]
fn test_fft_autograd() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let x_tensor = Tensor::from_vec(x_data.clone());
    let x = Var::<f32, MoiraiBackend>::new(x_tensor, true);
    let y = fft_1d_var(&x);

    assert_eq!(y.tensor.shape(), &[4]);

    // Expected FFT of [1, 2, 3, 4] is [10, -2+2i, -2, -2-2i]
    let expected = [
        Complex::new(10.0f32, 0.0),
        Complex::new(-2.0, 2.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-2.0, -2.0),
    ];

    let y_slice = y.tensor.as_slice();
    for i in 0..4 {
        assert!((y_slice[i].re - expected[i].re).abs() < 1e-4);
        assert!((y_slice[i].im - expected[i].im).abs() < 1e-4);
    }

    // Backward pass
    let grad_y_data = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(0.5, -0.5),
        Complex::new(0.0, -1.0),
    ];
    let grad_y = Tensor::from_vec(grad_y_data.clone());
    y.backward_with_seed(grad_y);

    // Expected dX = ifft_1d(dY) * N = ifft_1d(dY) * 4
    // dY = [1, i, 0.5-0.5i, -i]
    // ifft_1d(dY) = [0.375 + 0.125i, 0.125 - 0.375i, 0.375 - 0.125i, 0.125 + 0.375i]
    // dX = ifft_1d(dY) * 4 (since input is real, dX is real part of ifft_1d(dY) * 4)
    // ifft_1d(dY) * 4 = [1.5 + 0.5i, 0.5 - 1.5i, 1.5 - 0.5i, 0.5 + 1.5i]
    // Real part: [1.5, -1.5, 1.5, 2.5]
    let expected_dx = [1.5f32, -1.5, 1.5, 2.5];
    let dx = x.grad().unwrap();
    let dx_slice = dx.as_slice();
    for i in 0..4 {
        assert!(
            (dx_slice[i] - expected_dx[i]).abs() < 1e-4,
            "Mismatch at index {}: actual={}, expected={}",
            i, dx_slice[i], expected_dx[i]
        );
    }
}

#[test]
fn test_ifft_autograd() {
    let y_data = vec![
        Complex::new(10.0f32, 0.0),
        Complex::new(-2.0, 2.0),
        Complex::new(-2.0, 0.0),
        Complex::new(-2.0, -2.0),
    ];
    let y_tensor = Tensor::from_vec(y_data);
    let y = Var::<Complex<f32>, MoiraiBackend>::new(y_tensor, true);
    let x = ifft_1d_var(&y);

    assert_eq!(x.tensor.shape(), &[4]);

    // Expected IFFT is [1, 2, 3, 4]
    let expected = [1.0f32, 2.0, 3.0, 4.0];
    let x_slice = x.tensor.as_slice();
    for i in 0..4 {
        assert!((x_slice[i] - expected[i]).abs() < 1e-4);
    }

    // Backward pass
    let grad_x = Tensor::from_vec(vec![1.0f32, 2.0, 1.0, 0.0]);
    x.backward_with_seed(grad_x);

    // Expected dY = fft_1d(dX) / N = fft_1d(dX) / 4
    // dX = [1, 2, 1, 0]
    // fft_1d(dX) = [4, -2i, 0, 2i]
    // dY = [1, -0.5i, 0, 0.5i]
    let expected_dy = [
        Complex::new(1.0f32, 0.0),
        Complex::new(0.0, -0.5),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.5),
    ];

    let dy = y.grad().unwrap();
    let dy_slice = dy.as_slice();
    for i in 0..4 {
        assert!((dy_slice[i].re - expected_dy[i].re).abs() < 1e-4);
        assert!((dy_slice[i].im - expected_dy[i].im).abs() < 1e-4);
    }
}
