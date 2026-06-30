// ── Coeus FFT & Autograd Integration ──
// Facilitates FFT operations on Coeus Tensors and Vars using Apollo.

use crate::{Complex32, Complex64};
use coeus_autograd::{node::BackwardNode, var::Var, GradBuffer};
use coeus_core::{Complex, ComputeBackend, Float, MoiraiBackend, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Sealed trait for compile-time monomorphized FFT dispatch on Coeus types.
pub trait FftScalar: Float + eunomia::NumericElement {
    /// Internal Complex type used by Apollo FFT.
    type Complex;

    /// Direct 1-D forward FFT implementation.
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>>;

    /// Direct 1-D inverse FFT implementation.
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self>;
}

impl FftScalar for f64 {
    type Complex = Complex64;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        crate::fft_1d_slice_typed::<f64>(signal)
            .into_iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex64> = spectrum
            .iter()
            .map(|c| Complex64::new(c.re, c.im))
            .collect();
        crate::ifft_1d_slice_typed::<f64>(&spec)
    }
}

impl FftScalar for f32 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        crate::fft_1d_slice_typed::<f32>(signal)
            .into_iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex32> = spectrum
            .iter()
            .map(|c| Complex32::new(c.re, c.im))
            .collect();
        crate::ifft_1d_slice_typed::<f32>(&spec)
    }
}

impl FftScalar for half::f16 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        crate::fft_1d_slice_typed::<half::f16>(signal)
            .into_iter()
            .map(|c| Complex::new(half::f16::from_f32(c.re), half::f16::from_f32(c.im)))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex32> = spectrum
            .iter()
            .map(|c| Complex32::new(c.re.to_f32(), c.im.to_f32()))
            .collect();
        crate::ifft_1d_slice_typed::<half::f16>(&spec)
    }
}

impl FftScalar for half::bf16 {
    type Complex = Complex32;

    #[inline]
    fn fft_1d_impl(signal: &[Self]) -> Vec<Complex<Self>> {
        let input: Vec<f32> = signal.iter().map(|&x| x.to_f32()).collect();
        crate::fft_1d_slice_typed::<f32>(&input)
            .into_iter()
            .map(|c| Complex::new(half::bf16::from_f32(c.re), half::bf16::from_f32(c.im)))
            .collect()
    }

    #[inline]
    fn ifft_1d_impl(spectrum: &[Complex<Self>]) -> Vec<Self> {
        let spec: Vec<Complex32> = spectrum
            .iter()
            .map(|c| Complex32::new(c.re.to_f32(), c.im.to_f32()))
            .collect();
        crate::ifft_1d_slice_typed::<f32>(&spec)
            .into_iter()
            .map(half::bf16::from_f32)
            .collect()
    }
}

/// Trait for backend-specific execution of 1-D FFT operations.
pub trait FftDeviceOps<T: FftScalar>: ComputeBackend {
    /// Compute 1D Forward FFT.
    fn fft_1d(&self, signal: &Self::DeviceBuffer<T>) -> Self::DeviceBuffer<Complex<T>>;
    /// Compute 1D Inverse FFT.
    fn ifft_1d(&self, spectrum: &Self::DeviceBuffer<Complex<T>>) -> Self::DeviceBuffer<T>;
}

impl<T: FftScalar> FftDeviceOps<T> for MoiraiBackend {
    fn fft_1d(&self, signal: &Self::DeviceBuffer<T>) -> Self::DeviceBuffer<Complex<T>> {
        use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
        let slice = signal.as_slice();
        let out_vec = T::fft_1d_impl(slice);
        let mut out = coeus_core::storage::CpuStorage::allocate(out_vec.len());
        out.as_mut_slice().copy_from_slice(&out_vec);
        out
    }

    fn ifft_1d(&self, spectrum: &Self::DeviceBuffer<Complex<T>>) -> Self::DeviceBuffer<T> {
        use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
        let slice = spectrum.as_slice();
        let out_vec = T::ifft_1d_impl(slice);
        let mut out = coeus_core::storage::CpuStorage::allocate(out_vec.len());
        out.as_mut_slice().copy_from_slice(&out_vec);
        out
    }
}

impl<T: FftScalar> FftDeviceOps<T> for coeus_core::SequentialBackend {
    fn fft_1d(&self, signal: &Self::DeviceBuffer<T>) -> Self::DeviceBuffer<Complex<T>> {
        use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
        let slice = signal.as_slice();
        let out_vec = T::fft_1d_impl(slice);
        let mut out = coeus_core::storage::CpuStorage::allocate(out_vec.len());
        out.as_mut_slice().copy_from_slice(&out_vec);
        out
    }

    fn ifft_1d(&self, spectrum: &Self::DeviceBuffer<Complex<T>>) -> Self::DeviceBuffer<T> {
        use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
        let slice = spectrum.as_slice();
        let out_vec = T::ifft_1d_impl(slice);
        let mut out = coeus_core::storage::CpuStorage::allocate(out_vec.len());
        out.as_mut_slice().copy_from_slice(&out_vec);
        out
    }
}

/// 1-D forward FFT. Returns a Complex tensor.
#[inline]
pub fn fft_1d<T: FftScalar, B: ComputeBackend + FftDeviceOps<T> + Default>(
    signal: &Tensor<T, B>,
) -> Tensor<Complex<T>, B> {
    assert_eq!(signal.ndim(), 1, "fft_1d requires 1D input");
    let input = signal.to_contiguous();
    let out_storage = B::default().fft_1d(input.storage());
    Tensor::from_raw_parts(out_storage, coeus_core::Layout::new(signal.shape_cloned()))
}

/// 1-D inverse FFT from Complex component.
#[inline]
pub fn ifft_1d<T: FftScalar, B: ComputeBackend + FftDeviceOps<T> + Default>(
    spectrum: &Tensor<Complex<T>, B>,
) -> Tensor<T, B> {
    assert_eq!(spectrum.ndim(), 1, "ifft_1d requires 1D input");
    let spectrum_cont = spectrum.to_contiguous();
    let out_storage = B::default().ifft_1d(spectrum_cont.storage());
    Tensor::from_raw_parts(
        out_storage,
        coeus_core::Layout::new(spectrum.shape_cloned()),
    )
}

/// Autograd node for 1D Forward FFT on Coeus.
pub struct Fft1DNode<T: FftScalar, B: ComputeBackend + FftDeviceOps<T> + Default = MoiraiBackend> {
    /// Input variable being transformed.
    pub x: Var<T, B>,
    /// Accumulated gradient of the transformed output.
    pub output_grad: Arc<GradBuffer<Complex<T>, B>>,
}

impl<T: FftScalar, B: ComputeBackend + FftDeviceOps<T> + Default> BackwardNode<Complex<T>, B>
    for Fft1DNode<T, B>
where
    B: coeus_ops::BackendOps<T>,
{
    fn op_name(&self) -> &'static str {
        "fft_1d"
    }

    fn output_grad(&self) -> &Arc<GradBuffer<Complex<T>, B>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<Complex<T>, B>] {
        &[]
    }

    fn backward(
        &self,
        grad_out: &Tensor<Complex<T>, B>,
        _input_grads: &[Option<Arc<GradBuffer<Complex<T>, B>>>],
    ) {
        let backend = B::default();
        // dX = ifft_1d(grad_out) * N
        let d_x = ifft_1d(grad_out);
        let n_val = T::from_usize(grad_out.numel());
        let n_tensor = Tensor::full_on([1], n_val, &backend);
        let scaled_d_x = coeus_ops::mul(&d_x, &n_tensor, &backend);

        if let Some(ref g) = self.x.grad {
            coeus_ops::add_assign(g.write(), &scaled_d_x, &backend);
        }

        if self.x.creator.is_some() {
            let current_grad = self.x.grad().unwrap();
            self.x.backward_with_seed(current_grad);
        }
    }
}

/// Autograd node for 1D Inverse FFT on Coeus.
pub struct Ifft1DNode<T: FftScalar, B: ComputeBackend + FftDeviceOps<T> + Default = MoiraiBackend> {
    /// Input variable in the frequency domain.
    pub y: Var<Complex<T>, B>,
    /// Accumulated gradient of the reconstructed spatial output.
    pub output_grad: Arc<GradBuffer<T, B>>,
}

impl<T: FftScalar, B: ComputeBackend + FftDeviceOps<T> + Default> BackwardNode<T, B>
    for Ifft1DNode<T, B>
where
    B: coeus_ops::BackendOps<T>,
{
    fn op_name(&self) -> &'static str {
        "ifft_1d"
    }

    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<T, B>] {
        &[]
    }

    fn backward(&self, grad_out: &Tensor<T, B>, _input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        let numel = grad_out.numel();

        let scaled_d_y = backend.fft_1d(grad_out.storage());
        let mut fft_vec = vec![Complex::new(<T as eunomia::NumericElement>::ZERO, <T as eunomia::NumericElement>::ZERO); numel];
        backend.copy_to_host(&scaled_d_y, &mut fft_vec);

        // Scale by 1 / N
        let n_f64 = numel as f64;
        for c in &mut fft_vec {
            c.re = c.re / T::from_f64(n_f64);
            c.im = c.im / T::from_f64(n_f64);
        }

        if let Some(ref g) = self.y.grad {
            let mut current_g = vec![Complex::new(<T as eunomia::NumericElement>::ZERO, <T as eunomia::NumericElement>::ZERO); numel];
            let gl = g.write();
            backend.copy_to_host(gl.storage(), &mut current_g);
            for i in 0..numel {
                current_g[i].re = current_g[i].re + fft_vec[i].re;
                current_g[i].im = current_g[i].im + fft_vec[i].im;
            }
            backend.copy_to_device(&current_g, gl.storage_mut());
        }

        if self.y.creator.is_some() {
            let current_grad = self.y.grad().unwrap();
            self.y.backward_with_seed(current_grad);
        }
    }
}

/// Differentiable 1-D forward Fast Fourier Transform.
#[must_use]
pub fn fft_1d_var<T: FftScalar, B>(x: &Var<T, B>) -> Var<Complex<T>, B>
where
    B: ComputeBackend + FftDeviceOps<T> + Default + coeus_ops::BackendOps<T>,
{
    let backend = B::default();
    let out_tensor = fft_1d(&x.tensor);
    let requires_grad = x.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = Fft1DNode {
            x: x.clone(),
            output_grad,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<Complex<T>, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

/// Differentiable 1-D inverse Fast Fourier Transform.
#[must_use]
pub fn ifft_1d_var<T: FftScalar, B>(y: &Var<Complex<T>, B>) -> Var<T, B>
where
    B: ComputeBackend + FftDeviceOps<T> + Default + coeus_ops::BackendOps<T>,
{
    let backend = B::default();
    let out_tensor = ifft_1d(&y.tensor);
    let requires_grad = y.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = Ifft1DNode {
            y: y.clone(),
            output_grad,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
