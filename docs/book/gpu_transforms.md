# GPU Transforms

Apollo owns transform-domain mathematics. Hephaestus owns accelerator devices,
typed buffers, prepared pipelines, command streams, and dense WGPU FFT
algorithms. Leto supplies the shape and stride metadata shared by CPU and GPU
operands.

## Dense FFT

A dense accelerator FFT is prepared through Hephaestus rather than an Apollo
backend wrapper. The rank is encoded by `Layout<R>` and `FftOperands`:

```rust,no_run
use hephaestus_core::{
    ComputeDevice, FftDirection, FftOperands, FftOps, StridedView,
};
use hephaestus_wgpu::{WgpuDevice, WgpuFftOps};
use leto::Layout;

# fn run(device: &WgpuDevice) -> Result<(), Box<dyn std::error::Error>> {
let real = device.upload(&[1.0_f32, 0.0, -1.0, 0.0])?;
let imaginary = device.alloc_zeroed(4)?;
let layout = Layout::c_contiguous([4])?;
let plan = WgpuFftOps.prepare_fft(
    device,
    FftOperands {
        real: StridedView::new(&real, &layout),
        imaginary: StridedView::new(&imaginary, &layout),
    },
    FftDirection::Forward,
)?;
WgpuFftOps.dispatch_fft(device, &plan)?;
# Ok(())
# }
```

Preparation validates shape, strides, aliasing, scalar capability, and device
ownership before mutation. It also compiles and binds the required radix or
Bluestein pipelines. Retain the prepared plan with its fixed buffers; repeated
dispatch then performs no plan allocation or compilation.

Ranks one through three use the same API. The buffer scalar selects f32 or
native-f16 arithmetic; native f16 requires a `ShaderF16`-qualified device.

## Composition

`FftOps::encode_fft` records a prepared transform into an existing Hephaestus
command stream. Apollo NUFFT uses this boundary to order spread/load, FFT, and
extract/interpolate passes without an intermediate submission or host copy.
Forward and inverse plans live beside the reusable oversampled grids.

## Domain-specific transforms

Apollo's `WgpuTransformBackend<K>` remains the shared scaffold for non-FFT
transform kernels. A transform crate supplies its own zero-sized planner and
executor implementation, while the scaffold centralizes typed storage,
capability reporting, validation, and provider errors. It is not a dense FFT
backend and does not duplicate Hephaestus `FftOps`.

Accelerator implementations are verified against the corresponding Apollo CPU
operator under a derived finite-precision bound. Provider absence is reported
at acquisition; a present provider failure is never converted into a silent CPU
fallback.
