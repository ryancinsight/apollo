# Dense WGPU FFT migration

Apollo no longer exposes a dense WGPU FFT implementation. CPU FFT APIs remain
in `apollo-fft`; accelerator FFT plans are prepared and dispatched through
Hephaestus.

## Removed Apollo surface

- `GpuFft3d`
- `GpuFft3dBuffers`
- `GpuFft3dF16Native`
- `WgpuBackend`
- the `native-f16` feature

No forwarding aliases replace these items.

## Prepared Hephaestus plan

Use a Leto layout and fixed split-complex device buffers:

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
let prepared = WgpuFftOps.prepare_fft(
    device,
    FftOperands {
        real: StridedView::new(&real, &layout),
        imaginary: StridedView::new(&imaginary, &layout),
    },
    FftDirection::Forward,
)?;
WgpuFftOps.dispatch_fft(device, &prepared)?;
# Ok(())
# }
```

For a larger GPU operation, retain `WgpuPreparedFft<R, T>` with its fixed
buffers and call `encode_fft` on the operation's existing command stream. This
avoids per-call allocation, shader compilation, bind-group construction, and
provider selection.

Apollo NUFFT consumers retaining `NufftGpuBuffers1D` or
`NufftGpuBuffers3D` pass them by mutable reference to `with_buffers` methods.
The exclusive borrow prevents overlapping writes and lets the workspace retain
its host conversion capacity between calls.

`NufftWgpuError::Fft` is removed with Apollo's dense WGPU FFT provider. Match
`NufftWgpuError::Provider` for Hephaestus preparation, dispatch, allocation,
and transfer failures; Apollo CPU FFT errors no longer cross the NUFFT WGPU
boundary.

## Rank and precision

`WgpuFftOps` accepts Leto layouts of ranks one through three. The buffer scalar
selects f32 or native-f16 execution. Native f16 requires a device acquired with
`DeviceFeature::ShaderF16`; a missing capability is a typed acquisition or
preparation error, not a CPU fallback.

## Kwavers

Kwavers selects its CPU/Leto or Hephaestus backend at the solver operation
boundary. Its Hephaestus branch retains prepared provider plans and owns no
private dense FFT kernel. Existing Apollo CPU consumers continue to use
`FftPlan1D`, `FftPlan2D`, and `FftPlan3D` without migration.
