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

`ApolloError::Wgpu` is also removed because no Apollo FFT operation constructs
it after the dense provider deletion. Match the typed `HephaestusError` from
provider preparation or execution. Apollo's `WgpuError` remains a distinct
non-FFT transform-transport contract; it is not a replacement
`ApolloError` variant.

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
its host conversion capacity between calls. Construction also binds every
spread/load/extract/interpolate pipeline to its fixed provider buffers and
maximum-capacity dispatch grid. A warm call may change its logical sample
count up to that capacity; it updates the retained parameter buffer and encodes
the domain stages with the retained Hephaestus FFT in one grouped sequence.
No warm bind-group construction is required.

Buffer construction now consumes validated plan geometry instead of separate
raw dimensions:

```rust,ignore
// Before
NufftGpuBuffers1D::new(device, n, oversampled_len, max_samples)?;
NufftGpuBuffers3D::new(device, shape, oversampled_shape, max_samples)?;

// After
NufftGpuBuffers1D::new(device, &plan_1d, max_samples)?;
NufftGpuBuffers3D::new(device, &plan_3d, max_samples)?;
```

This is a breaking constructor migration: do not derive a second geometry
beside the plan. Construction can now fail while preparing retained domain and
FFT pipelines, reported through `NufftWgpuError::Provider`.

Apollo STFT consumers retain `StftGpuBuffers` for one signal/frame/hop
geometry. Internally the workspace prepares rank-two Hephaestus plans over
`[frame_count, frame_len]` with active axis `[1]`; the frame axis remains a
batch dimension. Apollo keeps only framing, Hann, interleave/deinterleave,
synthesis-window, and overlap-add kernels. The former Apollo radix-2 and
Bluestein descriptors, shaders, chirp workspace, and six-binding device-limit
request are removed. Hephaestus inverse normalization already applies
`1/frame_len`, so consumer synthesis kernels must not scale again.

The low-level `StftGpuKernel::execute_forward_fft`,
`execute_forward_fft_with_buffers`, `execute_inverse`, and
`execute_inverse_with_buffers` methods and the public `forward_chirp` and
`inverse_chirp` modules are removed with that implementation. Import
`FramedExecution` and call `execute_forward`, `execute_inverse`,
`make_buffers`, `execute_forward_with_buffers`, or
`execute_inverse_with_buffers` on `StftWgpuBackend`. No forwarding methods or
module aliases remain.

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
