# GPU Transforms

Apollo's `apollo-fft` provides a GPU transform backend based on wgpu,
exposing the same API surface as the CPU backend.

## `WgpuTransformBackend`

`WgpuTransformBackend` wraps a Hephaestus wgpu device and implements FFT
dispatch through WGSL compute shaders:

```rust,ignore
use apollo_fft::WgpuTransformBackend;

let backend = WgpuTransformBackend::new(&wgpu_device)?;
let plan = backend.plan_1d(4096, PrecisionProfile::default())?;
backend.execute(&plan, &mut input, &mut output)?;
```

## Shared Validation Helpers

`WgpuTransformBackend` exposes three validation helpers consumed by
`apollo-fft`, `apollo-gft`, and `apollo-validation`:

| Helper | Description |
|--------|-------------|
| `validate_plan(plan)` | Rejects empty or missing plans |
| `require_len(buf, n)` | Asserts buffer length == n |
| `validate_storage_profile(profile)` | Validates tier and precision |

## `GpuTransformExecutor` and `GpuTransformPlanner`

`GpuTransformPlanner` creates and caches `WgpuTransformPlan` instances.
The cache key is `(shape, PrecisionProfile, direction)`. Once compiled,
a plan can be executed many times without re-compiling the WGSL shader.

`GpuTransformExecutor` drives the dispatch loop: it binds input/output
buffers, dispatches the compute shader, and handles the synchronization
fence.

## `PrecisionProfile`

Controls the floating-point precision and normalization applied during
GPU transforms:

| Field | Description |
|-------|-------------|
| `compute` | On-chip computation precision |
| `storage` | Buffer storage precision |
| `normalization` | `Forward` / `Backward` / `Ortho` |
| `mode` | `PrecisionMode` (e.g., `Mixed`, `Full`) |

## Integration with Apollo GFT

`apollo-gft` (Graph Fourier Transform) uses `WgpuTransformBackend`
for GPU-accelerated spectral graph transforms. It adds
`validate_basis_len` for its graph-basis shape validation while
delegating generic error-validation to the shared helpers.
