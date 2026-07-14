# apollo-python

PyO3 bindings for the Apollo workspace.

## Architecture

```text
src/
  domain/          Python-facing domain boundary marker
  application/     binding orchestration boundary marker
  infrastructure/  host/runtime integration boundary marker
  lib.rs           PyO3 module, class, and function exports
```

`apollo-python` is an adapter crate. It owns Python type conversion,
contiguity validation, error translation into `PyValueError`, and PyO3 module
registration. The mathematical single sources of truth remain in the owning
Rust crates:

- `apollo-fft` owns CPU FFT plans, precision profiles, full-spectrum 1D/2D/3D
  execution, and real-to-complex helpers.
- `apollo-fft` owns WGPU backend capability discovery surfaced through
  `available_backends()` and `backend_capabilities()`.
- `apollo-nufft` owns exact and fast NUFFT execution.

The binding layer does not implement transform kernels. It validates Python
array layout, maps precision strings to Apollo `PrecisionProfile` values, and
delegates computation to the owning crate APIs.

## Mathematical Contract

FFT bindings preserve Apollo's normalization convention: forward transforms are
unnormalized and inverse transforms divide by the transformed volume. The
`rfft3` and `irfft3` functions expose Apollo's full-spectrum real transform
contract rather than a half-spectrum NumPy compatibility surface.

NUFFT bindings preserve the owning crate's uniform-domain contracts. Positions,
values, grid spacing, output shape, and kernel width are validated before
delegation; invalid contracts are returned as Python value errors.

## Precision Contract

Python callers select precision with:

- `high_accuracy`: `f64`/`complex128` storage and `f64` compute.
- `low_precision`: `f32`/`complex64` storage and `f32` compute.
- `mixed_precision`: `f32` Python input converted to `f16` storage for Apollo's
  mixed `f16`/`f32` path where exposed by the owning FFT plan.

Storage/profile mismatches are rejected before transform execution. WGPU
capability metadata reports the backend precision profiles that are actually
available on the host.

## Verification

This crate is verified through workspace checks and the PyO3 unit target:

- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`

The workspace tests cover Python binding compilation, backend capability
metadata shape, full-spectrum `rfft3`/`irfft3` value semantics, contiguity
rejection paths, and delegated CPU/WGPU/NUFFT API compatibility.

