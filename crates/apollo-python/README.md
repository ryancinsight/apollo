# apollo-python

PyO3 0.30 bindings for Python 3.9 or newer. The module requires the GIL;
free-threaded operation remains unverified. See workspace ADR 0076.

## Distribution

GitHub Releases tagged `apollo-fft-v<version>` build locked Linux, Windows,
and universal macOS wheels, install and import each artifact, attach the
validated wheel set and SHA-256 checksums to the release, and publish the same
wheels to `apollo-fft` on PyPI through OIDC Trusted Publishing. The tag
version must equal the `apollo-python` Cargo package version; Python metadata
reads that Cargo version rather than maintaining a second version field.

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


## Native Rust tests and wheel linking

Cargo tests embed the interpreter and execute the NumPy shape/value boundary
regression. Install NumPy for the selected interpreter, then run
`cargo test --locked -p apollo-python --lib` from the workspace root.
Maturin >= 1.9.4 selects extension-only linking for wheels through PyO3's
build environment; the build helper supplies the platform cdylib arguments.
The native CI lanes exercise the same boundary test alongside all existing
workspace tests. Wheel smoke tests retain their existing coverage.
