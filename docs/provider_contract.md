# Apollo Provider Contract

Apollo consumes provider crates through Git dependencies. Provider changes in
Moirai, Mnemosyne, Melinoe, Hermes, or Leto must be committed and pushed before
Apollo updates its dependency revision. Committed Apollo manifests must not use
local path overrides for provider work.

## Current Surface

- `moirai` is the active data-parallel provider in the Apollo workspace
  dependency table with default features disabled and `parallel` enabled.
- `mnemosyne` is the scratch-allocation provider in the Apollo workspace
  dependency table with default features disabled and `num-complex` enabled.
- `melinoe` is the branded zero-copy boundary provider in the Apollo workspace
  dependency table with default features disabled and `alloc` enabled.
- `hermes-simd` is the SIMD provider in the Apollo workspace dependency table
  with default features disabled and `std` enabled.
- `leto` is the strided-array and dense-matrix migration provider in the Apollo workspace
  dependency table with default features disabled and `std` plus
  `ndarray-compat` and `mnemosyne-alloc` enabled.
- `ndarray` remains the validation oracle and transitional public API substrate.
  The workspace dependency must not enable Rayon or matrixmultiply threading
  while Moirai is the intended parallel runtime.
- WGPU crates own GPU device buffers and dispatch. CPU scheduling and host-side
  allocation policy must remain decoupled from WGPU infrastructure types.

## Apollo Requirements for Moirai

- Monomorphized scheduler entry points for hot CPU paths; no `dyn Trait`,
  `Box<dyn Trait>`, or heap-erased closures in throughput-critical dispatch.
- Bounded work queues and deterministic chunking for transform axes,
  convolution batches, projection bins, and validation suites.
- Scoped non-`'static` tasks so Apollo can borrow plan scratch, twiddle tables,
  and input slices without cloning or promoting data into `Arc<Vec<T>>`.
- By-reference and by-value iterator adapters that do not require `T: Clone`
  unless the operation semantically clones the element.
- Caller-owned output collection APIs, such as collect-into-existing-buffer
  variants, so Apollo can reuse scratch capacity.
- Optional integration with Mnemosyne allocation policies without installing a
  process-wide allocator by default.

## Apollo Requirements for Mnemosyne

- Optional scratch allocators for aligned FFT/STFT/Radon/NUFFT workspaces and
  plan caches.
- Const-generic fixed scratch banks for same-typed role groups so Apollo can
  name transform roles locally while Mnemosyne owns pooled storage layout.
- Thread-local reusable regions for temporary buffers, with explicit reset
  semantics at transform boundaries.
- `Cow`-compatible borrowed views for twiddle tables, kernels, window
  functions, and validation fixtures.
- Zero-sized allocation policy markers and phantom-branded handles so policy
  selection is static and carries no runtime storage.
- No implicit global allocator requirement for Apollo library crates. Any
  global allocator mode must stay behind an opt-in binary or benchmark feature.

## Apollo Requirements for Melinoe

- Branded zero-copy slice views for scratch buffers, validation fixtures, and
  host-side staging slices.
- `Cow` boundary APIs that borrow by default and clone exactly once only when a
  caller needs retained ownership.
- Zero-sized policy markers for static borrow/retain choices so monomorphized
  Apollo call sites remove inactive branches.
- Capability tokens that encode read/write access without runtime flags,
  locks, or shared mutable state inside mathematical kernels.
- Deep vertical module ownership for branded cell, region, sync, atomic, and
  Cow surfaces so Apollo can depend on the public contract without reaching
  into provider internals.

## Apollo Requirements for Hermes

- Monomorphized SIMD vector kernels exposed through typed architecture markers
  such as `PreferredArch`, not runtime-erased hot-path dispatch.
- Zero-copy `SimdCow` and packed Cow state accessors so Apollo can assert
  borrowed-versus-owned storage contracts without matching provider internals.
- Native-precision vector arithmetic for Apollo kernels using Hermes vectors;
  hidden widen-compute-narrow paths are not acceptable for transform hot paths.
- Capability surfaces for lane count, alignment, mask, and execution mode that
  compile away through ZSTs, consts, and associated types.
- Public facade stability for Apollo's `hermes_simd::{PreferredArch, Vector}`
  usage in FWHT and FFT mixed-radix pointwise fallback kernels.
- Public interleaved complex kernels over primitive `[re, im, ...]` lanes so
  Apollo can keep `num-complex` at the domain boundary while reusing Hermes
  monomorphized SIMD logic.
- Runtime-selected interleaved complex dispatch for `f32` and `f64` so Apollo
  does not duplicate x86 feature detection or AVX/FMA pointwise kernels.

## Apollo Requirements for Leto

- Constructors and rank aliases matching Apollo's current `ndarray`
  `Array1`/`Array2`/`Array3` usage: `zeros`, `from_elem`, `from_vec`,
  `from_shape_vec`, `from_shape_fn`, and owned `into_vec`.
- Zero-copy immutable and mutable views over contiguous and strided layouts,
  with signed strides preserved for reverse views and storage-span validation
  before any external layout is accepted.
- `ndarray`-validated slicing semantics, including full axes, signed ranges,
  negative indices, negative strides, axis-dropping index selections, inserted
  axes, ellipsis expansion, implicit trailing axes, and retained single-element
  range stride metadata.
- Broadcast and axis-iteration semantics that match `ndarray` for read paths
  and reject mutable zero-stride aliasing.
- Transitional `ndarray-compat` conversions for validation only. Apollo uses
  `ndarray` to validate Leto behavior before replacing a downstream call site;
  core hot paths should move to Leto only after differential tests cover the
  relevant shape, stride, value, and mutation contracts.
- Mnemosyne-backed owned array constructors for Apollo output boundaries. The
  first migrated surface is the `apollo-fft` 1D Leto view API, which returns
  `MnemosyneStorage` and keeps `ndarray` only as the differential oracle.
- Dense graph/matrix descriptors that can replace `nalgebra::DMatrix` in
  Apollo domain models. Current migration coverage includes `apollo-gft`
  adjacency validation and combinatorial Laplacian construction; nalgebra
  remains only for symmetric eigendecomposition until Leto owns that solver
  contract.

## GPU Boundary

Moirai, Mnemosyne, Melinoe, Hermes, and Leto optimize CPU scheduling, host memory,
branded zero-copy access, host SIMD kernels, and host strided-array storage. WGPU
execution remains in `*-wgpu` infrastructure crates. GPU buffers, command
encoders, pipeline objects, and device futures must not leak into pure Apollo
domain models or CPU mathematical kernels.

## Verification

Run:

```powershell
cargo run -p xtask -- provider-audit
```

The audit reports Moirai, Mnemosyne, Melinoe, Hermes, Leto, Rayon, WGPU, `Arc`,
`Mutex`, `dyn`, clone-to-`Vec`, and `Cow` usage by crate. The evidence tier is
static source analysis; performance claims still require Criterion or
domain-specific benchmarks.
