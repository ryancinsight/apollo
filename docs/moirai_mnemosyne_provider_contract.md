# Moirai and Mnemosyne Provider Contract

Apollo consumes provider crates through Git dependencies. Provider changes in
Moirai or Mnemosyne must be committed and pushed before Apollo updates its
dependency revision. Committed Apollo manifests must not use local path
overrides for provider work.

## Current Surface

- `moirai` is the active data-parallel provider in the Apollo workspace
  dependency table with default features disabled and `parallel` enabled.
- `mnemosyne` is not yet an Apollo workspace dependency.
- `ndarray` still enables `rayon` and `matrixmultiply-threading`; this is an
  audit item because it keeps Rayon-linked parallelism in the dependency graph
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
- Thread-local reusable regions for temporary buffers, with explicit reset
  semantics at transform boundaries.
- `Cow`-compatible borrowed views for twiddle tables, kernels, window
  functions, and validation fixtures.
- Zero-sized allocation policy markers and phantom-branded handles so policy
  selection is static and carries no runtime storage.
- No implicit global allocator requirement for Apollo library crates. Any
  global allocator mode must stay behind an opt-in binary or benchmark feature.

## GPU Boundary

Moirai and Mnemosyne optimize CPU scheduling and host memory. WGPU execution
remains in `*-wgpu` infrastructure crates. GPU buffers, command encoders,
pipeline objects, and device futures must not leak into pure Apollo domain
models or CPU mathematical kernels.

## Verification

Run:

```powershell
cargo run -p xtask -- provider-audit
```

The audit reports Moirai, Mnemosyne, Rayon, WGPU, `Arc`, `Mutex`, `dyn`,
clone-to-`Vec`, and `Cow` usage by crate. The evidence tier is static source
analysis; performance claims still require Criterion or domain-specific
benchmarks.
