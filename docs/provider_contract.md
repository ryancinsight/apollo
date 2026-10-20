# Apollo Provider Contract

Apollo consumes provider crates through their Git default branches. Provider
changes in Eunomia, Moirai, Mnemosyne, Melinoe, Hermes, Leto, or Hephaestus
must be merged before Apollo refreshes its lockfile. `Cargo.lock` is the sole
reproducibility pin; committed Apollo manifests must not carry provider
revisions or local path overrides.

The resolved provider graph requires Rust 1.95. Every Apollo member inherits
that minimum version from the workspace package contract.

## Current Surface

- `moirai` is the active data-parallel provider in the Apollo workspace
  dependency table with default features disabled and `parallel` enabled.
- `mnemosyne` is the scratch-allocation provider in the Apollo workspace
  dependency table with default features disabled and `num-complex` enabled.
- `melinoe` is the branded zero-copy boundary provider in the Apollo workspace
  dependency table with default features disabled and `alloc` enabled.
- `hermes-simd` is the SIMD provider in the Apollo workspace dependency table
  with default features disabled and `std` enabled.
- `leto` is the active strided-array and dense-matrix provider in the Apollo
  workspace dependency table. Apollo code and manifests must not depend on the
  Rust `ndarray` crate; Leto owns owned arrays, views, shape metadata, and
  dense layout materialization.
- Runtime NumPy arrays remain only at Python/external-reference boundaries.
  Apollo library crates must not use the Rust `numpy` crate or `ndarray` as an
  internal array substrate while Leto is the intended array provider.
- `hephaestus-core` owns backend-neutral typed buffer, kernel, binding, launch,
  command-stream, synchronization, and transfer contracts;
  `hephaestus-wgpu` implements them for WGPU. Apollo transform crates own only
  transform parameters, formulas, dialect source, and Leto host boundaries.

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

- Constructors and rank aliases matching Apollo's current Leto
  `Array1`/`Array2`/`Array3` usage: `zeros`, `from_elem`, `from_vec`,
  `from_shape_vec`, `from_shape_fn`, and owned vector extraction.
- Zero-copy immutable and mutable views over contiguous and strided layouts,
  with signed strides preserved for reverse views and storage-span validation
  before any external layout is accepted.
- Leto-owned slicing semantics, including full axes, signed ranges, negative
  indices, negative strides, axis-dropping index selections, inserted axes,
  ellipsis expansion, implicit trailing axes, and retained single-element range
  stride metadata.
- Broadcast and axis-iteration semantics for read paths, plus rejection of
  mutable zero-stride aliasing.
- Apollo validation uses Leto-native value tests, analytical references, and
  external runtime references where appropriate. It does not reintroduce Rust
  `ndarray` compatibility as an oracle inside Apollo.
- Mnemosyne-backed slice and vector constructors for Apollo output boundaries.
  Vector construction transfers elements into a separate Mnemosyne allocation;
  allocation-sensitive generated outputs use Leto's
  `from_mnemosyne_shape_fn`, which initializes final provider storage once.
- Dense graph/matrix descriptors that can replace `nalgebra::DMatrix` in
  Apollo domain models. Current migration coverage includes `apollo-gft`
  adjacency validation and combinatorial Laplacian construction; nalgebra
  remains only for symmetric eigendecomposition until Leto owns that solver
  contract.

## Apollo Requirements for Hephaestus

- Typed device buffers and transfers through `ComputeDevice`; no raw device
  buffer handles in Apollo APIs.
- ZST-authored kernels through `KernelInterface` plus one real
  `KernelSource<L>` implementation per supported backend dialect.
- Prepared typed dispatch and ordered command streams through `KernelDevice`,
  `Binding`, and `DispatchGrid`, with binding contracts validated before
  submission.
- Concrete WGPU and CUDA mechanics remain inside their Hephaestus backend
  crates. Apollo must not reconstruct pipeline, bind-group, encoder, queue, or
  stream APIs locally.
- Unsupported dialects are compile-time missing implementations or typed
  capability errors; no silent backend fallback.

## GPU Boundary

Moirai, Mnemosyne, Melinoe, Hermes, and Leto optimize CPU scheduling, host memory,
branded zero-copy access, host SIMD kernels, and host strided-array storage.
Apollo GPU modules bind to Hephaestus contracts. Raw WGPU/CUDA buffers,
pipelines, bind groups, encoders, queues, streams, and futures remain inside
Hephaestus backend crates and do not leak into Apollo domain or infrastructure
types.

## Verification

Run:

```powershell
cargo run -p xtask -- provider-audit
```

The audit reports Moirai, Mnemosyne, Melinoe, Hermes, Leto, Hephaestus, Rayon,
WGPU mentions, `Arc`, `Mutex`, `dyn`, clone-to-`Vec`, and `Cow` usage by crate.
The evidence tier is static source analysis; performance claims still require
the native `apollo-bench` runner or a domain-specific benchmark with a recorded
baseline comparison.

## Eunomia 0.10 provider-cohort migration

Apollo FFT 0.29 and the corresponding family versions in CHANGELOG.md consume
Eunomia 0.10, Mnemosyne-memory 0.10, Hermes 0.9, Leto/Leto-ops 0.46,
Hephaestus 0.22 and Moirai 0.9.3. Direct Rust consumers update these compatible
provider requirements together to preserve one scalar/layout/device identity.
Operation signatures, arithmetic, tolerances and test domains stay unchanged.

A review lock may name actual reviewed upstream commits. Such a locked build
proves source compatibility, not the availability of an unmerged default or
an unpublished registry package. Merge the owning provider increments first,
refresh the final lock outside the Atlas development overlay, then execute
an unseeded external consumer with canonical Git sources and no path/patch
substitutions. Registry publication retains its separate dependency order and
release authority. The FFT correctness recovery and historical experiment
remain independently reviewable in PRs #671 and #655.

## Hephaestus 0.27.1 private compatibility preparation

FFT 0.30 and the corresponding transform versions in ADR 0077 require
Hephaestus ^0.27.1. Public accelerator consumers advance their direct provider
and Apollo requirements together. The real a735 review lock is private
source evidence while #509 remains draft, unmerged and unpublished. Preserve
the provider-first default/registry and no-overlay consumer gates above;
independent exact-head/ADR review and one owner-selected release source remain
pending. No numerical, performance, hardware, Python or publication acceptance
follows from admission. Initial service-blocked Apollo LaneBuffer work is
outside this increment and is not retried.

The authorized successor is a735c608dd686bc665ef48ac329aca4872b8508f,
tree cce4ff3202a4b15405d928cc9d29a0ec4db224ff. Consumer Leto/LetoOps remain
0.46.3 at 5051cd9577650a5712aa6a7effcd8955c9d8bcc7; the provider-own
0.46.1/7fd6256f lock is a separate check graph. Prior ec650 consumer receipts
remain scoped historical evidence, with successor checks and independent
review recorded separately in the private delivery.
