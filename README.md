# Apollo Workspace

Apollo is a standalone workspace for reusable Fourier transform planning, execution, validation, backend adapters, and Python bindings.

## Release contract

Apollo releases from Git tags with its committed `Cargo.lock`, pinned Rust
toolchain, and exact Atlas provider revisions. The workspace is not published
as independent crates.io packages because its first-party provider crates are
not available from that registry. See
[`docs/adr/0002-git-source-release-contract.md`](docs/adr/0002-git-source-release-contract.md).

Stage 2 moves Apollo beyond the initial compatibility cut:

- `apollo-fft` owns reusable dense CPU FFT plans, cache orchestration, shared contracts, and backend abstractions.
- `apollo-leto-interop` owns the shared, rank-polymorphic Leto host-boundary
  contract; transform crates do not depend on FFT implementation modules for
  view materialization or Mnemosyne-backed output construction.
- `apollo-bench` owns the native sequential measurement contract used by every
  benchmark binary; it reports normalized median/minimum samples without
  introducing a Rayon-backed harness.
- `apollo-dctdst` owns DCT/DST real-to-real transform plan metadata,
  verified direct kernels, inverse scaling, and caller-owned output execution.
- `apollo-dht` owns real-to-real Discrete Hartley Transform plans, coefficient storage, and self-inverse kernels.
- `apollo-hilbert` owns Hilbert transform plans, analytic-signal storage, envelope extraction, and phase extraction.
- `apollo-nufft` owns non-uniform FFT plans and direct-reference validation surfaces.
- `apollo-radon` owns parallel-beam Radon projections, adjoint backprojection, and filtered backprojection.
- `apollo-sht` owns spherical harmonic transform grid metadata, coefficient storage, Gauss-Legendre quadrature, and value-computing SHT plans.
- `apollo-sdft` owns sliding DFT streaming-window metadata, O(bin_count) update recurrence kernels, and streaming state.
- `apollo-mellin` owns Mellin scale-domain metadata and validation.
- `apollo-sft` owns the sparse Fourier transform single source of truth.
- Each transform crate owns its mathematical kernels and exposes accelerator
  execution through its `wgpu` feature. Hephaestus exclusively owns device,
  queue, buffer, pipeline, command, and transfer mechanics.
- `apollo-fft` owns the shader-backed 3D FFT algorithm with radix-2 and
  Bluestein/Chirp-Z axis strategies. Its `native-f16` scope reuses the same
  typed Hephaestus descriptor and command-stream implementation as f32 while
  retaining `enable f16;` shader arithmetic and a required `ShaderF16`
  capability contract.
- `apollo-nufft` owns exact and Kaiser-Bessel-gridded NUFFT algorithms;
  direct and fast execution use typed Hephaestus descriptors and compose
  oversampled FFT stages through the `apollo-fft` provider stream.
- `apollo-wavelet` owns discrete and continuous wavelet transform plans for multiresolution analysis.
- `apollo-validation` emits structured CPU, GPU, NUFFT, benchmark, and external-comparison reports.
- `apollo-python` exposes FFT, NUFFT, precision selection, and backend capability introspection for Python callers.

Mixed precision is now a first-class Apollo concept:

- CPU defaults to `high_accuracy` (`f64` storage and `f64` compute).
- CPU also supports opt-in `low_precision` (`f32` storage and `f32` compute).
- CPU also supports opt-in `mixed_precision` (`half::f16` storage with `f32` compute in the current FFT path).
- WGPU exposes `low_precision` (`f32` shaders) as the default GPU profile.
  Mixed `f16`-host / `f32`-GPU typed storage paths are available through each
  transform crate's `wgpu` feature except `apollo-ntt`, which uses exact `u32`
  modular residues.
- `apollo-fft` additionally supports native `f16` GPU arithmetic through its
  `native-f16` feature (`GpuFft3dF16Native`). Twiddle factors are computed in
  `f32` then narrowed to `f16`; per-output accumulation error is bounded by
  `O(log N)·ε_f16·‖input‖₁`, where `ε_f16 ≈ 9.77×10⁻⁴`.
- The authoritative per-crate precision surface is documented in `ARCHITECTURE.md` under the Mixed-Precision Capability Table.

## Crates

- `apollo-fft`: CPU FFT plans, cache management, shared types, and backend abstractions.
- `apollo-leto-interop`: rank-polymorphic zero-copy Leto view materialization
  and fallible Mnemosyne-backed output construction shared by transform crates.
- `apollo-czt`: chirp z-transform plans, direct reference execution,
  Bluestein convolution execution, and caller-owned output paths.
- `apollo-dctdst`: DCT/DST real-to-real transform plans, verified direct
  kernels, inverse scaling, and caller-owned output execution.
- `apollo-dht`: real-to-real Discrete Hartley Transform plans with forward/inverse kernel reuse.
- `apollo-frft`: fractional Fourier transform reference plans with finite
  integer-rotation state and caller-owned output execution.
- `apollo-fwht`: fast Walsh-Hadamard transform plans, in-place kernels, and
  caller-owned output execution.
- `apollo-gft`: graph Fourier transform domain validation, Laplacian spectral basis construction, reusable plans, and verification.
- `apollo-hilbert`: Hilbert transform plans with analytic-signal, envelope, and phase extraction.
- `apollo-mellin`: Mellin scale-domain moments, log-resampling, and log-frequency spectra.
- `apollo-ntt`: radix-2 number theoretic transform plans, modular residue
  normalization, in-place kernels, and caller-owned output execution.
- `apollo-nufft`: non-uniform FFT plans and exact direct references.
- `apollo-qft`: quantum state-dimension validation, dense unitary QFT kernels, reusable plans, and verification.
- `apollo-radon`: parallel-beam Radon transform plans with sinogram storage and filtered backprojection.
- `apollo-sdft`: sliding DFT streaming plans with direct initialization and recurrence updates.
- `apollo-sft`: sparse Fourier transform domain model, plan execution, direct recovery kernel, and verification.
- `apollo-sht`: spherical harmonic transform plans with Gauss-Legendre latitude quadrature and complex coefficient storage.
- `apollo-stft`: short-time Fourier transform plans with centered-frame
  reconstruction, overlap-add normalization, and caller-owned output buffers.
- The `wgpu` feature on each transform crate owns that domain's GPU kernels,
  capability surface, and CPU differential verification. GPU infrastructure is
  shared through `hephaestus-wgpu`; GPU algorithms are not separate crates.
- `apollo-wavelet`: DWT/CWT multiresolution transforms with Haar, Daubechies-4, Ricker, and DC-corrected real Morlet support.
- `apollo-validation`: parity, adversarial, benchmark, and external-reference runners. Includes 59 published-reference fixtures: FFT 4-point (Cooley-Tukey 1965), FFT inverse 4-point (Cooley-Tukey 1965), DHT 4-point (Bracewell 1983), DHT self-reciprocal (Bracewell 1983), DCT-II 2-point (FFTW REDFT10), DST-II 2-point (FFTW RODFT10), DCT-II inverse pair (Rao-Yip 1990), NTT impulse N=4 (Pollard 1971), NTT constant N=4, NTT impulse N=8, NTT polynomial convolution, NTT impulse N=16, NTT polynomial product N=16, NUFFT impulse at origin (Dutt-Rokhlin 1993), NUFFT quarter-period phase (Dutt-Rokhlin 1993), FWHT 2-point (Hadamard 1893), QFT 2-point (Shor 1994), CZT unit impulse is DFT (Rabiner-Schafer-Rader 1969), GFT path graph (Shuman 2013), FrFT unitary order-2 reversal (Candan 2000), DWT Haar one-level detail (Haar 1910/Mallat 1989), SDFT bin-zero unit impulse, SFT 1-sparse alternating tone (Gilbert et al. 2002), SHT monopole Y₀⁰ coefficient (Driscoll-Healy 1994), STFT rectangular-window impulse frame (Gabor 1946), Hilbert cosine-to-sine 4-point (Bedrosian 1963), Mellin constant-function first moment (Mellin 1896), Radon θ=0 column-impulse projection (Radon 1917), CZT inverse Vandermonde roundtrip N=4 (Rabiner-Schafer-Rader 1969; Björck-Pereyra 1970), Mellin inverse spectrum constant roundtrip N=32 (Mellin 1896), Hilbert instantaneous frequency constant tone N=64 (Boashash 1992), Haar DWT inverse perfect reconstruction N=4 (Mallat 1989), GFT K₂ path graph inverse roundtrip (Sandryhaila-Moura 2013), FrFT inverse roundtrip α=0.5 N=4 (Namias 1980), FWHT inverse roundtrip N=4 (Walsh 1923), QFT inverse roundtrip N=4 (Shor 1994), SHT Y₁⁰ dipole inverse roundtrip lmax=1 (Driscoll-Healy 1994), DHT inverse roundtrip N=4 (Bracewell 1983), SFT inverse roundtrip N=4 K=1 (Hassanieh et al. 2012), NTT inverse roundtrip N=4 (Pollard 1971), STFT Hann-WOLA inverse roundtrip frame=4 hop=2 (Allen-Rabiner 1977), DCT-IV self-inverse roundtrip N=2 (Makhoul 1980), DST-IV self-inverse roundtrip N=2 (Makhoul 1980), DCT-I self-inverse roundtrip N=3 (Makhoul 1980), DST-I self-inverse roundtrip N=2 (Makhoul 1980), NUFFT Type-1/Type-2 adjoint inner-product N=2 (Dutt-Rokhlin 1993), Radon Fourier Slice Theorem θ=0 on 2×2 image (Natterer 1986), SDFT sliding-update recurrence unit-impulse N=4 (Jacobsen-Lyons 2003), UnitaryFrFT order-4 identity N=4 (Candan 2000), DWT Daubechies-4 one-level known coefficients N=4 (Daubechies 1992), DWT Daubechies-4 inverse perfect reconstruction N=4 (Mallat 1989), CWT Ricker impulse peak value ψ(0)=2/(√3·π^¼) N=7 a=1 (Daubechies 1992/Marr-Hildreth 1980), CWT Ricker scale-normalization W(a=2)=ψ(0)/√2 N=7 (Daubechies 1992/Grossmann-Morlet 1984), DCT-III DC input N=4 → flat output [½,½,½,½] (Makhoul 1980), DST-III Nyquist input N=4 → alternating output [½,−½,½,−½] (Makhoul 1980), DCT-I N=3 forward [1,2,3] → [8,−2,0] (Rao & Yip 1990), and DST-I N=2 forward [1,3] → [4√3,−2√3] (Rao & Yip 1990).
- `apollo-python`: PyO3 bindings, NumPy interop, and backend introspection.

## Architecture

Apollo uses a vertical hierarchy per crate:

```text
src/
  domain/            contracts, metadata, validated configuration
  application/       plans, orchestration, transform execution
  infrastructure/    concrete kernels, backend transport, external probes
  verification/      value-semantic tests where crate-local test modules are not enough
```

Dependency direction is one-way:

```text
lib.rs -> application -> domain
lib.rs -> infrastructure -> application -> domain
```

`domain` never depends on `application` or `infrastructure`. Backend-specific code stays behind infrastructure or adapter crates. Public APIs are exposed through crate roots and narrow compatibility modules.

### Transform Ownership

- Dense FFT: `apollo-fft`. The 1D, 2D, and 3D CPU plans use Apollo-owned
  radix-2 and Bluestein FFT kernels with FFTW-compatible public inverse
  normalization. The direct DFT kernel remains a reference surface.
- NUFFT: `apollo-nufft`. NUFFT domain descriptors, Kaiser-Bessel kernels,
  exact direct references, and fast gridding plans live outside `apollo-fft`.
- Sparse FFT: `apollo-sft`.
- Real-to-real DCT/DST: `apollo-dctdst`.
- Discrete Hartley Transform: `apollo-dht`.
- Hilbert Transform: `apollo-hilbert`.
- Radon Transform: `apollo-radon`.
- Spherical harmonic transform: `apollo-sht`.
- Sliding DFT: `apollo-sdft`.
- Mellin transform: `apollo-mellin`.
- Wavelet transforms: `apollo-wavelet`.
- GPU FFT: `apollo-fft` with the `wgpu` feature. Radix-2 execution stages bit
  reversal, butterfly stages, and inverse scaling as typed provider passes.
- GPU NUFFT: `apollo-nufft` with the `wgpu` feature; its kernels stay outside
  the dense FFT domain while consuming Apollo's typed FFT provider stream.
- Other GPU transforms: the owning transform crate exposes a `wgpu` feature.
  CPU mathematical definitions remain the SSOT, and GPU implementations carry
  value-semantic CPU differential tests for their supported surfaces.
- Python bindings: `apollo-python`.
- Validation and external parity: `apollo-validation`.

SFT is consolidated into `apollo-sft`; `apollo-fft` does not contain an SFT implementation or SFT export path. This preserves SSOT and prevents duplicated sparse transform logic.

## Precision Profiles

Apollo exposes these precision descriptors through Rust and Python:

- `high_accuracy`
- `low_precision`
- `mixed_precision`

In Rust, they are represented by `PrecisionMode`, `StoragePrecision`, `ComputePrecision`, and
`PrecisionProfile`. Existing APIs keep their current default behavior; lower-precision paths are
opt-in via `with_precision(...)` constructors or the generic `*_typed(...)` helpers that dispatch on
`RealFftData`.

Apollo now also exposes explicit `*_f16` helpers for real-domain FFT storage. The maintainable
Rust surface is the generic typed API.

## Zero-Copy and Modularity Invariants

Apollo implements workspace-wide memory-efficiency and strict structural modularity constraints:

- **Zero-Copy Cow Promotion**: All `*_typed` and `*_typed_into` execution boundaries across GPU/WGPU transform crates route inputs and outputs through `std::borrow::Cow` and dynamically check types at execution boundaries using `std::any::TypeId`. When input/output layouts align with the compute backend's internal precision profile (e.g. `f32` or `Complex32`), unsafe reinterpretation casts bypass heap allocation and quantization loops.
- **Staging Buffer Pooling & Pipeline Caching**: Concrete GPU backends query a thread-safe compute pipeline cache (`pipeline_cache` in the shared `hephaestus-wgpu` substrate) to avoid recompilation on shader execution. GPU host-staging transfers reuse recycled staging buffers through `WgpuDevice`'s staging pool helper.
- **Modularity Limits**: Every source file in the workspace targets a maximum of 500 lines. Monolithic files exceeding this limit (such as `device.rs` and `kernel.rs` in WGPU transform crates) are refactored into structured sub-folders (e.g., `device/forward.rs`, `device/inverse.rs`, `device/helpers.rs`) and extend the parent struct definitions via decentralized `impl` blocks.

## Design Rules

- `apollo/` is intentionally **not** a member of the root `d:\\kwavers\\Cargo.toml` workspace.
- Shared transform invariants live in the owning crate and are re-exported instead of duplicated.
- `kwavers` consumes Apollo FFT/NUFFT through compatibility re-exports instead of owning reusable transform implementations.
- Solver-specific spectral helpers remain in `kwavers` until they prove broadly reusable.
- Bounded variation belongs in traits, newtypes, configuration structs, strategy types, or backend abstractions, not cloned public APIs.
- Validation assertions inspect computed values and compare them against analytical invariants or independent references.

## References

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`docs/THEORY.md`](./docs/THEORY.md)
- [`docs/VALIDATION.md`](./docs/VALIDATION.md)
- [`docs/MIGRATION_KWAVERS.md`](./docs/MIGRATION_KWAVERS.md)
