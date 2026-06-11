- `apollo-validation`: parity, adversarial, benchmark, and external-reference runners. Includes 51 published-reference fixtures:
SDFT sliding-update recurrence unit-impulse N=4 (Jacobsen-Lyons 2003), UnitaryFrFT order-4 identity N=4 (Candan 2000), CZT off-unit-circle Z-transform A=2 N=2 (Rabiner-Schafer-Rader 1969), and Hilbert envelope of pure cosine tone N=4 (Oppenheim-Schafer 2010).
# Apollo Workspace

Apollo is a standalone workspace for reusable Fourier transform planning, execution, validation, backend adapters, and Python bindings.

Stage 2 moves Apollo beyond the initial compatibility cut:

- `apollo-fft` owns reusable dense CPU FFT plans, cache orchestration, shared contracts, and backend abstractions.
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
- `apollo-fft-wgpu` owns the real shader-backed 3D WGPU FFT path with radix-2 and Bluestein/Chirp-Z axis strategies. The optional `native-f16` feature adds `GpuFft3dF16Native`, which executes all butterfly arithmetic directly in `f16` inside the shader when the adapter exposes `wgpu::Features::SHADER_F16`, covering both power-of-two (radix-2) and non-power-of-two (Bluestein chirp-Z) sizes.
- `apollo-nufft-wgpu` owns NUFFT GPU execution and ships exact direct Type-1 and Type-2 WGPU kernels for 1D and 3D, plus fast Kaiser-Bessel gridding paths for both 1D and 3D. The fast paths perform GPU spreading or interpolation, dispatch oversampled FFTs through `apollo-fft-wgpu`, and apply GPU deconvolution against the same Kaiser-Bessel metadata used by `apollo-nufft`.
- Per-transform WGPU crates own GPU backend boundaries for their respective mathematical domains. `apollo-fwht-wgpu` and `apollo-dht-wgpu` ship real 1D `f32` kernels, `apollo-dctdst-wgpu` ships the full real 1D `f32` DCT-II/DCT-III/DST-II/DST-III family, `apollo-czt-wgpu` ships direct complex forward and inverse CZT execution, `apollo-gft-wgpu` ships forward and inverse graph-basis execution, `apollo-hilbert-wgpu` ships forward analytic/quadrature Hilbert execution, `apollo-mellin-wgpu` ships forward and inverse Mellin log-frequency spectrum execution, `apollo-ntt-wgpu` ships forward and inverse NTT execution on its supported modulus surface, `apollo-qft-wgpu` ships forward and inverse dense unitary QFT execution, `apollo-radon-wgpu` ships forward parallel-beam projection, adjoint backprojection, and ramp-filtered backprojection (FBP) execution, `apollo-sdft-wgpu` ships forward direct-bin sliding DFT execution, `apollo-sft-wgpu` ships direct dense DFT sparse top-k execution and inverse reconstruction, `apollo-sht-wgpu` ships direct complex SHT forward/inverse execution using owner-derived basis and quadrature buffers, `apollo-stft-wgpu` ships FFT-accelerated forward Hann-windowed STFT and inverse WOLA reconstruction execution for both power-of-two and non-power-of-two frame lengths, and `apollo-wavelet-wgpu` ships forward and inverse Haar DWT execution.
- `apollo-wavelet` owns discrete and continuous wavelet transform plans for multiresolution analysis.
- `apollo-validation` emits structured CPU, GPU, NUFFT, benchmark, and external-comparison reports.
- `apollo-python` exposes FFT, NUFFT, precision selection, and backend capability introspection for Python callers.

GPU benchmark automation runs on a dedicated self-hosted workflow path rather than the hosted CPU CI path. The repository ships `.github/workflows/gpu-benchmarks.yml` plus `scripts/run_gpu_benchmarks.ps1` to execute the WGPU Criterion suites on a runner labeled `[self-hosted, gpu, apollo]`, stage outputs under `.benchmarks/gpu-runner/`, and upload the result bundle as a workflow artifact.

Mixed precision is now a first-class Apollo concept:

- CPU defaults to `high_accuracy` (`f64` storage and `f64` compute).
- CPU also supports opt-in `low_precision` (`f32` storage and `f32` compute).
- CPU also supports opt-in `mixed_precision` (`half::f16` storage with `f32` compute in the current FFT path).
- WGPU exposes `low_precision` (`f32` shaders) as the default GPU profile. Mixed `f16`-host / `f32`-GPU typed storage paths are available on all WGPU crates except `apollo-ntt-wgpu`, which uses exact `u32` modular residues instead.
- `apollo-fft-wgpu` additionally supports native `f16` GPU arithmetic via the `native-f16` feature (`GpuFft3dF16Native`). Twiddle factors are computed in `f32` then narrowed to `f16`; accumulation error is bounded by `O(log N)·ε_f16` where `ε_f16 ≈ 9.77×10⁻⁴`.
- The authoritative per-crate precision surface is documented in `ARCHITECTURE.md` under the Mixed-Precision Capability Table.

## Crates

- `apollo-fft`: CPU FFT plans, cache management, shared types, and backend abstractions.
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
- `apollo-fft-wgpu`: Real dense FFT WGPU backend and GPU parity surface.
- `apollo-nufft-wgpu`: exact direct and fast Kaiser-Bessel gridded Type-1/Type-2 NUFFT WGPU execution for 1D and 3D with `apollo-fft-wgpu`-backed oversampled FFT dispatch.
- `apollo-czt-wgpu`, `apollo-dctdst-wgpu`, `apollo-dht-wgpu`,
  `apollo-frft-wgpu`, `apollo-gft-wgpu`,
  `apollo-hilbert-wgpu`, `apollo-mellin-wgpu`, `apollo-ntt-wgpu`,
  `apollo-qft-wgpu`, `apollo-radon-wgpu`, `apollo-sdft-wgpu`,
  `apollo-sft-wgpu`, `apollo-sht-wgpu`, `apollo-stft-wgpu`, and
  `apollo-wavelet-wgpu`: per-transform WGPU backend boundaries with
  capability and plan descriptor contracts.
- `apollo-czt-wgpu`: direct complex forward and inverse CZT WGPU execution with CPU parity tests on the implemented `f32` surface.
- `apollo-dctdst-wgpu`: 1D real `f32` DCT-II/DCT-III/DST-II/DST-III WGPU execution with CPU parity tests.
- `apollo-dht-wgpu`: 1D real `f32` DHT WGPU execution with CPU parity tests.
- `apollo-fwht-wgpu`: 1D real `f32` FWHT WGPU execution with CPU parity tests.
- `apollo-gft-wgpu`: forward and inverse real graph Fourier basis WGPU execution with CPU parity tests on the implemented `f32` surface.
- `apollo-hilbert-wgpu`: forward analytic-signal and quadrature Hilbert WGPU execution with CPU parity tests; inverse recovers original signal DFT spectrum via the analytic-mask inversion rule.
- `apollo-mellin-wgpu`: forward and inverse Mellin log-frequency spectrum WGPU execution with CPU parity tests on the implemented `f32` surface.
- `apollo-ntt-wgpu`: forward and inverse modular NTT WGPU execution with CPU parity tests on the supported 32-bit modulus surface.
- `apollo-qft-wgpu`: forward and inverse dense unitary QFT WGPU execution with CPU parity tests on the implemented `f32` complex surface.
- `apollo-radon-wgpu`: forward parallel-beam Radon projection, adjoint backprojection (GPU WOLA, Natterer 2001 §II.2), and ramp-filtered backprojection (GPU FBP, Ram-Lak filter, Bracewell & Riddle 1967) WGPU execution with CPU parity tests.
- `apollo-sdft-wgpu`: forward and inverse direct-bin sliding DFT WGPU execution with CPU parity tests.
- `apollo-sft-wgpu`: forward dense direct DFT with deterministic top-k sparse projection and inverse sparse reconstruction with CPU parity tests.
- `apollo-sht-wgpu`: forward and inverse complex spherical harmonic transform WGPU execution with CPU parity tests on owner-derived quadrature and basis buffers.
- `apollo-stft-wgpu`: FFT-accelerated forward Hann-windowed STFT and inverse WOLA reconstruction (Allen-Rabiner 1977, Theorem 1) WGPU execution with CPU parity tests; power-of-two sizes use Radix-2 DIT and non-power-of-two sizes use the Bluestein/Chirp-Z path.
- `apollo-wavelet-wgpu`: forward and inverse Haar DWT WGPU execution with CPU parity tests on the implemented `f32` surface.
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
- GPU FFT: `apollo-fft-wgpu`. Radix-2 GPU execution stages bit reversal,
  butterfly stages, and inverse scaling as separate compute passes over
  linearized workgroup domains.
- GPU NUFFT: `apollo-nufft-wgpu`. NUFFT WGPU support is a separate transform
  domain boundary and does not live in the dense FFT backend crate.
- Other GPU transform domains: each transform has an owning `*-wgpu` crate.
  `apollo-czt-wgpu` now executes the direct complex forward CZT and the square-plan
  adjoint inverse on WGPU for the implemented `f32` kernel surface, validating
  parity and DFT-case roundtrip against the owning CPU crate.
  `apollo-hilbert-wgpu` now executes the analytic-signal Hilbert path on WGPU
  by direct DFT, analytic-spectrum masking, and inverse DFT over the
  implemented `f32` surface. Inverse remains unsupported there until the
  owning Hilbert crate defines and verifies an authoritative inverse or
  adjoint contract.
  `apollo-mellin-wgpu` now executes the forward Mellin log-frequency spectrum
  on WGPU by log-resampling onto the plan scale grid and applying the direct
  spectrum sum over the implemented `f32` surface, and executes the inverse by
  a two-pass GPU IDFT plus exp-resample pipeline validated against the owning
  CPU crate.
  `apollo-ntt-wgpu` now executes the forward and inverse NTT on WGPU by direct
  modular summation over the supported 32-bit modulus surface and validates
  parity against the owning CPU crate.
  `apollo-gft-wgpu` now executes the forward and inverse graph Fourier
  transform on WGPU by direct multiplication against the orthonormal spectral
  basis carried by the plan and validates parity against the owning CPU crate.
  `apollo-qft-wgpu` now executes the forward and inverse dense unitary QFT on
  WGPU by direct complex summation with `1 / sqrt(n)` normalization and
  validates parity against the owning CPU crate.
  `apollo-radon-wgpu` now executes the forward parallel-beam Radon projection,
  adjoint backprojection, and ramp-filtered backprojection on WGPU by direct
  image-space accumulation plus GPU filtering and validates parity against the
  owning CPU crate.
  `apollo-sdft-wgpu` now executes forward direct-bin sliding DFT evaluation on
  WGPU by direct `O(N*K)` summation over the current real-valued window and
  validates parity against `apollo-sdft::SdftPlan::direct_bins`. Inverse
  remains unsupported there because the crate exposes streaming bin evaluation
  rather than a reconstruction contract.
  `apollo-sht-wgpu` now executes forward and inverse complex SHT on WGPU by
  direct matrix summation over owner-derived Gauss-Legendre quadrature weights
  and spherical harmonic basis values, preserving `apollo-sht` as the SSOT for
  basis and quadrature definitions.
  `apollo-stft-wgpu` now executes both forward Hann-windowed STFT and inverse
  WOLA reconstruction on the GPU via a batched FFT / IFFT path: Radix-2 DIT for
  power-of-two frame lengths and Bluestein/Chirp-Z for non-power-of-two frame
  lengths. Forward validates parity against `apollo-stft::StftPlan::forward`;
  inverse validates against CPU WOLA reconstruction (Allen-Rabiner 1977,
  Theorem 1). Mixed-precision f16/f32 host-boundary paths are supported for
  both directions.
  `apollo-wavelet-wgpu` now executes forward and inverse multi-level Haar DWT
  on WGPU using a Mallat decomposition and validates forward coefficients and
  inverse reconstruction against `apollo-wavelet::DwtPlan` for the Haar
  surface. Daubechies4 and CWT remain CPU-only until their GPU contracts are
  implemented and verified.
  `apollo-dctdst-wgpu` now executes real 1D DCT-II, DCT-III, DST-II, and
  DST-III on WGPU and computes their inverses by the paired transform with
  `2 / n` normalization.
  `apollo-dht-wgpu` now executes the 1D real DHT on WGPU with direct Hartley
  summation and inverse normalization by `1 / n`.
  `apollo-fwht-wgpu` now executes the 1D real FWHT on WGPU with the same
  butterfly network as the CPU plan and inverse normalization by `1 / n`.
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

## GPU Benchmark Runner

Apollo's hosted CI remains CPU-only. WGPU benchmarks run through a separate manual workflow on a labeled self-hosted runner:

- Workflow: `.github/workflows/gpu-benchmarks.yml`
- Runner labels: `self-hosted`, `gpu`, `apollo`
- Script entry point: `scripts/run_gpu_benchmarks.ps1`
- Output bundle: `.benchmarks/gpu-runner/run-<run_id>/`

The workflow is manual (`workflow_dispatch`) so normal pull-request CI never blocks on GPU hardware availability. The runner script executes these Criterion suites:

- `apollo-fft-wgpu` `buffer_reuse`
- `apollo-nufft-wgpu` `buffer_reuse`
- `apollo-stft-wgpu` `stft_bench`
- `apollo-radon-wgpu` `radon_wgpu_bench`

Each run captures per-benchmark console logs, a machine-readable `manifest.json`, a human-readable `summary.md`, and the full `target/criterion` directory for artifact upload.

## References

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`docs/THEORY.md`](./docs/THEORY.md)
- [`docs/VALIDATION.md`](./docs/VALIDATION.md)
- [`docs/MIGRATION_KWAVERS.md`](./docs/MIGRATION_KWAVERS.md)
