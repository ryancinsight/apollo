# ADR 0006: Dense WGPU FFT ownership in Hephaestus

- Status: Accepted
- Date: 2026-07-15
- Revised: 2026-08-28
- Change class: [major] [arch]
- Board item: `APOLLO-FFT-HEPHAESTUS-CUTOVER-2026-08-28`

## Context

Apollo previously implemented a second dense WGPU FFT below
`apollo-fft::GpuFft3d`. It owned radix and Bluestein selection, WGSL, prepared
pipelines, split-complex device buffers, staging, and a three-dimensional-only
backend facade. Hephaestus now implements the same mathematical role through
the rank-generic `FftOps` seam for one-, two-, and three-dimensional f32 and
native-f16 operands. Retaining Apollo's implementation would duplicate the
algorithm, capability model, verification surface, and warm-plan state.

Leto owns host layouts and CPU array storage. Apollo owns its dense CPU FFT and
NUFFT mathematics. Hephaestus owns accelerator FFT algorithms and WGPU
resources. These roles do not require an Apollo WGPU FFT facade between a
consumer and Hephaestus.

## Decision

Delete Apollo's dense WGPU FFT plan, shaders, workspace, benchmark, native-f16
feature, and public `GpuFft3d`, `GpuFft3dBuffers`, `GpuFft3dF16Native`, and
`WgpuBackend` exports. No alias or forwarding adapter replaces them.

Accelerator consumers construct a Leto `Layout<R>`, pair fixed Hephaestus
split-complex buffers with `StridedView`, and prepare `WgpuPreparedFft<R, T>`
through `WgpuFftOps`. Backend selection occurs at the operation boundary. A
prepared plan is bound to its device, layout, direction, and buffers; repeated
dispatch encodes that plan into the consumer's existing command stream without
allocation, compilation, transfer, or capability probing.

`apollo-nufft` stores forward and inverse prepared plans beside its reusable
oversampled grids. Its spread/load, FFT, and extract/interpolate passes remain
one ordered Hephaestus stream. `apollo-validation` invokes Hephaestus directly
and compares its result with Apollo's Leto-backed CPU FFT. Apollo's CUDA FFT
surface is unchanged and remains governed by ADR 0030.

Provider absence is reported at acquisition. A present provider failure is
returned as a typed error; execution never silently falls back to Apollo CPU.

## Mathematical and evidence contract

For shape \(N_0,\ldots,N_{R-1}\), Hephaestus implements the unnormalized
forward transform

\[
X_k = \sum_x x_x\exp\left(-2\pi i\sum_d k_dx_d/N_d\right)
\]

and the positive-exponent inverse normalized by \(\prod_d N_d\). Root-of-unity
orthogonality gives \(\mathcal{F}^{-1}(\mathcal{F}(x))=x\) in exact
arithmetic. Apollo's validation uses Hephaestus's radix-stage operation count
with \(\gamma_k=ku/(1-ku)\): 31 rounding sites for the fixed 4×4×4 forward
case and 64 for its normalized round trip. The absolute bounds scale by the
represented input's L1 and L2 norms respectively. This is an analytical
finite-precision bound paired with empirical CPU differential and
inverse-roundtrip evidence, not a machine-checked proof.

## Migration

- Replace Apollo `GpuFft3d` construction with `WgpuFftOps::prepare_fft` over
  `FftOperands` and a Leto `Layout<R>`.
- Retain the returned `WgpuPreparedFft` with the fixed device buffers and call
  `encode_fft` for composed work or `dispatch_fft` for a standalone transform.
- Acquire `WgpuDevice` and any required `ShaderF16` capability from Hephaestus.
- Keep Apollo CPU plans for Leto-host execution; do not route accelerator
  failures through them.

## Revision note

The 2026-08-28 revision replaces the former descriptor-only delegation. The
accepted implementation evidence in Hephaestus made Apollo's algorithm and
resource layer obsolete, so ADRs 0028, 0029, and 0034 were deleted with their
subjects. Git history retains those superseded decisions.
