# ADR 0034: Dense FFT dispatch verification tree

- Status: Accepted
- Date: 2026-07-17
- Change class: [arch]

## Context

`apollo-fft` combined the typed Hephaestus dispatch implementation and two
device-present verification contracts in one 589-line leaf. The verification
code had no runtime responsibility and made the provider execution boundary
harder to audit against the deep vertical tree.

## Decision

Keep the dispatch implementation in `gpu_fft/dispatch.rs` and move its
device-present contracts to the private test leaf
`gpu_fft/verification/dispatch.rs`. The parent remains the only declaration
and continues to expose the same `GpuFft3d` API. No provider adapter, raw WGPU
surface, fallback, or compatibility export is introduced.

## Theorem and evidence boundary

The moved tests retain the dense-FFT inverse identity
\(\mathcal{F}^{-1}(\mathcal{F}(x))=x\) for the 2×2×2 delta field. The 2×3×2
Bluestein fixture retains its derived \(\gamma_{256}\) forward bound and
\(13\gamma_{256}\) round-trip bound, where
\(\gamma_k=ku/(1-ku)\) and \(u=2^{-24}\) for binary32. The refactor changes
only module placement; these equations remain empirical GPU differential
evidence, not machine-checked proofs.

## Consequences

- The dispatch leaf is below the 500-line topology target.
- Verification remains private and test-gated, so release APIs do not grow.
- Future GPU dispatch contracts have a canonical verification home without
  duplicating provider ownership in Apollo.
