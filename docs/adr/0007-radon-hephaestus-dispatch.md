# ADR 0007: Radon dispatch through Hephaestus

## Status

Accepted on 2026-07-15.

## Decision

`apollo-radon` retains the discrete projection, adjoint interpolation, and
Ram-Lak filter mathematics. It deletes direct WGPU pipeline, bind-group,
encoder, queue, transfer, and helper ownership. Three zero-sized typed
Hephaestus descriptors bind two read-only `f32` inputs and one read-write
`f32` output with a layout-asserted geometry parameter block.

Filtered backprojection records the filter kernel before the adjoint kernel in
one command stream. That ordered stream establishes filtered-sinogram
write-before-read. Leto is the host-array boundary; GPU buffers are
Hephaestus-owned.

## Mathematical contract

The discrete forward operator `R` deposits pixel-center masses to detector
bins with linear weights. Backprojection uses the transpose interpolation
weights, so `⟨R f, p⟩ = ⟨f, R* p⟩` in exact arithmetic. Filtered
backprojection computes `(pi / A) R*(h * p)` for `A` uniform angles and
Ram-Lak impulse response `h`. This states the discrete adjoint theorem; it
does not prove continuous inverse accuracy. CPU/GPU differential, adjoint, and
filtered-backprojection tests provide empirical finite-precision evidence.
