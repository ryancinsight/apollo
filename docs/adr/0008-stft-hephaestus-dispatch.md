# ADR 0008: STFT dispatch through Hephaestus

## Status

Accepted on 2026-07-15.

## Decision

`apollo-stft` retains frame planning, Hann-window arithmetic, FFT and
Bluestein mathematics, weighted overlap-add, and WGSL source. It deletes
direct WGPU pipeline, bind-group, encoder, queue, transfer, and helper-device
ownership. Typed Hephaestus descriptors represent each radix-2 and Bluestein
kernel, while an ordered command stream represents producer-before-consumer
execution. The overlap-add kernel is a flat descriptor because it has one
binding group; FFT and chirp stages use grouped descriptors because their
storage and parameter groups are distinct.

Leto remains the host-array boundary. `StftGpuBuffers` retains typed provider
storage for one radix-2 geometry. The Bluestein path requests six storage
bindings through the backend-neutral Hephaestus `DeviceLimits` contract: two
operation I/O buffers plus four chirp working or kernel buffers. The obsolete
`wgpu_backend` forwarding module and direct `apollo-wgpu-helpers`, `wgpu`, and
`pollster` dependencies are removed.

## Mathematical contract

For a complete frame DFT, root-of-unity orthogonality makes the normalized
inverse recover the windowed analysis frame exactly. Synthesis applies the
same window, so weighted overlap-add evaluates
`sum_m x[t] w[t-mH]^2 / sum_m w[t-mH]^2`, which equals `x[t]` wherever the
denominator is non-zero. Command stream order establishes the corresponding
device write-before-read dependencies. This is an exact-arithmetic theorem;
CPU differential and reconstruction tests are empirical finite-precision
evidence rather than a machine-checked proof.
