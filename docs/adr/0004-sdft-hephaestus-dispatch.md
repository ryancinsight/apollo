# ADR 0004: SDFT typed Hephaestus dispatch

- Status: Accepted
- Date: 2026-07-14
- Change class: arch

## Context

`apollo-sdft` owned raw WGPU pipelines, bind groups, encoders, submission, and
readback through `apollo-wgpu-helpers`. That duplicated the device-provider
boundary defined by ADR 0003 and exposed a forwarding module beside the
canonical GPU surface. The old inverse also accepted a partial spectrum even
though its reconstruction formula is an inverse only for a complete DFT.

## Decision

Apollo owns two zero-sized, typed SDFT kernel descriptors: real `f32` window
to complex-bin forward execution and complex-bin to complex-sample inverse
execution. The descriptors implement Hephaestus binding declarations, WGSL
source, dispatch geometry, and command-stream execution. Hephaestus owns
device acquisition, allocation, validation, pipeline preparation, binding,
submission, synchronization, and transfer. Leto owns host-array views and
Mnemosyne owns generated host storage.

The concrete accelerator accepts only `f32` real input and `Complex32` bin
storage, plus explicit `f16` storage conversion. The CPU `f64`/`Complex64`
storage contract is deliberately not admitted by the GPU traits, so a caller
cannot silently narrow an SDFT request. The obsolete `wgpu_backend` forwarding
module, raw device/queue accessors, and CPU-marker alias are removed.

Forward execution supports `1 <= K <= N` tracked bins. Inverse execution
requires `K = N`; a partial spectrum is a projection and does not identify an
arbitrary length-`N` window.

## Theorem: complete-bin reconstruction

For `X[k] = sum_(n=0)^(N-1) x[n] exp(-2 pi i k n / N)`, define
`x_hat[m] = (1/N) sum_(k=0)^(N-1) X[k] exp(2 pi i k m / N)`. Substitution gives

`x_hat[m] = sum_n x[n] (1/N) sum_k exp(2 pi i k (m-n) / N)`.

The finite root-of-unity sum is one when `m = n` and zero otherwise; therefore
`x_hat[m] = x[m]` in exact arithmetic. The provider enforces `K = N` before
dispatch so this theorem is the inverse operation's contract. Floating-point
execution is checked against the independent CPU direct-bin implementation and
the forward/inverse round trip.

## Rejected alternatives

- Retain `apollo-wgpu-helpers`: preserves a consumer-owned raw-device wrapper.
- Keep raw WGPU calls behind an SDFT-local adapter: recreates the provider
  interface instead of consuming Hephaestus directly.
- Accept partial-bin inverse reconstruction: presents a projection as an
  inverse and violates the stated theorem.
- Convert `f64` host storage implicitly: changes requested values at the
  accelerator boundary without an explicit caller decision.

## Failure modes and verification

Plan bounds and the complete-spectrum requirement are checked before device
allocation. Hephaestus validates typed binding layouts and manages all GPU
mechanics. Verification covers CPU differential forward values, the complete
DFT round trip, Leto contiguous and strided host views, reduced-storage
conversion, invalid plan and partial-inverse rejection, compilation,
documentation, provider audit, and a source scan that rejects raw WGPU,
`pollster`, and helper references in this crate.
