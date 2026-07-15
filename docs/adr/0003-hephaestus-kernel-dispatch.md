# ADR 0003: Hephaestus-owned GPU kernel dispatch

- Status: Accepted
- Date: 2026-07-13
- Change class: arch

## Context

Apollo owns spectral-transform mathematics and Leto host-array boundaries, but
its GPU modules directly construct WGPU buffers, pipelines, bind groups,
encoders, and submissions. `apollo-wgpu-helpers` wraps a Hephaestus device and
then exposes those raw WGPU types again. This aligns dependency versions but
does not invert the backend dependency or admit a second Hephaestus device
implementation without rewriting each transform.

Hephaestus already provides the required backend-neutral contracts:
`ComputeDevice` for typed buffers and transfers, `KernelInterface` and
`KernelSource` for authored kernels, `KernelDevice` for preparation and typed
dispatch, and `DispatchGrid` for launch geometry.

## Decision

Each Apollo GPU kernel is a zero-sized domain type implementing one
dialect-free `KernelInterface` and one `KernelSource<L>` per real backend
dialect. Transform orchestration binds generically to Hephaestus device and
kernel contracts. Leto owns host views and returned arrays; Apollo owns
transform parameters, formulas, and dialect source; Hephaestus owns device
allocation, compilation, binding, dispatch, synchronization, and transfer.

Migration proceeds by complete transform bounded contexts. The first increment
converts FWHT butterfly and inverse-scale execution, deletes its direct `wgpu`,
`pollster`, and `apollo-wgpu-helpers` edges, removes public raw-device access,
and updates every in-repository call site. No adapter or dual path remains in a
converted context. CUDA becomes real only when the same kernel interfaces gain
verified `KernelSource<CudaC>` implementations; unsupported dialects fail at
compile time rather than falling back.

## Rejected alternatives

- Retain `apollo-wgpu-helpers`: preserves a consumer-owned mirror of the
  Hephaestus device API and raw backend leakage.
- Depend directly on WGPU inside each transform: duplicates backend mechanics
  and makes CUDA a cloned algorithm path.
- Move transform WGSL into Hephaestus: transfers domain mathematics to the
  infrastructure provider and violates bounded-context ownership.
- Introduce a local Apollo backend wrapper: recreates the Hephaestus contract
  and is a compatibility shim.

## Failure modes and verification

Binding arity, access, and element sizes are validated by Hephaestus before
dispatch. Apollo validates transform shape and input length before allocation.
FWHT additionally verifies the independent identity `H_n² = nI`. Each
converted context must pass CPU differential and inverse-roundtrip tests on a
real device, negative contract tests, warning-denied compile/documentation
gates, provider audit, and a static scan proving its direct WGPU dependencies
and raw source calls are absent.
