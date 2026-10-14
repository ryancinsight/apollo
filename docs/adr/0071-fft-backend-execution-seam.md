# ADR 0071: `FftBackend` becomes an execution seam

Status: Proposed  \
Date: 2026-10-10  \
Change class: [arch] [minor]  \
Board item: `APOLLO-CAP-BACKEND-EXECUTION` (audit item 14)

## Context

`FftBackend` (`apollo-fft/src/domain/contracts/backend.rs`) constructs
plans only: three unbounded associated plan types, capability
reporting, and fallible `plan_1d/2d/3d` constructors, with no execute
methods. Its only consumer is Python capability reporting
(`apollo-python/src/bindings/backend.rs`), which never runs a
transform. Callers wanting backend-generic code — CPU and CUDA
swapped under one generic — cannot write it against the seam as
shipped.

The two implementors show both halves of the gap, measured
2026-10-10. `CpuBackend` pins `f64` (`Plan1D = FftPlan1D<f64>`)
although the plans are scalar-generic (`FftPlan1D<F:
MixedRadixScalar>`) with a rich execute surface (forward/inverse
over `Array1`, Leto views, and slices, normalized and unnormalized).
`CudaBackend` covers 1-D `f32` only (`Plan1D = CudaFft1d`, whose
`execute_forward_in_place`/`execute_inverse_in_place` take `&mut
Array1<Complex32>` and return `ApolloResult`), with `Plan2D/Plan3D =
()` whose constructors already fail honestly with
`BackendUnavailable`. So the plans can already execute; only the
seam cannot express it.

## Decision

Keep `FftBackend` as the planning/capability trait and bind its plan
types with per-dimension execution role traits, with the scalar as a
parameter:

- The plan types become generic associated types over the sealed
  scalar vocabulary: `type Plan1D<F: MixedRadixScalar>`,
  `Plan2D<F>`, `Plan3D<F>`, and the constructors take the scalar as a
  type parameter (`fn plan_1d<F>(&self, shape: Shape1D) ->
  ApolloResult<Self::Plan1D<F>>`). `CpuBackend` then yields
  `FftPlan1D<F>` for every `F` instead of fixing `f64`; CUDA keeps
  `CudaFft1d` as its 1-D plan for every `F` and its constructor
  rejects `F != f32` with the existing `BackendUnavailable` error —
  unsupported (backend, scalar, dimension) triples fail at plan time,
  never at execution.
- One execution role per dimension, over the backend's buffer type:

  ```rust
  trait FftPlanExec1D<F: MixedRadixScalar> {
      type Buffer: ?Sized;
      fn forward(&mut self, data: &mut Self::Buffer) -> ApolloResult<()>;
      fn inverse(&mut self, data: &mut Self::Buffer) -> ApolloResult<()>;
  }
  ```

  (and `FftPlanExec2D`/`FftPlanExec3D` likewise). `&mut self` matches
  the stricter implementor — CUDA's execution mutates stream state —
  and `ApolloResult` matches the fallible one; CPU implementations
  hold no state and return `Ok`. The buffer stays backend-native
  (host slices/views for CPU, the staged array for CUDA) rather than
  forcing one buffer vocabulary on both tiers. `()` implements no
  role, so generic execution code over CUDA 2-D/3-D fails to compile
  — correctly, since no such plan can exist.
- Generic callers bound `B::Plan1D<F>: FftPlanExec1D<F>` (etc.) and
  monomorphize per backend, scalar, and dimension. No dynamic
  dispatch, no scalar boxing: the seam is traits plus associated
  types throughout.

WGPU is out of scope per the audit (Hephaestus owns it); the
`BackendCapabilities::WGPU` constant keeps its all-false flags until
a landing implementation flips them alongside this seam.

## Alternatives rejected

- Delete `FftBackend` and keep `BackendCapabilities`: rejected
  because capability reporting without an execution seam leaves the
  audit's stated need — backend-generic transform code —
  unaddressed, and the plans already carry the execute surface the
  seam needs; the missing piece is small.
- One role trait with the dimensionality as a parameter: rejected
  because the plans are already per-dimension types with
  per-dimension buffer shapes; a single generic role would re-derive
  that structure instead of bounding it.
- `&self` execution with interior mutability on CUDA: rejected as an
  implementor rewrite in execution-seam clothing; `&mut self` is the
  honest shared denominator and costs CPU callers nothing but a
  binding modifier.
- Infallible execution (`()` return): rejected because CUDA
  execution genuinely fails (launch, transfer, device loss); the
  role must carry the failure both implementors can produce.

## Consequences

`FftBackend` gains generic associated types (stable since Rust 1.65;
the workspace is on 1.95), a breaking change for the two
implementors and the one consumer, all in-repo. The follow-up
implements the three roles for the CPU plans and `CudaFft1d`, makes
the constructors scalar-generic, and lands the audit's acceptance
oracle: one generic conformance test over `B: FftBackend`
instantiated for CPU (and CUDA when a device is present) passing the
DFT oracle.

## Verification

The executing change proves the seam by the audit oracle: the
generic conformance test passes on CPU, passes on CUDA hardware when
present, and CUDA 2-D/3-D plus non-`f32` CUDA 1-D fail at plan time
with `BackendUnavailable` rather than compiling into untestable
`()`.
