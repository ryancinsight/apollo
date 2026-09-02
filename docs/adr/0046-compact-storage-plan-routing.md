# ADR 0046: Route compact storage through cached plans

- **Status:** Accepted (2026-09-02)
- **Class:** [patch] [arch]
- **Item:** `ATLAS-APOLLO-STORAGE-ROUTE-MISSES-THE-PLAN-2026-09-02`
- **Driving evidence:** [`gap_audit.md#half-storage-routing-corrected`](../../gap_audit.md#half-storage-routing-corrected)

## Context

The public `FftPrecision` implementation for `Complex<f16>` promoted storage
through the execution dispatch entry. That entry selects a generic Stockham
route and does not retain the `FftPlan1D` base states or twiddle tables between
calls. A warm pinned probe measured the resulting storage/plan ratios as 1.99x
at 64, 2.03x at 128, 1.49x at 256, and 1.32x at 512; the earlier 12–16x
reading was a first-run-after-build artifact.

`PlanCacheProvider for f16` already maps compact storage to the cached `f32`
plan. The missing route is therefore an ownership and placement gap, not a
new kernel requirement.

## Decision

1. Keep `FftPrecision` as the public dispatch seam, but define its
   `Complex<f16>` implementation beside the other cached public complex API
   operations in `api/cfft.rs`.
2. Preserve the measured stack-resident route for lengths 2, 4, 8, 16, and
   32. For larger lengths, resolve `<f16 as PlanCacheProvider>::get_1d_plan`
   and execute the cached `f32` plan through the existing bulk bridge.
3. Remove the superseded execution-layer compact forward and inverse entry
   points. Kernel tests call the public seam so they cover the delivered route.

This keeps dependency direction unidirectional: the API layer selects
orchestration and storage policy, while execution kernels remain independent
of the plan cache.

## Rejected alternatives

- Calling `PlanCacheProvider` from `application/execution/kernel` was rejected
  because it reverses the execution-to-orchestration dependency direction.
- Adding another plan cache in the execution layer was rejected because it
  duplicates ownership and adds another contended, growing cache.
- Keeping the dispatch route was rejected because it preserves the measured
  plan-state miss.
- Constructing a new plan per call was rejected because it rebuilds the
  tables that the existing cache is intended to retain.

## Contract and verification

The forward, normalized inverse, and unnormalized inverse compact operations
retain their existing semantics. Lengths 0 and 1 remain no-ops, and the plan
length is validated by the existing plan entry points. The bridge uses its
thread-local reusable scratch pool, so one transform does not allocate a new
temporary buffer on each call.

The API regression test compares compact forward output at lengths 64, 128,
256, and 512 with the same cached `f32` plan after the storage conversion. The
existing impulse and round-trip tests continue to exercise the public compact
entry. The pinned probe supplies the performance evidence; its first complete
pass is discarded after a rebuild and the reported pass is warm.

## Consequences

The public names and signatures remain unchanged. The compact implementation
now shares the plan cache with ordinary `f32` transforms, while the small-size
optimization remains local to the execution kernel. Direct `f32` and `f64`
paths are unchanged.
