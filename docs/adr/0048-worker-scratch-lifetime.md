# ADR 0048: Worker scratch lifetime across submissions

- **Status:** Proposed
- **Date:** 2026-09-05
- **Class:** [patch] [arch]
- **Item:** [APOLLO-WORKER-WORKSPACE-LIFETIME](../../backlog.md#apollo-worker-workspace-lifetime)

## Problem

The census invokes a warm-up transform and a separate measured transform.
Multidimensional axis passes submit lane operations to Moirai. A worker can
become idle between submissions, at which point Apollo's registered hook
releases its Mnemosyne scratch banks. Unprovisioned banks release their complete
allocation. The next lane operation reacquires that storage.

At the merged baseline `f69f9c08`, a 4096-point planar lane requires
`64 * (64 + 8) * size_of::<Complex<f64>>() = 73,728` bytes. The census's
multidimensional allocation totals are integer multiples of this amount.
At 16384 points the corresponding amount is 278,528 bytes. This is source and
allocation-signature evidence; it does not attribute latency to individual
allocations.

The existing worker-retention probe checks two transforms inside one live task,
then observes release at worker quiescence. It establishes neither retention
nor allocation-free execution across independent submissions. Both zero idle
worker capacity and reuse across calls require a storage owner outside the
worker's temporary task lifetime.

## Recommended ownership

Keep Apollo's worker-idle release contract. Borrow lane scratch from storage
owned by the enclosing transform operation, with Mnemosyne supplying the
allocation and reusable storage. The coordinator retains ownership while
Moirai executes disjoint lane partitions; workers return borrows before the
synchronous submission completes. Leto continues to own layout movement.

Prefer the existing caller-thread plan-scratch roles as the storage owner
before adding a public workspace API. Determine the required capacity from
the transform decomposition and bounded partition count. For P concurrent
partitions and S scratch elements per partition, lane scratch is bounded by
`P * S * size_of::<Complex<T>>()`, in addition to existing transpose storage.
Check this bound against current peak and retained bytes before selecting P.

Thread scratch provision through one private generic execution path that can
borrow operation storage or use existing thread-local storage. Scalar,
direction, normalization and algorithm remain independent of storage ownership.
Do not duplicate FFT bodies to introduce the borrowed storage path.

## Alternatives and rejection criteria

- Permanently provision worker TLS: restores retained worker memory and defeats
  the existing release contract.
- Increase Moirai's idle spin period: changes scheduling and delays release;
  it cannot guarantee a storage lifetime across arbitrary caller gaps.
- Allocate each submission: preserves release but does not close allocation churn.
- Add a public workspace argument immediately: may be necessary if the existing
  owner cannot express the lifetime, but requires a separately classified API
  change and complete caller migration. Do not assume that break is required.

Reject a candidate that retains scratch on idle workers, weakens transform
oracles, introduces shared mutable lane storage, or increases measured peak
memory without a justified workload-level benefit. Nested transforms require
distinct active scratch roles; reentrancy must not alias an outstanding borrow.

## Verification

First extend the existing retained-footprint observer to distinguish two
submissions separated by condition-variable-confirmed idle hooks. Exercise
4096×16 and 4096×4×4 transforms with analytical spectra and normalized inverse
values. Record allocation bytes and post-idle capacity; use no sleeps or timing
assertions. This supplies the baseline before changing ownership.

Then compare the same cases using operation-owned scratch. Require disjoint
mutable storage, unchanged scalar/direction coverage, zero worker capacity
after idle, bounded caller retention and no allocation after caller warm-up.
Run the committed Nextest, reference census, peak-memory and replicated timing
gates. The design remains Proposed until those oracles establish the contract.
