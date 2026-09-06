# ADR 0048: Worker scratch lifetime across submissions

- **Status:** Accepted
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

## Ownership

Keep Apollo's worker-idle release contract. Borrow lane scratch from storage
owned by the enclosing transform operation, with Mnemosyne supplying the
allocation and reusable storage. The coordinator retains ownership while
Moirai executes disjoint lane partitions; workers return borrows before the
synchronous submission completes. Leto continues to own layout movement.

Reuse the inactive full-volume transpose companion. After gathering an axis,
the original input buffer is dead until the final transpose; it supplies lane
scratch while the gathered buffer holds the active values. A contiguous-axis
pass borrows the same caller-thread role used by a later transpose. Rank-two
and rank-three logical-view staging use other roles, preserving disjointness.

For lane length N and complete recursive workspace S, group G = ceil(S/N)
lanes. Moirai's existing paired-chunk operation lends disjoint G*N-element
groups from the active and inactive buffers. Each task reuses its scratch
group across its lanes. An incomplete final group executes after the join,
when it can borrow S elements of the full inactive buffer. Total companion
storage remains one volume; the design adds no retained allocation. A
degenerate volume smaller than S keeps the existing coordinator-local route.

The private FourStep body accepts borrowed scratch and validates its complete
recursive requirement before mutation. Its existing standalone route acquires
thread-local scratch at the boundary. Scalar, direction and normalization
remain monomorphized, with no duplicated FFT bodies or new public API. Safe
slice borrows express this non-escaping lifetime without a new GAT interface.

This increment covers the generic FourStep route beyond the existing sized
plan entries through 1024. Smaller scalar-specific kernels retain their
dispatch policy and scratch ownership. The expected binding constraint is
allocation/reclamation overhead between short independent submissions;
the numerical operation count and transpose traffic do not change. Grouping
can reduce available parallelism (4096×16 has eight groups instead of sixteen
lanes), so timing of that case is a rejection oracle, not an assumed win.

Cold-worker regression tests also expose two 116-byte allocations for the
planar table handle maps on each newly participating worker. Replace those
maps with two inline entries, one last length per direction. The global maps
still own every immutable table, so replacement cannot invalidate a running
transform's Arc. Retained handle metadata is fixed per worker. Alternating
lengths use the shared-map lookup instead of the local hit; measured timing
claims must identify the workload and cannot establish mixed-size throughput.

The same probe exposes allocation by test-only per-section cycle counters.
Their six existing labels bound the counter storage statically. Only the
reporting operation allocates a result vector; recording no longer contaminates
the production allocation measurement. Test inputs and allocation assertions
remain unchanged.

The cold-memory oracle rejects the first candidate: allocating scratch before
constructing the interleaved twiddle table overlaps the workspace with a full
temporary vector and its Arc copy. At 262,144 points, peak allocation rises
from 8,396,816 to 11,542,544 bytes. Stream the same row-major coefficients into
the final Arc, or unzip directly into the final planar vectors. This preserves
the existing coefficient evaluation and removes the intermediate matrix.

## Alternatives and rejection criteria

- Permanently provision worker TLS: restores retained worker memory and defeats
  the existing release contract.
- Increase Moirai's idle spin period: changes scheduling and delays release;
  it cannot guarantee a storage lifetime across arbitrary caller gaps.
- Allocate each submission: preserves release but does not close allocation churn.
- Add a public workspace argument immediately: may be necessary if the existing
  owner cannot express the lifetime, but requires a separately classified API
  change and complete caller migration. Do not assume that break is required.
- Add a separate partition workspace: retains P*S additional elements while
  the inactive transpose buffer already has sufficient capacity.

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
gates. These oracles establish the ownership contract; instrumentation limits
remain explicit below.

## Measured evidence and limits

The unchanged `engine_census` workload compares baseline `11017e9b` with this
item's implementation on Rust 1.97.0, Windows MSVC. Raw census, allocation,
host-load and comparison artifacts live under Atlas's retained output root,
`output/apollo-workspace-lifetime/`. These are allocator-window measurements,
not resident-set measurements; caller input buffers sit outside the window.

| Workload | Baseline allocations / bytes per warm call | Candidate |
| --- | ---: | ---: |
| 4096×16 | 15 / 1,105,920 | 0 / 0 |
| 4096×64 | 24 / 1,769,472 | 0 / 0 |
| 16384×16 | 15 / 4,177,920 | 0 / 0 |
| 65536×4 | 3 / 3,145,728 | 0 / 0 |
| 4096×4×4 | 12 / 884,736 | 0 / 0 |

At 262,144 points the final cold peak is 7,348,240 bytes, with unchanged
retained bytes at that length. Warm complex and real-half execution allocate
zero bytes at every census length. The census executable shrinks from
8,710,144 to 8,637,440 bytes. Eight counterbalanced timing runs cover 39 cases;
the comparison detects no supported regression, but variable worker timings
do not support a throughput improvement claim. Mixed-length table-handle
replacement is outside that timing claim.

Event-synchronized worker tests independently verify the analytical transform
values, zero allocation on subsequent submissions and zero idle scratch
capacity. Exact-size workspace tests cover both supported scalars, both
directions and inverse normalization; a sentinel detects writes beyond the
declared capacity. Undersized-workspace rejection preserves every input byte,
including NaN and signed-zero representations.

Independent review finds no ownership, partition, normalization or table-order
defect. Native tests are behavioral evidence, not a proof of absence of
undefined behavior. The Miri boundary test exceeds the unchanged 60-second
bound; Windows AddressSanitizer fails before execution because
`clang_rt.asan_dynamic_runtime_thunk-x86_64.lib` is unavailable. The
[instrumented verification item](../../backlog.md#apollo-workspace-instrumented-verification)
tracks this remaining coverage.

The final workspace Nextest run `c861fbcf-b915-419c-ae2a-ec9378f6bfc3`
passes 1,441 tests (31 skipped); release FFT run
`909f2b30-614b-4874-a12f-9ec2b07c7469` passes 545 (30 skipped).
All seven benchmark smoke targets pass within the committed bounds, and 196
semver checks find no public-contract change. The default dependency-policy
gate passes. An additional all-feature audit rejects the pre-existing
`cuda-oxide` license through Hephaestus; the
[CUDA provider item](../../backlog.md#apollo-cuda-crt-linkage) owns its removal.

**2026-09-05 revision:** accept operation-owned companion storage after the
allocation, analytical and independent-review oracles pass; retain the stated
limits on timing, mixed-length throughput and instrumented unsafe coverage.
