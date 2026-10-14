# ADR 0074: Sized fallback table ownership

Status: Accepted

The precise sized policy already selects four-step at the measured crossover.
On a host lacking its native base, the plan still classified that length as
power-of-two but executed through a transient four-step state. Batched caches
hold weak indexes by design; dropping the transient owner destroyed its tables
and made every warmed execution allocate again. The 1024 real-half case traced
29 allocations, and the 256 QFT round trip counted 14.

Bind that scalar policy in the existing private sealed seam and retain its
four-step owner at sized lengths only when a native base is absent. Preserve
the power-of-two strategy classification, thresholds, kernels, normalization,
and native base selection. Prepared executors borrow the retained state; static
plans continue to acquire the existing bounded dynamic owner. Do not strengthen
global weak indexes or add an unbounded table cache.

Validate native admission rather than assuming AVX on every host. A decline
preserves operands; the public fallback runs the original direct-value and
bitwise-layout workloads at the original bounds. Original allocation assertions
remain unchanged. A dedicated native ARM CI job protects these contracts.

The compatible Hephaestus PR #465 source supplies conservative dyld absence
classification. Only its three lock identities advance; Cargo's full locked
resolver verifies the graph, with every other provider pin preserved.
