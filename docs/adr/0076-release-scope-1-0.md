# ADR 0076: Apollo release scope for 1.0

## Status

Accepted (2026-11-12)

## Context

Apollo (workspace crates 0.17.2, `apollo-fft` and sibling transform crates)
is the Atlas stack's FFT and integral-transform provider over Eunomia, Leto,
Hermes, Hephaestus, Mnemosyne, and Moirai. Open work remains across native
admission gates, the reference scoreboard, memory discipline, capability
coverage, and source-size ratchets, and no milestone ladder exists to 1.0.

## Decision

Adopt a three-milestone ladder to 1.0 recorded in `backlog/milestones.md`:

- **M-fft-core** (0.18.0): native admission gates, rescue-port hygiene, memory
  discipline, and source-size ratchets verify on the provider cohort.
- **M-transforms-capability** (0.19.0): capability coverage, perf-gap leaves,
  and the reference scoreboard close against the capability audit oracles.
- **M-1.0** (1.0.0): stable transform release (authority-gated) with
  documented contracts and release gates satisfied.

In scope: FFT/transform kernels, capability slices, memory discipline,
perf-gap leaves, book chapters. Out of scope: AVX-512 bring-up (no hardware),
branch-ref cleanup owned by tooling, instrumented-budget work blocked on host
runtimes; any `publish = true` without release authority (veto point: M-1.0
never forces publication).

## Alternatives

- Plan 1.0 as a single release: rejected because memory and capability
  slices remain open across many crates.
- Keep roadmap implicit in backlog items: rejected because stack policy
  requires a runnable capability-oriented milestone ladder.

## Consequences

- Open items carry `milestone:` assignments and name the exit check they serve.
- Milestones close only when exit checks pass on fetched default commits; 1.0
  release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.
