# Rotated 64³ movement attribution

Driver: [APOLLO-ROTATED-MOVE-GEOMETRY-2026-09-10](../../../backlog.md#apollo-rotated-move-geometry).
Layout ownership remains [ADR 0040](../../adr/0040-leto-fft-layout-ownership.md).

## Question and intervention

At 64³, each of the chain's three geometries is a 64 × 4096 transpose.
Each of the rotated inverse's two geometries is 4096 × 64. Each move copies
4 MiB of complex values. The chain round trip makes six moves, the rotated
round trip four. The two rotated forward moves have the wide geometry.

The original provider already chooses blocked traversal using both source
pitch and destination width. Adding another destination-stride comparison
does not address a missing condition. A 64-element lane also does not use
the four-step companion (`generic_four_step_applies` requires length >1024).

The attribution varies only transpose task width while retaining the same
source-pitched copy primitive. With the original 64 KiB byte budget, a tall
move gives each task one destination row, hence one source column. Four
columns per task improve the tall move; eight do not. Raising the byte
budget to 256 KiB for wide moves regresses that control. The production
change therefore retains the byte budget and imposes a minimum of one
detected cache line's worth of source columns, clipped to the matrix width.
For this host and scalar, that minimum is four columns.

Task partition is the measured cause addressed by the change. Source-line
reuse explains the geometry-dependent result, but no hardware counter was
collected to establish the cache-traffic mechanism independently. Unaligned
origins and ragged pitches can still share boundary lines. Neither the
serial tile loop nor Apollo's FFT lane scheduling changes.

## Instrument and execution

The ignored `move_geometry_attribution` native test contains 13 cases:
three chain moves, two inverse moves, two serial controls, tall moves at
256/512 KiB per task, the wide 256 KiB control, production C-order and rotated
round trips, and a probe-only rotated inverse with four-column tasks.
Each move verifies every permuted value. Each pair verifies one fresh input
against the documented floating-point error bound before timing.

Each process executes four blocks, alternating ascending and descending case
order. Each case uses the existing regression instrument: 100 ms warm-up,
400 ms measurement target, 100 samples. The complete test takes approximately
26–27 seconds under the committed Nextest 30/60-second slow/termination budget.
CSV files preserve all cases and all sorted timing samples, in picoseconds.
Their 96.4799% median intervals are pointwise; the comparator recomputes
familywise intervals from the samples.

Environment: Windows x86-64, Intel Core Ultra 9 285K, 24 logical processors,
Rust 1.97.0, `bench-quick` (opt-level 3, LTO off, 16 codegen units). The test
process requests Windows high priority and inherits all-processor affinity.
Other processes and production scheduling policy are unchanged. Compiler
work remains active on this shared host. Reversed case and revision orders
and unchanged controls expose drift; they cannot prove sample independence
or stationarity, or establish performance on another machine.

Chronological revision order is production, baseline, baseline-reverse,
production-reverse. The first pair is candidate-first; the second is
baseline-first. Blocks within a process are not independent revision runs.

Baseline: Apollo `aaa11ddc` plus the instrument, with the original standalone
lock (Leto `c1e069bdcc7d8aa7186d5f28dbcb43552898b3d9`). Candidate: identical
Apollo source and providers except local Leto `c3c5a08` plus the destination-row
partition change. Temporary Cargo patches replace only `leto` and `leto-ops`;
the resulting lock diff removes exactly those two Git source lines. No local
patch or flattened lock is retained in the delivered consumer.

Source SHA-256 at measurement (working-file bytes):

- `pass_attribution.rs`: `BC3AE909F8E9BB0E2C6FD196E03CAFC0B5636FBCD850A1F67200A133FE6F0F19`
- `pass_attribution/move_geometry.rs`: `183D58C5D6F30AC3FC71A6E506F6FAD0C6BF0561DC1ECDFC675F5AE76F1E73CF`
- Leto `layout/complex/batch.rs`: `0AC12EC77D00EB14A6B02511F20085E7264A6272BAEA294D41081622CAECF793`

Run from outside the Atlas overlay, using the shared target directory:

```powershell
$env:CARGO_TARGET_DIR = 'D:\atlas\target'
$env:APOLLO_MEASUREMENT_HIGH_PRIORITY = '1'
cargo +1.97.0 nextest run --manifest-path D:\atlas\repos\apollo\Cargo.toml --locked -p apollo-fft --lib --cargo-profile bench-quick -E 'test(move_geometry_attribution)' --run-ignored all --no-capture
```

Each `MOVE-GEOMETRY block` marker precedes a complete CSV report. Save those
four reports with `python scripts/move_geometry.py <log> <destination>`.
The extractor preserves records, rejects incomplete or mismatched blocks,
and refuses to overwrite different evidence. Regenerating all four original
report directories produces no Git diff.
The local-provider experiment substitutes these two Cargo arguments for
`--locked` (and restores the original standalone lock after measurement):

```text
--config 'patch."https://github.com/ryancinsight/leto.git".leto.path="D:/atlas/repos/leto/crates/leto"'
--config 'patch."https://github.com/ryancinsight/leto.git".leto-ops.path="D:/atlas/repos/leto/crates/leto-ops"'
```

Rejected exploratory runs are not baselines: normal-priority runs showed
order-of-magnitude host contention; restricting this parallel workload to
eight performance cores changed the execution regime and increased task
latency. Serial controls in the retained runs also fluctuate as the calling
thread migrates between core classes. No serial-kernel speedup is claimed.

## Results and limits

The stable candidate-first comparison gives these ranges of block medians:

| Case | Baseline, µs | Production, µs |
|---|---:|---:|
| Tall inverse moves | 41.0–45.1 | 30.3–31.8 |
| Rotated round trip | 546.5–580.7 | 496.8–512.4 |
| C-order round trip | 554.5–582.1 | 554.6–582.1 |

Corresponding-block rotated medians fall 7.6–14.4%. Their pointwise intervals
separate in all four blocks. Across the full 52-case family, the existing
comparator supports both tall-move improvements in every block, and the
rotated-pair improvement in block 1; the wider familywise intervals overlap
for the other rotated blocks. Percentages describe these observations, not
a confidence interval for a universal speedup.

The first comparison flags two serial-control slowdowns under familywise
intervals, despite unchanged serial code. One wide control also has a
pointwise slowdown that does not recur in the reverse order. The reverse
production run gives tall moves around 30–31 µs and rotated pairs 483–498 µs.
The reverse baseline's unchanged C-order control rises to 593–797 µs with
broad intervals; that run is retained as evidence of host noise and excluded
from quantitative speedup claims.

The counterbalanced comparator reports `52 cases across 4 reports; no
supported regression`. The comparator does not detect a supported slowdown
in both observed orders; it does not turn the noisy baseline into a
controlled speedup result.
Run the existing `apollo-bench-compare` binary with:

```text
compare-counterbalanced
--baseline-first-baseline-directory baseline-reverse
--baseline-first-candidate-directory production-reverse
--candidate-first-baseline-directory baseline
--candidate-first-candidate-directory production
```

Independent review accepts the task-width intervention from the stable
comparison, same-process four-column controls, and repeated production
result. No global non-regression or serial-kernel claim follows.

## Correctness evidence

Leto's changed partition passes 945 native tests (360 core, 585 operations),
strict all-target Clippy, the no-default-features check, 31 doctests with one
existing ignored example, and warning-denied operations rustdoc. New cases
cover four scalar types, ragged widths, narrow matrices, and tasks crossing
batch boundaries. Allocation counters observe the calling thread only.
Production worker code and its allocation behavior are unchanged.

The probe's numerical oracle assumes normal-range IEEE round-to-nearest
AVX/FMA arithmetic. A separate 200-bit outward interval computation checks
all 192 FFT32/FFT64 combine coefficients against half-angle roots: complex
coefficient error <3.077u and <4.627u respectively, below the oracle's 8u
bound. It uses integer interval roots and products and exact decoding of
binary64 literals. This is computational evidence, not a machine-checked
proof of the FFT. The checked `twiddle_constants.rs` Git blob is
`aef4eefd5700ecce19f71f18d790a55a4eb5ec8e`. The final instrument changes only
documentation and a scoped Clippy expectation after the recorded runs.

The production provider is [Leto PR 192](https://github.com/ryancinsight/leto/pull/192),
commit `c13b7faa72723c1d3abc5f9b9b0726cce085dca0`. The consumer lock advances
only its two Leto entries; unrelated transitive updates from `cargo update`
are excluded to preserve the measured provider graph. Standalone `--locked`
native verification checks that exact graph.

Consumer gates on that graph pass: `cargo nextest run --locked -p apollo-fft
--all-features --profile ci` (629 passed, 43 ignored probes/environmental
cases), `cargo clippy --locked -p apollo-fft --all-targets --all-features --
-D warnings`, `cargo test --locked -p apollo-fft --all-features --doc`
(two compile-fail doctests), and warning-denied `cargo doc --locked
-p apollo-fft --all-features --no-deps`. The new ignored probe runs separately
under `bench-quick`, rather than being counted as covered by the ordinary
suite. The Windows priority FFI is exercised by those runs; it is not
Miri-covered. No production unsafe operation changes.

Final integration at `5c725de7` includes current main (`5281d60b`) and the Git
provider. Its separate `integrated/` reports confirm tall moves at
30.0–31.5 µs, rotated pairs at 482.5–509.8 µs, and C-order pairs at
554.6–570.2 µs. The optimized probe passes in 26.254 seconds; all 629 native
tests pass again (43 skipped), and strict all-target/all-feature Clippy
passes. These confirmation reports are not added to the original comparison
family. Main's intervening row-order change affects the 256-point base,
not this probe's length-32/64 lanes.
