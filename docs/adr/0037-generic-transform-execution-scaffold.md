# ADR 0037: Generic transform execution scaffold in apollo-fft

- Status: Accepted
- Revision 2026-08-02: Accepted on delivery of the shared layer with
  apollo-dht and apollo-fwht as the proving adopters (479/479 including
  both GPU verification suites on hardware).
- Date: 2026-08-02
- Refs: atlas backlog `ATLAS-SUBSTRATE-004`; atlas
  [ADR 0039 §4](../../../../docs/adr/0039-compute-substrate-topology.md)

## Context

Nineteen of the twenty-three Apollo crates carry the same
`application/execution/plan/<transform>/` and
`infrastructure/transport/gpu/` scaffold beside their transform
mathematics. Measured against each other the copies are one concern in
nineteen diverging states:

- `infrastructure/transport/gpu/domain/capabilities.rs` is
  **byte-identical** between apollo-dht and apollo-fwht (diff: 0 lines),
  and both already import `apollo_fft::PrecisionProfile` — the shared
  vocabulary already lives in apollo-fft.
- `domain/error.rs` differs only in the transform name embedded in
  message text (diff: 16 lines).
- `infrastructure/device.rs` and `verification.rs` differ by 225 and 181
  lines *after* normalizing the transform name — not because the
  transforms differ, but because the copies have **drifted**: apollo-dht
  carries the modern form (thread-local `ScratchPool` reuse,
  `execute_forward_into` caller-owned storage, Mnemosyne-native Leto
  outputs), while apollo-fwht still allocates per call and round-trips
  typed storage through `f32` vectors.

The drift is the cost of the duplication: improvements land in whichever
crate hosts the motivating work and nowhere else. The kernel layer
itself already went generic in the ADR 0034/atlas ADR 0039 arc — each
crate's `kernel.rs` implements `hephaestus_core::KernelInterface` +
`KernelSource` and is genuinely transform-specific (shader source,
parameter struct, pass sequence). The scaffold *around* it is not.

## Decision

One generic execution scaffold lands in **apollo-fft** (per atlas
ADR 0039 §4's no-new-package consequence, and because apollo-fft already
plays the shared-vocabulary role — 20 of 23 crates depend on it), under
a dedicated `transport` concern tree, and the transform crates adopt it
one at a time:

1. **Shared capability descriptor.** `WgpuCapabilities` moves verbatim
   to apollo-fft; the nineteen byte-identical copies are deleted.
2. **Parameterized error.** One `WgpuError` family carrying the
   transform label as data (populated from the transform marker), so
   messages keep their current text without nineteen enum clones.
3. **Typed plan descriptors.** `WgpuPlan<T>` generic over a zero-sized
   transform marker keeps per-transform type safety (a DHT plan cannot
   feed an FWHT device) with one implementation of the
   `len`/`is_empty`/validation surface.
4. **Generic device orchestration.** A `GpuTransform` trait owned by
   apollo-fft names what actually varies — the kernel types, the
   parameter struct construction, the pass sequence (forward, inverse,
   scale passes), and the storage contract — and one generic device type
   provides what today drifts: scratch-pool buffer reuse, `_into`
   caller-owned-storage execution, typed-storage dispatch, and
   Mnemosyne-native Leto outputs. Adopting crates inherit the modern
   dht-form behavior instead of their drifted copies.
5. **Generic verification.** The per-crate `verification.rs` harness
   becomes one generic module parameterized by the same trait.

The transform crate retains its kernel (`KernelInterface`/`KernelSource`
impls, shader sources, parameter structs) and its mathematical contract
— which is what actually varies — plus one `GpuTransform` impl wiring
them to the shared scaffold.

`mod helpers`/`mod utils` junk drawers in an adopting crate are removed
in that crate's adoption increment (atlas ADR 0039 §5).

## Revision 2026-08-02: CPU-tier storage vocabulary

The GPU-tier `GpuElement`/`GpuStorage` design extends one precision
tier up: `CpuElement` (`f64`, `Complex64`, with per-element scratch
pools and a capacity observer for reuse tests) and
`CpuStorage<E: CpuElement = f64>` live in apollo-fft's domain storage
module, un-gated, so the fourteen per-crate CPU conversion ladders
(`to_f64`/`from_f64`, `to_complex64`/`from_complex64`, private profile
consts, private `f64`/`Complex64` pools) consolidate the same way the
GPU copies did. Transform crates keep their plan-coupled dispatch
traits and bound them on the shared vocabulary (`HartleyStorage:
CpuStorage`, `QftStorage: CpuStorage<Complex64>`); dead reinterpret
views left over from the pre-adoption GPU paths are deleted rather
than migrated. Delivered with dht (real) and qft (complex) as the
proving pair under ATLAS-SUBSTRATE-005.

### Completion 2026-08-02

All fourteen crates share the vocabulary; zero conversion ladders
remain in the workspace. Three shapes emerged. Crates whose storage
traits carry plan-coupled dispatch keep them and gain `CpuStorage` as
a supertrait (dht, qft, sht, and the ten batch migrations). Crates
whose traits were pure conversion vocabulary lose them entirely, with
the bound applied directly at the call sites — sdft's two and stft's
four traits fell here, their apparent multiplicity being one direction
of one element each, which the shared trait already carries together.
apollo-nufft keeps a view-only trait: its `Complex32`/`Complex64`
reinterpret views let the device helpers skip a copy when the host
layout already matches an accelerator element, and the shared
vocabulary deliberately models conversions but no views. Dead views
elsewhere — left over from the pre-adoption GPU paths — are deleted.

## Exemptions

- **apollo-nufft** (revision 2026-08-02): exempted after fifteen
  adoptions proved the patterns. One backend serves two plan families
  (1D uniform domains and 3D uniform grids) across type-1/type-2,
  direct/fast, buffered, and diagnostic variants — under the
  single-marker scaffold that topology forks into two backends and
  rewrites its distinct `NufftWgpuError` vocabulary across ~1,100
  lines for shell-only savings; the parameterization would distort the
  shared layer, which is this document's recorded exemption test.
  apollo-ntt, the originally anticipated exemption, adopts the
  planner/extension form instead: its integer elements never touch the
  executor contract.

## Migration

apollo-dht (the modern template) and apollo-fwht (a drifted copy) adopt
first, proving the layer from both directions. The remaining seventeen
follow as one board item per crate; each adoption must leave the
existing per-transform tests passing untouched — unchanged transform
results are the acceptance oracle. apollo-ntt (integer domain, no
apollo-fft dependency today) adopts last, parameterizing the profile
vocabulary if its integer contract requires it; if the parameterization
would distort the shared layer, ntt keeps a local scaffold and this ADR
records the exemption.

## Alternatives rejected

- **A new apollo-core/apollo-execution crate.** Rejected by atlas
  ADR 0039's no-new-package consequence; apollo-fft already exports the
  shared vocabulary (`PrecisionProfile`), so a second foundation crate
  would split one role across two homes.
- **Leave the copies and align them by review.** The drift measured
  above is what review produced; nineteen copies of one concern cannot
  stay aligned by hand.
- **Macro-generate the scaffold per crate.** Generates nineteen
  compiled copies of what one generic layer expresses; degrades IDE
  support and type errors; forecloses none of the drift.

## Verification plan

- The generic layer carries its own unit tests plus the migrated
  verification harness.
- Each adoption increment: the crate's existing per-transform test
  suite passes untouched; `cargo nextest` for the crate, clippy
  `-D warnings`, doc build; file count falls; no `mod helpers`/`mod
  utils` remains in the adopted crate.
- The full-workspace gate runs at each merge as usual.
