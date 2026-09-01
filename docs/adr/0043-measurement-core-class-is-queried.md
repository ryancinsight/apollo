# 0043 — Measurement core class is queried, not assumed

- Status: Accepted (2026-09-01)
- Item: `ATLAS-APOLLO-CORE-CLASS-LABELS-2026-09-01`
- Driving evidence: `GetLogicalProcessorInformationEx` census (this change);
  per-processor calibration timing (this change); the three-engine table in
  [0041](0041-l1-resident-interleaved-base.md) read against its own labels
- **Revision 2026-09-01: the interim query is discharged.** This record
  originally carried a hand-rolled `GetLogicalProcessorInformationEx` query in
  `kernel::core_class`, declared interim and to be *deleted, not adapted*, once
  `themis-topology` exposed core efficiency class. It has: themis 0.10.1
  reports `CpuTopology::efficiency_classes` with the same absence discipline,
  so `core_class` — its Windows FFI, its non-Windows arm, and its local class
  enum — is deleted outright and the probes read themis. The decision below is
  unchanged; only its provider moved. Tracked as
  `ATLAS-APOLLO-CORE-CLASS-UPSTREAM-2026-09-01`. The selection reproduces
  cpu 1 / cpu 3 and [0042](0042-avx-stockham-backend-retained.md)'s table.

## Context

Every pinned probe in `apollo-fft` selected its two measurement processors from
a hardcoded literal and labelled the result from a hardcoded range:

```rust
// Logical 0..8 are P-cores and 8..24 E-cores on the Core Ultra 9 285K.
for cpu in [2u32, 12] {
    ...
    let core = if landed < 8 { "P" } else { "E" };
```

The comment is false on the host it names. Windows reports per-core
`EfficiencyClass` through `GetLogicalProcessorInformationEx`
(`RelationProcessorCore`); on this Core Ultra 9 285K the performance set is
`{0, 1, 10, 11, 12, 13, 22, 23}` — mask `0xc03c03` — not a contiguous `0..8`.
The layout is interleaved, so the two processors the probes chose are both
mislabelled: **cpu 2 is an efficiency core and cpu 12 is a performance core.**
Every table these instruments produced carried inverted column headers, and
every conclusion drawn from an asymmetric row was inverted with it.

Three independent corroborations were already sitting in the repository's own
records, unremarked because the labels were trusted:

- [0041](0041-l1-resident-interleaved-base.md)'s four-engine table has RustFFT
  (84.7 vs 181.8 ns), PhastFT (149.0 vs 330.0) and Apollo's own base (146.4 vs
  294.5) *all* running ~2x faster on the row labelled `E`. Three unrelated
  engines do not run twice as fast on an efficiency core.
- [0039](0039-one-dimensional-power-of-two-routing.md)'s `core_matrix` table
  has the "E-core" four-step (13.2 us) beating the "P-core" one (16.6 us).
- `gap_audit.md`'s 2048/8192 tables have every `(E)` row absolutely faster
  than its `(P)` counterpart.

An independent timing sweep taken for this change (fixed ILP-bound FP kernel,
best-of-7, two passes, 24 processors) partitions the host exactly as the mask
does: `{0, 1, 10, 11, 12, 13, 22, 23}` at 35.1–36.9 ms and every other
processor at 38.3–38.9 ms. The partition is reproducible and admits no
misclassification; the *magnitude* is workload-dependent and small on a
register-resident kernel, which is why a class query rather than a timing
heuristic is the authority for the label.

The root cause is not the wrong literal. It is that the instrument asserted a
property of the host it never measured, in a comment that could not be checked
by anything.

## Decision

Measurement processors are selected by `core_class::selected()`, and nothing
about core class is assumed:

- **Class is queried, and themis owns the query.** `CpuTopology::detect()`
  supplies each processor's `EfficiencyClass`. The value is a dense ordinal,
  not a flag: the highest rank is the performance tier and rank 0 the most
  efficient, so a host reporting one class is *uniform* — a reported result,
  labelled as such — and the probes find no second arm rather than inventing
  one. `efficiency_class_count()` is the absence oracle: `None` is "the
  platform did not say", which is distinct from a homogeneous `Some(1)`.
  Apollo keeps only what themis does not own: which processor represents each
  class, and how that choice is printed beside the numbers it produced.
- **The representative of each class is stated as a rule**, not a literal: the
  second processor of each class in index order. Skipping the first avoids
  processor 0, the conventional Windows interrupt and DPC target, and applies
  the same rule to both arms rather than special-casing one host's indices.
- **The axis is printed with the table.** Every probe emits the full
  per-processor class census, marking the processors it selected, before its
  own output. The label a reader sees is produced by the same run that produced
  the numbers, so a table can no longer mislabel its own axis.
- **Labels are spelled out** (`performance` / `efficiency`) rather than `P`/`E`,
  which are short enough to be transcribed into a document without their
  meaning.
- **A host with no class information is not measured.** The probes print that
  and return, rather than emitting a two-column table with invented headers.

The class query itself belongs to themis, which owns `CpuTopology` — NUMA
nodes and cache levels — and already calls `GetLogicalProcessorInformationEx`
for cache relationships, with the discipline this dimension needs
(`cache_levels()` returns typed absence rather than a machine-independent
guess). Hermes is not the home: it owns exact processor *binding* and
explicitly disclaims topology (`hermes/docs/adr/021-exact-processor-binding.md`:
"Processor selection remains an Apollo measurement-policy input; Hermes does
not choose a core class"), which is the line the split below follows.

## Consequences

- Every pinned-probe table produced before this change has inverted column
  headers wherever it reports two core classes, and single-core probes
  labelled "P-core" were measured on an efficiency core. [0042](0042-avx-stockham-backend-retained.md)
  is corrected against a re-run in this change; [0039](0039-one-dimensional-power-of-two-routing.md)
  and [0041](0041-l1-resident-interleaved-base.md) carry revision notes, and
  the surviving inventory is tracked as
  `ATLAS-APOLLO-INVERTED-CORE-CLAIMS-2026-09-01`.
- Conclusions of the form "X wins on **both** core types" are unaffected: they
  are label-independent by construction. The `ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD`
  = 256 reroute and the f64 N = 256/512 scalar reroute are both of this form
  and stand.
- Probe output is longer by the census block.
- The premise that cpu 2 is *anomalously* slow — slower than its class
  explains — is not supported. A calibration sweep taken while developing this
  change put cpu 2 at 1 936 000 ns against a 1 924 200–1 936 200 ns spread
  across the sixteen efficiency cores: it is an ordinary member of its class,
  0.6% off the class best. Its numbers were wrong because they were labelled
  performance, not because the processor is defective. The selection rule
  skips it for the reason it skips processor 0, not for a measured defect.

## Alternatives rejected

- **Fix the literal** (`[12, 2]` with the labels swapped): restores the correct
  answer for one host while leaving the instrument's authority a comment, which
  is the defect. It also silently breaks on any other machine.
- **Classify by timing**: an independent sweep's partition matched the mask
  exactly here, but its margin was 4.5% on a register-resident kernel and is
  workload- and load-dependent. Timing can corroborate a class; it is not sound
  as the source of one, and a per-processor calibration pass in every probe is
  more machinery than a correct instrument needs.
- **Query `GetSystemCpuSetInformation`**: supplies the same `EfficiencyClass`
  with more surface, and no per-CPU-set attribute this instrument needs.
- **Put the query in hermes**: contradicts hermes ADR 021 and its own module
  boundary ("Topology discovery belongs to themis").
- **Build a general topology abstraction in apollo**: duplicates the API themis
  is growing and would have to be deleted twice over. The interim module stays
  small enough to delete outright.
