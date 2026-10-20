# ADR 0077: Hephaestus 0.27.1 public-provider compatibility line

- Status: Proposed
- Date: 2026-10-20
- Item: APOLLO-HEPHAESTUS-0271-COMPATIBILITY [major]

## Context and contract

The accepted admission diagnostics retain old package versions and path-source
locks. They are not canonical release candidates. Apollo's old ^0.22.2
requirements reject the repaired 0.27.1 provider. FFT's public WGPU/CUDA
constructors, GpuTransformExecutor and WgpuError expose provider device/error
identities; sibling public GPU plans and kernel implementations share that
boundary. Unchanged Rust text does not make those nominal types compatible.

## Proposed decision and private preparation

Require ^0.27.1 (>=0.27.1,<0.28.0) for core/WGPU/CUDA without changing features
or source URLs. Use FFT 0.30 and the corresponding new pre-1.0 transform lines.
Update internal path+version requirements atomically. Keep apollo-alloc,
apollo-dctdst-core, apollo-fft-macros, apollo-leto-interop, apollo-bench and the
workspace version unchanged: their package/API content is outside this delta.
Validation's public reports and Box<dyn Error> boundary do not expose concrete
provider types; its changed dependencies require compatible patch 0.5.3 and
an actual patch API comparison. The private Python adapter keeps its own
lineage (0.3.0 -> 0.3.1); its Python operations, ABI floor,
PyO3 selection and wheel configuration remain unchanged by this migration.

| Package | Base | Prepared |
| --- | --- | --- |
| apollo-czt | 0.12.1 | 0.13.0 |
| apollo-dctdst | 0.12.1 | 0.13.0 |
| apollo-dht | 0.12.1 | 0.13.0 |
| apollo-fft | 0.29.3 | 0.30.0 |
| apollo-frft | 0.11.1 | 0.12.0 |
| apollo-fwht | 0.11.1 | 0.12.0 |
| apollo-gft | 0.12.1 | 0.13.0 |
| apollo-hilbert | 0.14.1 | 0.15.0 |
| apollo-mellin | 0.13.2 | 0.14.0 |
| apollo-ntt | 0.11.1 | 0.12.0 |
| apollo-nufft | 0.10.1 | 0.11.0 |
| apollo-python | 0.3.0 | 0.3.1 |
| apollo-qft | 0.11.2 | 0.12.0 |
| apollo-radon | 0.12.2 | 0.13.0 |
| apollo-sdft | 0.10.2 | 0.11.0 |
| apollo-sft | 0.11.2 | 0.12.0 |
| apollo-sht | 0.9.1 | 0.10.0 |
| apollo-stft | 0.11.1 | 0.12.0 |
| apollo-validation | 0.5.2 | 0.5.3 |
| apollo-wavelet | 0.12.1 | 0.13.0 |

The supplied canonical-preparation request authorizes this private proposal;
it does not supply independent ADR sign-off. Owner source convergence,
independent exact-head review and ADR sign-off remain gates before public
integration. No implementation checklist is opened as an approved release.
Current-main and accepted-FFI snapshots remain distinct alternatives, with
one future owner-selected release source. Never publish both under the same
Rust versions. This candidate is the separate accepted-FFI integration preparation, not a second release source.

## Migration and source identity

Direct Rust consumers advance Apollo transform and Hephaestus requirements
together, construct devices/errors from one Hephaestus 0.27.1 source, and
rebuild their accelerator callers. Coeus consumes FFT ^0.30.0 from the
owner-selected Apollo candidate. Do not translate duplicate provider types
through adapters or retain old requirements to obtain green resolution.

Manifests keep canonical unpinned Git sources. The renewed review lock records the real
provider commit a735c608dd686bc665ef48ac329aca4872b8508f (tree cce4ff3202a4b15405d928cc9d29a0ec4db224ff).
The previous ec650afef687e16f0563f351109a2c6ace22afe3 source and its accepted
ec650 consumer receipts remain immutable historical evidence. Hephaestus #509 is
draft, unmerged and unpublished: that exact-source lock is private source
evidence, not default-branch or registry availability. Provider-first merge
and publication, legitimate normalized package graphs and fresh no-overlay
Git/registry consumers remain separate gates. Cargo normalization uses
registry dependencies, whose source identities differ from Git packages.

## Alternatives, failures and verification

Reject lower floors, broadened incompatible ranges, manifest revision pins,
path overlays, fabricated registry receipts and renamed compatibility APIs.
Preserve all Rust, tests, fixtures, skips, profiles, workflows, numerical
bounds and performance/allocation obligations. Reproduce original refusal,
reject real 0.27.0 and admit exact 0.27.1. Verify one source identity for every
type-bearing provider; run native CPU, strict lint/docs/doctests, MSRV 1.95,
before/after API and standard-package checks on the selected exact head.

The successor normal-tail source/model correction has separate scoped review;
consumer admission does not certify its portable numerical contract. The
unchanged toward-zero handoff witnesses (367.359/328.240 ULP), six portable
countermodels, performance, physical hardware, final consumer/backend/Python
and broader cohort, publication and service holds remain. FFI runtime/wheels/common-FFI
are separate gates; compilation does not prove them. Coeus Miri remains
deferred and unverified. The initial service-blocked Apollo LaneBuffer task
is not retried. No push, PR edit, dispatch, device acquisition, merge,
publication, credential/settings or memory change occurs in this pass.

## Successor provider renewal — 2026-10-20

The separately authorized renewal selects the exact a735 provider above while
retaining the 0.27.1 floor, every package version, feature and consumer source.
The coordinator verifies the provider receipt
`a67101b6813ee27717cf0e43b1802ddf1ddbbe973afe7f828c7964c060d0c869`
(917 records) and actual source identity; no provider test is duplicated.
Consumer Leto/LetoOps stay at 0.46.3, Git 5051cd9577650a5712aa6a7effcd8955c9d8bcc7.
The provider's own source lock retains 0.46.1/7fd6256f; that is a distinct
provider-check graph and does not justify downgrading this consumer graph.

The four accepted ec650 preparations and their independent review remain
bound to ec650. New exact consumer/source/native/package/API evidence and
independent review are required for this successor. Status remains Proposed;
no owner source adoption, ADR sign-off or public lifecycle permission is
inferred. Both Apollo source alternatives remain private until one canonical
owner source is explicitly recorded. See the co-located renewal record.
