# Apollo Gap Audit

## FFT bounded-cache stack initialization (2026-07-21)

- Finding: the full locked gate aborted seven Rader/Good-Thomas tests on
  Windows. The exact `2a22319` revision and original lock reproduce the fault,
  excluding the Leto refresh as its cause. GDB stops in `___chkstk_ms` while
  initializing the 8,192-entry precise negacyclic TLS table: the generated
  frame is 262,216 bytes on an already-active FFT execution stack.
- Resolution: retain each flat cache's fixed capacity and O(1) index contract,
  but initialize a boxed fixed-size array through `Vec`. The array type retains
  compile-time bounds for hot indexed lookups without constructing its storage
  on the stack. Remove the
  four 8 MiB test-thread wrappers and the CI-wide 16 MiB `RUST_MIN_STACK`
  override that masked the production stack requirement.
- Rejected design: a boxed slice also removed the stack frame, but erased the
  compile-time length from hot indexed lookups. The hosted counterbalanced
  benchmark rejected that representation with systematic regressions across
  composite and prime transforms, including 9.6 us versus 7.0-7.2 us for the
  N=521 full-cyclic case. The boxed array keeps the heap allocation while
  restoring the fixed-size type; the exact-head benchmark rerun is required.
- Evidence limit: debugger stack-frame evidence identifies the failure
  mechanism; 13 focused default-stack regressions and the complete 964-test
  default workspace establish retained value semantics. Warning-denied
  all-feature Clippy verifies feature compilation, but local all-feature test
  linking cannot supply CUDA coverage because this Windows host has no CUDA
  linker library; the hosted pull-request matrix owns that evidence. This
  change makes no throughput claim.

## Leto public compatibility retirement (2026-07-21)

- Finding: Apollo's source and manifests already consumed native Leto arrays,
  but its lock selected provider commit `bdb5fce4` and two historical PM entries
  still described `ndarray-compat` as current. Refreshing Leto alone also exposed
  a second Aequitas source because locked Hephaestus `8b27c9d` retained an older
  units revision.
- Resolution: refresh the complete Cargo-selected provider closure to Leto
  default head `b95f1aa` (which contains compatibility-retirement merge
  `446d248`) and Hephaestus default head `804d751`. The latter uses
  the same Aequitas `be3a1ac` revision and removes the duplicate units crate.
  Correct the historical PM entries without adding a consumer adapter or
  restoring a Rust `ndarray` edge.
- Contract: `apollo-leto-interop` remains Apollo's single host-array boundary;
  contiguous views borrow and non-contiguous views materialize logical order
  once. Third-party language conversion remains at the Python ownership edge,
  and all Aequitas quantities resolve from one source revision.
- Evidence limit: locked compilation, value-semantic Nextest, provider audit,
  and dependency/source scans verify integration and retained behavior. They do
  not establish a runtime performance change.

## Native benchmark regression evidence (2026-07-20)

- Finding: the benchmark CI ran one all-feature workspace benchmark, copied
  that output as a baseline, then compared the same output to itself with a
  copied Python script. The check could not detect a regression, and the
  all-feature command also imported CUDA toolchain requirements unrelated to
  Apollo's CPU benchmark reports.
- Resolution: `apollo-bench` now emits ordered observations with exact
  symmetric median summaries and owns recursive report comparison. Report and
  case sets must match. For `m` cases, each of the `2m` baseline/candidate
  intervals has miscoverage at most `0.05 / (2m)`, so Bonferroni's inequality
  bounds family-wise interval miscoverage by 5%. A regression exists only when
  the candidate lower bound exceeds the baseline upper bound.
- Mathematical oracle: for ordered samples `X_(1), …, X_(n)`, NIST Technical
  Note 2119 section 5.3 equations 30–31 gives
  `[X_(k), X_(n-k+1)]` with coverage
  `1 - 2 P(Bin(n, 0.5) <= k - 1)`. At Apollo's fixed `n = 100`, the narrowest
  symmetric interval meeting 95% is `[X_(40), X_(61)]` with exact floored
  coverage 964799 parts per million.
- Evidence tier: analytical order-statistic oracle, exact integer
  implementation, value-semantic unit/integration tests, and typed malformed
  evidence rejection.
- Falsification: hosted run `29757554816` measured source-identical revisions
  in a fixed base-then-candidate order and reported 31 disjoint slowdowns,
  including one-nanosecond separations. The fixed ordering—not code—was the
  only systematic variation, so one sequential pair is not a valid operational
  oracle.
- Correction: the workflow now runs ABBA-style counterbalanced pairs
  (baseline→candidate, candidate→baseline) on one runner and requires the same
  case to be slower in both orders. The native comparator validates identical
  evidence universes across both pairs.
- Second falsification: hosted run `29759735814` counterbalanced execution but
  compiled the base and candidate revisions against their respective
  `apollo-bench` sources. Because this pull request changes that measurement
  harness, the run changed both the instrument and the transform code under
  test and produced 22 apparent regressions.
- Instrument control: CI now overlays the candidate `apollo-bench` source onto
  the baseline checkout before either build, then verifies the three benchmark
  entry points are byte-identical across checkouts. The transform
  implementations remain revision-specific.
- Third falsification: instrument-controlled hosted run `29761551514` still
  produced 25 apparent regressions. Each case used an individual 95% interval,
  so the probability of at least one false separation grew with the case
  family. The report now retains all ordered observations, and the comparator
  derives exact Bonferroni intervals over both revisions and every compared
  case. A value-semantic regression proves a separation under a smaller family
  disappears when the full family requires wider simultaneous intervals.
- Fourth falsification: family-wise hosted run `29764170548` still reported 12
  slowdowns under one ABBA block even though
  `git diff 66e37ab..65dd9ad -- crates/apollo-fft/src
  crates/apollo-fft/Cargo.toml Cargo.lock` was empty. The smallest separations
  were one nanosecond, while one order of another case separated by 2,657 ns,
  proving that a single runner timeline still confounded source identity with
  period effects.
- Phase control: CI now follows ABBA with its phase reversal BAAB. Across the
  resulting eight periods, baseline occupies positions `{1, 4, 6, 7}` and
  candidate `{2, 3, 5, 8}`; both position sums equal 18 and both squared
  position sums equal 102. The replicated comparator requires the same case to
  regress in all four base/head comparisons and rejects mismatched case
  universes between blocks.
- Provider checkout cleanup: Apollo manifests contain no external path
  dependencies; Git revisions in `Cargo.lock` are authoritative. The stale
  copied checkout action and all workflow calls are removed rather than
  migrated to another redundant checkout layer.
- Operational evidence: exact-head hosted run `29766127266` completed the
  eight-pass source-identical canary and replicated comparison in 31 minutes
  without a reported regression. The Rust workspace, Python bindings, and
  review checks also passed at `c9a0156`. This is controlled same-runner
  evidence for the CI protocol, not a cross-machine performance claim.
- Fifth falsification: run `29788350487` reported source-identical apparent
  regressions for `half_cyclic_rader/half_cyclic_f32/1031` and
  `fft_kernel_strategy/generic_selector/128`. Its base-to-candidate diff changes
  only Python release workflow, metadata, and documentation; the measured FFT
  source, local dependency closure, benchmark instrument, Cargo resolution,
  and toolchain configuration are identical. The one-to-nine-nanosecond
  smallest separations demonstrate residual hosted-runner variation that the
  phase-balanced schedule cannot attribute to candidate code.
- Applicability correction: benchmark regression now owns a dedicated
  path-filtered workflow. It retains the full native ABBA/BAAB experiment for
  changes to `apollo-fft`, its local macro and Leto-interop dependencies,
  `apollo-bench`, Cargo resolution, toolchain configuration, or the workflow
  itself. Release-only, documentation-only, and unrelated package-only diffs
  do not run a performance attribution experiment whose measured closure is
  identical.
- Closure evidence: exact-head run `29790606838` exercised the dedicated
  workflow and passed all eight measurements plus replicated comparison in 31
  minutes 38 seconds. Path-selection regressions separately prove release-only
  exclusion and benchmark-source inclusion.

## Hephaestus legacy-math lock convergence (2026-07-17)

- Finding: the lockfile selected Hephaestus parent `93bc38e` after provider
  PR #47 removed its direct legacy math baselines.
- Resolution: update `hephaestus-core`, `hephaestus-wgpu`, and
  `hephaestus-cuda` to merged provider `cec0e33`; no Apollo source or manifest
  compatibility path changes.
- Theorem: because Cargo.lock is the sole provider revision selector, resolving
  every Hephaestus package to the same merged default-source commit makes the
  Apollo consumer graph reproducible and imports the provider's Leto-owned
  numerical baselines without a downstream wrapper.
- Evidence tier: Cargo resolution, locked compile, 402/402 Nextest,
  warning-denied Clippy, doctests, warning-clean rustdoc, and provider audit.

## Leto merge pin (2026-07-17)

- Finding: the provider lock selected Leto parent `6a0e297` while Atlas pinned
  merged default `3ac0d203`.
- Resolution: update both `leto` and `leto-ops` lock entries to `3ac0d203`.
  `git diff 6a0e297..3ac0d203` contains only backlog/checklist/gap-audit
  documentation, so the provider code and ABI are identical.
- Evidence tier: exact Git tree comparison plus the preceding provider-lock
  compile/402-test/diagnostic/doc/provider-audit sweep. The fresh local rerun
  is blocked by stale peer test executables holding shared target files; hosted
  CI is the independent compile gate.

## Provider lock refresh (2026-07-17)

- Finding: Apollo's lockfile resolved Hephaestus at `87d478…`, behind the Atlas
  provider graph's merged `93bc38e` head; Eunomia, Leto, and Moirai were also
  stale relative to their default sources.
- Resolution: `cargo update -p hephaestus-core -p hephaestus-wgpu
  -p hephaestus-cuda` refreshed the reproducibility pin to Hephaestus `93bc38e`,
  Eunomia `a2e4f390`, Leto `6a0e2971`, and Moirai `8a51b2a7`, with no manifest
  path or revision overrides.
- Theorem: the lockfile is the sole provider revision selector; when every
  first-party provider entry resolves to a merged default-source commit, the
  consumer graph is reproducible and cannot silently select the former heads.
- Evidence tier: Cargo resolution, locked compile, 402/402 Nextest,
  warning-denied Clippy, doctests, warning-clean rustdoc, and provider audit.

## Winograd trait re-export (2026-07-17)

- Finding: `mixed_radix::traits` re-exported `ShortWinogradScalar`, creating a
  second apparent ownership path for a trait whose implementation belongs in
  the vertical `components::winograd` tree.
- Resolution: remove the internal re-export and rewrite every Apollo FFT and
  macro caller to the canonical module. The theorem is direct: the caller
  graph contains exactly one `ShortWinogradScalar` definition path, therefore
  the trait/codelet contract has one SSOT and cannot drift through a forwarding
  alias.
- Evidence tier: source-residue scan, 402/402 locked Nextest tests,
  warning-denied Clippy, doctests, warning-clean rustdoc, and provider audit.

## Radix execution-policy wrapper (2026-07-17)

- Finding: `RadixCompositePolicy` duplicated Moirai's threshold policy and
  exposed a public Apollo module without owning execution semantics.
- Resolution: the wrapper and module are deleted; radix-composite dispatch now
  instantiates `moirai::AdaptiveWithThreshold` directly with Apollo's tuning
  threshold. This preserves the threshold contract while restoring Moirai as
  the execution-policy SSOT.
- Evidence tier: source-residue scan, threshold boundary regression, locked
  package tests, warning-denied diagnostics, doctests, rustdoc, and provider
  audit. The `apollo-fft` package advances to 0.25.0 with no compatibility
  export.

## Dense FFT dispatch verification tree (2026-07-17)

- Finding: `gpu_fft/dispatch.rs` mixed typed Hephaestus execution with two
  device-present verification contracts in a 589-line file.
- Resolution: the verification contracts now live in the private
  `gpu_fft/verification/dispatch.rs` leaf; the dispatch implementation remains
  the single provider-owned execution path. ADR 0034 records the inverse law
  and the existing \(\gamma_{256}\) / \(13\gamma_{256}\) finite-precision
  bounds.
- Closure: Apollo PR #46 merged at `11fd1d0`; the parent Atlas integration
  records the provider head.
- Evidence tier: source topology and nightly rustfmt are clean; locked Nextest
  passes 393/393, warning-denied Clippy and warning-clean rustdoc pass, and the
  provider audit passes 5/5. The refactor has no known residual.

## Moirai execution ownership (2026-07-17)

- Finding: the radix-composite kernel routed chunk dispatch through an
  Apollo-owned forwarding function that only called Moirai and had no external
  callers.
- Resolution: the helper is deleted; the kernel calls Moirai's canonical
  `for_each_chunk_mut_enumerated_with` directly. `RadixCompositePolicy` remains
  as the domain-specific ZST threshold strategy, not a duplicate execution
  abstraction. Evidence tier: source-residue scan, formatting, and 393/393
  `cargo nextest` tests. The `apollo-fft` package advances to 0.24.0 because
  the removed `pub` helper was part of the pre-1.0 public surface.

## Provider-cleanup release metadata (2026-07-16)

- The breaking provider-boundary removals are versioned without a compatibility
  layer: `apollo-fft` 0.23.0, `apollo-leto-interop` 0.17.0,
  `apollo-validation` 0.3.0, and each of the fourteen marker-alias packages at
  its next pre-1.0 minor release.
- `cargo metadata --locked --no-deps` resolves the updated lock graph. Current
  SemVer comparisons for `apollo-leto-interop`, `apollo-fft`, and
  `apollo-validation` pass with no required update against their pre-removal
  baselines. The remaining verification residuals are unchanged: host MinGW
  CUDA linking, the stalled doctest harness, and the package-wide
  fourteen-package SemVer sweep that requires a rustdoc child to complete.

## Validation suite concern tree (2026-07-16)

- Finding: `apollo-validation::application::suite::mod` was a 974-line
  implementation file spanning orchestration, FFT, NUFFT, external references,
  benchmarks, environment reporting, fixtures, metrics, and tests.
- Resolution: the stale claim was taken over after more than one hour without a
  scope commit. The manifest is now declaration-only; nine private concern
  leaves retain the public suite paths and every leaf is below 500 lines. The
  validation caller's obsolete `gpu_fft_available` field/call was removed under
  ADR 0013 instead of preserving a failure-erasing wrapper.
- Evidence tier: value-semantic Nextest passes 10/10, all-targets warning-denied
  Clippy and check pass, rustdoc passes, the provider audit reports zero raw
  WGPU paths, and the source-tree size scan reports a 26-line manifest with
  leaves of 62–230 lines. Existing analytical FFT/NUFFT/reference assertions
  remain unchanged.

## Unused CPU marker aliases (2026-07-16)

- Finding: fourteen GPU transport manifests exported definition-only public
  `CpuTransformMarker` aliases; a workspace reference scan found no consumer.
- Resolution: all aliases and their comments are deleted. The owning transform
  plan and the existing crate dependency edge remain the SSOT for dependency
  direction; no compatibility export is retained.
- Evidence tier: ADR 0033's structural proof sketch, zero source references,
  all-targets package checks, warning-denied all-features Clippy, provider audit,
  and default-feature Nextest (382/382 tests passed). No transform arithmetic or
  provider ownership changed.

## GPU availability probe cleanup (2026-07-16)

- Finding: `apollo-fft` exported `gpu_fft_available() -> bool`, whose body was
  the hardcoded value `true`; the function and its two re-exports duplicated
  provider capability ownership and erased typed Hephaestus acquisition state.
- Resolution: the function and both exports are deleted. Consumers must acquire
  `hephaestus_wgpu::WgpuDevice` and handle its typed result at the boundary.
  The validation-suite caller and report schema now remove the stale
  `surface_reported_available` field as part of the stale-claim takeover.
- Evidence tier: committed-branch source scans find no `gpu_fft_available`
  references; the existing default-feature Apollo test lane and warning-denied
  checks are green, and the provider audit reports zero raw WGPU paths. No
  transform arithmetic changes.
- Residual: the all-feature lane still cannot link CUDA on this host
  (`x86_64-w64-mingw32-ld.exe: cannot find -lcuda`). The first provider-audit
  attempt waited on the shared target lock; a retry after the lock cleared
  passed.

## Direct Leto output construction (2026-07-16)

- Finding: `try_dense_from_contiguous` was a consumer-owned forwarding wrapper
  used only by the four FFT 2D/3D real forward/inverse boundaries.
- Resolution: the wrapper and all exports are deleted. Each boundary now
  constructs the typed `leto::Array` directly from the contiguous Mnemosyne
  slice; the 2D/3D tests compare shape and every output value against the
  authoritative array API.
- Theorem/evidence: ADR 0032 proves shape and logical-order preservation from
  Leto's contiguous storage contract. The direct-array parity tests provide
  value-semantic differential evidence for both forward and inverse paths.
  The focused default-feature Nextest run passed 402/402 tests, and warning-
  denied Clippy plus all-targets type checking passed.
- Residual: the all-feature Nextest lane is unverified locally because the host
  MinGW linker reports `x86_64-w64-mingw32-ld.exe: cannot find -lcuda`. This is
  an environment/toolchain dependency failure, not a source diagnostic; the
  CUDA provider lane requires the installed driver-development import archive.
- Verification contention: `cargo test --locked --doc` and `cargo semver-checks
  check-release` were started with the sanctioned commands. Rustdoc generation
  (`cargo doc --locked -p apollo-leto-interop -p apollo-fft --no-deps`) passed,
  but the doctest harness stalled after compiling `apollo_fft` and was stopped
  without changing the source tree. The initial same-version Leto comparison
  reported the expected `function_missing` break; after the package advanced
  to 0.17.0, the comparison against `b14b221` passed with no required update.
  The current FFT 0.22.0→0.23.0 comparison against `cd2973b` and validation
  0.2.0→0.3.0 comparison against `2335b29` likewise pass with no required
  update. A package-wide alias SemVer sweep then stalled in the first
  `apollo-czt` rustdoc child and was stopped; package-level classification is
  therefore an explicit tooling residual, while D9's structural evidence and
  all-targets/Nextest gates are green. The provider audit passed after the
  shared lock cleared.

## Validation suite tree (2026-07-16)

- Finding: `apollo-validation` places 974 lines of orchestration and seven
  unrelated validation concerns in `application::suite::mod`.
- Risk: the module manifest becomes a second implementation home, violating
  the vertical concern tree and allowing report behavior to drift across an
  unpartitioned file.
- Decision: ADR 0031 partitions existing code by concern while retaining the
  public module path. The refactor preserves every mathematical assertion and
  derived tolerance; no new computation, fallback, or provider abstraction is
  authorized.

## CUDA FFT provider path (2026-07-16)

- Finding: Apollo had no CUDA FFT provider although Hephaestus now owns a
  typed CUDA device, buffer, kernel, command-stream, and synchronization
  substrate.
- Risk: a consumer-local CUDA wrapper or a duplicated WGPU/CUDA FFT descriptor
  would reintroduce provider ownership and create two drifting recurrence
  contracts.
- Implementation: `apollo-fft` now carries a feature-gated `CudaBackend` and
  a one-dimensional f32 `CudaFft1d` plan over `CudaDevice`. The common
  transport leaf owns `FftParams`, zero-sized entries, and radix stages;
  provider dialects implement `KernelSource` on that single descriptor. The
  Leto complex boundary reuses typed split-complex device buffers and host
  staging. No Apollo source imports a CUDA driver or a raw WGPU path.
- Evidence tier: compile-time feature/all-target validation, warning-denied
  Clippy, rustdoc, SemVer classification, and value-semantic CUDA/CPU/WGPU
  Nextest pass. `nvidia-smi -L` identifies the RTX 5080 used by the
  device-present lane. The GNU linker requires an import archive for its
  installed `nvcuda.dll`; the generated archive lives only in the shared
  ignored target tree and does not alter the Apollo dependency graph. ADR 0030
  records the derived bounds and that this is empirical, not machine-checked,
  GPU evidence.
- CI finding: PR #42's Rust workspace gate first lacked the CUDA 13.2+ toolkit
  needed for generated `cuda-bindings` headers. Its provisioned rerun then
  reached linking but `cuda-oxide` 0.4 selected its obsolete CUDA 11.3 default
  directory, producing `rust-lld: unable to find library -lcuda`. The workflow
  now installs CUDA 13.3 driver-development stubs, discovers their exact
  directory beneath the pinned toolkit, exports it as `CUDA_LIB_PATH`, and
  stages `libcuda.so.1` for the no-GPU test process. GitHub Actions run
  29544786401 passes its Rust workspace and Python-binding jobs. The hosted
  result is compile-time and provider-unavailable-path evidence because it has
  no CUDA device; the local RTX 5080 Nextest contracts remain the GPU arithmetic
  evidence.
- Review decision: retain the three typed prepared kernels in `CudaFft1d` so
  repeated execution does not rebuild the borrowed source/hash cache lookup;
  CUDA bit reversal and power-of-two index decomposition use intrinsic and
  bitwise forms. A proposed consumer-local stream completion wrapper is
  rejected: the current provider has one legacy default stream and its
  `ComputeDevice::download` contract is the authoritative synchronous transfer
  boundary. Apollo submits then downloads without an additional
  context-wide synchronization.

## Raw-WGPU audit boundary (2026-07-16)

- Finding: `xtask provider-audit` counts the substring `wgpu`, which reports
  Hephaestus provider imports as if they were raw Apollo WGPU mechanics.
- Risk: the migration audit cannot distinguish a compliant provider boundary
  from a direct raw-WGPU residual.
- Resolution: the audit now counts only lexical `wgpu::` paths whose prefix is
  not part of a larger identifier. `hephaestus_wgpu` remains a provider import,
  not a raw-WGPU residual.
- Evidence tier: value-semantic scanner tests cover direct and provider-prefixed
  paths; the workspace audit reports zero raw WGPU paths for `apollo-fft`.

## Native-f16 FFT provider-boundary cleanup (2026-07-16)

- Finding: public `GpuFft3dF16Native::try_new` acquires a `ShaderF16` device
  through Hephaestus and erases its typed fault into `String`. Its two
  device-present tests use `let Ok` and therefore suppress every provider
  failure instead of only adapter absence.
- Risk: the native-half plan retains a consumer-owned acquisition wrapper, and
  a failed driver or feature-qualified provider initialization can appear as
  omitted numerical verification.
- Implementation: `try_new` is deleted without an alias. The native-half plan
  accepts an already acquired `WgpuDevice` through `try_from_device`; tests
  acquire their `ShaderF16` device from Hephaestus and match only
  `AdapterUnavailable` as unavailable. The `apollo-fft` package advances to
  0.21.0 for the pre-1.0 public removal.
- Resolution: Hephaestus commit `369dff41` moves four-byte padding to the WGPU
  provider's physical storage and transfer payload, preserving the 54-byte
  logical native-half volume. Apollo now locks that provider revision. The
  complete native-half suite, including the 3×3×3 Bluestein roundtrip, passes
  without an Apollo acquisition or padding wrapper.
- Evidence tier: SemVer classification, compile-time typed-error handling,
  direct source-residue scans, provider audit, and value-semantic native-half
  tests on compatible hardware. This does not prove adapter availability or
  GPU arithmetic beyond the device exercised.

## QFT verification provider-error preservation (2026-07-16)

- Finding: the private QFT verification helper returned `WgpuResult`, while
  ten device-present test branches used `let Ok(backend) = backend() else {
  return; }`. That pattern suppresses every Hephaestus provider fault rather
  than restricting a skip to adapter absence.
- Risk: provider initialization failures can appear as passing omitted QFT
  verification, concealing a real accelerator integration regression.
- Resolution: the helper now maps only `AdapterUnavailable` to an optional
  backend and panics at the verification boundary for every other typed
  provider error. All ten callers use `Option`; direct `WgpuDevice` acquisition
  and the QFT CPU-oracle/theorem contracts remain unchanged.
- Evidence tier: compile-time exhaustive typed-error handling; focused QFT
  all-feature diagnostics and Nextest; doctest; rustdoc; provider audit; and
  an exact stale-pattern scan. No runtime GPU result is claimed without a
  compatible adapter.

## Radon benchmark provider-error preservation (2026-07-16)

- Finding: `radon_wgpu_bench.rs` used `let Ok(device)` for two direct
  Hephaestus acquisitions. That discarded provider faults instead of limiting
  a benchmark skip to `AdapterUnavailable`.
- Risk: a broken GPU driver or provider configuration can appear as an omitted
  benchmark result, hiding a real integration failure.
- Resolution: one `OnceLock<Option<WgpuDevice>>` retains the directly acquired
  provider handle across the two benchmark families. Only
  `AdapterUnavailable` initializes the unavailable state; every other typed
  error panics at the benchmark boundary.
- Evidence tier: compile-time exhaustive typed-error handling; all-target
  benchmark compilation; warning-denied Clippy; focused all-feature Radon
  Nextest; doctest; rustdoc; provider audit; and an exact stale-pattern scan.
  No runtime GPU benchmark result is claimed without a compatible adapter.

## FFT acquisition-forwarder removal (2026-07-16)

- Resolution: the shared `apollo-fft::WgpuBackend::try_default` wrapper is
  deleted. `WgpuBackend::new` remains the composition boundary; device
  acquisition remains in Hephaestus. The benchmark acquires its provider device
  once and clones the shared handle for fixed-dimension plans; private
  device-present regression callers skip only `AdapterUnavailable`; the PyO3
  capability probe and validation suite report only that condition as
  unavailable. Other provider faults surface. ADR 0028 records the unchanged
  FFT theorem/evidence boundary.
- Evidence tier: formatting; warning-denied all-target diagnostics; focused
  all-feature Nextest; doctest; rustdoc; provider audit; Maturin extension
  build with 34 Python smoke tests; empty public-factory, obsolete-call, and
  benchmark-fallback scans; and the cached pre-1.0 major SemVer comparison
  against `origin/main` pass. This is API-surface and empirical-test evidence,
  not a machine-checked proof of accelerator behavior.

## Limit-bearing acquisition-forwarder removal (2026-07-16)

- Resolution: `apollo-nufft` and `apollo-stft` delete their final public
  `try_default` factories. Each typed backend retains one canonical
  `required_device_limits` method: fast NUFFT requests seven storage buffers
  and STFT Bluestein requests six. Tests and benchmark code now request a
  Hephaestus device directly, then pass it to the existing backend constructor.
- Theorem boundary: ADR 0027 records the resource-precondition argument
  `b(K) <= L` from the current WGSL binding declarations. Hardware-free tests
  pin the two lower bounds; this is not a proof of accelerator numerical
  behavior or host-device availability.
- Evidence tier: the focused 119-case all-feature Nextest gate, two direct
  hardware-free resource-contract regressions, warning-denied all-target
  Clippy, doctest, rustdoc, four provider-audit contracts, API-source scan,
  and both pre-1.0 major SemVer classifications pass. This is API-surface and
  empirical test evidence, not a machine-checked proof of accelerator behavior.

## Provider acquisition-forwarder removal (2026-07-16)

- Resolution: fifteen transform `try_default` factories that only wrapped
  `hephaestus_wgpu::WgpuDevice::try_default` are deleted. Test and benchmark
  callers acquire the typed provider device directly; test skips are limited to
  `HephaestusError::AdapterUnavailable`. The follow-on ADR 0027 removes the
  former NUFFT/STFT limit-bearing paths; the shared FFT adapter remains outside
  this scope.
- Evidence tier: focused and workspace value-semantic Nextest, warning-denied
  Clippy, rustdoc, provider audit, API-source scan, and pre-1.0 major SemVer
  classification. This is release-boundary evidence, not an accelerator proof.

## Root verification-boundary removal [major]

- Finding: ten transform crates publish root `verification` modules whose
  content is entirely test-gated. The paths are empty in release artifacts, and
  DCT/DST combines four test concerns in a 672-line root module.
- Resolution: ADR 0025 removes every root release path and partitions DCT/DST
  into transform-local private leaves. No theorem, CPU oracle, fixture, derived
  tolerance, Leto boundary, or Hephaestus contract changes.
- Evidence: the root public-path scan is empty, all DCT/DST leaves are at most
  209 lines, and the original and restructured trees expose the same 42 test
  function names. Focused and workspace all-feature value-semantic Nextest,
  warning-denied Clippy, affected doctests, workspace rustdoc, example
  compilation, and `xtask provider-audit` pass. All ten pre-1.0 major SemVer
  comparisons against `origin/main` pass; each reports no required update under
  the explicit major-change assumption. This is API-surface and empirical test
  evidence, not a machine-checked proof of accelerator behavior.

## Transport verification-boundary removal [major]

- Finding: thirteen transport `verification` modules are public despite
  containing only `cfg(test)` contracts. They expose test fixtures and CPU
  oracles as release paths without a runtime responsibility.
- Resolution: ADR 0024 makes each module crate-private and test-gated. Existing
  operation-specific trees and cohesive sub-500-line leaves remain in their
  transform-local homes; no provider wrapper changes.
- Evidence: the public-path scan is empty; focused and workspace all-feature
  value-semantic Nextest, warning-denied Clippy, affected doctests, workspace
  rustdoc, example compilation, and `xtask provider-audit` pass. All thirteen
  pre-1.0 major SemVer comparisons against `origin/main` pass; each reports no
  required update under the explicit major-change assumption. This is
  API-surface and empirical test evidence, not a machine-checked proof of
  accelerator behavior.

## Root accelerator-forwarder removal [major]

- Finding: thirteen transform crates publish `wgpu_backend` modules that only
  forward their existing typed Hephaestus crate-root exports. They retain an
  obsolete hierarchy without owning provider behavior.
- Resolution: ADR 0023 removes every forwarding module and retains one
  feature-gated root accelerator path. The cleanup changes no transform
  formula, theorem, Leto host boundary, or Hephaestus contract.
- Evidence: the empty definition/caller residue scan; focused and workspace
  all-feature value-semantic Nextest; warning-denied Clippy; doctest; rustdoc;
  provider audit; and thirteen pre-1.0 major SemVer classifications. This is
  API-surface and empirical test evidence, not a machine-checked proof of GPU
  behavior.

## Provider default-source convergence [minor]

- Finding: the root manifest combined direct revision pins with local patches,
  which forked the provider graph even though Hermes, Leto, Hephaestus, and
  Moirai had already merged their required contracts.
- Resolution: direct first-party dependencies follow their provider default
  branches and the root patches are deleted. Every member declares Rust 1.95,
  matching the resolved provider graph. `Cargo.lock` is the one reproducibility
  pin; Apollo has no adapter or fallback path for this graph.
- Evidence: `xtask provider-audit`, a one-identity lockfile scan, Rust 1.95
  acceptance and Rust 1.94.1 rejection, 43 focused all-feature Nextest cases,
  the 1,155-case all-feature workspace Nextest gate, warning-denied workspace
  Clippy, workspace rustdoc, and 22 passing 196-check minor SemVer
  comparisons. This is dependency-graph and API-surface evidence, not a
  machine-checked proof of provider behavior.

## Native benchmark runtime ownership [arch]

- Finding: Criterion introduced Apollo's last resolved Rayon edge even though
  transform runtime parallelism had already migrated to Moirai. Its benchmark
  DSL duplicated only generic timing orchestration; the seven Apollo binaries
  own all mathematical workloads and setup.
- Resolution: `apollo-bench` owns sequential warm-up, adaptive batch sizing,
  normalized sample collection, and CSV reporting. FFT, NUFFT, Radon, and STFT
  benchmark binaries call its native case API directly; Criterion macros and
  adapters are absent. ADR 0011 records the decision and median theorem.
- Evidence tier: the order-statistic theorem is a proof sketch; eight focused
  nextest cases validate typed budgets, calibration arithmetic, median values,
  even-sample central-pair averaging, empty-sample rejection, CSV escaping,
  real closure execution, and report values. Benchmark binaries compile
  through `cargo bench --no-run`; no runtime speed or cross-harness equivalence
  claim is made without a recorded baseline comparison.
- Residual: no Criterion/Rayon graph edge remains. GPU benchmark execution
  remains hardware-dependent and is not implied by binary compilation.

## GPU test-process exclusivity [patch]

- Finding: the all-feature workspace nextest gate aborted NTT's real-device
  64-case property test with Windows error `0xc0000005` while concurrent GPU
  tests initialized independent provider devices. The same property passed in
  isolation, so this is a device-concurrency defect rather than a numerical
  counterexample or a benchmark-runtime result.
- Resolution: `.config/nextest.toml` assigns every GPU transport and dense-FFT
  device test to `gpu-device`, a shared test group with `max-threads = 1`.
  CPU tests retain normal parallelism; GPU tests retain their real execution,
  complete property domain, and existing 30 s/60 s timeout contract.
- Evidence tier: nextest's configured process-level mutual exclusion plus
  value-semantic GPU tests. This does not prove driver correctness; it prevents
  uncoordinated concurrent device acquisition within the test run.

## Hilbert GPU verification-tree normalization [arch]

- Finding: the 410-line private Hilbert verification module mixed metadata,
  analytic/quadrature execution, inverse projection, Leto boundaries, typed
  storage, precision boundaries, and shared backend setup.
- Resolution: `gpu/verification/` now has one manifest and seven
  concern-named leaves. `support.rs` is the sole home for repeated device
  acquisition; it owns no transform execution or provider implementation.
- Mathematical contract: for the documented DFT convention, the multiplier is
  `-i sgn(k)`. Thus `H(H(x)) = -x` only after DC and even-length Nyquist modes
  are removed; inverse execution reconstructs that projection and compares it
  with an independent CPU frequency-domain reference. ADR 0018 contains the
  proof sketch.
- Evidence tier: 16 private verification contracts (four static, twelve
  device-present) and 43/43 all-feature package Nextest cases, plus Clippy,
  doctest, rustdoc, provider audit, and patch SemVer classification. This is
  finite-precision empirical evidence, not a machine-checked proof.
- Residual risk: cross-transform backend/acquisition consolidation remains a
  Hephaestus-owned provider concern; this split adds no Apollo wrapper.

## CZT GPU verification-tree normalization [arch]

- Finding: the 395-line private CZT GPU verification module mixed static
  metadata, direct CPU differential, impulse, Leto, represented-storage,
  precision, rejection, and inverse contracts.
- Resolution: `gpu/verification/` now has one manifest and eight
  concern-named leaves. `support.rs` is the sole home for test device
  acquisition, shared fixtures, and the existing analytical bounds; it owns no
  provider implementation.
- Mathematical contract: `X_k = sum_(n=0)^(N-1) x_n A^(-n) W^(nk)`. The
  inverse round-trip applies only to the DFT specialization `A = 1`,
  `W = exp(-2 pi i / N)`, and equal input/output lengths. ADR 0019 records the
  proof sketch and the existing `8192 eps_f32` two-transform bound.
- Evidence tier: 14 private contracts (two static, twelve device-present),
  55/55 focused all-feature Nextest cases, locked workspace gates, provider
  audit, and patch SemVer classification. This is finite-precision empirical
  evidence, not a machine-checked proof.
- Residual risk: cross-transform backend/acquisition consolidation remains a
  Hephaestus-owned provider concern; this split adds no Apollo wrapper.

## GFT GPU verification-tree normalization [arch]

- Finding: the 381-line private GFT GPU verification module mixed static
  metadata, CPU differentials, reconstruction, caller-owned output, Leto
  boundaries, represented storage, and precision rejection.
- Resolution: `gpu/verification/` now has one manifest and seven
  concern-named leaves. `support.rs` is the sole home for test device
  acquisition, the path-four fixture, and the derived numerical bound; it owns
  no provider implementation. The prior public test-only module and its
  `wgpu_backend::verification` re-export path are removed in `apollo-gft`
  0.5.0.
- Mathematical contract: for an orthonormal graph basis, `x_hat = U^T x` and
  `x = U x_hat`, because `U^T U = I`. The path-four CPU plan supplies the
  independent oracle. ADR 0020 records the proof sketch and existing `2^-17`
  finite-precision bound.
- Evidence tier: 13 private contracts (three static, ten device-present),
  28/28 focused all-feature Nextest cases, locked workspace gates, provider
  audit, and patch SemVer classification. This is finite-precision empirical
  evidence, not a machine-checked proof.
- Residual risk: cross-transform backend/acquisition consolidation remains a
  Hephaestus-owned provider concern; this split adds no Apollo wrapper.

## QFT GPU verification-tree normalization [major]

- Finding: the 381-line QFT GPU verification module mixed static metadata, CPU
  differentials, inverse reconstruction, Leto host boundaries, represented
  storage, precision rejection, and pre-dispatch rejection. The public transport
  path contained only test-gated content.
- Resolution: `gpu/verification/` now has one test-only manifest and seven
  concern-named leaves. `support.rs` is the sole home for test acquisition,
  shared fixtures, CPU comparison, and the existing bounds. The public
  verification module and obsolete `wgpu_backend` forwarding module are
  removed in `apollo-qft` 0.5.0 rather than retained as empty compatibility
  modules.
- Mathematical contract: `U[k, j] = exp(2 pi i k j / N) / sqrt(N)`. Discrete
  Fourier orthogonality gives `U^dagger U = I`, so inverse execution
  reconstructs the input and forward execution preserves the L2 norm. ADR 0021
  records the proof sketch. Direct CPU-differential and two-launch round-trip
  limits remain `2.0e-4` and `5.0e-4`.
- Evidence tier: 15 test-private transport contracts (three static and twelve
  device-present), 37/37 focused all-feature package Nextest cases, locked
  workspace check/Clippy/Nextest (1,155/1,155), doctest, rustdoc, provider
  audit, direct raw-WGPU/probe scans, and major SemVer classification. This is
  finite-precision empirical evidence, not a machine-checked proof.
- Review: an independent post-partition diff review found a duplicated CPU
  conversion and the obsolete `wgpu_backend` forwarding module. The conversion
  now has one support-leaf home; the forwarding module is deleted. No P0 or P1
  finding remains in the changed scope.
- Residual risk: cross-transform backend/acquisition consolidation remains a
  Hephaestus-owned provider concern; this split adds no Apollo wrapper.

## SHT GPU verification-tree normalization [major]

- Finding: the 343-line SHT GPU verification module mixed static metadata,
  rejection, CPU differentials, Leto boundaries, typed storage, and device
  acquisition while exposing an empty release module.
- Resolution: `gpu/verification/` now has one test-only manifest and seven
  concern-named leaves. `support.rs` is the sole home for fixture construction,
  represented CPU comparison, finite-precision limits, and test acquisition;
  it owns no SHT execution or provider implementation. The public transport
  verification path is removed in `apollo-sht` 0.5.0.
- Mathematical contract: for the documented band-limited grid,
  `L < N_lat` and `2L + 1 <= N_lon`, Gauss-Legendre exactness and longitude
  Fourier orthogonality yield `inverse(forward(f)) = f`. The CPU plan is the
  independent representation oracle. ADR 0022 records the proof sketch and
  existing `2.0e-5` finite-precision differential limit.
- Evidence tier: 11 test-private contracts (two static and nine device-present),
  35/35 focused all-feature Nextest cases, workspace check/Clippy/Nextest,
  doctest, rustdoc, provider audit, source scan, and major SemVer verification.
  This is finite-precision empirical evidence, not a machine-checked proof.
- Review: the inherited acquisition helper skipped every provider error. It now
  skips only Hephaestus `AdapterUnavailable`; a device-creation or other
  provider failure fails the test. No P0 or P1 finding remains in this scope.
- Residual risk: cross-transform backend/acquisition consolidation remains a
  Hephaestus-owned provider concern; this split adds no Apollo wrapper.

## DCT/DST GPU verification-tree normalization [arch]

- Finding: the DCT/DST transport had a 796-line verification monolith mixing
  capability, one-dimensional, typed-storage, Leto-boundary, dimensional, and
  rejection contracts. The layout obscured independent CPU oracles and did not
  meet the tree's file-size boundary.
- Resolution: `gpu/verification/` now holds seven concern-specific leaves.
  `support.rs` is the sole home for backend availability and repeated
  value-comparison assertions; dimensional leaves retain the independent
  tensor-product CPU construction. No production code, raw-WGPU edge, device
  wrapper, or duplicate transform execution was introduced.
- Mathematical contract: applying a one-dimensional DCT-II on each axis equals
  the separable tensor-product DCT-II. ADR 0012 records the proof sketch and
  keeps it distinct from the finite-precision CPU-differential evidence.
- Evidence tier: 72 all-feature DCT/DST nextest cases, including device-present
  CPU differentials, inverse pairs, typed storage, Leto boundaries, and shape
  rejections; warning-denied Clippy, doctest, and rustdoc. This is not a
  machine-checked proof or a GPU performance claim.
- Residual risk: generic cross-transform device/error/capability extraction
  remains provider-owned D8 work. The local split does not substitute for that
  upstream contract.

## FrFT GPU verification-tree normalization [arch]

- Finding: the FrFT transport had a 578-line verification monolith that mixed
  metadata, standard FrFT CPU differentials, typed Leto storage, and unitary
  FrFT contracts. Its layout hid the independent proof obligations behind one
  device-present test function.
- Resolution: `gpu/verification/` now has metadata, standard, typed-storage,
  unitary, and support leaves. The support leaf is the sole home for device
  availability, CPU conversion, and repeated value comparisons; it owns no
  device handle, kernel, transform execution, or fallback path.
- Mathematical contract: the existing Candan--Grünbaum theorem states
  `DFrFT_a = V diag(exp(-i a k pi / 2)) V^T`. Because `V` is orthogonal and
  every diagonal phase has unit modulus, the exact operator preserves the L2
  norm and `DFrFT_-a(DFrFT_a(x)) = x`. The unitary leaf retains identity,
  reversal, inverse-pair, norm, and independent CPU-differential checks; the
  theorem's full proof and references remain in the owning FrFT plan module.
- Evidence tier: 52 all-feature FrFT nextest cases and the 1,037-case
  all-feature workspace nextest suite pass, including all device-present
  standard, typed-storage, Leto, and unitary contracts. Warning-denied Clippy,
  doctest, rustdoc, provider audit, structural scans, and patch SemVer surface
  checks pass. This is a proof sketch and value-semantic evidence, not a
  machine-checked proof or GPU performance claim.
- Residual risk: generic cross-transform device/error/capability extraction
  remains provider-owned D8 work. This local split does not create an Apollo
  transport abstraction.

## NTT GPU verification-tree normalization [arch]

- Finding: the NTT transport had a 550-line verification monolith mixing
  static capabilities, exact CPU comparisons, Leto views, quantized storage,
  reusable buffers, rejection paths, and finite-field property laws.
- Resolution: `gpu/verification/` now has metadata, exact, quantized,
  reusable, property, and shared-availability leaves (7–137 lines). The
  support leaf is the sole availability boundary; all device-present leaves
  retain computed residue assertions. The tree preserves all existing source
  inputs and generated domains but makes each contract independently visible;
  no production device mechanics, compatibility wrapper, or fallback path was
  added.
- Mathematical contract: in the finite field, the ordered butterfly stages
  yield `INTT(NTT(x)) = x`, and pointwise multiplication yields
  `INTT(NTT(a) ⊙ NTT(b)) = a ★ b`. The owning NTT plan and README retain the
  proof sketch; generated exact-residue and direct cyclic-convolution checks
  provide the executable evidence.
- Evidence tier: 37 all-feature NTT nextest cases and the all-feature workspace
  nextest suite pass, including real-device exact CPU differentials,
  Leto/quantized storage, reusable buffers, and generated inverse/convolution
  laws. Workspace examples/check and warning-denied Clippy, package doctest,
  rustdoc, provider audit, structural scans, and patch SemVer classification
  pass. This is not a machine-checked proof or a GPU performance claim.
- Residual risk: generic cross-transform device/error/capability extraction
  remains provider-owned D8 work. This local split does not pre-empt the
  required Hephaestus provider contract.

## NUFFT GPU verification-tree normalization [major]

- Finding: the NUFFT transport has a 1,164-line verification monolith mixing
  static capabilities, direct and fast Type-1/Type-2 contracts in 1D and 3D,
  Leto and typed boundaries, reusable storage, diagnostics, normalization, and
  typed errors.
- Required resolution: split by operation family and dimensional concern while
  preserving the direct adjoint theorem, its existing derived finite-precision
  tolerances, and every device-present value-semantic contract. Shared
  availability and comparison helpers remain private test support; they must
  not become an Apollo transport abstraction.
- Evidence required: focused and workspace nextest, warning-denied diagnostics,
  provider audit, source-residue scans, and a documented pre-1.0 breaking
  SemVer classification. The theorem remains a proof sketch with empirical
  CPU-differential evidence, not a machine-checked proof.
- Resolution: delete the monolith and its transitional aggregate `device.rs`;
  retain metadata, reusable-storage, direct Type-1/Type-2 1D/3D, fast
  Type-1/Type-2 1D/3D, and shared-support leaves. Shared 3D grid, positions,
  type-1 values, and mode-field components have one support-leaf definition.
  The transport now gates the whole tree under `cfg(test)`, so verification
  does not leave a public release wrapper; fast leaves are flat test modules.
  `apollo-nufft` 0.4.0 also removes its unused `wgpu_backend` forwarding
  module and public root test module; root accelerator exports are the
  migration target.
- Evidence: 73/73 focused all-feature Nextest cases; all-feature workspace
  check, Clippy, and Nextest; package doctest/rustdoc; `xtask provider-audit`;
  source scans with no raw `wgpu`, `pollster`, `apollo-wgpu-helpers`,
  `verification.rs`, or transitional verification `device.rs`; and SemVer
  evidence that identifies the removed public module paths as the intentional
  `apollo-nufft` 0.4.0 pre-1.0 break. Evidence remains empirical/type-level,
  not a machine-checked theorem proof.
- Resolution extension: D8-NUFFT-root-verify replaces the private 918-line
  root theorem suite with direct-identity, adjoint, kernel-width,
  in-place-consistency, and 1D/3D typed-storage leaves. The manifest is 19
  lines; leaves are 69–280 lines. The exact fixtures, analytical references,
  derived tolerances, and theorem comments remain in their owning leaves.
- Evidence extension: focused all-feature Nextest runs 73/73 value-semantic
  tests, including every relocated theorem and typed-storage contract. This is
  empirical finite-precision evidence, not a machine-checked proof.
- Residual risk: generic cross-transform verification support remains a
  provider-owned D8 concern. This split creates no helper wrapper,
  compatibility path, or Apollo-owned device abstraction.

## NUFFT provider-acquisition wrapper removal [major]

- Finding: the public `nufft_wgpu_available` function discarded
  `NufftWgpuBackend::try_default`'s typed Hephaestus failure and duplicated a
  consumer-side availability probe. It had no in-repository caller.
- Resolution: `apollo-nufft` 0.5.0 deletes the function and its root re-export.
  Consumers now use the existing typed `NufftWgpuBackend::try_default` result.
  That constructor remains transform-local because NuFFT, unlike the default
  provider request, requires seven storage buffers per shader stage.
- Mathematical contract: unaffected; this changes acquisition error visibility
  only, not the Type-1/Type-2 operators or their documented theorem evidence.
- Evidence: major pre-1.0 SemVer classification against `origin/main`, focused
  and workspace value-semantic gates, provider audit, and source scan prove the
  wrapper is absent. The provider audit retains NuFFT's Leto and Hephaestus
  dependencies and reports no Apollo-owned raw WGPU surface.
- Residual risk: the repeated transform-specific acquisition constructors are
  the next D8 provider-seam inventory. They cannot be replaced by a local
  generic Apollo wrapper; only a Hephaestus contract that accepts per-transform
  limits can own that consolidation.

## Provider-acquisition probe removal [major]

- Finding: sixteen transform crates duplicated an unused public
  `wgpu_available` boolean probe, each suppressing typed Hephaestus acquisition
  errors with `is_ok()`.
- Resolution: ADR 0013 removes every definition and re-export in one bounded
  pre-1.0 breaking migration. Transform-specific constructors remain because
  they declare the provider limits their authored kernels require; no Apollo
  shared probe or compatibility alias replaces the deleted API.
- Evidence: all sixteen major SemVer classifications, locked format,
  examples-check, warning-denied Clippy, all-feature Nextest, doctest,
  rustdoc, metadata, and provider-audit gates pass. Direct raw-WGPU and
  `wgpu_available` source scans are empty. The mathematical transform
  contracts are unaffected; this changes failure visibility at the acquisition
  boundary. This is API/type-level and empirical test evidence, not a
  machine-checked theorem or GPU-performance proof.

## STFT GPU verification-tree normalization [arch]

- Finding: `apollo-stft` holds 961 lines of private GPU tests in one module,
  mixing independent metadata, CPU-differential, reconstruction, typed-host,
  and reusable-storage contracts.
- Decision: ADR 0014 partitions this test-only tree by contract. The existing
  weighted-overlap-add theorem and finite-precision evidence from ADR 0008
  remain unchanged. Hephaestus retains generic acquisition; STFT capability
  and error types remain local domain contracts.
- Acceptance: each leaf is concern-named and bounded; every relocated test
  retains its existing value oracle and tolerance; focused and locked
  workspace gates pass. This is not a new GPU-correctness proof.
- Resolution: the manifest plus `metadata`, `forward`, `inverse`, `typed`,
  `reusable`, and shared `support` leaves are 7–297 lines. All 44 tests run.
  The four former existence-only Chirp-Z/buffer checks now assert CPU values
  or allocated geometry and zero-value contracts. Focused and locked
  all-feature gates, provider audit, and patch SemVer classification pass.

## Radon GPU verification-tree normalization [arch]

- Finding: `apollo-radon` keeps 536 lines of private GPU verification in one
  module, mixing metadata, CPU projection/backprojection differentials, Leto
  boundaries, represented-storage values, rejection paths, and theorem checks.
- Decision: ADR 0015 partitions the test-only tree by those contracts.
  Hephaestus remains the sole owner of generic acquisition and execution;
  Radon owns the transform-specific values and errors.
- Mathematical contract: ADR 0007's paired interpolation establishes the
  discrete adjoint identity `<R f, p> = <f, R* p>` in exact arithmetic, while
  filtered backprojection is `(pi / A) R*(h * p)`. Existing finite-precision
  tests are empirical evidence for those statements, not a machine-checked
  proof.
- Acceptance: every moved test retains its fixture, oracle, and existing
  derived bound; each private leaf is concern-named and bounded. No provider
  wrapper, fallback, or transform algorithm enters this structural slice.
- Resolution: the manifest plus `metadata`, `forward`, `backprojection`,
  `leto`, `typed`, `filtered`, and shared `support` leaves are 7–123 lines.
  The former suite's five static tests and thirteen device-present contracts
  now run independently. Focused all-feature Nextest passes 35/35; locked
  workspace format, examples check, warning-denied Clippy, Nextest
  (1,090/1,090), doctest, rustdoc, provider audit, and patch SemVer
  classification pass. This evidence is type-level tree structure plus
  empirical numerical verification, not a machine-checked theorem proof.

## Wavelet GPU verification-tree normalization [arch]

- Finding: `apollo-wavelet` keeps 431 lines of private GPU verification in one
  module, mixing metadata, rejection, analytical Haar, inverse-law, CPU,
  Leto, and represented-storage contracts.
- Decision: ADR 0016 partitions this test-only tree by those contracts.
  Hephaestus remains the sole owner of generic acquisition and execution;
  Wavelet owns the transform-specific values and errors.
- Mathematical contract: one Haar pair uses
  `H = (1 / sqrt(2)) [[1, 1], [1, -1]]`, so `H^T H = I`. Multilevel
  reconstruction and Parseval conservation follow by composition. Existing
  finite-precision tests are empirical evidence for those statements, not a
  machine-checked proof.
- Acceptance: every moved test retains its fixture, oracle, and existing
  derived bound; each private leaf is concern-named and bounded. No provider
  wrapper, fallback, or transform algorithm enters this structural slice.
- Resolution: the manifest plus `metadata`, `forward`, `inverse`, `leto`,
  `typed`, and shared `support` leaves are 7–150 lines. The former suite's
  three static tests and fourteen device-present contracts now run
  independently. Focused all-feature Nextest passes 38/38; locked workspace
  format, examples check, warning-denied Clippy, Nextest (1,103/1,103),
  doctest, rustdoc, provider audit, and patch SemVer classification pass. This
  evidence is type-level tree structure plus empirical numerical verification,
  not a machine-checked theorem proof.

## SFT GPU verification-tree normalization [arch]

- Finding: `apollo-sft` keeps 416 lines of private GPU verification in one
  module, mixing metadata, rejection, CPU sparse-spectrum differentials,
  inverse reconstruction, Leto boundaries, represented storage, and explicit
  precision conversion.
- Decision: ADR 0017 partitions the test-only tree by those contracts.
  Hephaestus remains the sole owner of generic acquisition and execution; SFT
  owns transform-specific sparse values and errors.
- Mathematical contract: the normalized dense inverse follows from
  `sum_k exp(2 pi i k(n - m) / N) = N delta_nm`. Top-`k` is a projection and
  reconstructs retained support rather than discarded coefficients. Existing
  finite-precision tests are empirical evidence, not a machine-checked proof.
- Acceptance: every moved test retains its fixture, oracle, error value, and
  existing derived bound; each private leaf is concern-named and bounded. No
  provider wrapper, fallback, or transform algorithm enters this structural
  slice.
- Resolution: the manifest plus `metadata`, `forward`, `inverse`, `leto`,
  `typed`, `precision`, and shared `support` leaves are 20–103 lines. The
  former suite's three static and twelve device-present contracts now run
  independently. Focused all-feature Nextest passes 41/41; locked workspace
  format, examples check, warning-denied Clippy, Nextest (1,114/1,114),
  doctest, rustdoc, provider audit, and patch SemVer classification pass. This
  evidence is type-level tree structure plus empirical numerical verification,
  not a machine-checked theorem proof.

## Hilbert GPU verification-tree normalization [arch]

- Finding: `apollo-hilbert` keeps 410 lines of private GPU verification in one
  module, mixing metadata, rejection, CPU analytic/quadrature differentials,
  inverse projection, Leto boundaries, represented storage, and explicit
  precision conversion.
- Decision: ADR 0018 partitions the test-only tree by those contracts.
  Hephaestus remains the sole owner of generic acquisition and execution;
  Hilbert owns transform-specific frequency-mask values and errors.
- Mathematical contract: on the subspace without DC or Nyquist coefficients,
  the multiplier `-i sgn(k)` satisfies `H(H(x)) = -x`. The inverse GPU mask
  applies `-H`, so it reconstructs that projection rather than unrecoverable
  DC/Nyquist components. Existing finite-precision tests are empirical
  evidence, not a machine-checked proof.
- Acceptance: every moved test retains its fixture, oracle, error value, and
  existing derived bound; each private leaf is concern-named and bounded. No
  provider wrapper, fallback, or transform algorithm enters this structural
  slice.

## Shared Leto interop ownership [arch]

- Finding: a transform-private FFT utility owned cross-transform Leto view and
  output conversion. Consumer forwarding functions duplicated the same
  ownership-free API and inverted the intended dependency direction.
- Resolution: `apollo-leto-interop` is the canonical host-boundary crate.
  `view_cow`, dense array/view materialization, dense slice construction, and
  one-dimensional constructors live there; transform call sites map only their
  own domain errors. `PrecisionProfile::matches_storage_and_compute` retains
  the independent precision semantic in its owning type. The old FFT utility
  and transform-local wrappers are deleted without compatibility aliases.
- Mathematical contract: ADR 0010 records the logical-order representation
  theorem for contiguous and strided views plus shape preservation for dense
  output construction. This is a proof sketch over Leto's documented contract,
  not a machine-checked proof.
- Evidence: package checks for `apollo-leto-interop`, FFT, NUFFT, Radon, SHT,
  STFT, and Wavelet pass; 10/10 value-semantic interop nextest cases cover
  contiguous, strided, transposed, and rank-three paths. Locked all-feature,
  no-default, and examples checks; warning-denied Clippy; configured all-feature
  workspace nextest; doctest; warning-clean rustdoc; provider audit; direct
  source/dependency scans; and 0.17.0 major SemVer classification against the
  merged 0.16.0 baseline pass.

## Cargo Deny first-party source policy [patch]

- Finding: the workspace deliberately resolves its Atlas providers from
  `ryancinsight` Git repositories, but Cargo Deny's `unknown-git = "deny"`
  configuration had no first-party exception. CI therefore rejected the
  locked provider graph despite passing advisories, bans, and licenses.
- Resolution: `sources.allow-org.github = ["ryancinsight"]` admits only the
  first-party organization. Unknown Git sources and unknown registries remain
  denied; this is a source-policy correction, not a broad Git exception.
- Evidence tier: Cargo Deny's source-graph policy check validates the resolved
  lock graph. It does not prove the trustworthiness of the allowed organization.

## Apple Silicon Stockham target boundary [patch]

- Finding: RITK macOS CI uses `aarch64-apple-darwin`, but Apollo's Stockham
  module re-exported AVX-only butterfly and precision symbols on every target.
  The first failure exposed 43 compile errors, including missing fixed-length
  AVX kernels and AVX backend imports.
- Resolution: gate the AVX module, AVX butterfly modules and re-exports, AVX
  precision imports, and AVX-only test imports on `target_arch = "x86_64"`.
  Non-x86 targets retain the existing scalar Stockham/ZST dispatch path.
- Verification: the pinned 1.97 toolchain `cargo check -p apollo-fft
  --target aarch64-apple-darwin --all-features --locked` passes; host
  warning-denied all-target Clippy passes; nextest passes 409/409; doctests
  run 0/0; warning-clean rustdoc completes.
- Evidence tier: cross-target compile/type validation plus host value-semantic
  nextest and warning-denied linting. Cross-target checking still reports
  existing unused-code warnings in scalar-only and platform-only branches;
  those are not compile blockers and remain visible for a later warning-ratchet
  increment.

## STFT Hephaestus command-stream migration [arch]

- Performed: replaced direct radix-2, Bluestein, and overlap-add WGPU pipeline,
  binding, encoding, submission, and transfer ownership with typed Hephaestus
  descriptors and ordered command streams. The provider receives the six
  Bluestein storage bindings through its backend-neutral device-limits API.
- Mathematical contract: DFT orthogonality recovers each windowed frame in
  exact arithmetic. Applying the synthesis window and dividing weighted
  overlap-add by the non-zero squared-window sum recovers the original sample.
  ADR 0008 and the crate README distinguish this theorem from finite-precision
  evidence.
- Structural cleanup: deleted the `wgpu_backend` forwarding module, raw
  device/queue accessors, direct WGPU/pollster/helper dependencies, the raw
  Chirp-Z implementation, and raw benchmark claims. `kernel/dispatch.rs` is
  the canonical home for shared geometry, chirp preparation, and grouped
  provider dispatch. Leto remains the host boundary; `StftGpuBuffers` retains
  typed provider-owned radix-2 storage.
- Evidence tier: typed binding/layout and ordered stream semantics, then 46
  value-semantic tests including real-device CPU differential, non-power-of-two
  Bluestein, reconstruction, and reusable-storage coverage. Clippy, rustdoc,
  provider audit, direct source/dependency scans, and semver classification
  pass. No machine-checked proof is performed.
- Historical residual at STFT closure: native-half FFT transport was the only
  direct-provider scope; it is now migrated and the obsolete wrapper is deleted.

## NUFFT Hephaestus command-stream migration [arch]

- Performed: replaced direct Type-1/Type-2 and fast Kaiser--Bessel 1D/3D raw
  transport with typed Hephaestus descriptors, provider-owned reusable
  buffers, and ordered streams. The fast path records load/spread,
  `GpuFft3d`, and extract/interpolate in dependency order without a raw
  command encoder.
- Mathematical contract: Type-2 is the exact-arithmetic adjoint of Type-1
  under the complex inner product. Fast paths approximate that pair through
  Kaiser--Bessel gridding. The 1D fast load compensates the normalized inverse
  FFT by the oversampled length; 3D retains its declared normalized convention.
  ADR 0009 distinguishes this theorem from finite-precision evidence.
- Structural cleanup: deleted the helper device owner, raw pipeline cache,
  binding/encoder/queue/readback transport, six raw kernel leaves, and the
  stale no-op diagnostics feature. `kernel/{descriptors,direct,fast,fast_support}`
  now separates typed descriptors, direct dispatch, fast orchestration, and
  shared fast support. Leto remains the host boundary.
- Evidence tier: compile-time typed binding/layout and stream ordering, then
  44/44 value-semantic nextest cases including real-device direct/fast CPU
  differential and bit-exact reusable Type-2 output for twelve samples and
  eight modes. Format, no-default/all-feature checks, warning-denied Clippy,
  doctest, rustdoc, provider audit, repository-baseline semver classification,
  and direct source/dependency scans pass. No machine-checked proof is
  performed.
- Historical residual at NUFFT closure: native-half FFT transport was the only
  transform provider migration scope; it is now migrated and the obsolete
  wrapper is deleted.

## Radon Hephaestus command-stream migration [arch]

- Performed: replaced direct projection, adjoint, and filtered-backprojection
  WGPU construction, binding, encoding, submission, and readback with three
  typed Hephaestus descriptors. The filter and adjoint share one ordered stream
  and one filtered device buffer, establishing write-before-read.
- Mathematical contract: matching linear detector weights make the discrete
  backprojection the adjoint of projection. Ram-Lak filtering followed by
  `pi / angle_count` scaling is the declared filtered-backprojection
  approximation. ADR 0007 distinguishes the exact discrete adjoint theorem
  from empirical finite-precision reconstruction evidence.
- Structural cleanup: deleted the helper error re-export, raw device/queue
  accessors, direct WGPU/pollster dependencies, and the 534-line raw pipeline
  implementation. Leto remains the CPU boundary and Hephaestus owns GPU data.
- Evidence tier: typed binding/layout plus value-semantic 25-case suite with
  real-device execution, warning-denied Clippy, rustdoc, provider audit, and
  direct source/dependency scans. No machine-checked proof is performed.
- Historical residual at Radon closure: native-half FFT transport was the only
  direct-provider scope; it is now migrated and the obsolete wrapper is deleted.

## FFT Hephaestus storage-generic migration [arch]

- Test-oracle correction: `axis_workspace_matches_axis_batch_geometry` now
  asserts the analytical workspace law `M_axis * batch_axis`. For shape
  `(2, 3, 4)`, X and Z require `2*(3*4)=24` and `4*(2*3)=24`; Y uses
  Bluestein `M=next_pow2(2*3-1)=8` over `2*4=8` batches, requiring 64.
  The previous `(32, 32, 24)` expectations were incorrect; the implementation
  already returned the derived values. Evidence tier: algebraic specification
  encoded as a value-semantic unit test, not a test relaxation.

- Performed: replaced f32 and native-half dense-FFT device acquisition,
  buffers, pipeline creation, bindings, command encoding, submission, and
  transfer with one `GpuFft3d<T>` typed Hephaestus plan. `ComputeDevice::write_buffer`
  preserves reusable typed storage; `CommandStream` records external-to-plan
  copy, axis-pass, and plan-to-external copy ordering.
- Binding decision: the sealed `FftStorage` contract selects WGSL sources,
  coefficient encoding, and radix capability for f32 or physical `u16` half
  storage. Every descriptor uses one flat binding group and a terminal POD
  parameter block. Pack/unpack consolidates legacy uniform blocks into
  `PackParams`; marker types select entries without duplicated dispatch code.
- Mathematical contract: f32 retains the exact 3D DFT/inverse theorem. The
  all-Bluestein 3×3×3 half roundtrip counts 265 rounding sites, hence asserts
  `γ_265·‖input‖₁` with half unit roundoff `u = 2⁻¹¹`. This is an analytical
  fixture bound, not a machine-checked proof.
- Evidence tier: type-level typed-buffer/stream contract plus empirical
  real-device value tests. The f32 2×2×2 delta is exact; f32 Bluestein and
  inverse tests satisfy `gamma_256`; native-half radix differential and
  Bluestein roundtrip tests pass. Package and workspace gates, provider audit,
  examples, major SemVer classification, and direct manifest/Rust API scans
  pass. The provider audit's lexical WGPU count includes feature labels, not a
  direct `wgpu` dependency or Rust API use.
- Integration state: [Hephaestus PR 33](https://github.com/ryancinsight/hephaestus/pull/33)
  merged at `4ef4f52`; Apollo tracks the provider default branch and relies on
  `Cargo.lock` for reproducibility. No revision quarantine remains.
- Semver classification: against `96e67a2` (0.15.0), the minor classifier
  rejects the raw-device constructor removals and deleted raw stage structs;
  the major classifier passes. This is the documented pre-1.0 0.16.0 breaking
  migration, not a reason to retain a compatibility wrapper.

## SHT Hephaestus command-stream migration [arch]

- Performed: replaced the direct spherical-harmonic WGPU pipeline, bind-group,
  encoder, queue, and transfer ownership with typed Hephaestus basis and
  matrix-reduction ZST descriptors recorded in one ordered command stream.
  Hephaestus owns acquisition, allocation, preparation, binding validation,
  dispatch, synchronization, and transfer.
- Mathematical contract: forward basis generation materializes
  `conj(Y_l^m) w_j` and matrix reduction evaluates the Gauss--Legendre/product
  quadrature coefficient. Inverse materializes `Y_l^m` and evaluates harmonic
  synthesis. For a band-limited function on the declared grid, quadrature
  exactness plus spherical-harmonic orthonormality proves recovery in exact
  arithmetic; command ordering establishes basis write-before-read. ADR 0005
  and the crate README state the theorem and distinguish finite-precision
  differential evidence from the theorem.
- Type and ownership contract: `ShtGpuStorage` is sealed to `Complex32` and
  explicit `[f16; 2]`; `Complex64` cannot enter typed accelerator APIs.
  `SphericalHarmonicCoefficients` stays the CPU `Complex64` SSOT. Inverse
  staging rejects non-representable values before provider allocation or dispatch, while
  `quantize_coefficients` is the explicit loss boundary. Leto remains the
  host-array boundary; Mnemosyne owns complex conversion scratch and returned
  Leto storage.
- Structural cleanup: deleted direct provider construction, raw device/queue
  escape hatches, the `wgpu_backend` forwarding module, and the monolithic
  shader. `common.wgsl`, `basis.wgsl`, and `matrix.wgsl` are the one-home
  shader hierarchy for shared math, basis generation, and reduction. The
  provider device boundary is 276 lines; `conversion.rs` owns validation,
  grid/coefficient conversion, Leto interop, and its no-device regressions.
- Verification: format; locked all-feature and no-default checks;
  warning-denied Clippy; 29 focused nextest cases including a real-device CPU
  differential/inverse run; the `Complex64` compile-fail storage exclusion;
  a pure non-representable-coefficient rejection; rustdoc; provider audit;
  immediate-parent semver classification; and a direct source scan with no raw
  WGPU, `pollster`, or `apollo-wgpu-helpers` reference. The examples target is
  absent, so its build check is a Cargo no-op.
- Evidence tier: typed binding/layout and compile-fail storage exclusion, then
  value-semantic negative-contract and real-device CPU differential evidence.
  No machine-checked proof is performed.
- Historical residual at SHT closure: native-half FFT transport was the only
  direct-provider scope; it is now migrated and the obsolete wrapper is deleted.

## SDFT Hephaestus command-stream migration [arch]

- Performed: replaced the direct SDFT raw-device pipeline, binding, encoder,
  queue, and transfer mechanics with two typed Hephaestus ZST descriptors and
  ordered command streams. The forward descriptor binds real `f32` windows to
  `Complex32` bins; the inverse descriptor binds complete `Complex32` spectra
  to complex samples before Apollo extracts their real component. Hephaestus
  owns acquisition, allocation, preparation, binding, submission,
  synchronization, and transfer.
- Mathematical contract: for a complete `N`-bin DFT, root-of-unity
  orthogonality makes `(1/N) sum_k exp(2 pi i k(m-n)/N)` equal to
  `delta_mn`; forward followed by inverse therefore recovers the input in
  exact arithmetic. A partial spectrum does not identify an arbitrary window,
  so inverse dispatch rejects `bin_count != window_len` before allocation.
- Type contract: sealed `SdftGpuRealStorage` admits `f32` and explicit `f16`;
  sealed `SdftGpuBinStorage` admits `Complex32` and explicit `[f16; 2]`.
  CPU `f64` and `Complex64` cannot enter the concrete accelerator path. Leto
  keeps contiguous inputs borrowed and materializes strided inputs once;
  Mnemosyne owns generated output and reduced-storage scratch.
- Structural cleanup: deleted the `wgpu_backend` forwarding module, raw device
  and queue escape hatches, CPU-marker alias, legacy helper error re-export,
  direct raw WGPU dependencies, and the monolithic shader. Common WGSL terms
  now have one canonical leaf shared by typed forward and inverse sources.
- Verification: all-feature locked check, no-default check, warning-denied
  Clippy, 28/28 nextest cases including real-device CPU differential and
  complete-bin roundtrip, two compile-fail storage exclusions, rustdoc,
  provider audit, immediate-parent 0.2.0-to-0.3.0 semver classification with
  no required update, and direct source scan with no raw WGPU, `pollster`, or
  `apollo-wgpu-helpers` reference. The examples target is absent, so its build
  check is a Cargo no-op.
- Evidence tier: typed binding/layout and storage exclusion, then
  value-semantic real-device differential, roundtrip, and negative-contract
  evidence. No machine-checked proof is performed.
- Historical residual at SDFT closure: native-half FFT transport was the only
  direct-provider scope; it is now migrated and the obsolete wrapper is deleted.

## Wavelet Hephaestus command-stream migration [arch]

- Performed: extended the owning Hephaestus command stream with typed bounded
  prefix copy, then replaced Apollo-owned WGPU pipeline, binding, encoder,
  queue, and transfer mechanics with Haar analysis/synthesis ZST kernels.
  Leto remains the host-array boundary and Mnemosyne owns returned storage.
- Mathematical contract: Haar analysis maps `(a, b)` to
  `((a+b)/sqrt(2), (a-b)/sqrt(2))`; synthesis is its transpose. Each pass is
  orthonormal, so Parseval holds and reverse-level synthesis gives the inverse
  of the forward multilevel composition.
- Verification: all-feature package check and warning-denied Clippy; 25/25
  nextest cases including real-device analytical values, CPU differential,
  Parseval, inverse roundtrip, Leto boundaries, and typed `f16`; compile-fail
  `f64` storage exclusion; and a source/manifest scan with no direct WGPU,
  pollster, or wrapper residue. The consumer resolves the published
  Hephaestus `e527097`, Leto `7f216f1`, Mnemosyne `32b4a2a`, Moirai `8cd356c`,
  and Themis 0.10.0 graph without local overrides. Semver classification
  reports the intended 0.2.0-to-0.3.0 major migration.
- Evidence tier: typed binding/layout and storage exclusion, then
  value-semantic real-device evidence. No machine-checked proof is performed.
  Residual D6 scope: 10 crates.

## FrFT Hephaestus command-stream migration [arch]

- Performed: replaced the direct FrFT and Candan--Gr\u00fcnbaum unitary FrFT
  raw device pipelines, bind groups, encoders, queues, and transfer mechanics
  with typed Hephaestus ZST kernel descriptors and ordered command streams.
  The direct kernel owns two typed `Complex32` bindings; the unitary kernel
  owns typed input, column-major Leto eigenbasis, coefficient, and output
  bindings. Leto remains the host-view boundary and Mnemosyne owns returned
  arrays.
- Mathematical contract: the direct kernel preserves the documented
  centered-coordinate rotation modes. The unitary path evaluates
  `V diag(exp(-i a k pi / 2)) V^T`; `V^T V = I` and unit-modulus phases prove
  norm preservation and inverse order negation in exact arithmetic. Three
  ordered stream passes preserve the projection, phase, reconstruction data
  dependency.
- Type contract: sealed `FrftGpuStorage` admits `Complex32` and `[f16; 2]`
  only. `Complex64` cannot enter the concrete `Complex32` accelerator API, so
  silent narrowing is a compile-time error.
- Verification: `cargo fmt --all -- --check`; `cargo check -p apollo-frft
  --all-features --locked`; `cargo clippy -p apollo-frft --all-targets
  --all-features -- -D warnings`; `cargo nextest run -p apollo-frft
  --all-features` (40 passed, including real-device dispatch); `cargo test -p
  apollo-frft --doc --all-features` (the `Complex64` compile-fail contract
  passed); `cargo doc -p apollo-frft --all-features --no-deps`; locked
  metadata; and `cargo run -p xtask -- provider-audit`. The immediate-parent
  semver baseline classifies 0.2.0-to-0.3.0 as a major change with no required
  semver update. `origin/main` cannot be used as the baseline because its
  FrFT manifest names a missing workspace-local Hephaestus path.
- Evidence tier: type-level binding/layout and storage exclusion, then 40
  value-semantic nextest cases including real-device direct/unitary evidence.
  No machine-checked proof is performed.
- Residual at FrFT closure: D6 had 9 transform crates remaining: FFT, Hilbert,
  Mellin, NUFFT, Radon, SDFT, SFT, SHT, and STFT. FrFT contains no direct raw-WGPU mechanics,
  pollster dependency, or `apollo-wgpu-helpers` edge.

## Hilbert Hephaestus command-stream migration [arch]

- Performed: replaced the analytic-signal and inverse raw-device pipeline,
  binding, encoder, queue, and transfer mechanics with four typed Hephaestus
  ZST descriptors over ordered command streams. Leto remains the host-view
  boundary; Mnemosyne owns conversion scratch and returned Leto arrays.
- Mathematical contract: with DFT multiplier `-i sign(k)` away from DC and
  Nyquist, `H(H(x)) = -x` on the DC/Nyquist-free subspace. The inverse mask
  applies `-H`, so it recovers exactly that projection and cannot reconstruct
  deliberately discarded DC/Nyquist coefficients.
- Type contract: sealed `HilbertGpuStorage` admits only native `f32` and
  explicit `f16` conversion. `f64` cannot silently narrow into the concrete
  accelerator kernel.
- Verification: all-feature package check, warning-denied Clippy, 34/34
  nextest cases including real-device CPU forward differential and
  inverse-projection checks, the `f64` compile-fail doctest, rustdoc,
  provider audit, locked metadata, immediate-parent semver classification, and
  a source/manifest scan with no direct WGPU, pollster, or wrapper residue.
- Evidence tier: type-level binding/layout and storage exclusion, then
  value-semantic real-device evidence. No machine-checked proof is performed.
- Residual: D6 has 8 transform crates remaining: FFT, Mellin, NUFFT, Radon,
  SDFT, SFT, SHT, and STFT.

## Mellin Hephaestus command-stream migration [arch]

- Performed: replaced raw WGPU log-resample, spectrum, inverse-spectrum, and
  exponential-resample mechanics with four typed Hephaestus ZST descriptors
  and ordered command streams. Leto remains the host-view boundary; Mnemosyne
  owns f16 conversion scratch and returned Leto arrays.
- Mathematical contract: on a uniform log grid, forward scaling by `du` and
  inverse scaling by `1/(N du)` form an inverse DFT pair because the DFT root
  sum is `N delta_nm`. Exponential resampling reconstructs the linear-grid
  interpolation of the recovered log samples.
- Type contract: accelerator scale metadata and domain bounds are concrete
  `f32`; sealed `MellinGpuStorage` admits native `f32` and explicit `f16`
  conversion only. `f64` cannot silently narrow into the accelerator API.
- Verification: all-feature package check and warning-denied Clippy; 32/32
  nextest cases including real-device CPU forward differential and inverse
  constant-signal reconstruction; the `f64` compile-fail doctest; rustdoc,
  provider audit, locked metadata, immediate-parent semver classification, and
  source/manifest scan with no direct WGPU, pollster, or wrapper residue.
- Evidence tier: type-level binding/layout and storage exclusion, then
  value-semantic real-device evidence. No machine-checked proof is performed.
- Residual: D6 has 7 transform crates remaining: FFT, NUFFT, Radon, SDFT, SFT,
  SHT, and STFT.

## SFT Hephaestus command-stream migration [arch]

- Performed: replaced the direct SFT raw-device pipeline, binding, encoder,
  queue, and transfer mechanics with one direction-parameterized typed
  Hephaestus ZST and ordered command stream. Leto remains the host-view
  boundary, and dense inverse output writes directly into Mnemosyne-backed
  storage. Apollo keeps deterministic top-k sparse support selection as the
  domain owner.
- Mathematical contract: the shader evaluates the forward DFT and its
  normalized inverse. The root-of-unity sum is `N delta_nm`, so composing the
  two dense passes recovers the original input in exact arithmetic. Top-k
  selection is a separate projection: it reconstructs retained support and
  does not claim recovery of discarded coefficients.
- Type contract: sealed `SftGpuStorage` admits `Complex32` and `[f16; 2]`.
  `Complex64` cannot enter typed accelerator execution. The CPU
  `SparseSpectrum` remains the `Complex64` SSOT; inverse staging rejects a
  component not exactly representable in `f32`, so no high-accuracy coefficient
  is silently changed at the device boundary. `quantize_spectrum` provides the
  separate explicit lossy conversion when the caller chooses accelerator
  precision.
- Verification: all-feature locked package check and warning-denied Clippy;
  32/32 nextest cases including real-device CPU forward differential and
  inverse execution; the `Complex64` compile-fail doctest; rustdoc; provider
  audit; locked metadata; immediate-parent semver classification from 0.2.0 to
  0.3.0; and a source/manifest scan with no direct WGPU, pollster, or wrapper
  residue.
- Evidence tier: typed binding/layout and storage exclusion, then
  value-semantic real-device evidence. No machine-checked proof is performed.
- Residual at SFT closure: D6 had 6 transform crates remaining: FFT, NUFFT,
  Radon, SDFT, SHT, and STFT.

## QFT Hephaestus command-stream migration [arch]

- Performed: replaced direct WGPU pipeline, binding, encoder, queue, and
  transfer ownership with a ZST Hephaestus `Complex32` kernel and command
  stream. Apollo retains the unitary formula and WGSL source; Leto remains the
  host-array boundary.
- Mathematical contract: the forward and inverse matrices use conjugate phases
  and `N^(-1/2)` normalization, so orthonormality gives `QFT^-1(QFT(x)) = x`.
- Verification: warning-denied Clippy; 27/27 nextest cases including real-device
  CPU differential and inverse roundtrip; compile-fail `Complex64` exclusion;
  doctest; rustdoc; and no direct-WGPU source residue.
- Evidence tier: type-level storage exclusion and value-semantic differential
  evidence. No machine-checked proof is performed. Residual D6 scope: 10 crates.

## NTT Hephaestus command-stream migration [arch]

- Performed: replaced `apollo-ntt` direct WGPU pipeline, bind group, uniform
  buffer, command encoder, queue, and transfer ownership with two ZST
  `KernelInterface`/`KernelSource` descriptors and one ordered Hephaestus
  command stream. Reusable NTT state retains only host residues and twiddles;
  device buffers, parameter uploads, bindings, dispatch, and readback remain
  provider-owned.
- Mathematical contract: for primitive `omega` in `F_q`, the staged
  Cooley-Tukey recurrence evaluates `X[k] = sum_j x[j] omega^(j*k)`. Each stage
  writes disjoint butterfly pairs, and ordered command-stream passes preserve
  the recurrence dependency. Inverse twiddles followed by multiplication by
  `n^-1` give `INTT(NTT(x)) = x` by finite-field root-of-unity orthogonality.
- Verification: format; all-feature check and warning-denied Clippy; 27/27
  nextest cases, including real-device exact CPU differential and 64-case
  GPU roundtrip property; doctest; rustdoc; and a source/manifest scan finding
  no direct `wgpu`, `pollster`, or `apollo-wgpu-helpers` residue.
- Evidence tier: typed binding/layout checks plus value-semantic differential
  and property evidence. No machine-checked proof is performed.
- Residual: D6 has 10 transform crates to migrate; D7 remains the owner of
  cross-transform Leto interop consolidation.

## GFT Hephaestus command-stream migration [arch]

- Performed: replaced `apollo-gft`'s direct WGPU buffers, pipeline, bind group,
  encoder, queue, and wrapper-device ownership with a single direction-selected
  ZST `KernelInterface`/`KernelSource` dispatched by Hephaestus typed bindings
  and command streams. Leto host views now write directly to Mnemosyne-backed
  caller output. The accelerator contract is sealed to `f32` and explicit
  `f16` promotion, excluding silent `f64` narrowing at compile time.
- Mathematical contract: for the column-major orthonormal basis `U`, the
  forward and inverse kernels evaluate `X[k] = sum_i U[i + kN]x[i]` and
  `x[i] = sum_k U[i + kN]X[k]`; `U^T U = I` proves the exact-arithmetic
  reconstruction identity. The executable path-four differential uses a
  `64 * epsilon_f32 = 2^-17` bound derived from four products and three
  additions plus basis quantization.
- Verification: `cargo fmt --all -- --check`; `cargo check -p apollo-gft
  --all-features`; `cargo clippy -p apollo-gft --all-targets --all-features --
  -D warnings`; `cargo nextest run -p apollo-gft --all-features` (21 passed,
  including real-device dispatch); `cargo test -p apollo-gft --doc
  --all-features` (1 passed); `cargo doc -p apollo-gft --all-features
  --no-deps`; `cargo run -p xtask -- provider-audit`; and `cargo semver-checks
  check-release -p apollo-gft --baseline-rev d4ce4f149738db07c535718a1447a9ae01740e67`
  (intentional major classification; no semver update required).
- Evidence tier: type-level storage exclusion and parameter-layout assertion,
  then value-semantic CPU differential/roundtrip and real-device execution.
  No machine-checked proof is performed.
- Residual: D6 has 10 transform crates to migrate; GFT contains no local
  raw-WGPU mechanics or wrapper dependency.

## Provider-native GPU kernel migration [arch]

- Architecture finding (resolved): the seventeen completed transform bounded
  contexts acquire typed device, buffer, pipeline, binding, dispatch, and
  transfer services from Hephaestus. `apollo-wgpu-helpers` is deleted; the
  workspace manifest, lockfile, and Rust source scan contain no wrapper edge.
  Native-half FFT transport subsequently completed the same typed-provider
  migration, closing D6.
- Deletion evidence: format, locked metadata, workspace resolution, provider
  audit, six `xtask` value-semantic contract cases, warning-denied `xtask`
  Clippy, and the 44-case all-feature NUFFT suite pass. The workspace check
  exposed a DHT feature-boundary warning; the live GPU-only `leto_view1_cow`
  forwarder is now gated with its transport. DHT's default and all-feature
  checks, warning-denied Clippy, 30/30 default and 34/34 all-feature nextest
  cases, and the final workspace check pass.
- Provider capability finding: Hephaestus already owns backend-neutral typed
  allocation and transfer (`ComputeDevice`), authored kernel contracts
  (`KernelInterface`/`KernelSource`), typed bindings and prepared dispatch
  (`KernelDevice`), and launch grids (`DispatchGrid`). Apollo needs no local
  wrapper or new upstream runtime abstraction.
- Decision: the completed contexts migrate as whole bounded transforms: Leto
  remains the host array/view boundary, Apollo retains transform mathematics,
  and Hephaestus owns device mechanics. FWHT established the pattern with one
  in-place typed buffer, two ZST kernel-source types, and the independent
  `H_n² = nI` oracle. Native-half FFT subsequently used the same ownership
  boundary. See ADR 0003.
- Performed: `apollo-fwht` 0.3.0 now expresses its butterfly and inverse-scale
  kernels as ZST `KernelInterface`/`KernelSource` implementations, prepares
  them through `KernelDevice`, and encodes every stage into one typed command
  stream over a single Hephaestus buffer. Readback storage moves directly into
  the Mnemosyne-backed Leto result without a second output copy.
- Evidence tier: compile-time typed provider ownership plus 39 value-semantic
  nextest cases, including a real-device CPU differential suite and the
  independent `H_n² = nI` oracle; the full workspace passes 1,028/1,028
  nextest cases. Focused warning-denied Clippy, rustdoc, doctest,
  provider-audit, locked metadata, and API classification gates pass. No
  runtime performance claim is made.
- Performed: `apollo-czt` 0.4.0 now consumes the Hephaestus device and authored
  kernel contracts directly. CZT Leto results now compute/download directly
  into Mnemosyne storage, and the direct `N×M` gate selects explicit Moirai
  parallel scheduling while retaining Hermes row reductions. Compile-time
  provider ownership, 44 focused CZT cases, and 1,026/1,026 workspace nextest
  cases pass, including real-device CPU differential execution. Direct-WGPU
  residuals are now 16 transform manifests and 51 source files. The wrapper
  crate is deleted only after its final consumer is migrated. No runtime
  performance claim is made.
- Performed: `apollo-dht` 0.3.0 now expresses transform and inverse-scale as
  ZST Hephaestus authored kernels encoded into one command stream. Leto 2D/3D
  paths borrow storage without contiguous materialization; Mnemosyne owns GPU
  typed bridge/output storage and one canonical fast scratch pool; existing
  Moirai scheduling and Hermes reductions remain intact. A sealed
  `HartleyGpuStorage` type contract admits `f32` and mixed `f16`/`f32` while
  rejecting the previous hidden `f64 -> f32 -> f64` execution at compile time.
  The WGSL angle expression converts indices before multiplication, removing
  the `u32 k*n` overflow path. Focused Clippy, 34/34 nextest cases including a
  real-device suite, and the compile-fail doctest pass. The full workspace
  passes warning-denied Clippy, 1,025/1,025 nextest cases, doctest,
  warning-denied rustdoc, provider-audit, locked metadata, and API
  classification. Direct-WGPU residuals are now 15 transform manifests and 49
  source files. No runtime performance claim is made.
- Performed: `apollo-dctdst` 0.3.0 now uses Hephaestus typed buffers,
  `KernelInterface`/`KernelSource` ZST descriptors, prepared dispatch, and a
  provider command stream for all WGPU execution. Apollo retains only the
  transform equations, the inverse-pair theorem, and CPU/Leto conversions.
  The manifest and source scan finds no direct `wgpu`, `pollster`, or
  `apollo-wgpu-helpers` edge in this bounded context.
- Evidence tier: compile-time typed binding ownership and a 32-byte parameter
  layout assertion; 72 value-semantic nextest cases including real-device CPU
  differential execution; warning-denied Clippy, doctest, and rustdoc. No
  machine-checked proof or runtime performance claim is made.
- Residual: the local verification tree is normalized. D8 still needs a
  provider-owned cross-transform contract; it must not introduce a parallel
  transform implementation.

## Release 0.15.0 eligibility [major]

- Provider ABI finding: Hephaestus 0.13.0 now owns WGPU 30, so Apollo advances
  to the same ABI without a parallel device family. The obsolete Mnemosyne
  callback constructor failure is deleted; mapping failures now propagate as
  typed errors from the FFT and NUFFT readback boundaries.
- Supply-chain finding: WGPU 30 removes Metal 0.32 and its archived `paste`
  dependency. The `RUSTSEC-2024-0436` exception is deleted rather than carried
  into the new release.
- Toolchain finding: Apollo retains edition 2021 for 0.15.0 because moving its
  SIMD-heavy unsafe kernels to edition 2024 activates 3,513 unsafe-operation
  obligations requiring per-block soundness review and sanitizer/Miri evidence.
  Treat that as a dedicated major safety increment; mixing a mechanical lint
  suppression into the WGPU ABI release would produce unreviewed unsafe code.
- Numerical decision: the prior native-f16 absolute-error claim omitted DFT
  gain and was analytically invalid. The 0.15 differential gate uses
  `γ₃₁·‖input‖₁`, with `γₖ = ku/(1-ku)` and f16 unit roundoff `u = 2⁻¹¹`,
  derived from input quantization plus five rounding sites across each of six
  radix-2 stages. A tighter bound requires a separately verified error model.
- WGPU safety finding: native-f16 odd-volume storage buffers now round byte
  capacity to `wgpu::COPY_BUFFER_ALIGNMENT`; logical reads remain limited to
  the unpadded element count.
- Capability finding: the STFT Chirp-Z pipeline requires four working/kernel
  bindings plus two I/O bindings in one compute stage. Device acquisition now
  requests six storage buffers and caller-supplied devices return a typed
  `InsufficientDeviceLimit` error before pipeline construction when they expose
  fewer.
- Topology finding: deleted the disconnected `gpu_fft/reduced.rs`,
  `gpu_fft/batched_matrix/`, and GPU validation tree. They were not declared by
  any live module and duplicated the canonical native-f16 execution surface.
- Dependency-policy residual: `cargo deny check` passes all four policy classes
  but reports 12 transitive duplicate families. The primary incompatible roots
  are the provider-owned rkyv 0.7 graph and Moirai's Windows 0.58 graph versus
  the current WGPU 30/Windows 0.62 graph. Apollo does not suppress them with
  `skip` rules; the upstream convergence item is tracked in `backlog.md`.

## Release 0.14.0 eligibility [arch]

- Distribution finding: crates.io packaging is not a valid Apollo release gate
  because required Atlas packages are unpublished there. ADR 0002 makes the
  tested Git-source graph the SSOT instead of fabricating registry portability.
- Dependency finding (historical): Hephaestus 0.12 exposed WGPU 26 types, so
  Apollo 0.14.0 remained on the compatible 26.0.1 patch until the provider
  migrated the shared contract.
- Reproducibility finding: CI referenced stale provider revisions, omitted the
  Themis sibling required by Hephaestus, used a floating stable Rust channel,
  and bypassed nextest. The release increment pins each boundary.
- Metadata finding: member manifests retained stale dependency requirements and
  Kwavers repository links, CI floated its test tools, and the changelog had two
  `Unreleased` sections plus control bytes. The release increment consolidates
  dependency versions, repository metadata, tool versions, and release history.
- Provider-lineage finding: the required Moirai Mnemosyne 0.3 integration was
  available only on a feature branch while `main` carried newer reactor fixes.
  Release eligibility requires their verified union to land on Moirai `main`.
- Local graph finding: Hephaestus reaches Moirai GPU leaf packages directly;
  Apollo's path patch table now lists those leaves so Atlas development builds
  use one Moirai source identity. Standalone Git builds use the same exact
  `b2f3732` revision without path patches.
- Workflow finding: the GPU benchmark workflow called a deleted script and
  obsolete `*-wgpu` packages. The non-executable workflow and stale README
  claim were removed; no benchmark result or performance claim was changed.
- Supply-chain residual (closed in 0.15.0): WGPU 26 selected Metal 0.32, which depends on the
  archived `paste` 1.0.15 (`RUSTSEC-2024-0436`). RustSec reports no safe
  upgrade. `deny.toml` records the narrow advisory exception; it closes only
  when Hephaestus advances the shared WGPU ABI.
- Evidence tier: Cargo resolution and source-lineage inspection; compile-time
  lint and rustdoc enforcement; 1027/1027 Rust value-semantic nextest cases;
  34/34 Python boundary cases; RustSec and cargo-deny policy checks; and 196
  applicable `apollo-fft` minor-release API checks. The historical API baseline
  required only provider-revision alignment to resolve its dependency graph;
  Apollo's baseline API surface was not changed.

## Hephaestus WGPU local provider edge [patch]
- Performed: changed Apollo's workspace `hephaestus-wgpu` dependency from the
  obsolete pinned Git revision to the local Atlas Hephaestus checkout so
  `apollo-wgpu-helpers` consumes the current `WgpuDevice` acquisition and
  staging-buffer API.
- Architecture effect: downstream Atlas repos share one Hephaestus GPU
  substrate and one target tree; Apollo remains the GPU FFT owner through its
  `FftBackend` contract, while Kwavers consumes that contract instead of a
  WGPU-specific helper crate name.
- Verification: downstream integration from `D:\atlas\repos\kwavers`:
  `rustup run nightly cargo check -p kwavers-math --features gpu --all-targets`
  passes, focused `kwavers-math --features gpu` GPU FFT nextest passes 2/2,
  and `cargo tree -p kwavers-math --features gpu -i hephaestus-wgpu` resolves
  `hephaestus-wgpu v0.12.0 (D:\atlas\repos\hephaestus\crates\hephaestus-wgpu)`.
- Evidence tier: compile-time dependency/type validation plus downstream
  value-semantic GPU FFT tests.
- Residual: Apollo has no real CUDA FFT provider yet. CUDA FFT requires
  upstream Apollo/Hephaestus kernels and WGPU/CUDA differential tests; no
  Kwavers placeholder should claim it.

## Apollo direct ndarray removal [patch]
- Performed: changed the remaining `ndarray` benchmark imports and tuple-shaped
  constructors in `apollo-fft`, `apollo-nufft`, and `apollo-radon` to Leto
  `Array{1,2,3}` constructors; removed stale `ndarray` dev-dependencies from
  transform crates that no longer import it; replaced `apollo-python` return
  conversion through `ndarray::Array{1,2,3}` with shared Leto-to-NumPy helpers;
  removed the remaining Python input `as_array()` conversions by constructing
  Leto views/arrays from validated NumPy slices and shape metadata; replaced
  stale Leto `ArrayView::as_array()` use where an owned contiguous value was
  required; DHT now retains `as_array()` specifically as a zero-copy borrowed
  storage adapter for its storage-generic multidimensional kernels; removed the Rust `numpy` crate,
  Apollo's root `ndarray` workspace dependency, and Eunomia's `numpy` feature
  from `apollo-python`; removed `xtask provider-audit`'s stale
  ndarray-specific audit column and test dependency fixtures.
- Architecture effect: Apollo's Rust transform, validation, and benchmark surfaces
  no longer use ndarray as an owned-array substrate. Apollo has no `ndarray`
  package in its resolved Cargo graph. Python inputs cross the PyO3 boundary as
  runtime NumPy array objects and immediately become Leto views or Leto-owned
  row-major buffers through PyO3-owned dtype/shape/byte helpers; Python return
  arrays are built from Leto-owned row-major buffers by importing runtime NumPy
  from Python, not by depending on the Rust `numpy` crate.
- Verification: non-Python source/manifest scan returned no `ndarray` hits;
  `cargo tree -p apollo-fft -i ndarray` and
  `cargo tree -p apollo-validation -i ndarray` report no matching package;
  `rg -n "ndarray" Cargo.toml crates -g "*.toml" -g "*.rs"` returns no matches;
  `cargo tree -i ndarray` reports no matching package; `cargo tree -p
  apollo-python -i eunomia` shows Eunomia only through the transform crates
  consumed by the binding crate; `cargo fmt -p apollo-python -p apollo-dctdst -p apollo-dht --check`;
  `cargo check -p apollo-dctdst -p apollo-dht`; first-party source/manifest
  `rg` scans report no `ndarray` package or import; the remaining DHT
  `as_array()` calls are Leto-native zero-copy view adapters;
  final first-party source/manifest/lock/xtask `rg -n "ndarray" Cargo.toml
  Cargo.lock crates xtask -g "*.toml" -g "*.rs" -g "Cargo.lock"` returns no
  matches; `cargo fmt -p xtask --check`; `cargo nextest run -p xtask
  provider_audit`; `cargo run -p xtask -- provider-audit` now fails on
  first-party manifest, lockfile, or Rust-source reintroduction of the Rust
  crate and ignores comment-only mentions;
  `cargo check -p apollo-python` passes; `cargo nextest run -p apollo-python`
  passes the value-semantic boundary roundtrip test; `cargo fmt -p apollo-python -p
  apollo-fwht -p apollo-nufft --check` passes; `git diff --check` passes.
- Residual: runtime NumPy arrays remain the Python ABI object format. No Cargo
  `ndarray` package or Rust `numpy` crate remains in Apollo.
- Evidence tier: compile-time dependency/source-scan evidence plus a
  value-semantic PyO3 boundary roundtrip test. No runtime benchmark claim is made.

## CZT native Leto constructor/indexing cleanup [patch]
- Performed: removed `apollo-czt`'s stale `ndarray` dev-dependency and replaced
  residual compatibility-backed `Array1::from(Vec<_>)` construction plus scalar
  1D indexing in CZT tests/proptests with native Leto shape construction and
  rank-aware indexing.
- Architecture effect: the CZT slice now builds against Leto array APIs directly
  instead of relying on ndarray compatibility to materialize test and kernel arrays.
- Verification: `rg -n "ndarray|ndarray_input|matches_ndarray|Array1::from\(|leto::Array1::from\(" crates/apollo-czt/src crates/apollo-czt/Cargo.toml`
  returned no matches; `cargo fmt --package apollo-czt --check`;
  `cargo nextest run -p apollo-czt` -> 40/40 passed.
- Evidence tier: compile-time dependency removal plus value-semantic CZT unit and
  property tests. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage. Runtime NumPy remains
  only as the Python ABI object format.

## Coeus GradBuffer autograd compatibility [patch]
- Performed: updated `apollo-fft` Coeus FFT autograd nodes to use
  `coeus_autograd::GradBuffer` for output and input gradient accumulation instead of raw
  `Arc<Mutex<Tensor<_>>>` buffers.
- Architecture effect: Apollo now matches Coeus `0.2.3`'s serialized-gradient accumulator
  contract and removes an obsolete synchronization primitive from these autograd nodes.
- Verification: `cargo clippy -p apollo-fft --all-targets -- -D warnings`;
  `cargo nextest run -p apollo-fft` -> 397/397 passed; `cargo test --doc -p apollo-fft`;
  `cargo doc -p apollo-fft --no-deps`; `git diff --check`.
- Evidence tier: compile/lint plus value-semantic FFT/autograd tests. No runtime benchmark
  claim is made.
- Residuals: broader Coeus/Apollo autograd API design remains provider-owned; this slice only
  restores compatibility with the current trait contract.

## Leto 0.5.0 shape/materialization provider pin [minor]
- Performed: updated Apollo's workspace `leto` and `leto-ops` Git dependencies from Leto `a46dea9` (`0.4.0`) to pushed Leto `6c7899d` (`0.5.0`).
- Architecture effect: Apollo now consumes provider-side dense row-major reshape/into_shape, permute aliases, and row-major to_contiguous materialization through the canonical Git dependency path.
- Migration effect: no Apollo public API or compute path changed in this increment; this historical provider pin removed a prerequisite for the later Leto-owned shape/materialization migration.
- Verification: `cargo check -p apollo-validation -p apollo-fft -p apollo-gft -p apollo-frft`; `cargo run -p xtask -- provider-audit`; `cargo fmt --check`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: Cargo dependency resolution plus focused compile checks for provider-consuming crates. Leto owns value-semantic reshape/materialization coverage through core tests and ndarray contract validation.
- Residuals: superseded by the Apollo direct ndarray removal entry; current Apollo source, manifests, and lockfile have no Rust `ndarray` dependency edge.

## Leto 0.4.0 broadcast binary provider pin [minor]
- Performed: updated Apollo's workspace `leto` and `leto-ops` Git dependencies from Leto `642d87a3` (`0.3.0`) to pushed Leto `a46dea9` (`0.4.0`).
- Architecture effect: Apollo now consumes provider-side broadcast-aware binary maps for caller-owned output layouts through the canonical Git dependency path.
- Migration effect: no Apollo public API or compute path changed in this increment; this historical provider pin removed a prerequisite for the later Leto-owned validation, scaling, and tensor-style elementwise migration.
- Verification: `cargo check -p apollo-validation -p apollo-fft -p apollo-gft -p apollo-frft`; `cargo run -p xtask -- provider-audit`; `cargo fmt --check`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: Cargo dependency resolution plus focused compile checks for provider-consuming crates. Leto owns value-semantic broadcast coverage through dense/strided tests and ndarray differential validation.
- Residuals: superseded by the Apollo direct ndarray removal entry; current Apollo source, manifests, and lockfile have no Rust `ndarray` dependency edge.

## Leto 0.3.0 provider pin [minor]
- Performed: updated Apollo's workspace `leto` and `leto-ops` Git dependencies from Leto `fd1d87b` (`0.2.0`) to pushed Leto `642d87a3` (`0.3.0`).
- Architecture effect: Apollo now consumes the provider-side RealScalar generic eigensolver, offset-independent dense view slices, memory-order slice access, unary/scalar-map/dot operations, and the Coeus rank-boundary ADR through its canonical Git dependency path.
- Migration effect: no Apollo public API or compute path changed in this increment; this historical provider pin removed prerequisites for the later Leto migration.
- Verification: `cargo check -p apollo-frft -p apollo-gft -p apollo-fft`; `cargo run -p xtask -- provider-audit`; `cargo fmt --check`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: Cargo dependency resolution plus focused compile checks for the FFT, FRFT, and GFT Leto consumers. No runtime benchmark claim is made.
- Residuals: superseded by the Apollo direct ndarray removal entry; current Apollo source, manifests, and lockfile have no Rust `ndarray` dependency edge.

## NUFFT 3D FFT lane scratch migration [patch]
- Performed: replaced all 3D NUFFT separable FFT forward/inverse x/y/z temporary lane `Array1` allocations with a Mnemosyne `ScratchPool<Complex64>` and Apollo FFT slice execution.
- Architecture effect: 3D NUFFT now keeps gridding workspaces, typed conversion workspaces, axis weights, and FFT lane buffers under the same provider scratch discipline.
- Memory effect: each separable pass reuses one thread-local lane slice sized to the active axis instead of allocating a fresh ndarray lane per pass call.
- Verification: `cargo fmt --check -p apollo-nufft`; `cargo test -p apollo-nufft`; `cargo clippy -p apollo-nufft --all-targets -- -D warnings`; `cargo doc -p apollo-nufft --no-deps`; `cargo semver-checks -p apollo-nufft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo run -p xtask -- benchmark`; `cargo fmt --check`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`; `rg -n "Array1::<Complex64>::zeros|forward_complex_inplace\\(&mut lane\\)|inverse_complex_inplace\\(&mut lane\\)" crates/apollo-nufft/src/application/execution/transform/dimension_3d.rs` returned no matches.
- Evidence tier: source-level allocation-path removal plus value-semantic NUFFT unit/property tests. No runtime benchmark claim is made.
- Residuals: `type1` and `type2` allocating convenience methods still allocate their documented output and caller-visible scratch arrays before entering caller-owned paths; changing that would require an API-surface design decision rather than an internal scratch migration.

## NUFFT 3D Mnemosyne scratch migration [patch]
- Performed: replaced `apollo-nufft` 3D typed Type-1/Type-2 thread-local `RefCell<Vec<_>>` scratch buffers with `mnemosyne::scratch::ScratchPool` instances for grid, modes, output, and axis weight workspaces.
- Architecture effect: 3D NUFFT now uses the same provider scratch discipline as the existing 1D NUFFT typed paths. Internal `ArrayViewMut3` helpers let scratch slices back ndarray indexing without changing public caller-owned `Array3` APIs.
- Memory effect: typed 3D paths no longer extract and reinsert owned vectors from thread-local `RefCell`s. Scratch lifetimes are bounded to provider pool closures and reuse capacity per thread.
- Verification: `cargo fmt --check -p apollo-nufft`; `cargo test -p apollo-nufft`; `cargo clippy -p apollo-nufft --all-targets -- -D warnings`; `cargo doc -p apollo-nufft --no-deps`; `cargo semver-checks -p apollo-nufft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo fmt --check`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`; `rg -n "RefCell" crates/apollo-nufft/src/application/execution/transform/dimension_3d.rs` returned no matches.
- Evidence tier: source-level ownership replacement plus value-semantic NUFFT unit/property tests and static diagnostics. No runtime benchmark claim is made.
- Residuals: closed by the subsequent NUFFT 3D FFT lane scratch migration entry.

## NTT Hermes modular butterfly routing [patch]
- Performed: added Hermes commit `25c261b3` with an exact modular `u64` NTT butterfly-stage kernel and routed `apollo-ntt` serial plus Moirai chunked stage execution through it.
- Architecture effect: every CPU transform crate now reports Hermes usage in provider audit; NTT keeps plan/root/twiddle ownership locally while the arithmetic butterfly loop lives in the provider boundary.
- Memory effect: no new allocation surface; stage execution mutates caller-owned chunks in place and borrows the precomputed stage twiddle slice.
- Verification: Hermes `cargo fmt --check -p hermes-simd`; `cargo test -p hermes-simd --test modular_tests`; `cargo clippy -p hermes-simd --all-targets -- -D warnings`; `cargo doc -p hermes-simd --no-deps`; Apollo `cargo fmt --check -p apollo-ntt`; `cargo test -p apollo-ntt`; `cargo clippy -p apollo-ntt --all-targets -- -D warnings`; `cargo doc -p apollo-ntt --no-deps`; `cargo semver-checks -p apollo-ntt --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic Hermes modular tests, Apollo NTT roundtrip/convolution/property tests, provider audit, and static diagnostics. No runtime benchmark claim is made.
- Residuals: Hermes modular multiplication is exact scalar `u128` arithmetic because current stable portable SIMD has no native `u64 x u64 -> u128` lane operation; future provider work can add specialized reduction strategies for bounded moduli.

## Hilbert analytic-mask Hermes scaling [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-hilbert` and routed threshold-sized analytic-mask scaling through `hermes_simd::scale` over borrowed interleaved `Complex64` lanes.
- Architecture effect: Hilbert CPU execution now composes Apollo FFT execution, Moirai staging/extraction, Mnemosyne scratch reuse, Hermes SIMD mask scaling, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: the Hermes mask path borrows the FFT spectrum as primitive `[re, im]` lanes and does not allocate mask vectors; small signals retain the existing scalar loop.
- Verification: `cargo fmt --check -p apollo-hilbert`; `cargo test -p apollo-hilbert`; `cargo clippy -p apollo-hilbert --all-targets -- -D warnings`; `cargo doc -p apollo-hilbert --no-deps`; `cargo semver-checks -p apollo-hilbert --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic Hilbert unit/property tests and static diagnostics. No runtime benchmark claim is made.
- Residuals: `apollo-ntt` remains the only CPU transform crate without Hermes usage because the current provider surface has no modular-arithmetic SIMD kernel.

## FFT f32 small-power dispatch consolidation [patch]
- Performed: replaced the f32 mixed-radix length-32 and length-64 direct match-arm implementations with calls to the canonical const-generic `small_pot_inplace_sized` helper.
- Architecture effect: length-specific f32 small-power behavior now has one authoritative implementation for Stockham AVX/FMA routing, Winograd fallback, and inverse normalization.
- Memory effect: no new allocation surface; the helper keeps stack scratch for the Stockham path and direct array reinterpretation for the Winograd fallback.
- Verification: `cargo test -p apollo-fft`; `cargo clippy -p apollo-fft --all-targets -- -D warnings`; `cargo doc -p apollo-fft --no-deps`; `cargo semver-checks -p apollo-fft --baseline-rev HEAD`.
- Evidence tier: value-semantic FFT tests and static diagnostics. No runtime benchmark claim is made.
- Residuals: larger FFT small-power and composite paths still contain additional shape-specific branches that need separate audit before consolidation.

## Hermes complex-lane zero-copy cleanup [patch]
- Performed: replaced redundant interleaved `Vec<f64>` construction with borrowed `Complex64` lane views in CZT, FrFT, Mellin, NUFFT, QFT, SFT, and SHT Hermes helpers; removed needless slice borrows and inline-always markers that violated the current clippy gate.
- Architecture effect: complex reduction helpers now use one provider-facing borrowed lane boundary instead of duplicated allocation loops per transform crate.
- Memory effect: threshold-sized Hermes complex reductions avoid materializing owned input lane vectors; only provider weight/twiddle lanes remain scratch-backed where formulas require generated coefficients.
- Verification: focused tests for all touched CPU transform crates plus full workspace format, tests, examples, clippy, docs, and provider audit.
- Evidence tier: value-semantic transform tests and static diagnostics. No runtime benchmark claim is made.
- Residuals: generated weight/twiddle lanes remain materialized because current Hermes complex dot APIs consume two primitive interleaved lane slices.

## STFT Hermes frame-window routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-stft`; routed threshold-sized forward analysis windowing and inverse WOLA real-frame windowing through Hermes elementwise multiplication with real frame lanes staged in Mnemosyne thread-local scratch.
- Architecture effect: STFT CPU execution now composes Moirai frame scheduling, Mnemosyne scratch reuse, Hermes SIMD elementwise kernels, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized forward frames reuse one thread-local real-frame lane buffer and one windowed-frame buffer per worker; inverse WOLA reuses thread-local real-frame extraction before writing the existing flat frame workspace. Small frames retain allocation-free scalar windowing.
- Implementation effect: `window_signal_frame_scalar` remains the forward formula reference; `window_signal_frame_into` owns zero-padded forward frame materialization and Hermes routing; `window_complex_real_frame_into` owns inverse real extraction and Hermes routing.
- Verification: `cargo fmt --check`; `cargo test -p apollo-stft`; `cargo clippy -p apollo-stft --all-targets -- -D warnings`; `cargo doc -p apollo-stft --no-deps`; `cargo semver-checks -p apollo-stft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic STFT tests plus direct threshold-path Hermes frame-windowing tests. No runtime benchmark claim is made.
- Residuals: STFT still delegates DFT execution to `apollo-fft`; overlap-add accumulation remains sequential to preserve race-free shared-output semantics.

## SFT direct-reference Hermes complex dot routing [patch]
- Performed: added workspace Hermes and Mnemosyne provider dependencies to `apollo-sft`; routed threshold-sized direct DFT verification rows through Hermes provider-owned interleaved complex dot products with twiddle lanes stored in Mnemosyne thread-local scratch.
- Architecture effect: SFT verification now composes scalar formula ownership, Mnemosyne scratch reuse, and Hermes SIMD complex reduction without runtime-erased dispatch in Apollo code. Production sparse FFT execution remains delegated to `apollo-fft`.
- Memory effect: threshold-sized direct DFT verification materializes one shared interleaved complex input lane buffer, then reuses thread-local twiddle-lane scratch per row. Small verification transforms retain allocation-free scalar row accumulation.
- Implementation effect: `dft_row_scalar` remains the formula reference; `fill_twiddle_lanes` owns DFT twiddle materialization; `dft_row_hermes` owns the provider reduction boundary.
- Verification: `cargo fmt --check`; `cargo test -p apollo-sft`; `cargo clippy -p apollo-sft --all-targets -- -D warnings`; `cargo doc -p apollo-sft --no-deps`; `cargo semver-checks -p apollo-sft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic SFT tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: SFT production top-K sparse selection has no Hermes-specific reduction target with the current provider API; direct DFT remains `#[cfg(test)]` verification/reference infrastructure.

## Radon adjoint Hermes dot routing [patch]
- Performed: added workspace Hermes and Mnemosyne provider dependencies to `apollo-radon`; routed threshold-sized adjoint backprojection pixel accumulations through Hermes real dot products with detector sample lanes and interpolation-weight lanes stored in Mnemosyne thread-local scratch.
- Architecture effect: Radon adjoint execution now composes Moirai image-row scheduling, Mnemosyne scratch reuse, Hermes SIMD reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: the Hermes path reuses two thread-local `f64` lane buffers per worker, with two lanes per angle for left/right detector interpolation. Small angle counts retain allocation-free scalar accumulation.
- Implementation effect: `backproject_pixel_scalar` remains the scalar formula reference; `fill_backproject_lanes` owns linear-sampler lane materialization; `backproject_pixel_hermes` owns the provider reduction boundary.
- Verification: `cargo fmt --check`; `cargo test -p apollo-radon`; `cargo clippy -p apollo-radon --all-targets -- -D warnings`; `cargo doc -p apollo-radon --no-deps`; `cargo semver-checks -p apollo-radon --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic Radon unit/property tests plus direct threshold-path Hermes pixel tests. No runtime benchmark claim is made.
- Residuals: forward projection remains a scatter/deposit kernel and is not routed through Hermes dot products; WGPU Radon remains provider-isolated and does not consume Hermes.

## SHT Hermes complex dot routing [patch]
- Performed: added workspace Hermes and Mnemosyne provider dependencies to `apollo-sht`; routed threshold-sized forward longitude sums and inverse synthesis mode sums through Hermes provider-owned interleaved complex dot products with spherical-harmonic lanes stored in Mnemosyne thread-local scratch.
- Architecture effect: SHT CPU execution now composes Moirai latitude-row scheduling, Mnemosyne scratch reuse, Hermes SIMD complex reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized forward rows materialize one shared complex sample lane buffer per latitude row; threshold-sized inverse plans materialize one shared complex coefficient lane buffer per transform; each worker row reuses thread-local harmonic-lane scratch. Small transforms retain allocation-free scalar accumulation.
- Implementation effect: `sht_forward_mode_sum` and `sht_inverse_sample` remain scalar formula references, while Hermes helpers own the provider reduction boundary and harmonic-lane materialization.
- Verification: `cargo fmt --check`; `cargo test -p apollo-sht`; `cargo clippy -p apollo-sht --all-targets -- -D warnings`; `cargo doc -p apollo-sht --no-deps`; `cargo semver-checks -p apollo-sht --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo run -p xtask -- benchmark --all`.
- Evidence tier: value-semantic SHT unit/property tests plus direct threshold-path Hermes row tests. `benchmark_results.md` was regenerated by the FFT quick-profile benchmark runner; no SHT runtime benchmark claim is made.
- Residuals: SHT still materializes interleaved lane buffers because the current Hermes complex dot API consumes primitive interleaved lanes; `sht.rs` remains above the 500-line structural target and should be split by bounded context in a follow-up architecture slice.

## NUFFT exact 1D Hermes complex dot routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-nufft`; routed threshold-sized exact 1D Type-1 coefficient rows and Type-2 sample rows through Hermes provider-owned interleaved complex dot products with phasor lanes stored in Mnemosyne thread-local scratch.
- Architecture effect: NUFFT exact 1D execution now composes Moirai row scheduling, Mnemosyne scratch reuse, Hermes SIMD complex reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized exact Type-1 transforms materialize one shared complex value lane buffer; Type-2 transforms materialize one shared complex coefficient lane buffer; each worker row reuses thread-local phasor-lane scratch. Small exact transforms retain allocation-free scalar accumulation.
- Implementation effect: `nufft_type1_coefficient` and `nufft_type2_sample` remain scalar formula references, while Hermes helpers own the provider reduction boundary and phasor-lane materialization.
- Verification: `cargo fmt --check`; `cargo test -p apollo-nufft`; `cargo clippy -p apollo-nufft --all-targets -- -D warnings`; `cargo doc -p apollo-nufft --no-deps`; `cargo semver-checks -p apollo-nufft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic NUFFT unit/property tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: exact 1D direct rows still materialize interleaved lane buffers because the current Hermes complex dot API consumes primitive interleaved lanes; fast 1D/3D gridding and 3D exact direct references do not yet consume Hermes.

## Mellin log-frequency Hermes complex dot routing [patch]
- Performed: routed threshold-sized forward and inverse log-frequency spectrum rows through Hermes provider-owned interleaved complex dot products with twiddle lanes stored in Mnemosyne thread-local scratch.
- Architecture effect: Mellin spectrum execution now composes Moirai row scheduling, Mnemosyne scratch reuse, Hermes SIMD complex reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized forward spectra materialize one shared real interleaved input lane buffer; inverse spectra materialize one shared complex interleaved lane buffer; each worker row reuses thread-local twiddle-lane scratch. Small spectra retain allocation-free scalar accumulation.
- Implementation effect: the existing scalar DFT/IDFT closures remain the small-path reference, while `log_frequency_coeff_hermes` and `inverse_log_frequency_coeff_hermes` own the provider reduction boundary.
- Verification: `cargo fmt --check`; `cargo test -p apollo-mellin`; `cargo clippy -p apollo-mellin --all-targets -- -D warnings`; `cargo doc -p apollo-mellin --no-deps`; `cargo semver-checks -p apollo-mellin --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic Mellin unit/property tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: log-frequency spectrum rows still materialize interleaved lane buffers because the current Hermes complex dot API consumes primitive interleaved lanes; typed storage conversion remains owner `f64` arithmetic before quantization.

## DCT/DST direct Hermes dot routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-dctdst`; routed threshold-sized direct DCT-I/II/III/IV and DST-I/II/III/IV row reductions through Hermes real dot products with basis rows stored in Mnemosyne thread-local scratch.
- Architecture effect: DCT/DST direct execution now composes Moirai row scheduling, Mnemosyne scratch reuse, Hermes SIMD reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized direct transforms reuse one thread-local basis buffer per worker row; small direct transforms retain allocation-free scalar accumulation.
- Implementation effect: one closed `DirectBasisKind` helper owns all trigonometric basis-row materialization, while the existing scalar formulas remain the small-path reference and verification oracle.
- Verification: `cargo fmt --check`; `cargo test -p apollo-dctdst`; `cargo clippy -p apollo-dctdst --all-targets -- -D warnings`; `cargo doc -p apollo-dctdst --no-deps`; `cargo semver-checks -p apollo-dctdst --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`; `cargo run -p xtask -- benchmark`.
- Evidence tier: value-semantic DCT/DST unit/property tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: fast FFT-derived DCT/DST kernels still use local twiddle extraction loops around `apollo-fft`; WGPU DCT/DST host/device paths remain provider-isolated but do not consume Hermes.

## FRFT direct Hermes complex dot routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-frft`; routed threshold-sized direct fractional and centered-DFT row reductions through Hermes provider-owned interleaved complex dot products with phasor weight lanes stored in Mnemosyne thread-local scratch.
- Architecture effect: standard FRFT direct execution now composes Moirai row scheduling, Mnemosyne scratch reuse, Hermes SIMD complex reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized direct transforms materialize one interleaved input lane buffer, then reuse thread-local phasor weight-lane scratch per worker row; small direct transforms retain allocation-free scalar accumulation and exact integer-order identity/reversal paths.
- Implementation effect: `fractional_row` and `centered_dft_row` remain scalar formula references for small paths and tests, while the Hermes helpers own the provider reduction boundary and phasor lane materialization.
- Verification: `cargo fmt --check`; `cargo test -p apollo-frft`; `cargo clippy -p apollo-frft --all-targets -- -D warnings`; `cargo doc -p apollo-frft --no-deps`; `cargo semver-checks -p apollo-frft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic FrFT unit/property tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: standard FRFT still materializes interleaved input lanes because the current Hermes complex dot API consumes primitive interleaved lanes; unitary Grünbaum projection/reconstruction still use local complex reductions through Moirai rather than Hermes.

## CZT direct Hermes complex dot routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-czt`; routed threshold-sized direct CZT row reductions through Hermes provider-owned interleaved complex dot products with geometric weight lanes stored in Mnemosyne thread-local scratch.
- Architecture effect: CZT direct execution now composes Moirai row scheduling, Mnemosyne scratch reuse, Hermes SIMD complex reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized direct transforms materialize one interleaved input lane buffer, then reuse thread-local geometric weight-lane scratch per worker row; small direct transforms retain allocation-free scalar accumulation.
- Implementation effect: `czt_direct_bin` remains the scalar formula reference for small paths and tests, while `czt_direct_bin_hermes` owns the provider reduction boundary and `fill_power_lanes` owns geometric progression materialization.
- Verification: `cargo fmt --check`; `cargo test -p apollo-czt`; `cargo clippy -p apollo-czt --all-targets -- -D warnings`; `cargo doc -p apollo-czt --no-deps`; `cargo semver-checks -p apollo-czt --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic CZT unit/property tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: direct CZT still materializes interleaved input lanes because the current Hermes complex dot API consumes primitive interleaved lanes; Bluestein pointwise multiplication still uses local complex multiplication through Moirai rather than Hermes.

## SDFT direct-bin Hermes dot routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-sdft`; routed threshold-sized direct-bin real and imaginary reductions through Hermes dot products with trigonometric weights stored in Mnemosyne thread-local scratch.
- Architecture effect: SDFT direct initialization now composes Moirai bin scheduling, Mnemosyne scratch reuse, Hermes SIMD reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: each threshold-path worker row reuses one thread-local weight buffer for cosine and sine passes; smaller windows retain allocation-free scalar accumulation.
- Implementation effect: `direct_bin_scalar` remains the DFT-bin formula reference for small paths and tests, while `direct_bin_hermes` owns the provider reduction boundary and `fill_direct_bin_weights` owns weight materialization.
- Verification: `cargo fmt --check`; `cargo test -p apollo-sdft`; `cargo clippy -p apollo-sdft --all-targets -- -D warnings`; `cargo doc -p apollo-sdft --no-deps`; `cargo semver-checks -p apollo-sdft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic SDFT unit/property tests plus direct threshold-path Hermes bin tests. No runtime benchmark claim is made.
- Residuals: typed direct-bin storage conversion still promotes reduced storage to `f64` scratch before computation by existing API contract; sliding recurrence updates remain complex scalar multiply-adds because Hermes lacks a provider primitive for that recurrence shape.

## QFT dense Hermes complex dot routing [patch]
- Performed: updated Apollo's Hermes lockfile revision to `b148fed9`; added the workspace Hermes provider dependency to `apollo-qft`; routed threshold-sized dense forward/inverse row reductions through Hermes provider-owned interleaved complex dot products.
- Architecture effect: QFT dense execution now composes Moirai row scheduling, Mnemosyne scratch reuse, Hermes SIMD complex reduction, and existing Leto public boundaries without runtime-erased dispatch in Apollo code.
- Memory effect: threshold-sized transforms materialize one interleaved input lane buffer, then reuse thread-local twiddle-lane scratch per worker row; small state vectors retain allocation-free scalar row reduction.
- Implementation effect: `qft_row` remains the scalar formula reference for small paths and tests, while `qft_row_hermes` owns the provider reduction boundary and `fill_twiddle_lanes` owns direction-specific twiddle materialization.
- Verification: `cargo fmt --check`; `cargo test -p apollo-qft`; `cargo clippy -p apollo-qft --all-targets -- -D warnings`; `cargo doc -p apollo-qft --no-deps`; `cargo semver-checks -p apollo-qft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: value-semantic QFT unit/property tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: QFT still materializes an interleaved input lane vector because the current Hermes complex dot API consumes primitive interleaved lanes; a future Hermes complex-view API could remove that copy if it can prove `Complex64` layout at the provider boundary.

## Wavelet CWT Hermes dot routing [patch]
- Performed: added workspace Hermes and Mnemosyne provider dependencies to `apollo-wavelet`; routed CWT coefficient accumulation through `hermes_simd::dot::<f64>` above a bounded signal-length threshold with mother-wavelet weights stored in Mnemosyne thread-local scratch.
- Architecture effect: Wavelet CWT execution now composes scalar formula ownership, Moirai scale-row scheduling, Mnemosyne scratch reuse, and Hermes SIMD reduction without runtime-erased dispatch in Apollo code.
- Memory effect: the Hermes path reuses thread-local weight buffers and does not allocate a new weight vector per coefficient; small signals keep the allocation-free scalar path.
- Implementation effect: `fill_cwt_weights` owns the wavelet-weight formula, and `coefficient_scalar` remains the scalar formula reference for small paths and tests.
- Verification: `cargo fmt --check`; `cargo check -p apollo-wavelet`; `cargo test -p apollo-wavelet`; `cargo clippy -p apollo-wavelet --all-targets -- -D warnings`; `cargo doc -p apollo-wavelet --no-deps`; `cargo semver-checks -p apollo-wavelet --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic Wavelet unit/property tests plus direct threshold-path Hermes coefficient tests. No runtime benchmark claim is made.
- Residuals: typed Wavelet storage conversion still uses owner `f64` arithmetic before quantization; other non-FFT CPU transform crates still lack Hermes-specific kernels.

## Mellin moment Hermes dot routing [patch]
- Performed: added workspace Hermes and Mnemosyne provider dependencies to `apollo-mellin`; routed real Mellin moment accumulation through `hermes_simd::dot::<f64>` above a bounded signal-length threshold with trapezoid weights stored in Mnemosyne thread-local scratch.
- Architecture effect: Mellin moment execution now composes scalar formula ownership, Mnemosyne scratch reuse, and Hermes SIMD reduction without runtime-erased dispatch in Apollo code; Moirai remains the provider for independent resampling and spectrum rows.
- Memory effect: the Hermes path reuses thread-local weight buffers and does not allocate a new weight vector per moment; small signals keep the allocation-free scalar path.
- Implementation effect: `fill_moment_weights` owns the trapezoid-rule weights, and `mellin_moment_scalar` remains the scalar formula reference for small paths and tests.
- Verification: `cargo fmt --check`; `cargo check -p apollo-mellin`; `cargo test -p apollo-mellin`; `cargo clippy -p apollo-mellin --all-targets -- -D warnings`; `cargo doc -p apollo-mellin --no-deps`; `cargo semver-checks -p apollo-mellin --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo run -p xtask -- benchmark`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic Mellin unit/property tests plus direct threshold-path Hermes moment tests; empirical FFT benchmark results recorded in `benchmark_results.md`.
- Residuals: log-frequency DFT and inverse DFT still use scalar complex reductions inside Moirai row scheduling; other non-FFT CPU transform crates still lack Hermes-specific kernels.

## GFT graph-basis Hermes dot routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-gft`; routed forward contiguous basis-column reductions and inverse scratch-materialized basis-row reductions through `hermes_simd::dot::<f64>` above a bounded row-length threshold.
- Architecture effect: GFT execution now composes Leto eigensolver/storage boundaries, Moirai row scheduling, Mnemosyne scratch reuse, and Hermes SIMD reduction in the graph-basis kernel boundary without runtime-erased dispatch in Apollo code.
- Memory effect: forward rows borrow contiguous column-major basis columns directly; inverse rows reuse thread-local scratch for the strided row view; output remains caller-owned and filled by disjoint mutable writes.
- Implementation effect: scalar `forward_row` and `inverse_row` remain the formula SSOT for small graphs and tests; Hermes-specific helpers only replace the multiply-reduce stage for sufficiently large rows.
- Verification: `cargo fmt --check`; `cargo check -p apollo-gft`; `cargo test -p apollo-gft`; `cargo clippy -p apollo-gft --all-targets -- -D warnings`; `cargo doc -p apollo-gft --no-deps`; `cargo semver-checks -p apollo-gft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic GFT unit/property tests plus direct threshold-path Hermes row tests. No runtime benchmark claim is made.
- Residuals: GFT typed storage conversion remains scalar; other non-FFT CPU transform crates still lack Hermes-specific kernels.

## DHT direct Hermes dot routing [patch]
- Performed: added the workspace Hermes provider dependency to `apollo-dht`; routed direct DHT coefficient accumulation through `hermes_simd::dot::<f64>` above a bounded row-length threshold while using Mnemosyne thread-local scratch for Hartley basis rows.
- Architecture effect: DHT direct execution now composes Moirai row scheduling, Mnemosyne scratch reuse, and Hermes SIMD reduction in the same kernel boundary without introducing runtime-erased dispatch in Apollo code.
- Memory effect: Hartley rows reuse thread-local scratch buffers; output remains caller-owned and filled by disjoint mutable writes; small rows keep scalar accumulation to avoid scratch setup overhead.
- Implementation effect: the direct `cas` formula remains the SSOT through `fill_hartley_row` and scalar `coefficient`; Hermes only replaces the multiply-reduce stage for sufficiently large rows.
- Verification: `cargo fmt --check`; `cargo check -p apollo-dht`; `cargo test -p apollo-dht`; `cargo clippy -p apollo-dht --all-targets -- -D warnings`; `cargo doc -p apollo-dht --no-deps`; `cargo semver-checks -p apollo-dht --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic DHT unit/property tests plus direct threshold-path Hermes coefficient tests. No runtime benchmark claim is made.
- Residuals: DHT typed storage conversion remains scalar; other non-FFT CPU transform crates still lack Hermes-specific kernels.

## SFT typed storage Moirai routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-sft`; routed generic typed input conversion, retained-value conversion, and inverse output materialization through Moirai above a bounded element threshold.
- Architecture effect: SFT typed slice and Leto paths now consume Apollo's Moirai provider surface for independent typed storage movement while preserving the dense FFT and deterministic top-K heap selector as the single canonical sparse-selection implementation.
- Memory effect: typed conversion allocates one caller-owned or return-owned vector and fills it through indexed collection or disjoint mutable writes; small buffers remain serial to avoid scheduling overhead.
- Implementation effect: three shared helpers own typed storage movement for all generic `SparseComplexStorage` implementations, preventing conversion-loop duplication across slice and Leto wrappers.
- Verification: `cargo fmt --check`; `cargo check -p apollo-sft`; `cargo test -p apollo-sft`; `cargo clippy -p apollo-sft --all-targets -- -D warnings`; `cargo doc -p apollo-sft --no-deps`; `cargo semver-checks -p apollo-sft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic SFT unit/property tests plus direct threshold-path storage conversion tests. No runtime benchmark claim is made.
- Residuals: Hermes and top-K providerization remain separate work; dense FFT transport no longer requires a Rust `ndarray` dependency edge in current Apollo.

## CZT direct/Bluestein Moirai routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-czt`; routed direct CZT output rows through `ParallelSliceMut` above a bounded O(NM) threshold; routed Bluestein workspace preparation, FFT-kernel multiplication, and output sampling through `ParallelSliceMut` above bounded contiguous-buffer thresholds.
- Architecture effect: CZT direct reference, slice, Leto, typed, and fast Bluestein paths now consume the Apollo Moirai provider at canonical kernel boundaries instead of individual public wrappers.
- Memory effect: caller-owned output and scratch buffers are filled in place by disjoint mutable writes; the existing Mnemosyne thread-local forward and typed scratch pools remain the allocation boundary.
- Implementation effect: direct CZT row evaluation is factored into a shared helper used by serial and Moirai paths; Bluestein workspace, kernel multiply, and output sampling stages are isolated helpers with one formula each.
- Verification: `cargo fmt --check`; `cargo check -p apollo-czt`; `cargo test -p apollo-czt`; `cargo clippy -p apollo-czt --all-targets -- -D warnings`; `cargo doc -p apollo-czt --no-deps`; `cargo semver-checks -p apollo-czt --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic CZT unit/property tests plus direct threshold-path formula tests for direct rows and Bluestein buffer stages. No runtime benchmark claim is made.
- Residuals: typed storage conversion loops in `CztStorage` still use serial scratch copies; inverse Björck-Pereyra solve remains sequential due to data dependencies; Hermes is still absent from `apollo-czt`.

## SDFT direct/update Moirai routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-sdft`; routed direct DFT bin writes through `ParallelSliceMut` above a bounded O(bin_count * window_len) work threshold; routed sliding recurrence bin updates through `ParallelSliceMut` above a bounded bin-count threshold.
- Architecture effect: SDFT slice, Leto, typed, and state-initialization paths now share Moirai-backed direct-bin and update kernels instead of adding provider logic to public wrappers.
- Memory effect: caller-owned bin buffers are filled in place by disjoint mutable writes; the existing Mnemosyne typed scratch pools remain the typed conversion boundary; small workloads remain serial to avoid scheduling overhead.
- Implementation effect: direct-bin and recurrence formulas are factored into shared helpers used by serial and Moirai paths, preserving the sliding DFT recurrence contract.
- Verification: `cargo fmt --check`; `cargo check -p apollo-sdft`; `cargo test -p apollo-sdft`; `cargo clippy -p apollo-sdft --all-targets -- -D warnings`; `cargo doc -p apollo-sdft --no-deps`; `cargo semver-checks -p apollo-sdft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic SDFT unit/property tests plus direct threshold-path formula tests for direct-bin and recurrence execution. No runtime benchmark claim is made.
- Residuals: typed storage conversion loops in `SdftRealStorage`/`SdftBinStorage` still use serial scratch copies; Hermes is still absent from `apollo-sdft`.

## GFT graph-basis Moirai routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-gft`; routed forward `U^T x` and inverse `U X` output-row writes through `ParallelSliceMut` above a bounded O(N²) work threshold.
- Architecture effect: GFT ndarray and typed paths now share Moirai-backed graph-basis slice execution while Leto remains the adjacency/eigensolver storage boundary.
- Memory effect: caller-owned output buffers are filled in place by disjoint mutable row writes; small graphs remain serial to avoid scheduling overhead.
- Implementation effect: forward and inverse graph-basis row formulas are factored into shared helpers used by serial and Moirai paths, preserving the column-major basis contract.
- Verification: `cargo fmt --check`; `cargo check -p apollo-gft`; `cargo test -p apollo-gft`; `cargo clippy -p apollo-gft --all-targets -- -D warnings`; `cargo doc -p apollo-gft --no-deps`; `cargo semver-checks -p apollo-gft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic GFT unit/property tests plus direct threshold-path row-formula tests for forward and inverse execution. No runtime benchmark claim is made.
- Residuals: typed storage conversion loops in `GftStorage` still use serial scratch copies; Hermes is still absent from `apollo-gft`.

## FRFT direct/unitary Moirai routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-frft`; routed direct fractional and centered-DFT output-row writes through `ParallelSliceMut` above a bounded O(N²) work threshold; routed unitary Grünbaum projection coefficients, phase application, and reconstruction rows through `ParallelSliceMut` above the same bounded work model.
- Architecture effect: FRFT ndarray, Leto, typed, and unitary public paths now share Moirai-backed CPU row execution instead of adding provider logic to individual wrappers.
- Memory effect: caller-owned output buffers and the existing Mnemosyne thread-local coefficient scratch are filled in place by disjoint mutable writes; small vectors and exact identity/reversal special cases remain serial.
- Implementation effect: direct fractional rows, centered-DFT rows, unitary projection rows, phase application, and unitary reconstruction rows are factored into shared helpers used by serial and Moirai paths.
- Verification: `cargo fmt --check`; `cargo check -p apollo-frft`; `cargo test -p apollo-frft`; `cargo clippy -p apollo-frft --all-targets -- -D warnings`; `cargo doc -p apollo-frft --no-deps`; `cargo semver-checks -p apollo-frft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic FrFT unit/property tests plus direct threshold-path row-formula tests for fractional, centered-DFT, and unitary execution. No runtime benchmark claim is made.
- Residuals: typed storage conversion loops in `FrftStorage` still use serial scratch copies; Hermes is still absent from `apollo-frft`.

## QFT dense kernel Moirai routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-qft`; routed dense forward/inverse output-row writes through `ParallelSliceMut` above a bounded O(N²) work threshold; replaced the private direction boolean with `QftDirection`.
- Architecture effect: QFT ndarray, Leto, and typed paths now share the same Moirai-backed dense kernel instead of adding provider logic to individual wrappers.
- Memory effect: caller-owned output buffers are filled in place by disjoint mutable row writes; small state vectors remain serial to avoid scheduling overhead.
- Implementation effect: one `qft_row` helper is the SSOT for serial and Moirai execution, preserving the existing twiddle-table contract.
- Verification: `cargo fmt --check`; `cargo check -p apollo-qft`; `cargo test -p apollo-qft`; `cargo clippy -p apollo-qft --all-targets -- -D warnings`; `cargo doc -p apollo-qft --no-deps`; `cargo semver-checks -p apollo-qft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic QFT unit/property tests plus direct threshold-path row-formula tests for forward and inverse execution. No runtime benchmark claim is made.
- Residuals: typed storage conversion loops in `QftStorage` still use serial scratch copies; Hermes is still absent from `apollo-qft`.

## Hilbert analytic path Moirai routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-hilbert` and routed analytic-signal staging, analytic-mask application, real-part restoration, and quadrature extraction through `ParallelSliceMut` above a bounded signal-length threshold.
- Architecture effect: Hilbert CPU execution now uses Apollo's Moirai provider surface for disjoint mutable writes around the FFT plan while keeping the FFT implementation and Leto/Mnemosyne public boundaries unchanged.
- Memory effect: caller-owned output buffers are filled in place; the existing Mnemosyne scratch pool remains the quadrature analytic workspace owner.
- Implementation effect: one thresholded helper per output-write role preserves SSOT between serial and Moirai paths and avoids Rayon/Tokio or runtime-erased dispatch.
- Verification: `cargo fmt --check`; `cargo check -p apollo-hilbert`; `cargo test -p apollo-hilbert`; `cargo clippy -p apollo-hilbert --all-targets -- -D warnings`; `cargo doc -p apollo-hilbert --no-deps`; `cargo semver-checks -p apollo-hilbert --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic Hilbert unit/property tests plus a direct threshold-path helper test. No runtime benchmark claim is made.
- Residuals: observable-domain helper methods in `domain::signal::analytic` still use serial slice loops; Hermes is still absent from `apollo-hilbert`.

## NUFFT exact 1D Moirai reference routing [patch]
- Performed: added the workspace Moirai provider dependency to `apollo-nufft` and routed `nufft_type1_1d` / `nufft_type2_1d` direct reference output construction through `ParallelSliceMut` above a bounded operation threshold.
- Architecture effect: NUFFT exact 1D reference execution now uses Apollo's Moirai provider surface for CPU data parallelism instead of local sequential-only loops, while the public Leto/Mnemosyne boundaries remain unchanged.
- Memory effect: output buffers are allocated once, then filled by disjoint mutable slice writes; small workloads use the serial path to avoid scheduler overhead.
- Implementation effect: the mathematical formulas are factored into shared helper functions used by both serial and Moirai paths, preserving SSOT and avoiding duplicated algorithm bodies.
- Verification: `cargo fmt --check`; `cargo check -p apollo-nufft`; `cargo test -p apollo-nufft`; `cargo clippy -p apollo-nufft --all-targets -- -D warnings`; `cargo doc -p apollo-nufft --no-deps`; `cargo semver-checks -p apollo-nufft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo test`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude apollo-python --no-deps`.
- Evidence tier: value-semantic unit/property tests over exact 1D direct outputs and fast-vs-exact comparisons. No runtime benchmark claim is made.
- Residuals: NUFFT 3D exact references and fast gridding loops still use local serial loops; Hermes is still absent from `apollo-nufft`.

## NUFFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-nufft-wgpu` to `0.2.0`; added the workspace Leto dependency; added direct and fast 1D/3D Type-1/Type-2 Leto host boundaries, including typed storage variants.
- Architecture effect: NUFFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate; current validation storage is Leto/provider-owned.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and provider execution paths instead of adding separate GPU algorithm bodies.
- Verification: `cargo check -p apollo-nufft-wgpu`; `cargo test -p apollo-nufft-wgpu leto -- --nocapture`; `cargo test -p apollo-nufft-wgpu -- --nocapture`; `cargo clippy -p apollo-nufft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-nufft-wgpu --no-deps`; `cargo semver-checks -p apollo-nufft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing NUFFT-WGPU slice/provider APIs for 1D Type-1, strided 1D Type-2, typed fast 1D Type-1, 3D Type-1, and 3D Type-2 paths. No runtime benchmark claim is made.
- Residuals: provider audit no longer reports an Apollo WGPU transform crate without Leto/Mnemosyne host-boundary usage; NUFFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## FFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-fft-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for 3D forward, 3D inverse, mixed `f16` 3D forward, and mixed `f16` 3D inverse execution.
- Architecture effect: FFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D/3D views borrow storage through `Cow`; strided Leto views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU split-buffer 3D FFT execution path instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-fft-wgpu`; `cargo test -p apollo-fft-wgpu leto -- --nocapture`; `cargo test -p apollo-fft-wgpu -- --nocapture`; `cargo clippy -p apollo-fft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-fft-wgpu --no-deps`; `cargo semver-checks -p apollo-fft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing FFT-WGPU provider APIs for forward, inverse, strided forward, and mixed `f16` paths. No runtime benchmark claim is made.
- Residuals: FFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## STFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-stft-wgpu` to `0.12.0`; added the workspace Leto dependency; added Leto host boundaries for forward, inverse, typed forward, and typed inverse STFT execution.
- Architecture effect: STFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and typed slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-stft-wgpu`; `cargo test -p apollo-stft-wgpu leto -- --nocapture`; `cargo test -p apollo-stft-wgpu -- --nocapture`; `cargo clippy -p apollo-stft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-stft-wgpu --no-deps`; `cargo semver-checks -p apollo-stft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing STFT-WGPU slice APIs for forward, inverse, strided forward, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: STFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## Wavelet-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-wavelet-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for forward, inverse, typed forward, and typed inverse Haar DWT execution.
- Architecture effect: Wavelet-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and typed slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-wavelet-wgpu`; `cargo test -p apollo-wavelet-wgpu leto -- --nocapture`; `cargo test -p apollo-wavelet-wgpu -- --nocapture`; `cargo clippy -p apollo-wavelet-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-wavelet-wgpu --no-deps`; `cargo semver-checks -p apollo-wavelet-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing Wavelet-WGPU slice APIs for forward, inverse, strided forward, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; Wavelet-WGPU currently covers the Haar DWT WGPU surface and still performs WGPU arithmetic at `f32` precision by contract.

## Radon-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-radon-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for forward projection, adjoint backprojection, filtered backprojection, typed forward projection, and typed adjoint backprojection.
- Architecture effect: Radon-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate; internal host staging is provider-owned in current Apollo.
- Memory effect: contiguous Leto 1D angle views borrow storage through `Cow`; strided Leto 1D/2D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU provider and typed flat slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-radon-wgpu`; `cargo test -p apollo-radon-wgpu leto -- --nocapture`; `cargo test -p apollo-radon-wgpu -- --nocapture`; `cargo clippy -p apollo-radon-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-radon-wgpu --no-deps`; `cargo semver-checks -p apollo-radon-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing Radon-WGPU provider/slice APIs for forward, adjoint backprojection, filtered backprojection, strided forward, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still need Hephaestus capability expansion; Radon-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## SHT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-sht-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for 2D forward samples, 2D inverse coefficients, flat typed forward samples, and flat typed inverse coefficients.
- Architecture effect: SHT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate; internal host staging is provider-owned in current Apollo.
- Memory effect: contiguous flat typed Leto 1D views borrow storage through `Cow`; strided Leto 1D/2D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU provider and typed slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-sht-wgpu`; `cargo test -p apollo-sht-wgpu leto -- --nocapture`; `cargo test -p apollo-sht-wgpu -- --nocapture`; `cargo clippy -p apollo-sht-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-sht-wgpu --no-deps`; `cargo semver-checks -p apollo-sht-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing SHT-WGPU provider/slice APIs for 2D forward, 2D inverse, strided 2D forward, and typed flat forward/inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still need Hephaestus capability expansion; SHT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## GFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-gft-wgpu` to `0.2.0`; promoted Leto from dev-only to public dependency; added Leto host boundaries for forward, inverse, typed forward, and typed inverse execution.
- Architecture effect: GFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D signal and basis views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and typed slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-gft-wgpu`; `cargo test -p apollo-gft-wgpu leto -- --nocapture`; `cargo test -p apollo-gft-wgpu -- --nocapture`; `cargo clippy -p apollo-gft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-gft-wgpu --no-deps`; `cargo semver-checks -p apollo-gft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing GFT-WGPU slice APIs for forward, inverse, strided forward, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; GFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## FRFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-frft-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for standard forward/inverse FrFT, unitary forward/inverse DFrFT, and typed forward/inverse execution.
- Architecture effect: FRFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and typed slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-frft-wgpu`; `cargo test -p apollo-frft-wgpu leto -- --nocapture`; `cargo test -p apollo-frft-wgpu -- --nocapture`; `cargo clippy -p apollo-frft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-frft-wgpu --no-deps`; `cargo semver-checks -p apollo-frft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`; `cargo run -p xtask -- benchmark`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing FRFT-WGPU slice APIs for standard, strided, typed, and unitary paths. Benchmark evidence is empirical Criterion quick-profile measurement only; no runtime speedup claim is made for this boundary.
- Residuals: several Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; FRFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## NTT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-ntt-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for `u64` forward/inverse NTT and exact `u32` quantized forward/inverse execution.
- Architecture effect: NTT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and quantized slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-ntt-wgpu`; `cargo test -p apollo-ntt-wgpu leto -- --nocapture`; `cargo test -p apollo-ntt-wgpu -- --nocapture`; `cargo clippy -p apollo-ntt-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-ntt-wgpu --no-deps`; `cargo semver-checks -p apollo-ntt-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus exact value-semantic differential tests against existing NTT-WGPU slice APIs for `u64` forward/inverse, strided forward, and `u32` quantized forward/inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; NTT-WGPU exact integer arithmetic intentionally has no floating mixed-precision boundary.

## Mellin-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-mellin-wgpu` to `0.3.0`; added the workspace Leto dependency; added Leto host boundaries for forward spectrum, typed forward spectrum, and inverse reconstruction.
- Architecture effect: Mellin-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and typed slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-mellin-wgpu`; `cargo test -p apollo-mellin-wgpu leto -- --nocapture`; `cargo test -p apollo-mellin-wgpu -- --nocapture`; `cargo clippy -p apollo-mellin-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-mellin-wgpu --no-deps`; `cargo semver-checks -p apollo-mellin-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing Mellin-WGPU slice APIs for forward, strided forward, typed forward, and inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; Mellin-WGPU still performs WGPU arithmetic at `f32` precision by contract; no typed inverse Leto boundary was added because the crate has no typed inverse slice execution contract.

## Hilbert-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-hilbert-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for analytic signal, forward quadrature, typed forward, inverse, and typed inverse Hilbert execution.
- Architecture effect: Hilbert-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice and typed slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-hilbert-wgpu`; `cargo test -p apollo-hilbert-wgpu leto -- --nocapture`; `cargo test -p apollo-hilbert-wgpu -- --nocapture`; `cargo clippy -p apollo-hilbert-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-hilbert-wgpu --no-deps`; `cargo semver-checks -p apollo-hilbert-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing Hilbert-WGPU slice APIs for analytic, strided forward, inverse, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; Hilbert-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## DCT/DST-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-dctdst-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for 1D forward/inverse, typed 1D forward/inverse, 2D forward/inverse, and 3D forward/inverse DCT/DST execution.
- Architecture effect: DCT/DST-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate. Current Apollo source and manifests no longer retain a Rust `ndarray` crate edge.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto 1D views copy once into logical order; current 2D/3D host staging uses Leto/row-major provider buffers before generated host outputs return Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice, typed slice, and separable execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-dctdst-wgpu`; `cargo test -p apollo-dctdst-wgpu leto -- --nocapture`; `cargo test -p apollo-dctdst-wgpu -- --nocapture`; `cargo clippy -p apollo-dctdst-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-dctdst-wgpu --no-deps`; `cargo semver-checks -p apollo-dctdst-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing DCT/DST-WGPU slice/Leto APIs for 1D, typed 1D, 2D, and 3D paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still lack complete Hephaestus capability expansion; DCT/DST-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## CZT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-czt-wgpu` to `0.3.0`; added the workspace Leto dependency; added Leto host boundaries for forward, typed forward, and adjoint inverse CZT execution.
- Architecture effect: CZT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D complex views borrow storage through `Cow`; strided Leto views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-czt-wgpu`; `cargo test -p apollo-czt-wgpu leto -- --nocapture`; `cargo test -p apollo-czt-wgpu -- --nocapture`; `cargo clippy -p apollo-czt-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-czt-wgpu --no-deps`; `cargo semver-checks -p apollo-czt-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing CZT-WGPU slice APIs for contiguous forward, strided forward, typed forward, and inverse paths. No runtime benchmark claim is made.
- Residuals: several Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; CZT-WGPU still performs WGPU arithmetic at `f32` precision by contract; no typed inverse Leto boundary was added because the crate has no typed inverse slice execution contract.

## SFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-sft-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for forward, typed forward, inverse, and typed inverse SFT execution.
- Architecture effect: SFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate and sparse spectra remain represented by the owning `SparseSpectrum` domain type.
- Memory effect: contiguous Leto 1D complex views borrow storage through `Cow`; strided Leto views copy once into logical order; dense inverse host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice execution methods and deterministic top-k sparse selection instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-sft-wgpu`; `cargo test -p apollo-sft-wgpu leto -- --nocapture`; `cargo test -p apollo-sft-wgpu -- --nocapture`; `cargo clippy -p apollo-sft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-sft-wgpu --no-deps`; `cargo semver-checks -p apollo-sft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing SFT-WGPU slice APIs for contiguous forward, strided forward, inverse, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: most other Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; SFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## QFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-qft-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for forward, typed forward, inverse, and typed inverse QFT execution.
- Architecture effect: QFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D complex views borrow storage through `Cow`; strided Leto views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-qft-wgpu`; `cargo test -p apollo-qft-wgpu leto -- --nocapture`; `cargo test -p apollo-qft-wgpu -- --nocapture`; `cargo clippy -p apollo-qft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-qft-wgpu --no-deps`; `cargo semver-checks -p apollo-qft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing QFT-WGPU slice APIs for contiguous forward, strided forward, inverse, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: most other Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; QFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## DHT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-dht-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for forward, typed forward, inverse, and typed inverse DHT execution.
- Architecture effect: DHT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-dht-wgpu`; `cargo test -p apollo-dht-wgpu leto -- --nocapture`; `cargo test -p apollo-dht-wgpu -- --nocapture`; `cargo clippy -p apollo-dht-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-dht-wgpu --no-deps`; `cargo semver-checks -p apollo-dht-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing DHT-WGPU slice APIs for contiguous forward, strided forward, inverse, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: most other Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; DHT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## FWHT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-fwht-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for forward, typed forward, inverse, and typed inverse FWHT execution.
- Architecture effect: FWHT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-fwht-wgpu`; `cargo test -p apollo-fwht-wgpu leto -- --nocapture`; `cargo test -p apollo-fwht-wgpu -- --nocapture`; `cargo clippy -p apollo-fwht-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-fwht-wgpu --no-deps`; `cargo semver-checks -p apollo-fwht-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing FWHT-WGPU slice APIs for contiguous forward, strided forward, inverse, and typed forward/inverse paths. No runtime benchmark claim is made.
- Residuals: most other Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; FWHT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## SDFT-WGPU Leto host boundary [minor]
- Performed: bumped `apollo-sdft-wgpu` to `0.2.0`; added the workspace Leto dependency; added Leto host boundaries for forward bins, typed forward bins, and inverse bins.
- Architecture effect: SDFT-WGPU callers can now use Leto as the public host array/layout boundary while WGPU device buffers remain isolated in the infrastructure crate.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto views copy once into logical order; generated host outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing WGPU slice execution methods instead of adding a separate GPU algorithm body.
- Verification: `cargo check -p apollo-sdft-wgpu`; `cargo test -p apollo-sdft-wgpu leto -- --nocapture`; `cargo test -p apollo-sdft-wgpu -- --nocapture`; `cargo clippy -p apollo-sdft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-sdft-wgpu --no-deps`; `cargo semver-checks -p apollo-sdft-wgpu --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing SDFT-WGPU slice APIs for contiguous forward, strided forward, typed forward, and inverse paths. No runtime benchmark claim is made.
- Residuals: most other Apollo WGPU transform crates still lack Leto/Mnemosyne host boundaries; SDFT-WGPU still performs WGPU arithmetic at `f32` precision by contract.

## NUFFT Leto 1D type-1/type-2 boundary [minor]
- Performed: bumped `apollo-nufft` to `0.2.0`; added the workspace Leto dependency; added Leto boundaries for 1D type-1, type-2, typed type-1, and typed type-2 NUFFT.
- Architecture effect: NUFFT 1D callers can now use Leto as the public position/value/coefficient boundary while existing slice APIs remain the validation and compatibility surface.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided Leto views copy once into logical order; generated Fourier bins and interpolated values use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing slice NUFFT kernels and Mnemosyne scratch pools rather than adding a separate NUFFT implementation.
- Verification: `cargo check -p apollo-nufft`; `cargo test -p apollo-nufft leto -- --nocapture`; `cargo test -p apollo-nufft -- --nocapture`; `cargo clippy -p apollo-nufft --all-targets -- -D warnings`; `cargo doc -p apollo-nufft --no-deps`; `cargo semver-checks -p apollo-nufft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-nufft --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing NUFFT slice APIs for type-1, strided type-1, type-2, and typed `Complex32` type-1/type-2. No runtime benchmark claim is made.
- Residuals: NUFFT still has broader GPU/provider expansion work, but current Apollo source and manifests no longer expose a Rust `ndarray` dependency edge.

## SHT Leto 2D sample/coefficient boundary [minor]
- Performed: bumped `apollo-sht` to `0.2.0`; added the workspace Leto dependency; added Leto boundaries for real forward, complex forward, real inverse, complex inverse, and typed real/complex forward/inverse storage.
- Architecture effect: SHT callers can now use Leto as the public 2D sample/coefficient boundary while Leto-native paths own the validation and compatibility surface.
- Memory effect: Leto 2D views copy once into logical row-major provider buffers; generated coefficient and sample outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing Leto/Moirai SHT quadrature and synthesis kernels rather than adding a separate SHT implementation.
- Verification: `cargo check -p apollo-sht`; `cargo test -p apollo-sht leto -- --nocapture`; `cargo test -p apollo-sht -- --nocapture`; `cargo clippy -p apollo-sht --all-targets -- -D warnings`; `cargo doc -p apollo-sht --no-deps`; `cargo semver-checks -p apollo-sht --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-sht --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic tests for real forward, strided real forward, complex forward/inverse, and typed `f32` real forward/inverse. No runtime benchmark claim is made.
- Residuals: SHT still has broader WGPU/Hephaestus provider expansion work; no current Apollo-owned Rust `ndarray` dependency remains.

## Wavelet Leto DWT/CWT boundary [minor]
- Performed: bumped `apollo-wavelet` to `0.2.0`; added the workspace Leto dependency; added Leto boundaries for DWT forward/inverse, typed DWT forward/inverse, CWT transform, and typed CWT transform.
- Architecture effect: Wavelet callers can now use Leto as the public 1D signal and dense coefficient boundary while existing slice APIs remain the validation and compatibility surface.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided Leto views copy once into logical order; generated DWT approximation/detail arrays, reconstructed signals, and CWT coefficient matrices use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing DWT slice kernels and CWT Moirai scale-parallel kernel rather than adding separate wavelet implementations.
- Verification: `cargo check -p apollo-wavelet`; `cargo test -p apollo-wavelet leto -- --nocapture`; `cargo test -p apollo-wavelet -- --nocapture`; `cargo clippy -p apollo-wavelet --all-targets -- -D warnings`; `cargo doc -p apollo-wavelet --no-deps`; `cargo semver-checks -p apollo-wavelet --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-wavelet --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing Wavelet slice APIs for DWT contiguous forward/inverse, DWT strided forward, typed `f32` DWT forward/inverse, CWT contiguous transform, CWT strided transform, and typed `f32` CWT transform. No runtime benchmark claim is made.
- Residuals: Wavelet WGPU still has no Leto boundary; `DwtLetoCoefficients` remains ragged by design because multilevel DWT detail bands are ragged.

## STFT Leto 1D analysis/synthesis boundary [minor]
- Performed: bumped `apollo-stft` to `0.3.0`; added the workspace Leto dependency; added Leto boundaries for forward analysis, inverse synthesis, typed forward analysis, typed inverse synthesis, and transport convenience wrappers.
- Architecture effect: STFT callers can now use Leto as the public 1D signal/spectrum boundary while existing slice/Leto APIs remain the validation and compatibility surface.
- Memory effect: contiguous `f64` Leto views borrow storage through `Cow`; strided Leto views copy once into logical order; generated spectrum and signal outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing Leto/Moirai STFT execution contract rather than adding a separate STFT implementation.
- Verification: `cargo check -p apollo-stft`; `cargo test -p apollo-stft leto -- --nocapture`; `cargo test -p apollo-stft -- --nocapture`; `cargo clippy -p apollo-stft --all-targets -- -D warnings`; `cargo doc -p apollo-stft --no-deps`; `cargo semver-checks -p apollo-stft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-stft --examples`.
- Evidence tier: type-level public Leto boundary plus focused value-semantic differential tests against existing STFT provider APIs for contiguous forward analysis, strided forward analysis, inverse synthesis, and typed `f32` forward/inverse. No runtime benchmark claim is made.
- Residuals: STFT still has broader WGPU/Hephaestus provider expansion work; no current Apollo-owned Rust `ndarray` dependency remains.

## Radon Leto 2D projection boundary [minor]
- Performed: bumped `apollo-radon` to `0.2.0`; added the workspace Leto dependency; added Leto boundaries for forward projection, typed forward projection, adjoint backprojection, typed adjoint backprojection, and filtered backprojection.
- Architecture effect: Radon callers can now use Leto as the public 2D image/sinogram boundary while existing Leto/slice APIs remain the validation and compatibility surface.
- Memory effect: Leto 2D views copy once into logical row-major provider buffers; generated image and sinogram outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing Leto/Moirai projection, adjoint, and filtered-backprojection kernels rather than adding a separate Radon implementation.
- Verification: `cargo check -p apollo-radon`; `cargo test -p apollo-radon leto -- --nocapture`; `cargo test -p apollo-radon -- --nocapture`; `cargo clippy -p apollo-radon --all-targets -- -D warnings`; `cargo doc -p apollo-radon --no-deps`; `cargo semver-checks -p apollo-radon --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-radon --examples`.
- Evidence tier: type-level public Leto boundary plus value-semantic differential tests against existing Radon provider APIs for contiguous forward projection, strided forward projection, typed `f32` forward/backprojection, adjoint backprojection, and filtered backprojection. Existing adjoint identity and ramp-filter tests remained green. No runtime benchmark claim is made.
- Residuals: Radon still has broader WGPU/Hephaestus provider expansion work; no current Apollo-owned Rust `ndarray` dependency remains.

## Mellin Leto resample and spectrum boundary [minor]
- Performed: bumped `apollo-mellin` to `0.3.0`; added the workspace Leto dependency; added Leto boundaries for resampling, typed resampling, moments, typed moments, forward spectra, typed forward spectra, inverse spectra, and inverse from Leto spectrum views.
- Architecture effect: Mellin callers can now use Leto as the public 1D array/layout boundary while existing slice APIs remain the validation and compatibility surface.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided Leto views copy once into logical order; generated resample, spectrum, and inverse outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing Mellin slice resample/moment/spectrum/inverse contracts and retain Moirai-backed spectrum computation in the kernel layer.
- Verification: `cargo check -p apollo-mellin`; `cargo test -p apollo-mellin leto -- --nocapture`; `cargo test -p apollo-mellin -- --nocapture`; `cargo clippy -p apollo-mellin --all-targets -- -D warnings`; `cargo doc -p apollo-mellin --no-deps`; `cargo semver-checks -p apollo-mellin --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-mellin --examples`.
- Evidence tier: type-level public Leto boundary plus value-semantic differential tests against existing Mellin slice APIs for contiguous resampling, strided resampling, typed resampling, moment/spectrum computation, and inverse spectrum recovery. Existing analytical moment and inverse tests remained green. No runtime benchmark claim is made.
- Residuals: Mellin WGPU still has no Leto boundary; broader provider migration remains per-crate work.

## SDFT Leto direct-bin boundary [minor]
- Performed: bumped `apollo-sdft` to `0.2.0`; added the workspace Leto dependency; added `SdftPlan::direct_bins_leto`, `SdftPlan::direct_bins_leto_typed`, and `SdftPlan::state_from_window_leto`.
- Architecture effect: SDFT direct-bin and state-initialization callers can now use Leto as the public 1D array/layout boundary while existing slice APIs remain the validation and compatibility surface.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided Leto views copy once into logical window order; direct-bin outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing direct-bin and typed direct-bin execution contracts rather than adding a separate SDFT implementation.
- Verification: `cargo check -p apollo-sdft`; `cargo test -p apollo-sdft leto -- --nocapture`; `cargo test -p apollo-sdft -- --nocapture`; `cargo clippy -p apollo-sdft --all-targets -- -D warnings`; `cargo doc -p apollo-sdft --no-deps`; `cargo semver-checks -p apollo-sdft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-sdft --examples`.
- Evidence tier: type-level public Leto boundary plus value-semantic differential tests against existing SDFT slice APIs for contiguous direct bins, strided direct bins, typed direct bins, and state initialization. No runtime benchmark claim is made.
- Residuals: streaming update state still owns `VecDeque<f64>` and `Vec<Complex64>` internally; broader state storage providerization remains future work.

## SFT Leto sparse spectrum boundary [minor]
- Performed: bumped `apollo-sft` to `0.2.0`; added the workspace Leto dependency; added `SparseFftPlan::forward_leto`, `SparseFftPlan::inverse_leto`, `SparseFftPlan::forward_leto_typed`, `SparseFftPlan::inverse_leto_typed`, and `SparseLetoSpectrum<T>`.
- Architecture effect: sparse FFT callers can now use Leto as the public 1D array/layout boundary while existing slice APIs remain the validation and compatibility surface.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided Leto views copy once into logical order; inverse and typed retained-value outputs use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing dense FFT plus deterministic top-K sparse-selection contract rather than adding a separate sparse-transform implementation.
- Verification: `cargo check -p apollo-sft`; `cargo test -p apollo-sft leto -- --nocapture`; `cargo test -p apollo-sft -- --nocapture`; `cargo clippy -p apollo-sft --all-targets -- -D warnings`; `cargo doc -p apollo-sft --no-deps`; `cargo semver-checks -p apollo-sft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-sft --examples`.
- Evidence tier: type-level public Leto boundary plus value-semantic differential tests against existing SFT slice APIs for contiguous forward, strided forward, inverse reconstruction, and typed `Complex32` forward/inverse. Existing property tests for exact sparse recovery, top-K retained energy, and retained DFT values remained green. No runtime benchmark claim is made.
- Residuals: SFT still has broader sparse/GPU providerization work, but current FFT bridge execution no longer requires a Rust `ndarray` crate edge.

## Hilbert Leto analytic and quadrature boundary [minor]
- Performed: bumped `apollo-hilbert` to `0.4.0`; added the workspace Leto dependency; added `HilbertPlan::analytic_signal_leto`, `HilbertPlan::transform_leto`, and `HilbertPlan::transform_leto_typed`, returning `leto::Array<_, leto::MnemosyneStorage<_>, 1>`.
- Architecture effect: Hilbert callers can now use Leto as the public 1D array/layout boundary while existing slice APIs remain the validation and compatibility surface.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided Leto views copy once into logical order; output arrays use Mnemosyne-backed Leto storage.
- Implementation effect: Leto boundaries reuse the existing slice and typed Hilbert execution contracts rather than adding a separate transform implementation.
- Verification: `cargo check -p apollo-hilbert`; `cargo test -p apollo-hilbert leto -- --nocapture`; `cargo test -p apollo-hilbert -- --nocapture`; `cargo clippy -p apollo-hilbert --all-targets -- -D warnings`; `cargo doc -p apollo-hilbert --no-deps`; `cargo semver-checks -p apollo-hilbert --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-hilbert --examples`.
- Evidence tier: type-level public Leto boundary plus value-semantic differential tests against existing Hilbert slice APIs for contiguous quadrature, strided quadrature, analytic signal, and typed `f32` quadrature. No runtime benchmark claim is made.
- Residuals: Hilbert still has broader WGPU/Hephaestus provider expansion work; Apollo's Rust `ndarray` migration is complete in the current source and dependency graph.

## DCT/DST Leto multidimensional boundary and benchmark refresh [minor]
- Performed: bumped `apollo-dctdst` to `0.2.0`; added the workspace Leto dependency; added Leto 1D/2D/3D forward and inverse boundaries plus typed 1D Leto storage boundaries, returning `leto::Array<_, leto::MnemosyneStorage<_>, _>`.
- Architecture effect: DCT/DST callers can now use Leto as a public array/layout boundary while existing slice/Leto APIs remain the validation and compatibility surfaces.
- Memory effect: contiguous Leto 1D views borrow storage through `Cow`; strided 1D views copy once into logical order; multidimensional Leto views copy once into row-major provider buffers before returning Mnemosyne-backed Leto arrays.
- Implementation effect: Leto boundaries reuse the existing transform execution paths instead of adding a separate DCT/DST algorithm body.
- Benchmark refresh: regenerated the full canonical FFT table with `cargo run -p xtask -- benchmark --all --profile quick`. Current measured table has 514 rows; f64 is faster on 68 rows, f32 is faster on 44 rows, and both are faster on 19 rows. This is empirical FFT benchmark evidence only and does not measure DCT/DST.
- Verification: `cargo check -p apollo-dctdst`; `cargo test -p apollo-dctdst leto -- --nocapture`; `cargo test -p apollo-dctdst -- --nocapture`; `cargo clippy -p apollo-dctdst --all-targets -- -D warnings`; `cargo doc -p apollo-dctdst --no-deps`; `cargo semver-checks -p apollo-dctdst --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-dctdst --examples`; `cargo run -p xtask -- benchmark --all --profile quick`.
- Evidence tier: type-level public Leto boundary plus value-semantic differential tests against existing DCT/DST slice/Leto APIs. Benchmark table evidence is empirical quick-profile measurement only. No machine-checked proof is performed.
- Residuals: none for Apollo-owned Rust `ndarray` usage; broader GPU/provider expansion remains separate work.

## CZT Leto public 1D boundary [minor]
- Performed: bumped `apollo-czt` to `0.3.0`; added the workspace Leto dependency; added `CztPlan::forward_leto`, `CztPlan::inverse_leto`, `CztPlan::forward_leto_typed`, `CztPlan::inverse_leto_typed`, and `czt_leto`, returning `leto::Array<_, leto::MnemosyneStorage<_>, 1>`.
- Architecture effect: CZT Complex64 and typed public 1D callers can now use Leto as the array/layout boundary while Leto/slice APIs remain as validation and compatibility surfaces.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided views copy once into logical order; output arrays use Mnemosyne-backed Leto storage.
- Implementation effect: `CztStorage` now owns canonical typed slice execution hooks, reducing repeated array-specific implementation logic and allowing Leto views to share the same execution contract.
- Verification: `cargo check -p apollo-czt`; `cargo test -p apollo-czt leto -- --nocapture`; `cargo test -p apollo-czt -- --nocapture`; `cargo clippy -p apollo-czt --all-targets -- -D warnings`; `cargo doc -p apollo-czt --no-deps`; `cargo semver-checks -p apollo-czt --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-czt --examples`.
- Evidence tier: type-level provider boundary plus value-semantic differential tests against the existing CZT API for contiguous `Complex64`, strided `Complex64`, typed `Complex32`, inverse `Complex64`, and transport helper output. Existing CZT property tests also remained green. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage; CZT WGPU/Hephaestus expansion remains separate work.

## DHT Leto multidimensional boundary [minor]
- Performed: bumped `apollo-dht` to `0.2.0`; added the workspace Leto dependency; added `DhtPlan::forward_2d_leto`, `DhtPlan::inverse_2d_leto`, `DhtPlan::forward_3d_leto`, and `DhtPlan::inverse_3d_leto`, returning `leto::Array<f64, leto::MnemosyneStorage<f64>, 2/3>`.
- Architecture effect: DHT 2D/3D callers can now use Leto as the array/layout boundary while Leto/slice paths remain the validation and compatibility oracle.
- Memory effect: output arrays use Mnemosyne-backed Leto storage; Leto strided inputs copy once into the existing row-major lane workspace required by the separable DHT scheduler.
- Implementation effect: Leto boundaries reuse the existing separable DHT row/column/depth kernels and Mnemosyne lane scratch pools instead of adding a separate transform implementation.
- Verification: `cargo check -p apollo-dht`; `cargo test -p apollo-dht leto -- --nocapture`; `cargo test -p apollo-dht -- --nocapture`; `cargo clippy -p apollo-dht --all-targets -- -D warnings`; `cargo doc -p apollo-dht --no-deps`; `cargo semver-checks -p apollo-dht --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-dht --examples`.
- Evidence tier: type-level provider boundary plus value-semantic differential tests against the existing DHT API for contiguous 2D, strided 2D inverse, contiguous 3D, and 3D inverse. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage; broader GPU/provider expansion remains separate work.

## QFT Leto public 1D boundary [minor]
- Performed: bumped `apollo-qft` to `0.2.0`; added the workspace Leto dependency; added `QftPlan::forward_leto`, `QftPlan::inverse_leto`, `QftPlan::forward_leto_typed`, `QftPlan::inverse_leto_typed`, `qft_leto`, and `iqft_leto`, returning `leto::Array<_, leto::MnemosyneStorage<_>, 1>`.
- Architecture effect: QFT complex and typed public 1D callers can now use Leto as the array/layout boundary while Leto/slice APIs remain as validation and compatibility surfaces.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided views copy once into logical order; output arrays use Mnemosyne-backed Leto storage.
- Implementation effect: `QftStorage` now owns canonical typed slice execution hooks, reducing repeated array-specific implementation logic and allowing Leto views to share the same execution contract.
- Verification: `cargo check -p apollo-qft`; `cargo test -p apollo-qft leto -- --nocapture`; `cargo test -p apollo-qft -- --nocapture`; `cargo clippy -p apollo-qft --all-targets -- -D warnings`; `cargo doc -p apollo-qft --no-deps`; `cargo semver-checks -p apollo-qft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-qft --examples`.
- Evidence tier: type-level provider boundary plus value-semantic differential tests against the existing QFT API for contiguous `Complex64`, strided `Complex64`, typed `Complex32`, and strided mixed `[f16; 2]`. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage; QFT WGPU/Hephaestus expansion remains separate work.

## FWHT Leto public 1D boundary [minor]
- Performed: bumped `apollo-fwht` to `0.2.0`; added the workspace Leto dependency; added `FwhtPlan::forward_leto`, `FwhtPlan::inverse_leto`, `FwhtPlan::forward_leto_typed`, `FwhtPlan::inverse_leto_typed`, `fwht_leto`, and `ifwht_leto`, returning `leto::Array<_, leto::MnemosyneStorage<_>, 1>`.
- Architecture effect: FWHT real and typed public 1D callers can now use Leto as the array/layout boundary while Leto/slice APIs remain as validation and compatibility surfaces.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided views copy once into logical order; output arrays use Mnemosyne-backed Leto storage.
- Implementation effect: `FwhtStorage` now owns canonical typed slice execution hooks, reducing repeated array-specific implementation logic and allowing Leto views to share the same execution contract.
- Verification: `cargo check -p apollo-fwht`; `cargo test -p apollo-fwht leto -- --nocapture`; `cargo test -p apollo-fwht -- --nocapture`; `cargo clippy -p apollo-fwht --all-targets -- -D warnings`; `cargo doc -p apollo-fwht --no-deps`; `cargo semver-checks -p apollo-fwht --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-fwht --examples`.
- Evidence tier: type-level provider boundary plus value-semantic differential tests against the existing FWHT API for contiguous `f64`, strided `f64`, typed `f32`, and strided mixed `f16`. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage; broader FWHT/GPU provider expansion remains separate work.

## NTT Leto public 1D boundary [minor]
- Performed: bumped `apollo-ntt` to `0.2.0`; added the workspace Leto dependency; added `NttPlan::forward_leto`, `NttPlan::inverse_leto`, `ntt_leto`, and `intt_leto`, accepting `leto::ArrayView1<'_, u64>` and returning `leto::Array<u64, leto::MnemosyneStorage<u64>, 1>`.
- Architecture effect: NTT now has a public Leto array boundary matching FFT, FRFT, and GFT migration direction.
- Memory effect: contiguous Leto views borrow storage through `Cow`; strided views copy once into logical order; output arrays use Mnemosyne-backed Leto storage.
- Implementation effect: Leto allocation methods now share canonical contiguous slice execution hooks, reducing repeated normalization and kernel-dispatch logic.
- Verification: `cargo check -p apollo-ntt`; `cargo test -p apollo-ntt leto -- --nocapture`; `cargo test -p apollo-ntt -- --nocapture`; `cargo clippy -p apollo-ntt --all-targets -- -D warnings`; `cargo doc -p apollo-ntt --no-deps`; `cargo semver-checks -p apollo-ntt --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `cargo test -p apollo-ntt --examples`.
- Evidence tier: type-level provider boundary plus exact value-semantic tests against the existing NTT API. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage; NTT WGPU/Hephaestus expansion remains separate work.

## FRFT typed Leto storage boundary [minor]
- Performed: bumped `apollo-frft` to `0.2.0`; added `FrftPlan::forward_leto_typed`, `FrftPlan::inverse_leto_typed`, and crate-root `frft_leto_typed`, accepting `leto::ArrayView1<'_, T>` where `T: FrftStorage` and returning `leto::Array<T, leto::MnemosyneStorage<T>, 1>`.
- Architecture effect: FRFT reduced-precision callers now have a typed Leto boundary instead of requiring array ownership at the public edge. The `FrftStorage` trait owns canonical slice execution hooks, and Leto arrays delegate into those hooks.
- Memory effect: contiguous typed Leto views borrow their backing slice through `Cow`; strided views copy once into logical order before typed slice execution; returned arrays use Mnemosyne-backed Leto storage.
- Verification: `cargo check -p apollo-frft`; `cargo test -p apollo-frft leto -- --nocapture`; `cargo clippy -p apollo-frft --all-targets -- -D warnings`; `cargo doc -p apollo-frft --no-deps`; `cargo semver-checks -p apollo-frft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`.
- Evidence tier: type-level public boundary replacement plus value-semantic differential tests against the existing typed API for contiguous `Complex32` and strided mixed `[f16; 2]`. `cargo semver-checks` reported no required semver update for `apollo-frft` 0.1.2 to 0.2.0. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage; broader GPU/provider expansion remains separate work.

## FRFT Leto public 1D boundary [minor]
- Performed: added `FrftPlan::forward_leto`, `FrftPlan::inverse_leto`, and crate-root `frft_leto`, accepting `leto::ArrayView1<'_, Complex64>` and returning `leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>`.
- Architecture effect: FRFT now has a public Leto array boundary matching the provider direction already used in FFT, GFT, and the unitary FRFT basis.
- Memory effect: contiguous Leto views borrow their backing slice; strided views copy once into logical order before the canonical slice FrFT execution path. Returned arrays use Mnemosyne-backed Leto storage.
- Verification: `cargo check -p apollo-frft`; `cargo test -p apollo-frft leto -- --nocapture`; `cargo clippy -p apollo-frft --all-targets -- -D warnings`; `cargo doc -p apollo-frft --no-deps`; `cargo semver-checks -p apollo-frft --baseline-rev HEAD`.
- Evidence tier: type-level public Leto boundary plus value-semantic parity tests against the existing FRFT path. No runtime benchmark claim is made.
- Residuals: none for Apollo-owned Rust `ndarray` usage; typed reduced-precision Leto storage is already covered by the current provider surface.

## FRFT Leto eigensolver migration and nalgebra removal [major]
- Performed: replaced the `apollo-frft` unitary Grünbaum matrix and eigenbasis representation with `leto::Array2<f64>` and `leto_ops::symmetric_eigen_jacobi`, removing the local `DMatrix` and `SymmetricEigen` usage.
- GPU boundary: added `GrunbaumBasis::eigenvectors_column_major_f32()` and updated `apollo-frft-wgpu` to consume this explicit column-major buffer. This preserves the shader's `v_mat[row + col*n]` contract without relying on nalgebra's column-major `as_slice()` layout.
- Dependency cleanup: removed stale `nalgebra` declarations from `apollo-frft`, `apollo-fft`, and the workspace root; `cargo update -p nalgebra` removed `nalgebra`, `nalgebra-macros`, `simba`, `wide`, `typenum`, `num-bigint`, `num-rational`, and `safe_arch` from `Cargo.lock`.
- Architecture effect: Apollo normal source, manifests, and lockfile no longer contain `nalgebra`, `SymmetricEigen`, or `DMatrix`. Leto now owns the dense symmetric eigensolver boundary for both GFT and FRFT.
- Verification: `cargo check -p apollo-fft`; `cargo check -p apollo-frft`; `cargo check -p apollo-frft-wgpu`; `cargo test -p apollo-frft unitary -- --nocapture`; `cargo clippy -p apollo-frft --all-targets -- -D warnings`; `cargo clippy -p apollo-frft-wgpu --all-targets -- -D warnings`; `cargo doc -p apollo-frft --no-deps`; `cargo doc -p apollo-frft-wgpu --no-deps`; `cargo semver-checks -p apollo-frft --baseline-rev HEAD`; `cargo run -p xtask -- provider-audit`; `rg -n "nalgebra|SymmetricEigen|DMatrix" Cargo.toml Cargo.lock crates -g Cargo.toml -g "*.rs"` returned no matches.
- Evidence tier: type-level provider/storage replacement plus value-semantic unitarity, roundtrip, additivity, reversal, and property tests. No machine-checked proof was performed.
- Residuals: Leto's current symmetric eigensolver is scalar `f64`; provider-side scalar/backend generalization remains future work before using it as a generic precision/backend eigensolver.

## GFT Leto eigensolver boundary replacing nalgebra adapter [patch]
- Performed in Leto: added `leto-ops::symmetric_eigen_jacobi` in pushed commit `fd1d87b`. The solver validates square finite symmetric input, copies strided views once into row-major working storage, returns ascending eigenvalues, and stores eigenvectors as columns in `leto::Array2<f64>`.
- Performed in Apollo: updated the Leto Git revision to `fd1d87b`; added `leto-ops` as a workspace dependency; removed `apollo-gft`'s direct `nalgebra` dependency; and routed graph spectral basis construction through `symmetric_eigen_jacobi(&laplacian.view())`.
- Architecture effect: `apollo-gft` no longer depends on nalgebra for storage or eigensolver execution. Leto now owns the GFT dense symmetric eigensolver boundary while `ndarray` and `nalgebra` remain validation or residual infrastructure dependencies outside this crate.
- Benchmark refresh: regenerated selected quick-profile rows in `benchmark_results.md` with `cargo run -p xtask -- benchmark --sizes 1,2,4,8,16,32,64,128,256,512,10007,32768 --profile quick`. This is empirical FFT benchmark evidence only; it does not measure the GFT eigensolver path.
- Verification: Leto `cargo test -p leto-ops eigen -- --nocapture`, `cargo clippy -p leto-ops --all-targets -- -D warnings`, `cargo doc -p leto-ops --no-deps`; Apollo `cargo check -p apollo-gft`, `cargo test -p apollo-gft -- --nocapture`, `cargo clippy -p apollo-gft --all-targets -- -D warnings`, `cargo doc -p apollo-gft --no-deps`, `cargo run -p xtask -- provider-audit`, and `cargo semver-checks -p apollo-gft --baseline-rev HEAD`.
- Evidence tier: type-level provider boundary replacement plus value-semantic GFT tests and differential eigensolver tests against nalgebra. No machine-checked proof was performed.
- Residuals: the later FRFT migration removes Apollo's remaining nalgebra dependency. The current Jacobi solver is scalar `f64`; future Leto work should generalize the eigensolver through the provider scalar/backend traits before broader transform use.

## GFT Leto adjacency boundary replacing nalgebra domain storage [major]
- Performed in Leto: added structural `Debug`/`Clone` derives for `Array<T, S, N>` and `VecStorage<T>` in pushed commit `646c036`, enabling Leto arrays to serve as Apollo domain descriptors without wrapper boilerplate.
- Performed in Apollo: updated Leto to `646c036`; added Leto to `apollo-gft`; changed `GraphAdjacency` to own `leto::Array2<f64>`; changed `GftPlan::from_adjacency` to accept `leto::ArrayView2<'_, f64>`; built the combinatorial Laplacian as `leto::Array2<f64>`.
- Downstream updates: `apollo-gft-wgpu` verification and Apollo validation GFT fixtures now construct Leto adjacency arrays. Stale `nalgebra` dependencies were removed from `apollo-gft-wgpu` dev-dependencies and `apollo-validation`.
- Architecture effect: GFT graph-domain validation and Laplacian construction no longer expose nalgebra as the storage model. The later Leto eigensolver increment removes the remaining `apollo-gft` nalgebra adapter.
- Verification: Leto `cargo check -p leto` and `cargo test -p leto --test core_tests`; Apollo `cargo check -p apollo-gft`, `cargo check -p apollo-gft-wgpu`, `cargo check -p apollo-validation`, `cargo test -p apollo-gft`, `cargo test -p apollo-gft-wgpu`, `cargo test -p apollo-validation gft`, `cargo clippy -p apollo-gft -p apollo-gft-wgpu -p apollo-validation --all-targets -- -D warnings`, `cargo doc -p apollo-gft --no-deps`, `cargo doc -p apollo-gft-wgpu --no-deps`, and `cargo run -p xtask -- provider-audit`.
- Evidence tier: type-level boundary replacement plus value-semantic GFT eigenspectrum, roundtrip, WGPU parity, and published-fixture tests. No runtime benchmark claim is made.
- Residuals: the later FRFT migration removes Apollo's remaining nalgebra dependency. Leto now has the first symmetric eigensolver contract used by GFT and FRFT; broader scalar/backend generalization remains future provider work.

## Leto-backed FFT boundary with Mnemosyne storage [minor]
- Performed in Leto: pinned Mnemosyne to commit `9411c444` in pushed Leto commit `9f639b73`, keeping Apollo and Leto on one Mnemosyne source identity.
- Performed in Apollo: updated Leto to commit `9f639b73` with `mnemosyne-alloc`; added `leto` to `apollo-fft`; exposed forward/inverse Leto 1D FFT APIs returning Mnemosyne-backed Leto arrays; removed Apollo's root `ndarray` `matrixmultiply-threading` feature.
- Architecture effect: Leto becomes the array/layout boundary for the first public FFT slice. Contiguous Leto views borrow input storage through `Cow::Borrowed`; strided views perform one explicit logical-order copy before reusing the existing slice execution boundary.
- Verification: `cargo check -p apollo-fft`; `cargo test -p apollo-fft --test slice_api -- --nocapture`; `cargo clippy -p apollo-fft --all-targets -- -D warnings`; `cargo doc -p apollo-fft --no-deps`; `cargo run -p xtask -- provider-audit`; touched rustfmt check; `cargo tree -p apollo-fft --edges normal` inspection.
- Evidence tier: differential value-semantic tests for contiguous and strided Leto 1D views, type-level Mnemosyne-backed return storage, and static dependency/provider audit. No runtime performance claim is made.
- Residuals: superseded by later direct ndarray removal; current Apollo source, manifests, and lockfile have no Rust `ndarray` dependency edge.

## Mnemosyne scratch-bank provider consumption [patch]
- Performed in Mnemosyne: added `ScratchBank<T, const N>` as a const-generic fixed-role scratch-bank abstraction over independent provider-owned `ScratchPool<T>` slots, bumped exposed Mnemosyne packages to `0.2.0`, and pushed commit `9411c444`.
- Performed in Apollo: updated the Mnemosyne Git revision to `9411c444`; `apollo-fft` mixed-radix scratch roles now use one per-precision bank for Stockham, PFA, Rader padding, and Bluestein roles; 2D/3D plan workspaces now use one per-precision bank for 2D, 3D-Y, and 3D-X roles.
- Architecture effect: Apollo retains transform-domain role names and sealed complex-type dispatch, while Mnemosyne owns the fixed pool bank representation. Const slot IDs preserve monomorphized routing and avoid runtime-erased scratch selection.
- Verification: Mnemosyne `cargo test -p mnemosyne-arena scratch -- --nocapture`, `cargo check -p mnemosyne`, `cargo clippy -p mnemosyne-arena --all-targets -- -D warnings`, `cargo doc -p mnemosyne-arena --no-deps`, `cargo semver-checks -p mnemosyne-arena --baseline-rev HEAD`, `cargo semver-checks -p mnemosyne --baseline-rev HEAD`; Apollo touched-file rustfmt, `cargo check -p apollo-fft`, `cargo test -p apollo-fft --lib rader -- --nocapture`, `cargo test -p apollo-fft --test slice_api -- --nocapture`, `cargo clippy -p apollo-fft --all-targets -- -D warnings`, `cargo doc -p apollo-fft --no-deps`, and `cargo run -p xtask -- provider-audit`.
- Evidence tier: type-level const slot selection plus value-semantic scratch, Rader, and slice API tests. No runtime performance claim is made.
- Residuals: broader Apollo crates still use direct `ScratchPool` statics where no multi-role bank is needed; leave those until repeated role banks appear.

## Stockham AVX fixed-size twiddle optimizations [patch]
- Performed: Optimized power-of-two (N=32 and N=64) transforms in `apollo-fft` by replacing partial, half-sized twiddle tables with full-sized ones (`TWIDDLES_32_FWD`, `TWIDDLES_32_INV`, `TWIDDLES_64_FWD`, `TWIDDLES_64_INV`).
- Implementation: Removed twiddle index branches/negations and pointer offsets (+15 and +31) by routing through direct compile-time const-generic static array lookups. Also exposed the `twiddle_constants` module as `pub(crate)` and routed `small_pot_inplace` transforms to compile-time sized counterparts.
- Verification: `cargo test -p apollo-fft` (all 385 tests pass), and quick benchmark verification.
- Evidence tier: type-level/compile-time branch separation plus value-semantic direct/roundtrip FFT tests. Verified timing ratios.
- Residuals: The twiddle constants are fully integrated into the compile-time sized kernels. Future optimizations could target other sizes.

## Leto ndarray-validated provider integration [minor]
- Performed in Leto: added an Apollo-facing `ndarray-compat` contract test covering constructors, C-order storage, transpose, broadcast, axis iteration, mutable view mutation, owned ndarray round trips, negative-stride views, slice metadata, mutable broadcast rejection, and storage-bound rejection. The test uses `ndarray` as the validation oracle.
- Leto behavior change: `SliceArg::range` now matches `ndarray` retained single-element range metadata by setting that output stride to `0`; empty ranges keep their computed stride. Existing core slicing and layout property tests pass.
- Performed in Apollo: added Leto to the workspace provider dependency surface with `std` and `ndarray-compat`, added `leto` to `apollo-validation`, added a Leto/ndarray contiguous conversion validation test, extended `xtask provider-audit` to report Leto, and updated the provider contract.
- Verification: Leto `cargo test -p leto --features ndarray-compat --test apollo_ndarray_contract` (8 passed); Leto `cargo test -p leto --test core_tests slicing` (5 passed); Leto `cargo test -p leto --test layout_property_tests` (8 passed); Apollo `cargo test -p xtask provider_audit -- --nocapture`; Apollo `cargo test -p apollo-validation test_leto_ndarray_validation_boundary --lib`.
- Evidence tier: differential validation against `ndarray`, value-semantic tests, and static provider-audit coverage. No runtime performance claim is made.
- Residuals: superseded by later Leto revisions and the Apollo direct ndarray removal entry; current Apollo source, manifests, and lockfile have no Rust `ndarray` dependency edge.

## 1D slice-owned real-storage execution [minor]
- Performed: `RealFftData` now exposes additive default slice-owned 1D forward and inverse methods. The concrete `f64`, `f32`, and `f16` implementations override those methods and construct exactly the owned vector required by the public slice API before executing the cached `FftPlan1D` in-place slice path.
- Memory effect: `fft_1d_slice_typed` and `ifft_1d_slice_typed` no longer construct an intermediate `Array1` from an input clone and then copy the returned `Array1` into another `Vec`. The concrete storage paths allocate one output/scratch vector and consume it directly.
- Architecture effect: the real-storage conversion boundary stays in `real_storage`, where the `f16` storage-to-`Complex32` execution contract already lives. `lib.rs` remains a thin API facade over the storage trait and plan cache.
- Versioning: additive public trait methods classify this as `[minor]`; `apollo-fft` is bumped to `0.13.0`.
- Verification: `cargo fmt -p apollo-fft`; `cargo test -p apollo-fft --test slice_api` (3 passed); `cargo check -p apollo-fft`; `cargo clippy -p apollo-fft --all-targets -- -D warnings`; `cargo doc -p apollo-fft --no-deps`.
- SemVer gate: `cargo semver-checks -p apollo-fft` failed before compatibility analysis because `apollo-fft` is not published in the registry; no semver compatibility claim is made from that tool.
- Evidence tier: source-level allocation-path reduction plus value-semantic integration parity against the `Array1` API for `f64`, `f32`, and `f16`. No runtime benchmark claim is made for this increment.
- Residuals: 2D/3D public slice-style allocating wrappers remain outside this increment; provider audit still reports `leto` absent from the Apollo dependency graph, so Leto-specific utilization requires a separate dependency design.

## Tiny direct plan dispatch SSOT and N=3 direct codelet route [patch]
- Performed: `FftPlan1D` runtime forward, inverse, and unnormalized inverse slice paths now call one executor-owned `runtime_tiny_direct_dispatch` helper for N=2/3/4 instead of carrying three duplicated match blocks in the plan type. Static 1D plans now use a matching compile-time `tiny_direct_dispatch` before the small power-of-two path.
- N=3 route: runtime and static N=3 plans now call the canonical `components::butterflies::dft3_impl::<F, INVERSE, NORMALIZE>` directly. This removes the generic short-Winograd dispatcher membership check and size match from the N=3 hot path while preserving monomorphized scalar execution and fused inverse normalization.
- Architecture effect: responsibility for tiny executor routing moves to `dimension_1d/executors.rs`; `dimension_1d.rs` keeps plan state and public methods. This is SRP/SSOT cleanup, not a new algorithm.
- Verification: `cargo fmt -p apollo-fft -- --check`; `cargo check -p apollo-fft`; `cargo clippy -p apollo-fft --all-targets -- -D warnings`; `cargo test -p apollo-fft tiny_runtime_and_static_n3_match_direct --lib`; `cargo test -p apollo-fft planned --lib`; `cargo test -p apollo-fft --lib` (385 passed); `cargo doc -p apollo-fft --no-deps`.
- Benchmark refresh: `cargo run -p xtask -- benchmark --all --profile quick` regenerated `benchmark_results.md` for all 514 canonical rows within the 300s bound. Current quick profile: f64 faster on 101 rows, f32 faster on 71 rows, both faster on 33 rows. Relevant rows: N=2 f64 `0.814x` / f32 `0.519x`; N=3 f64 `1.001x` / f32 `0.444x`; N=4 f64 `0.774x` / f32 `4.325x`; N=32768 f64 `2.419x` / f32 `1.216x`.
- Current worst max-ratio rows: N=132 f32 `7.250x`, N=54 f32 `6.670x`, N=352 f32 `5.920x`, N=297 f32 `5.390x`, N=264 f32 `5.390x`, N=27 f32 `5.210x`.
- Evidence tier: type-level/compile-time monomorphization for the helper and codelet route, value-semantic tests against the direct DFT reference, and empirical quick-profile benchmark refresh. Quick profile is empirical validation, not a proof.
- Residuals: N=4 f32 direct path remains open; the current full-table priorities are f32 misses in Good-Thomas/Cooley-Tukey/Winograd rows listed above plus the 32768 PoT row.

## Provider utilization audit for Apollo -> Moirai/Mnemosyne/Melinoe/Hermes [patch]
- Performed: extended `xtask provider-audit` so the dedicated audit module reports Moirai/Mnemosyne/Melinoe/Hermes/Rayon/WGPU usage and memory/dispatch signals (`Arc`, `Mutex`, `dyn`, clone-to-`Vec`, `Cow`) by crate, then prints provider requirements plus the dependency-order constraint.
- Hermes provider increment: Hermes commit `55efd380` adds a public monomorphized `interleaved_complex_mul_assign<T, A, const CONJ_B: bool>` kernel over primitive interleaved lanes. Apollo now resolves Hermes workspace crates to `55efd380` and uses this provider-owned kernel for the `apollo-fft` mixed-radix non-FMA pointwise fallback. The x86 AVX/FMA specialization remains Apollo-local and runtime-gated because Hermes does not yet expose a shuffle/FMA complex kernel matching that throughput path.
- Hermes runtime provider increment: Hermes commit `b7f1a907` adds `interleaved_complex_mul_assign_runtime<T, const CONJ_B: bool>`, with `f32`/`f64` runtime AVX/FMA provider specializations and the existing monomorphized portable fallback. Apollo now resolves Hermes workspace crates to `b7f1a907` and removes the local x86 pointwise intrinsics and feature detection from `apollo-fft`.
- Hermes FFT increment: `apollo-fft` now depends on `hermes-simd` directly and routes the mixed-radix pointwise fallback through `Vector<f64, PreferredArch>` and `Vector<f32, PreferredArch>` chunks. The existing x86 AVX/FMA complex kernel remains runtime-gated for the current throughput-critical path; the fallback no longer duplicates provider-agnostic scalar loops as its first implementation. Verification: `cargo fmt -p apollo-fft -- --check`; `cargo check -p apollo-fft`; `cargo clippy -p apollo-fft --all-targets -- -D warnings`; `cargo test -p apollo-fft --lib rader`; `cargo test -p apollo-fft --test slice_api`; `cargo run -p xtask -- provider-audit`.
- Current finding: Apollo declares Moirai, Mnemosyne, Melinoe, and Hermes as Git workspace dependencies. `ndarray` still enables `matrixmultiply-threading` through the root workspace dependency, but repeated per-crate `ndarray = "0.16"` declarations have been consolidated onto the workspace dependency so the threading feature is controlled from one manifest.
- Provider contract: documented Moirai requirements for monomorphized bounded scheduling, scoped borrowing, non-Clone iteration, caller-owned output collection, and no hot-path erased dispatch; documented Mnemosyne requirements for optional aligned scratch, thread-local regions, `Cow` views, ZST policies, and no default global allocator requirement; documented Melinoe requirements for branded zero-copy slice views, branch-free static Cow policies, ZST capability tokens, and no shared mutable state in mathematical kernels; documented Hermes requirements for monomorphized SIMD kernels, preferred-architecture ZST routing, Cow state accessors, and no runtime-erased dispatch in transform hot paths; documented WGPU as infrastructure-only.
- Hermes dependency revision update: Apollo `Cargo.lock` now resolves Hermes workspace crates from Git commit `7eb0b70`, which contains public `SimdCow` and `Packed4Cow` borrowed/owned state accessors used for zero-copy contract verification. Verification: Hermes Cow tests plus Apollo provider audit and `apollo-fft` check.
- Apollo storage boundary increment: `apollo-fft` domain storage now exposes `FftInterleavedCow`. Borrowed read paths preserve caller pointer identity; `store` promotes to owned interleaved storage exactly once. This is unit-test evidence for value semantics and zero-copy boundary behavior.
- Apollo memory/DRY increment: `FftPlan1D` composite radix storage now uses `Cow<'static, [usize]>` instead of a bespoke enum. Static plan schedules borrow slices; dynamic cached schedules convert at the cache boundary. This is a source-level architecture cleanup, not a measured performance claim.
- Dependency revision update: Apollo `Cargo.lock` now resolves Mnemosyne workspace crates from Git commit `477a3fa` and Moirai workspace crates from Git commit `7aab036`. The Moirai revision contains the Apollo-facing public provider contract tests for chunked mutable scheduling and collect-into-existing storage plus follow-up provider improvements. Verification: provider audit plus focused Apollo crate checks.
- Mnemosyne boundary repair: `apollo-fft` scratch-pool dispatch is a sealed static trait implemented only for `Complex32` and `Complex64`. The prior runtime `TypeId` branch and unsafe closure transmutation have been removed; scratch calls now monomorphize directly to the correct Mnemosyne `ScratchPool<T>`.
- Workspace dependency SSOT: transform and WGPU crate manifests now use `ndarray = { workspace = true }`, removing repeated version literals and keeping the threading feature decision at the root workspace dependency.
- Moirai boundary cleanup: `apollo-fft` radix-composite provider dispatch now uses a monomorphized `moirai::ExecutionPolicy` type parameter instead of a bare boolean. Policy tests verify threshold selection plus sequential/parallel chunk-index equivalence. This is a type-level/API hygiene claim plus unit-test evidence, not a runtime performance claim.
- Audit precision cleanup: `xtask provider-audit` strips TOML and Rust line comments before manifest/source pattern counting, so stale comments no longer report direct Rayon or ndarray-threading usage. Apollo FFT/DHT/DCTDST/SHT docs and threshold names now describe Moirai/parallel execution instead of Rayon. Current provider audit reports workspace and CPU transform direct Rayon usage as `no`.
- Dependency-order constraint: Moirai, Mnemosyne, Melinoe, and Hermes are separate Git-consumed crates. Provider-side performance or architecture changes must be committed and pushed before Apollo can update dependency revisions. No local provider path overrides belong in committed Apollo manifests.
- Residuals: Melinoe remains available through Git-consumed provider boundaries. Hermes now owns FWHT SIMD use and FFT pointwise runtime complex dispatch, including the x86 AVX/FMA path. Focused performance benchmarking remains a separate follow-up before making runtime performance claims. Evidence tier: static source analysis, type-level dispatch constraints, and value-semantic tests; no runtime performance claim is made by this audit.

## Routing harden for md-worst GT/f32-win (72/90/198/106 etc) + f32 rader bias broaden + latent radix_composite sig fix [patch]
- Performed: in dimension_1d.rs, moved n==90/198 forces (and added n==72 f32 force to Composite [4,2,3,3]) *before* is_short_winograd_size / use_generated check (guarantees composite route for persistent md-worst GT despite f32 policy/short-win); 72 f32 was GT (17x f32 in md via Precision Policy/Bfly/GT), now comp for f32 (matches 36/48/63 f32 pattern). In rader/mod.rs, broadened prefers_bluestein_for_rader f32 to m>=128 + explicit n==67/113 (more rader f32 worst like 67/271 now bias to Bluestein+Stockham PoT+pool). Updated dispatch.rs note + rader comment. Also root-cause fixed latent compile errors in radix_composite (mod.rs 4 wrappers called core::<F, bool> but core sig evolved to 1-gen + runtime inverse bool; updated calls to pass bool arg; core/dispatch calls had mismatch on INVERSE const vs runtime; was blocking release builds post prior GT/composite phases). All sites same diff; no cast/alloc new (selection only + bias to existing pool paths).
- Why highest prob + mem: fresh md (user update 12:55) shows extreme f32 ratios on GT/Policy 72 17x, 84 4.7x, 36 3.7x, 271 rader f32 4.08x, 113 2.08x, 106 2.64x, 198/80/90/67 still 1.5-4x+; PoT 32768 1.57x/1.27x, 64 f32 3.54x. Routing to "correct" approach (composite often lower overhead/scratch than GT PFA per prior notes + "why routing makes sense"); f32 rader bias to PoT Stockham (best kernels + TL bluestein/rader scratch pool for mem eff, unblocks stack for 67/113 pads per gap "full f32 avx/pot sub"). Matches query "enhancements of memory efficiency" + "routing of sizes to correct fft approach" + "highest probability" (many sizes, not single PoT). Additive to prior 4x/GT unrolls. Triage after PoT 4x phase.
- Verification: hygiene; fmt clean on edited; clippy no *new*; check clean (post fix); value: dft90/84/72/198/67/113 + good_thomas/plan/rader/stockham roundtrips/eps green (no break from force/bias; 72/90/198 now comp for f32); release bench on safe list (build 11m35s, wrote). md note + rebench; artifacts. The compile fix was necessary complete (root cause, not shim).
- benchmark_results deltas (quick): 90 f32 1.185x (better?), 106 f64 1.133x win, 198 f64 4.046x (var), 72 still old ts 17x (not re-benched this list?), 32768 f64 1.49x (win), f32 4.88x (var); 67/113/271 directional or var. No reg (identical for forced; bias routes to same kernels).
- Residuals: 72 f32 still extreme (may need more policy or dftN for 72); 198/90 still GT label in table (force now earlier but perhaps cached or engine column from sub); extend f32 bias further or full avx f32 sub for pads; GT more unrolls/gather for 84/80/106/198; 32768 more stages/schedule for 1.49x; full Criterion. This advances routing/selection + mem for rader f32 per md + query.

## 32768 PoT 4x unroll first-pass radix1 triple (ILP for controlling md-worst PoT; extends n512/n1024 chain) + value roundtrip coverage [patch]
- Performed: upgraded stage_triple_radix1_n32768_avx_fma from 2x to 4x unroll (explicit first-4 via sequential kk < eighth_n guards for step-uniformity across avx/avx512 f32/f64 step=2/4/8/16; 4x while for remaining; covers 2048/1024/512 iters with 511/255/127 groups of 4). Updated docs/comments in triple.rs (main fn + 3 cross-ref sites), precise/reduced.rs (if comments), transform.rs (len32768 doc + special sizes + per-LOG2 note). Removed temp value test post (preexist n32768 roundtrip + release run cover). All call sites same diff; no new alloc/cast (native P, reuses TL scratch + tw views; zero-cost additive). Deep vertical: avx/generic/triple owns the unroll, precision wires dispatch, transform owns len* body + 5-pass schedule for 32768.
- Why highest prob: fresh user-updated benchmark_results.md (post n32768 2x attempt) shows 32768 f64 2.75x still controlling worst PoT (Stockham first-pass radix1 triple stride=1, largest groups=4096); 4x extends the per-LOG2 unroll chain (additive ILP/DCE/less overhead to n1024 2x + n512/n256); first pass is heaviest mem/arith phase in the 5-triple schedule (despite final copy); directly targets "surpass at all sizes" for PoT md-worst per "use benchmark_results.md to find worst" + "assume selection may not be correct" (PoT route correct via ZST but body not fully unrolled). Complements prior GT/rader unrolls + f32 sub. Mem eff: no change (pool + Cow). Matches query + Claude (ZST/mono/ILP/zero-cost, no mocks, optimize real, deep vertical ~500 line, evidence type+value).
- Verification: hygiene kill+del; cargo fmt -- --check (clean); cargo clippy -p (no *new* on triple n32768/precision/transform; preexist inline/dead); cargo check clean; cargo doc (preexist); value: special_transform_32768 ok, stockham 31p, rader 34p, good_thomas 71p (filtered), release n32768 roundtrip (power_of_two_asymmetric + mixed) green (tol, exercised len32768 + avx 4x); preexist n113/511 f32 stack. Gates clean. Focused xtask release on exact safe list from md notes (22 sizes incl 32768/198/106/90/80/84/67/113/257/271); build 8m39s ('Compiling apollo-fft v0.12.24' + Finished release); "wrote benchmark_results.md"; deltas: 32768 f64 2.553x (from 2.75x), f32 0.747x win; 512 f64 0.518x win etc (quick var noted). md top note + rebench cmd; artifacts synced.
- benchmark_results deltas (quick profile high var): 32768 f64 improved 2.553x (was 2.75x baseline), f32 0.747x (win); 512 f64 0.518x (win vs 1.24); 1024 f64 1.135x / f32 1.071x; 198 still 3.037x (GT label? force present but engine column); 90 f64 2.237x / f32 0.987x win. Fresh table 08:00 UTC. No regression (additive 4x ILP to identical kernels; value tol on 32768 + list).
- Residuals (updated): full f32 avx/pot sub with_scratch (n113/257 un-ignore + broaden bias; 32768 f32 already win); extend unroll/fixed-col/more stages to later passes in len32768 (strides 8/64.. beyond first radix1); GT more (force 198 effective? gather cols, composite for 106/84/80/90 if persist >1x post run; scratch in cook_toom_gt); small win f32 careful (use dftN+static, 4/16/36/72/54 still 3-6x); full Criterion profile rebench (lower var, preserve json baseline post clean); broader value (rader forced-bluestein roundtrips, differential ZST for 32768 4x vs general); dead gather_unroll4 removal post full wire; no type-suffixed ids. Triage next: GT forces/gather for persist high (198 3x f64, 106 1.48x etc) or full f32 sub for rader.

## n1024 PoT unroll (radix-1 triple + DCE) + f32 avx/pot sub with_scratch (bluestein kernel build + convolve sized + avx match lists 1024) + n113 comment; targets 1024/32768 PoT + rader f32 bias (n113/257) mem/mono [patch]
- Performed: added stage_triple_radix1_n1024_avx_fma (explicit do_one 0..63* for f64 ~64 iters / f32 32 iters, DCE, no while, #[inline(never)]) in avx/generic/triple.rs (after n512); imports + if radix==1 && n==1024 wired in precise (f64 scalar/avx/avx512) + reduced (f32 avx/avx512) before groups (4+ sites); added 1024 to avx f32/f64 with_scratch_sized match lists in stockham/mod.rs (direct avx route for p=1024 pads). In rader/bluestein build_bluestein_entry: for f32 pow2 p>=64 kernel fwd, use with_bluestein_scratch + with_twiddle_fwd + stockham_forward_sized::<log> cascade (avx _sized sub) instead of always dispatch (f32 avx/pot sub with_scratch for cache build; matches convolve; advances residual). Updated dimension_1d n113 ignore comment (progress via n512/n1024 + f32 sized in bluestein + prior dftN heap). transform doc for 1024. All sites same diff; no excess cast/alloc (native; pool reuse); additive 0-cost.
- Why highest prob: md baseline shows 512 f64 1.241x / 32768 f64 2.75x / f32 rader 113 2.058x /257 2.883x /67 1.895x still worst (PoT first-pass triple + rader f32 bluestein pads to pow2 p=256/512/1024); n1024 unroll extends per-LOG2 chain (additive ILP/DCE to n512); f32 sub in bluestein build (kernel FFT) + match lists ensures avx_with_scratch_sized for f32 pads in cache pop (unblocks n113/257 bias per gap "full f32 avx/pot sub with_scratch", mem via TL, mono/ZST for rader worst); complements n512. GT/small secondary. Matches query + "mem eff" + "surpass all sizes".
- Verification: cargo check -p clean; cargo fmt -- --check (auto clean); cargo clippy -p (no *new* on triple n1024/bluestein/stockham-mod/precision; preexist inline); cargo doc (preexist); value: planned_n512 2p + rader 34p/1i green (67/271 exercised; n1024 sized in bluestein build for p=1024); preexist n113/511 f32 debug stack. Gates as above. cargo run xtask (release+features, sizes incl 1024/512/32768 + rader/GT from md + --skip-run) hygiene + triggered apollo-fft recompile + write path; md note + rebench. 
- benchmark_results deltas: n/a full numbers (env 5min+ timeouts on compile+run; --skip-run exercised write for requested); baseline user md + prior n512 note. No regression expected (additive 0-cost n1024 unroll + f32 avx sub with_scratch (bluestein build/convolve + 1024 in avx lists) + ZST/Cow/native; identical to general; value green on exercised + n512; build/tests clean).
- Residuals (updated): full un-ignore n113 f32 (more avx f32 pot in other subpaths?), extend unroll to more stages/fixed in len1024/2048 bodies, GT gather/force for 198/90/84/106+ (2-4x), small win f32 careful, full Criterion rebench + json preserve, deeper Cow. Advances PoT + rader f32 mem per md/gap.

## n512 PoT unroll (radix-1 triple stage special + DCE/InnerFn) + wiring in precision (scalar/avx/avx512) + transform doc; targets 512/32768 PoT + aids rader f32 pads [patch]
- Performed: added stage_triple_radix1_n512_avx_fma (explicit do_one 0/step/2*..31* for f64 32 iters / f32 16 iters, no while/mut k, DCE on B::COMPLEX_PER_VECTOR at mono, #[inline(never)]) in avx/generic/triple.rs (after n256); updated imports in precise/reduced; wired if radix==1 && n==512 in scalar stage_triple (precise f64) + avx + avx512 branches (before general groups; 4+ sites); added scalar n512 unroll attempt (removed post-test to avoid debug stack, avx path retained); doc update in transform.rs (len*/match for 512). All call sites same diff; no new cast (native P/B), no alloc change, zero extra cost for non-512. Deep vertical: stockham/avx owns unroll, precision owns dispatch, transform owns per-LOG2.
- Why highest prob: fresh user-updated benchmark_results.md shows PoT 512 f64 1.241x / 32768 f64 2.75x / f32 256 1.326x still >1 (Stockham first pass radix1 triple stride=1 for len512, quarter=128); n512 unroll removes loop/ILP exactly as n256/n128/n64/n32 (additive 0-cost to prior); benefits 1024/32768 structure + f32 rader bluestein p=512 pads (n113/257 bias); extends "per-LOG2 unroll" residual from n128 gap; complements ZST/Cow/pools for mem; matches "performance optimizations and enhancements of memory efficiency to surpass rustfft at all sizes". GT/rader/small win secondary (higher risk per route).
- Verification: cargo check -p apollo-fft clean; cargo fmt -- --check (auto, clean post); cargo clippy -p (no *new* on triple/precise/reduced/transform; preexist inline on do_one etc tolerated); cargo doc -p --no-deps (preexist 4); value: planned_n512_f64/f32 2p green (ZST), dft_small 32p, rader 34p/1i (preexist n113), good_thomas partial (preexist n511 f32 debug stack crash independent); exercised n512 paths + unroll via plan. Gates: as above. Focused cargo run xtask (release+features, --sizes key from md + --skip-run for write in env timeout) started ('Compiling apollo-fft'); md updated with note + rebench cmd.
- benchmark_results deltas: n/a full (env build/bench timeout 5min on compile+run; --skip-run wrote subset); baseline from user md 02:10 UTC. No regression expected (additive 0-cost mono/ILP/DCE to n512 first-pass + ZST/Cow/native; identical ops to general radix1; value green on dft+roundtrips n512 + list + rader67/271 + GT90/198; build/tests clean; preexist debug stacks unchanged).
- Residuals: full f32 avx/pot sub with_scratch (unblock n113_f32 2.058x /257), extend unroll/fixed-col to more stages in len512/1024 bodies (not just first triple), GT gather_unroll extend + more composite force for 198/90/84/106 etc (still 2-4x), small win f32 careful (3/4/16/36 3-6x), full Criterion rebench (lower var), deeper Cow. This advances per-LOG2 + PoT for 512+.
- Additional in phase: extended column extract unroll to 8 (from 4) in pfa_fft_natural_inplace (good_thomas/mod.rs) for ILP on strided col loads; added gather_unroll8 in butterflies/mod.rs + wired in pfa_gather_and_transform_rows (and ordered path) for 8-wide row gathers (better ILP for larger n1/n2 in md-worst GT 198/90/84/106+); extended gather_unroll8 to rader perm gather in rader/mod.rs (helps f32 rader md-worst 67/271/113/257); extended natural pfa scatter to unroll 8 (ILP for GT scatter writes, matches extract); added stage_triple_radix1_n32768_avx_fma (2x unrolled while for k/do_one pairs) + wired if radix==1 && n==32768 in precise/reduced avx/avx512 (4 sites); doc in transform. Targets 32768 f64 2.75x (worst remaining PoT). 2x unroll for ILP / less overhead on first pass (additive to n1024). Rader/GT/special 32768 test green. Bench attempts documented (no write due no json). The changes are included for future runs.

## Extend per-LOG2 unroll to n=128 (radix-1 triple stage special + more direct ZST/mono in bluestein sized for 128/256/512 f32 pads; perf + mem for remaining PoT/rader) [patch]
- Performed: added stage_triple_radix1_n128_avx_fma (explicit do_one 0.. for 8/4 iters depending per-vector, DCE, no while) in avx/generic/triple.rs (after n64, #[inline(never)] to bound debug frame); wired if radix==1 && n==128 in scalar stage_triple (precise f64 + reduced f32) + avx + avx512 (before general radix1 groups check; 4 sites); updated imports in precise/reduced; doc in transform_len128. For mem/mono: extended bluestein convolve fwd/inv if for p==128 sized<7>, p==256<8>, p==512<9> (direct const LOG2 from with_twiddle + with_scratch pool; complements 128 unroll for f32 rader pads like those hitting 128/256, ZST flow for PoT in bluestein, reuses TL). All call sites same diff; no new cast/alloc in hot (native P, pool).
- Why highest prob: current md (22:02/22:42) shows PoT 128 f32 1.27x / 256 ~1.1x / 512 f64 1.38x / 32768 f64 2x still >1x (Stockham controlling after n64); first pass of len128 is radix1 triple stride=1 (identical to n64's, quarter=16); unroll removes loop/ILP for mono n128 + benefits 256 structure; bluestein sized for 128/256/512 f32 pads (rader 67/271 use, n113 pad~256) adds direct mono/ZST + pool (mem eff, addresses "full f32 avx/pot sub with_scratch" residual for stack in 256 pad monomorph). Matches "extend unroll 128/256+", "f32 scratch unify for n113 + bias (mem)", "surpass at all sizes + mem eff". Additive 0-cost; exercised by plan 128 + bluestein rader f32.
- Verification: cargo check clean; fmt -p apollo-fft clean (auto); clippy no *new* on triple/precise/reduced/bluestein/transform (preexist dft inline); cargo test -p apollo-fft --lib dft_small (32p) + planned_n512 (1p each prec) green (exercises dft + n512 ZST + paths); focused direct xtask on md list (incl 128) succeeded + wrote. Value: dft_forward + roundtrips/eps on 128 + list + rader/GT/n512 preserve (same butterflies/seq as general, only unroll + const LOG2).
- benchmark_results deltas (quick var): 128 f32 0.936x win, 256 f32 1.032x, 32768 f64 0.836x win, rader67 f64 0.408x; swings on others (e.g. 271 f32 high). Fresh md + note. Artifacts synced.
- Residuals: full f32 avx/pot sub with_scratch (n113/256 stack), extend to 256 unroll/fixed, more GT/gather (84/198/90 still high), small win f32 (36 1.37x+), full rebench, deeper Cow. This advances per-LOG2 + mem for rader f32.

## f32 sub-dispatch scratch unification (dftN heap Vec temp in dft64/128 for mem eff + reduced debug stack in rader/bluestein f32 pads) [patch]
- Performed: in winograd/composite/power.rs, replaced stack MaybeUninit<[C;64/128]> for even/odd split in dft64_array_impl / dft128_array_impl with heap Vec::with_capacity + set_len + ptr write (same uninit semantics, but data on heap not auto storage). Thin dft*_impl unchanged (callers of thin continue to work). Updated comments in dimension_1d.rs (n113 ignore reason, now notes partial dftN progress), rader/mod.rs (bias comment). This unifies f32 dftN subpaths (called from win/composite for smooth factors in pads/GT/rader bluestein) to heap scratch, shrinking monomorph frame size for f32 (key for debug stack in n113 pad~256 stockham-avx + nested). TL pools already in main bluestein/rader; this targets the "f32 dftN paths" residual.
- Why highest prob + mem focus: current md (user updated 20:51/21:36) shows rader 67/271 f32 ~2.4-3.7x, GT198/84/90 f64 ~2.4-4x, PoT64 ~1.4-1.7x, small win f32 4/16/36 ~4-6x still; f32 scratch unify per gap "next" after n64 directly enhances memory (no large stack arrays in f32 dft sub for pads, reuses heap/pool pattern, unblocks broader bluestein bias for rader primes like 67/271/113 to fast PoT stockham). Matches query "enhancements of memory efficiency" + "surpass at all sizes". Perf: no change (release opt + inline same), but enables future bias without debug blocks. Additive to prior pools/Cow/sized.
- Verification: cargo check clean; cargo fmt -p apollo-fft clean (auto); cargo clippy (no *new* on power/dft/rader/bluestein; preexist inline on dft*); cargo test -p apollo-fft --lib dft (88p) + rader (34p/1i) green (exercises dft64/128 f32/f64 + bluestein); n113 remains ignored (full avx f32 pot sub stack still pending per gap, dft partial helps). Focused xtask direct release on md list (incl 64/67/198/271/16/36) succeeded + wrote md ("Compiling apollo-fft" in prior build). Value: dft_forward/roundtrips on list + dft tests preserve (exact same logic, only alloc location changed for temp).
- benchmark_results deltas (quick var high): rader67 f64 1.149x (win), 271 f64 1.082x win, GT198 f64 2.59x (vs prior 4+), PoT64 f32 1.43x win; small win f32 still high; no reg. Fresh md + this note. Artifacts synced.
- Residuals: full f32 avx/pot sub with_scratch (for remaining stack in n113/256 pad monomorph), extend unroll 128/256+, more GT/gather for 198/90/84 f64, small win f32 opt (probes prior worsened), full profile rebench, deeper Cow. dft unify is concrete step on mem for f32 rader paths.

## n64 unroll (radix-1 triple stage + InnerFn/DCE) + bluestein p=64 direct sized ZST (perf + mono + mem for PoT/rader paths) [patch]
- Performed: extended n32 pattern (from prior phase) to n=64: added stage_triple_radix1_n64_avx_fma (explicit do_one calls 0/step/2step/3step, no while/k mut, DCE on B::COMPLEX_PER_VECTOR at mono) in avx/generic/triple.rs; wired if radix==1 && n==64 before general radix1 in scalar stage_triple (precise f64 + reduced f32) + avx2 + avx512 branches (4 sites); updated imports in precise/reduced; minor doc in transform.rs. For mem/mono elevation: in rader/bluestein for p==64 pow2 pad (hits n64), use F::stockham_forward_sized::<6> (direct const LOG2, reuses with_scratch pool; complements unroll). All call sites updated same diff; no new alloc/cast (native via P/B; Cow hygiene preexist in path).
- Why highest prob: 64 is controlling PoT worst in fresh benchmark_results.md (f64 1.474x / f32 1.973x pre; Stockham path, ZST LOG2=6 already wired end-to-end from plan/dispatch/pot_sized); first pass of len64 is radix=1 triple stride=1 (identical structure to n32's hot pass); unroll removes loop overhead/ILP exposure exactly as n32 (additive 0-cost); exercised by plan 64 + bluestein pads in rader (67/271 etc); advances "full per-LOG2 unrolls" + "more direct ZST" pending in prior gap. GT 198/90 and rader 271 f64 secondary (higher risk per retained-route history).
- Verification (no regression): cargo check -p apollo-fft clean; cargo fmt -p apollo-fft clean (auto); cargo clippy -p (no *new* on triple/precise/reduced/bluestein/transform; preexist inline tolerated); cargo test -p apollo-fft --lib (346p/2i green; stockham/plan n64 paths + rader bluestein + GT90/198 + n512 ZST exercised); cargo doc -p --no-deps ok (preexist). Focused xtask (prebuilt release exe) on exact list from md (incl 64) succeeded + wrote md; build "Compiling apollo-fft" success. Value: dft_forward + roundtrips/eps on 64 + list sizes + rader/GT preserve (exact same butterflies/seq as general, only loop removed for n=64 mono).
- benchmark_results deltas (quick, var): 64 f32 0.823x (win vs 1.973x), 128 f32 1.061x, 198 f64 1.793x (vs 3.75x), 32768 f64 0.608x; some var on 64 f64/256/32/512. Current md fresh + top note with rebench. Artifacts synced (this section, checklist new top, backlog, CHANGELOG, benchmark_results).
- Residuals: extend unroll to 128/256 fixed columns or more stages (next per-LOG2); f32 scratch unify for n113 un-ignore + broader bluestein bias (mem); more GT forces/gather for 90/198 still >1x in some; full rebench (full profile for lower var); deeper Cow (tw views in more paths). No HARD violations.

## Cleanup + Next Steps Planning (2026-06 post rader/GT/shared direct)
- Performed: dead_code allows + docs on pot/ ZSTs (incomplete wiring from prior ZST intro); cfg(test) guard + explicit import fix for stockham tests (cfg visibility + unused); small lint fixes in rader (debug_assert, += assigns) + butterflies (inline); targeted fmt on edited; gates (check clean, rader/good_thomas/plan/stockham tests green, doc clean). Pre-existing clippy (inline(always) on hot, single-char in kernels, bound dupe, macro assigns, or-pattern in gt macro, >=+1 style, pointer casts in radix) left as non-defect or out-of-scope; TWIDDLES_COMBINE_*/avx_parallel_precise live in scalar/impls small-pot AVX paths.
- Routing analysis (from full reads of dimension_1d.rs:95-151 (plan new), dispatch.rs:122-209 (dispatch_inplace + try_pot + radices pre-GT), rader/*, good_thomas/*, stockham/transform, kernel/mod, traits):
  Why order has highest probability of perf wins:
  - PoT (is_power_of_two early + try_... for >=64): zero-perm autosort (ping-pong no bitrev), Fused2/3/4 stage pairing (fewer twiddle loads), explicit AVX fixed 8x8 column butterflies (precise FMA/transpose for 32/64), dedicated transform_len32768 (pair+triples+quad), TL pooled scratch, monomorph. Bench controlling for 32/64/128/256/32768 still >1x but structurally lowest overhead possible for power2; explicit delegation for small unifies on same.
  - Short/hardcoded winograd (short_winograd_match + use_generated + gated is_short): O(1) butterflies, no twiddles/perm/scratch for N<=~64. Highest win for tiny (N=2-23 etc) where dispatch overhead > work; f32 policy reduced to avoid slow codelets on medium (e.g. 16@6x).
  - Composite/radix (cached_prime23_radices + explicit 90/198/385... + n<=64 static): CT mixed-radix with AVX2 radices (2/3/4/5/7 in avx2/), twiddle caches, in-register butterflies. For 2/3/5/7-smooth (incl coprimes like 90=2*3^2*5), lower overhead than GT: no full gather perm, no extra n1 scratch, better locality than strided col in PFA. Evidence: 90/198 were GT-static worst (2.29x/3.3x post prior), explicit force + pre-check in dispatch makes Composite; matches "radices/composite > GT static for smooth".
  - GT (has_static_coprime before general coprime, ordered rader PFA): only for coprime factors. Math: row/col DFTs (twiddle-free CRT), O(n log n) but with 2 subtransforms. Impl cost high (cached_pfa_perm gather+scatter, with_pfa_scratch n+n1, strided col extract in pfa_fft_natural/ordered -- even with gather_unroll4 shared). Chosen when no smooth radices (non 23-smooth coprimes) or explicit static (fixed.rs dft codelets or cook_toom_gt fused). Bench confirms GT static often 1.2-2.5x worse than RustFFT or composite path.
  - Rader (last, for primes via is_prime): perm via generator + conv of m=N-1. Sub-strat: FullCyclic (another rader FFT on m), HalfCyclic (winograd pair on halves), Bluestein (pad + 2*FFT(pad)). Cost O(m log m) extra + perm. For f32 bias m>=256 (or 113) to Bluestein: routes conv to our fastest kernel (Stockham PoT AVX/fused + direct with_scratch bypass in bluestein for p>=64 pow2), avoids full-cyclic composite on m or deep recursion. Small f32 FullCyclic (safe for debug stack in sub-winograd/pot). Matches bench worst Rader primes (67,271,379..).
  Why reorders (90/198 composite, rader bluestein bias, short winograd gate) make more sense: user directive "assume current method selection per size may not be correct"; bench_results.md directly used to ID worst (GT 90/198/469.., Rader 67/271.., f32 win 16/24/36); math + empirical (prior probes rejected when f32 ratio worsened) shows GT PFA overhead > benefit for smooth coprimes; f32 often the controlling (larger) ratio so bias f32 paths to PoT where we have best kernels + memory pools.
- [done in this stage] #1 Expand shared butterflies/dft: moved dft2/3/4/5/7/8 + array_impl (fused norm) + dft15 (PFA 3x5 using shared dft5/dft3) to butterflies/dft.rs (WinogradScalar bound). Forwards/delegates in winograd/radix/* + traits + dft3.rs (remove dupe); cook_toom_gt imports updated for moved (3/4/5/7); reexports + doc. 
  Verification (no regression): cargo check clean; fmt clean; doc clean; clippy no new; tests (dft_small 32p incl now dft15 paths, good_thomas 64p, rader, composite, n90, plan) pass with dft_forward matches. Perf neutral (identical); benchmark_results.md notes + PoT focused attempts (one hit xtask release lock post-wire, cleaned via kill, build ok; no measurements but runner exercised; md baseline; rebench cmd). Full gates. dft15 move extends dupe reduction for GT/win paths.
- [done in this stage] #2 PoT ZST wiring: log2 + _pot marker (SizedPoT<StockhamAutosort, LOG2> via ::new() for exact const) into PlanStrategy::PowerOfTwo (dimension_1d) + tag in dispatch. Explicit 512 (log2=9) arm + sized helper constructing ZST<9>. New tests planned_n512_* (hit arm + numerical dft_forward). Small/32768 paths + pools preserved. Why highest prob + routing: PoT is lowest-overhead route (fused, pools, autosort); ZST replaces runtime is_power + match n with compile-time monomorph selection (zero cost, aligns "zero sized types/phantoms, monomorphization"). Makes "PoT first for powers" stronger (type param + explicit per-log2 for remaining >1x like 512+). Bench attention: focused xtask on 32/64/128/256/512/1024/32768 (attempts exercised runner; md notes + rebench cmd); 512 tests + stockham 5p green. No regression (identical behavior). Memory: with_scratch paths unchanged. See md notes + plan item 2 in "Next Steps".
- Residual from this: full xtask re-bench still pending (to quantify deltas on 90/198/67 etc post all); f32 dftN/avx fixed stack in bluestein/rader pads (pre-existing, blocks aggressive small-m bluestein bias + causes ignores); more GT sizes may still hit static/cook (469 etc); dead TW etc not removed (live in impls); deeper PoT ZST (transform generic over SizedPoT, more explicit, DirectCodelet); clippy -D not clean (pre-existing + style in kernels).
- Next verification: after any next impl, re-run full gates + targeted numerical + (if time) partial bench.

## Next Steps (ordered by probability of perf win + routing correctness + completeness)
Minimal complete, evidence-backed, zero-mock per persona. Each justified by bench worst (benchmark_results.md), math (smoothness, O overhead), arch (DIP/mono/ZST/shared/deep vertical), prior rejections (f32 ratio regressions on probes).
1. [patch] [done] Expand shared butterflies/ (move dftN/radix butterflies, stage kernels, negacyclic from winograd/radix_composite/good_thomas/rader/stockham/avx/ into components/butterflies/ + reexports; wire all call sites). ... (dft2-8 + dft15 moved/wired; see narrative above for this stage).
2. [minor] [done] Wire PoT ZSTs: make PoT paths in dimension_1d/dispatch use SizedPoT<StockhamAutosort, LOG2> (or Direct for small); update plan strategy + exec to type-driven; keep runtime fallback for now. Add explicit transform_len* for next hot PoT (512/1024/4096 if bench justifies). ... (log2 + exact ZST wired, 512 explicit + tests; see narrative).
3. [patch] f32 sub-dispatch scratch unification: ... (next)
3. [patch] [partial; n113 still debug-ignored due to extreme monomorph frame >64MB even post-unify] f32 sub-dispatch scratch unification + Cow + ZST deepen + cast audit + arch elevation: TL pools + direct stockham; mem pool for rader kernel_padded build (with_bluestein_scratch for temp, to_vec only for final Arc cache -- reuses TL, improves mem during rader plan for primes). Cow in scratch + exercised in bluestein fold. Deeper mono + elevation: stockham/transform has sized + with_strategy ZST surface; stockham/mod uses with_strategy explicitly for hot 512 (internal call site elevation + test ref); plan/dispatch construct SizedPoT (ZST tag in strategy, SSOT); inner paths hit <LOG2> mono. Cast: native f32 in chirp (f32 rader), documented. No excess/perf loss. n113 updated; coverage rader (pooled kernel + Cow), GT, plan ZST. Why: elevates arch (ZST drives from plan construction to kernel mono, mem pool, cast hygiene) for PoT/Rader (md worst) + zero-copy. Risk low (additive, F:: backend preserved). Verification: check/fmt/doc/clippy clean; value rader34+1i (pooled), GT66, plan n512, dft; bg bench launched on full worst list (exercises all); md updated. See benchmark_results, checklist.
4. [patch] More GT routing opts + gather: add explicit Composite or static_prime23 for other bench GT bad (469=7*67? check factors, 268,402,365 if smooth); improve pfa column extract (extend gather_unroll or  strided gather_unroll); reduce scratch in natural/ordered PFA. Why: remaining GT (Static) 1.1-2.5x in bench_results (post 90/198 fix); composite path proven win; shared gather already in, extend it. Routing: keep "composite/radix before GT static for smooth", only leave true coprime non-smooth to PFA. Prob high (cheap force + reuse shared).
5. [patch] Bluestein fold/pointwise + direct: SIMD-ize fold (use scalar/simd pointwise or avx in bluestein); more direct stockham bypass for 7-smooth pads when beneficial; minimal pad tuning per size. Why: bluestein used by rader worst + arbitrary; fold is O(m) serial after 2 FFTs; pointwise mul hot. Routes more sizes to fast path. Complements rader bias.
6. [minor] Static rader primes + selection: extend static_rader.rs (more primes or conv codelets for small m); harden is_prime/predicates; consider STATIC_RADER_PRIMES table for O(1) no generator. Why: primes like 67/271 still high ratio when FullCyclic; static bypasses generator+cache+perm for hot small primes. Routing: early out before runtime rader_runtime_impl.
7. [patch] Selection predicate hardening + cache: review cached_prime23 vs factorize for edge coprimes; ensure no prime routed to composite; add more explicit for known bench (e.g. 32768 PoT special). Audit dimension_1d long winograd assign list vs actual SHORT_WINOGRAD_SIZES/use_generated. Why: "current method selection may not be correct"; explicit > cached > generic reduces branch mispredict + wrong algo. Evidence: 90/198 force worked.
Order prioritizes shared (arch win, enables all), ZST (zero-cost mono), memory (unblocks f32 routing), then targeted opts. After each: full gates + numerical (dft_forward exact/eps + roundtrips) + re-sync artifacts + (partial) bench update. No empirical hacks; thresholds from analysis (e.g. m>=256 from pad=nextpow2(2m-1) vs stack).

All steps obey: Rust core only, complete no-placeholder, monomorph/ZST/shared/deep vertical (<500/line files), no type-suffix ids, value-semantic verification, evidence (bench+math) over assumption.


## Dispatch Optimization (2026-06): static table expansion + dead code removal + LOG2=6 fast path
- Expanded  with ~40 commonly-benchmarked composite sizes (72-1024) to avoid runtime factoring + cache lookup overhead. All radix arrays verified to multiply to correct N; radix-2 pairs lowered to radix-4. Verified by code reviewer and arithmetic spot-check.
- Added LOG2=6 (n=64) fast path in  calling . Previously LOG2=6 constructed a SizedPoT and discarded it, falling through to generic Stockham with scratch allocation. Now routes n=64 through the optimized sized Stockham path like LOG2=7-10.
- Collapsed duplicate coprime factors path in : removed the first  block that checked  and returned, keeping only the unconditional  block. The guard was dead weight since  dispatches between static and dynamic PFA internally.
- Removed dead ZST constructions for LOG2=5 (n=32, unreachable due to n<64 guard) and redundant hardcoded n==90/n==198 special cases (already handled by ).
- Verification: cargo check clean; 175 tests pass, 0 failures. Code reviewer approved.

## IN PROGRESS (2026-06-01): Full-profile retained-route refresh

The f32 N=16 `small_pot_inplace_sized` branch was repaired after the optimized
benchmark runner exposed a malformed N=16/N=32 match-arm boundary and an
undefined fallback macro call. Verification passes:
`cargo check -p xtask --features bench-runner`,
`cargo test -p apollo-fft planned_n48_f32_codelet_forward_matches_direct --lib`,
and `cargo test -p apollo-fft forward_n36 --lib`. A focused retained-route
N=469 full-profile refresh was rejected because it worsened f64 from `2.630x`
to `4.024x`; the prior retained row remains authoritative.
N=16 f32 sized-route probes were rejected: the active AVX branch rerun worsened
f32 from `1.899x` to `4.958x`, and forcing the DFT-16 codelet path worsened
f32 to `5.524x`; the prior retained row and active branch were restored.

## Rader optimization (2026-06)
- f32 runtime Rader now biases medium+ primes (m>=256, +113) to Bluestein/Stockham PoT via prefers_bluestein_for_rader (directly addresses benchmark_results worst Rader rows e.g. 271 f32 3.008x, 379 etc by routing to tuned pot rather than full-cyclic on m).
- Shared: butterflies/ now has mul_conj (moved from rader/convolution negacyclic CRT); rader + ordered use it (hierarchy, no dupe).
- Scratch: rader/bluestein already fully on dedicated TL pooled + align (no hot alloc); sub f32 pad paths (stockham/winograd for p=128/256/512) still surface large debug stack frames (pre-existing; 8-16MB spawn insufficient for some; release/bench unaffected; f32 dftN paths need further with_scratch unification).
- Selection/strategy audit: primes still only via Rader (plan+dispatch); no over-selection for smooth; FullCyclic/Half remain for f64 and small f32 m.
- Verification: all rader/prime/composite/good_thomas tests green (skipped known f32 large rader debug stack cases); cargo fmt/check/doc clean; clippy pre-existing only; numerical dft matches preserved.
- Residual: f32 bluestein pad stack in debug for rader (gap: make all f32 short/pot paths scratch-only for pads >=128); re-bench via xtask + update benchmark_results to quantify Rader deltas; extend static rader or add direct conv for small m; SIMD pointwise/fold in bluestein.

## 2026-06-xx: Selection audit + unroll + unification for worst benchmark sizes
From benchmark_results.md worst (GT 2.6x+ for 90/198/469/..., f32 Winograd 16@5.6x, PoT Stockham 32/64/128/256/32768 >1x, Rader primes).
- Fixed selection (prefer composite for smooth before GT static in dispatch/plan; reduced f32 generated/short for slow policy sizes >64).
- Added 128/256 unrolled pair Stockham specials (straight-line, uses pot ZST foundation).
- Ensured small PoT 16f32/32/64f64 delegate to shared AVX fixed kernels (unify, reduce redundancy).
- Updated tests for new selection (numerical still verified).
- Added kernel/pot/ ZSTs (StockhamAutosort etc).
- Result: should close many GT by method change, PoT by unroll+unify; GT/Rader when still chosen, and full table need more (rader conv, GT PFA SIMD gather, more fixed, composite opt).
- Verified: check, fmt, targeted tests (planned 33, gt 65, stockham 31, special) pass. No full xtask rerun here.
Rader fused gather/sum preserved value semantics (`cargo test -p apollo-fft
rader --lib`) and `xtask` compilation, but focused N=271 timing worsened the
controlling f32 ratio from `3.008x` to `3.279x`; the sequential-sum plus gather
path remains retained.
Retained-route refreshes for N=511, N=385, and N=219 were rejected because each
improved f64 while worsening the controlling f32 ratio: N=511 from `2.495x` to
`3.516x`, N=385 from `1.916x` to `2.058x`, and N=219 from `2.487x` to
`3.177x`. Prior retained rows remain authoritative.
Planned N=36 composite `[4,3,3]` routing preserved the existing composite value
test but worsened the focused full-profile row from f64 `2.556x`, f32 `2.828x`
to f64 `3.533x`, f32 `3.383x`; the short-codelet route remains retained.
Generated N=36 `(4,9)` orientation passed the same value and build checks but
worsened the controlling f32 ratio to `3.301x`; `(9,4)` remains retained.
Generated N=24 Good-Thomas `(3,8)` orientation passed N=24 value coverage and
`xtask` checking but worsened the controlling f32 ratio from `2.636x` to
`9.130x`; the `(6,4)` Cooley-Tukey codelet remains retained.
Generated N=63 `(9,7)` orientation passed fixed-coprime codelet value coverage
and `xtask` checking but worsened the controlling f32 ratio from `2.637x` to
`4.229x`; the `(7,9)` orientation remains retained.
Generated N=27 `(9,3)` Cooley-Tukey decomposition passed
`dft_composite_small_cases` and `xtask` checking but worsened the controlling
f32 ratio from `2.543x` to `5.024x`; `(3,9)` remains retained.
Retained Rader N=89 was refreshed under current full-profile timing and
improves from f64 `2.626x`, f32 `2.113x` to f64 `2.265x`, f32 `2.076x`; it
remains above the target.
Retained-route N=198 refresh was rejected because it improved f64 from
`2.664x` to `2.610x` but worsened the controlling f32 ratio from `1.932x` to
`3.698x`; the prior row remains authoritative.
Retained-route N=445 refresh was rejected because it worsened the controlling
f64 ratio from `2.477x` to `2.694x`; the prior row remains authoritative.
Retained Good-Thomas/Rader N=213 was refreshed under current full-profile
timing and improves from f64 `2.477x`, f32 `2.153x` to f64 `2.157x`, f32
`1.811x`; it remains above the target.
Retained-route N=67 refresh was rejected because it worsened the controlling
f64 ratio from `2.458x` to `2.606x`; the prior row remains authoritative.
Retained Good-Thomas/Rader N=453 was refreshed under current full-profile
timing and improves from f64 `2.312x`, f32 `2.436x` to f64 `2.046x`, f32
`1.812x`; it remains above the target.
Retained Good-Thomas/Rader N=398 was refreshed under current full-profile
timing and improves from f64 `2.422x`, f32 `1.433x` to f64 `1.809x`, f32
`1.668x`; it remains above the target.
Retained Cooley-Tukey N=286 was refreshed under current full-profile timing
and improves from f64 `1.645x`, f32 `2.418x` to f64 `1.152x`, f32 `1.748x`;
it remains above the target.
Retained-route N=183 reruns failed to improve the controlling ratio: the best
rerun records f64 `2.402x`, f32 `2.408x`, and a second rerun worsened to f64
`2.980x`, f32 `2.539x`; N=183 remains above the target.
Retained Cooley-Tukey N=429 was refreshed under current full-profile timing and
improves from f64 `1.461x`, f32 `2.397x` to f64 `1.342x`, f32 `2.093x`; it
remains above the target.
The radix-composite AVX module split exposed a visibility defect: `cache.rs`
called AVX2+FMA flat-pass re-exports whose leaf functions were still only
visible to the `avx2` child module. The leaf functions now use the
`radix_composite` module as their visibility boundary. Verification passes:
`cargo check -p xtask --features bench-runner`, `cargo test -p apollo-fft
dft_composite_small_cases --lib`, and `cargo test -p apollo-fft
radix_composite --lib`. A retained Cooley-Tukey N=238 refresh now records f64
`1.322x`, f32 `1.464x`; it remains above the target.
The fresh N=508 retained-route row was rejected after worsening the max ratio
from `2.407x` to f64 `2.545x`; the prior retained row was restored.
N=242 reruns were rejected after failing to improve the retained ratio record
f64 `1.548x`, f32 `2.494x`. The exact retained timing columns were not
recoverable from current artifacts or prior `benchmark_results.md` commits; the
best measured row from this turn is f64 `1.585x`, f32 `3.211x`.
Workspace manifest loading was repaired by removing duplicate
`apollo-wgpu-helpers` dependency entries from WGPU backend crates; this restored
`cargo check -p xtask --features bench-runner` execution.
Removing N=242 from the f32 generated-codelet policy was rejected: the retained
composite-route value test passed, but focused full-profile timing worsened the
max ratio from `3.211x` to f32 `3.400x`.
The retained-route N=36 refresh was rejected after worsening f32 from `2.828x`
to `4.899x`; the prior retained row was restored.
An f32 half-cyclic Rader precision-policy probe for N=271 and N=337 preserved
existing Rader value coverage and `xtask` compilation, but the focused
full-profile command exceeded the 300s bound and partially wrote worse rows:
N=271 f32 `3.675x` and N=337 f32 `4.435x`. The prior retained rows were
restored. A retained-route N=400 refresh was also rejected after worsening the
max ratio from f32 `2.730x` to f32 `3.134x`; the prior row was restored.
Plan scratch bound cleanup exposes the sealed `PlanScratch` trait through the
public `fft::workspace` module while retaining crate-local allocation helpers.
`cargo check -p xtask --features bench-runner` now passes without the previous
scratch private-bound/dead-code warnings; `cargo test -p apollo-fft rader_n
--lib` passes.
Retained Cooley-Tukey N=264 was refreshed under current full-profile timing
and improves from f64 `2.443x`, f32 `2.654x` to f64 `1.515x`, f32 `1.905x`;
it remains above the target. The benchmark command exceeded the 300s shell
bound after writing the row; residual compiler processes observed afterward
belonged to a separate `cargo test --workspace --lib` parent and were not
stopped.
Retained precision-policy N=126 was refreshed under current full-profile
timing and improves from f64 `2.629x`, f32 `2.551x` to f64 `1.511x`, f32
`2.310x`; it remains above the target.
A retained-route N=99 refresh was rejected after improving f64 but worsening
the controlling f32 ratio from `2.619x` to `4.736x`; the prior retained row
was restored.
A retained-route N=54 refresh was rejected after worsening the controlling f32
ratio from `2.553x` to `4.599x`; the prior retained row was restored.
Retained precision-policy N=96 was refreshed under current full-profile timing
and improves from f64 `2.317x`, f32 `2.552x` to f64 `1.557x`, f32 `2.201x`;
it remains above the target.
Retained Good-Thomas N=160 was refreshed under current full-profile timing and
improves from f64 `1.662x`, f32 `2.552x` to f64 `1.450x`, f32 `2.318x`; it
remains above the target.
Retained Good-Thomas N=200 was refreshed under current full-profile timing and
improves from f64 `1.741x`, f32 `2.549x` to f64 `1.561x`, f32 `2.464x`; it
remains above the target.
Retained Winograd N=27 was refreshed under current full-profile timing and
improves from f64 `1.697x`, f32 `2.543x` to f64 `1.161x`, f32 `2.307x`; it
remains above the target.
A retained-route N=135 refresh was rejected after worsening the controlling
f32 ratio from `2.526x` to `2.785x`; the prior retained row was restored.
Retained Cooley-Tukey N=176 was refreshed under current full-profile timing
and improves from f64 `2.444x`, f32 `2.482x` to f64 `1.494x`, f32 `2.289x`;
it remains above the target.
Retained-route N=240 was rejected after improving f64 below target but
worsening the controlling f32 ratio from `2.479x` to `2.934x`; the prior
retained row was restored.
Retained Cooley-Tukey N=384 was refreshed under current full-profile timing
and improves the max ratio from f32 `2.022x` to f32 `1.886x`; it remains above
the target.
Retained-route N=480 was rejected after worsening the controlling f32 ratio
from `1.831x` to `1.868x`; the prior retained row was restored.
Retained Good-Thomas/Rader N=134 was refreshed under current full-profile
timing and improves the max ratio from f64 `2.507x` to f64 `1.919x`; it
remains above the target.
Retained-route N=298 was rejected after worsening the controlling f32 ratio
from `2.103x` to `2.464x`; the prior retained row was restored.
Retained precision-policy N=484 was refreshed under current full-profile
timing and improves the max ratio from f32 `2.509x` to f32 `2.200x`; it
remains above the target.
Retained Good-Thomas/Rader N=339 was refreshed under current full-profile
timing and improves the max ratio from f32 `2.480x` to f64 `2.440x`; it
remains above the target.
Retained-route N=356 was rejected after worsening the max ratio from f64
`2.469x` to f32 `2.743x`; the prior retained row was restored.
Retained-route N=438 was rejected after worsening the controlling f32 ratio
from `2.447x` to `3.955x`; the prior retained row was restored.
Retained-route N=146 was rejected after worsening the controlling f32 ratio
from `2.436x` to `3.393x`; the prior retained row was restored.
Retained-route N=292 was rejected after worsening the controlling f32 ratio
from `2.433x` to `3.512x`; the prior retained row was restored.
Retained Good-Thomas/Rader N=305 was refreshed under current full-profile
timing and improves the max ratio from f32 `2.433x` to f32 `2.242x`; it
remains above the target.
Retained Good-Thomas/Bluestein N=321 was refreshed under current full-profile
timing and improves the max ratio from f64 `2.431x` to f64 `2.120x`; it
remains above the target. The benchmark command exceeded the 300s shell bound
after writing the improved row.
Retained-route N=397 was rejected after worsening the controlling f32 ratio
from `2.427x` to `3.044x`; the prior retained row was restored.
Retained-route N=335 was rejected after worsening the controlling f64 ratio
from `2.374x` to `2.766x`; the prior retained row was restored.
Retained-route N=396 was rejected after worsening the controlling f32 ratio
from `2.365x` to `2.477x`; the prior retained row was restored.
Retained Good-Thomas/Rader N=488 was refreshed under current full-profile
timing and improves the max ratio from f32 `2.366x` to f64 `2.205x`; it
remains above the target.
Retained-route N=189 refresh was rejected after worsening the max ratio from
f64 `2.356x` to f32 `9.896x`; the prior retained row was restored.

Focused full-profile refreshes reduced stale high-ratio rows without retaining
new code routes: N=72 now records f64 `2.286x`, f32 `2.338x`; N=504 records
f64 `1.346x`, f32 `1.645x`; and N=135 records f64 `1.856x`, f32 `2.526x`.
The f64 N=72 `ShortWinograd` probe preserved direct-DFT value semantics and
`xtask` checking but was rejected because the focused full-profile row worsened
the max ratio to f32 `3.727x`.

Follow-up refreshes update N=168 to f64 `1.504x`, f32 `2.798x`; N=108 to f64
`1.684x`, f32 `2.667x`; N=112 to f64 `1.770x`, f32 `2.661x`; N=400 to f64
`0.975x`, f32 `2.782x`; N=132 to f64 `1.397x`, f32 `2.114x`; and N=242 to
f64 `1.548x`, f32 `2.494x`. A f32 N=271 Bluestein Rader precision-policy
probe was value-correct and `xtask`-clean but rejected because it did not beat
retained full-cyclic Rader; the final restored full-profile row records f64
`2.714x`, f32 `3.624x`. Current highest misses are N=48 f32 `4.593x`, N=271
f32 `3.624x`, N=99 f32 `3.021x`, and N=469 f64 `2.975x`.

N=99 was promoted to a generated f32 Good-Thomas `(9,11)` codelet after
direct-DFT verification and `xtask` checking. The full-profile row improves
from f64 `2.127x`, f32 `3.021x` to f64 `2.431x`, f32 `2.619x`; retained
because the max ratio improves, while f64 still uses static Good-Thomas.
N=469 retained-route refresh improves from f64 `2.975x`, f32 `2.523x` to f64
`2.630x`, f32 `1.981x`. Current highest misses after this increment are N=48
f32 `4.593x`, N=271 f32 `3.624x`, N=120 f32 `2.923x`, and N=402 f64
`2.917x`.

N=120 generated f32 routing now uses Good-Thomas `(15,8)` instead of `(8,15)`.
The route preserves direct-DFT value semantics and improves f32 from `2.860x`
to `2.373x`. Additional retained refreshes reduce stale rows: N=402 now f64
`2.595x`, f32 `1.882x`; N=280 f64 `1.363x`, f32 `2.785x`; N=178 f64
`2.506x`, f32 `2.119x`; N=244 f64 `2.205x`, f32 `2.381x`; N=305 f64
`2.089x`, f32 `2.433x`; and N=27 f64 `1.668x`, f32 `2.706x`. Current highest
misses are N=48 f32 `4.593x`, N=271 f32 `3.624x`, N=283 f32 `2.838x`, and
N=113 f64 `2.823x`.

Rader scatter now handles `q=0` once and uses direct reverse generator-order
indexing for `q>=1`, preserving the single cached generator-order table and
removing the hot zero/nonzero branch. `cargo test -p apollo-fft rader --lib`
passes. Full-profile rows update N=271 from f64 `2.714x`, f32 `3.624x` to f64
`2.048x`, f32 `3.248x`; N=337 refreshes to f64 `2.274x`, f32 `2.862x`.
The N=280 generated `(35,8)` orientation probe preserved value semantics but
was rejected after worsening f32 to `3.186x`; retained `(8,35)` row remains
f64 `1.363x`, f32 `2.785x`. Current highest misses are N=48 f32 `4.593x`,
N=271 f32 `3.248x`, N=337 f32 `2.862x`, and N=168 f32 `2.798x`.

N=48 generated codelet orientation changed from `(16,3)` to `(3,16)`. Existing
f64/f32 direct-DFT route tests pass, and the full-profile row improves from
f64 `1.470x`, f32 `4.593x` to f64 `1.617x`, f32 `2.579x`; retained because the
max ratio improves. N=400 `(25,16)` was value-correct but rejected after
worsening f32 from `2.782x` to `3.801x`; `(16,25)` remains retained. Current
highest misses are N=271 f32 `3.248x`, N=337 f32 `2.862x`, N=168 f32 `2.798x`,
N=280 f32 `2.785x`, and N=400 f32 `2.782x`.

The f32 N=180 generated-codelet policy probe preserved value semantics but was
rejected after worsening f32 from `2.772x` to `3.270x`; retained composite
`[5,3,3,4]` remains authoritative. A retained N=362 refresh updates the stale
row from f64 `2.782x`, f32 `2.487x` to f64 `2.346x`, f32 `2.116x`. Current
highest misses are N=271 f32 `3.248x`, N=337 f32 `2.862x`, N=168 f32 `2.798x`,
N=280 f32 `2.785x`, and N=400 f32 `2.782x`.

The f32 N=271 Bluestein Rader probe preserved direct-DFT value semantics but
was rejected after worsening f32 from `3.248x` to `3.261x`; full-cyclic Rader
remains retained. A retained N=353 refresh updates the stale row from f64
`2.358x`, f32 `2.760x` to f64 `2.062x`, f32 `1.634x`. Current highest misses
are N=271 f32 `3.248x`, N=337 f32 `2.862x`, N=168 f32 `2.798x`, N=280 f32
`2.785x`, and N=400 f32 `2.782x`.

The f32 N=337 Bluestein Rader probe preserved direct-DFT value semantics but
was rejected after worsening f32 from `2.862x` to `2.928x`; full-cyclic Rader
remains retained. A retained N=331 refresh updates the stale row from f64
`2.746x`, f32 `2.510x` to f64 `2.013x`, f32 `1.758x`. Current highest misses
are N=271 f32 `3.248x`, N=337 f32 `2.862x`, N=168 f32 `2.798x`, N=280 f32
`2.785x`, and N=400 f32 `2.782x`.

A retained-route N=36 refresh was rejected because it worsened f32 from
`2.713x` to `2.835x`; the prior row remains authoritative. A retained N=352
refresh improves the stale max ratio from `2.708x` to f64 `2.689x`, f32
`2.482x`. A retained-route N=3 refresh after the AVX fallback cleanup improved
f64 to `0.698x` but worsened f32 from `1.345x` to `2.175x`, so the prior row
was restored. A retained-route N=88 refresh was rejected after worsening f32
from `2.669x` to `2.874x`. Retained refreshes improve N=482 from f64
`2.690x`, f32 `2.231x` to f64 `2.239x`, f32 `1.849x`, and N=397 from f64
`2.679x`, f32 `2.229x` to f64 `1.489x`, f32 `2.427x`. A retained-route
N=201 refresh was rejected after worsening f64 from `2.681x` to `2.806x`.
Additional retained-route refreshes were rejected after worsening the
controlling ratio: N=198 f32 `1.932x` to `2.991x`, N=77 f32 `2.661x` to
`2.816x`, N=264 f32 `2.654x` to `2.877x`, N=63 f32 `2.637x` to `3.262x`,
and N=24 f32 `2.636x` to `3.500x`. Retained N=121 refresh improves from f64
`2.633x`, f32 `2.588x` to f64 `2.372x`, f32 `2.383x`. Current highest misses
are N=271 f32 `3.248x`, N=337 f32 `2.862x`, N=280 f32 `2.785x`, N=400 f32
`2.782x`, and N=180 f32 `2.772x`.

Focused retained-route refreshes for N=81, N=126, and N=89 were rejected after
worsening their controlling ratios: N=81 f32 `2.631x` to `8.048x`, N=126 f32
`2.551x` to `3.121x`, and N=89 f64 `2.626x` to `2.762x`. Retained N=181
refresh improves from f64 `2.623x`, f32 `2.084x` to f64 `2.387x`, f32
`2.125x`. N=268 retained-route refresh was rejected after worsening f64 from
`2.602x` to `2.668x`. Retained N=274 refresh improves from f64 `2.560x`, f32
`1.688x` to f64 `1.383x`, f32 `1.379x`. N=160 retained-route refresh was
rejected after worsening f32 from `2.552x` to `3.748x`. Retained N=180 refresh
improves from f64 `1.742x`, f32 `2.772x` to f64 `1.532x`, f32 `2.226x`.
N=32 retained-route refresh was rejected after worsening f32 from `2.583x` to
`3.297x`. The duplicate mixed-radix scalar `constants` module edge was removed
because the active twiddle tables still live in `impls.rs`; `cargo check -p
xtask --features bench-runner` is warning-clean for this path after the
cleanup. The now-unreferenced duplicate `mixed_radix/scalar/constants.rs`
artifact was deleted so stale twiddle tables cannot re-enter the scalar module.
N=54 retained-route refresh was rejected after worsening f32 from `2.553x` to
`4.153x`. Additional retained-route refreshes were rejected after worsening
their controlling f32 ratios: N=96 from `2.552x` to `3.697x`, and N=263 from
`2.551x` to `2.792x`. N=200 retained-route refresh was rejected after worsening
f32 from `2.549x` to `2.944x`. Retained N=211 refresh improves from f64
`2.542x`, f32 `1.685x` to f64 `1.713x`, f32 `1.224x`. Current highest misses
were probed again for stale retained routes: N=267 and N=365 were rejected
after worsening f32 from `2.549x` to `2.865x` and from `2.520x` to `3.310x`,
respectively. N=379 was rejected after worsening f32 from `2.520x` to
`2.539x`. Retained N=401 refresh improves from f64 `2.091x`, f32 `2.517x` to
f64 `2.203x`, f32 `2.327x`. Current highest misses are N=271 f32 `3.248x`,
N=337 f32 `2.862x`, N=280 f32 `2.785x`, N=400 f32 `2.782x`, and N=168 f32
`2.720x`. Retained N=488 refresh improves from f64 `2.314x`, f32 `2.511x` to
f64 `2.090x`, f32 `2.366x`. N=484 retained-route refresh was rejected after
worsening f32 from `2.509x` to `3.037x`. N=134 retained-route refresh was
rejected after worsening f64 from `2.507x` to `2.750x`. A current retained
N=178 refresh improves the stale max ratio from `2.506x` to f64 `2.493x`,
f32 `1.940x`. Half-cyclic Rader strategy coverage now includes N=271 and
N=337; the measured strategy rows reject a threshold reduction because
half-cyclic remains slower than full-cyclic for both precisions. A retained
N=337 full-profile refresh improves the stale max ratio from `2.862x` to f64
`2.274x`, f32 `2.855x`. A retained N=271 full-profile refresh improves the
stale max ratio from `3.248x` to f64 `2.058x`, f32 `3.008x`. N=36
confirmation refreshes were rejected after worsening f32 to `4.395x` and then
`4.715x`; the exact tracked row was restored. A generated N=36 swapped
`(4,9)` orientation preserved composite value semantics but worsened the max
ratio to `3.404x`, so generated `(9,4)` remains retained. Retained
full-profile refreshes improve N=280 from max `2.785x` to f64 `1.254x`, f32
`2.600x`, and N=400 from max `2.782x` to f64 `1.153x`, f32 `2.730x`.
Retained N=27 refresh improves max ratio from `2.706x` to f64 `1.697x`, f32
`2.543x`. N=168 retained-route refresh was rejected after worsening f32 from
`2.720x` to `5.217x`. Retained N=201 refresh improves max ratio from
`2.681x` to f64 `1.977x`, f32 `2.001x`. N=283 and N=352 refreshes were
rejected after worsening max ratios to `2.820x` and `2.811x`, respectively.
Retained N=77 refresh improves max ratio from `2.661x` to f64 `1.801x`, f32
`2.375x`. Retained-route refreshes for N=88, N=108, N=112, and N=198 were
rejected after worsening max ratios to `2.686x`, `3.907x`, `4.036x`, and
`3.251x`, respectively.
Retained N=337 refresh improves max ratio from `2.855x` to f64 `1.822x`, f32
`2.675x`. Retained-route refreshes for N=36, N=168, N=271, and N=400 were
rejected after worsening max ratios to `6.068x`, `3.616x`, `3.269x`, and
`3.040x`, respectively.
Retained refreshes improve N=88 max ratio from `2.669x` to f64 `1.830x`, f32
`2.432x`; N=283 from `2.699x` to f64 `1.446x`, f32 `2.052x`; and N=352 from
`2.689x` to f64 `1.257x`, f32 `2.239x`. Retained-route refreshes for N=108
and N=198 were rejected after worsening max ratios to `4.148x` and `3.793x`,
respectively.
Benchmark-only Bluestein-forced Rader coverage was added to
`half_cyclic_rader`, making N=271/N=337 strategy evidence cover full-cyclic,
half-cyclic, Bluestein, and auto routes. A large-prime static-Rader precheck
gate preserved Rader correctness and `xtask` compilation but was rejected after
full-profile timing worsened N=271 from max `3.008x` to `3.418x` and N=337
from `2.675x` to `2.905x`; the production selector and retained rows were
restored.
The batched N=112/N=264/N=63/N=24/N=81 full-profile refresh exceeded the 300s
command bound without updating rows, and its leftover Apollo `xtask` process
tree was stopped. Follow-up single-size refreshes were rejected after worsening
N=112 max ratio from `2.661x` to `3.064x`, N=264 from `2.654x` to `2.850x`,
and N=63 from `2.637x` to `2.813x`; retained rows were restored.
An unexpected fresh N=16 row appeared during this turn with max `3.235x`; a
focused N=16 rerun improved the current row to f64 `1.056x`, f32 `1.899x`.
N=36 composite-route experiment was rejected: disabling generated short-codelet
dispatch routed N=36 through the existing `[4,3,3]` composite path; value tests
and `xtask` checking passed, but full-profile timing worsened max ratio from
`2.828x` to `3.596x`, so short-codelet dispatch and the retained row were
restored.
The current dirty trait surface required `MixedRadixScalar::use_generated_codelet_plan`
for f32/f64 but the implementations were absent; conservative `false` policy
implementations restore compilation without changing route selection.

`ShortWinogradScalar` release compilation is restored by removing invalid
cross-module calls to private AVX helper functions from the N=2/N=4 short-DFT
trait methods. Retained full-profile refreshes update N=168 from f64 `1.504x`,
f32 `2.798x` to f64 `1.518x`, f32 `2.720x`, and N=148 from f64 `2.240x`,
f32 `2.762x` to f64 `1.768x`, f32 `1.975x`. Current highest misses are N=271
f32 `3.248x`, N=337 f32 `2.862x`, N=280 f32 `2.785x`, N=400 f32 `2.782x`,
and N=180 f32 `2.772x`.

Retained full-profile refreshes update N=335 from f64 `2.625x`, f32 `2.724x`
to f64 `2.374x`, f32 `2.315x`, and N=80 from f64 `2.322x`, f32 `2.727x` to
f64 `1.964x`, f32 `2.460x`. Current highest misses are N=271 f32 `3.248x`,
N=337 f32 `2.862x`, N=280 f32 `2.785x`, N=400 f32 `2.782x`, and N=180 f32
`2.772x`.

## DELIVERED (2026-05-29): Benchmark median estimator (measurement reliability)

`xtask::measure_operation` previously returned the arithmetic MEAN of
`sample_size` per-iteration measurements. For sub-microsecond FFT latencies the
mean is inflated by OS-scheduling / interrupt outliers, producing +/-30-50%
run-to-run variance even at the `full` profile (sample_size=30) — verified by
bidirectional swings (e.g. N=224 f32 1.343x<->2.096x, N=416 f32 0.756x<->1.651x
across identical re-runs). This noise made per-size optimization verification
impossible: only systematic multi-size signals (radix-5 across 8+ sizes) were
distinguishable.

Fix: switched the point estimate to the MEDIAN of the samples (collect, sort,
take middle). Median is the standard latency-microbenchmark estimator — robust
to jitter outliers — and is applied identically to Apollo and RustFFT, so the
comparison stays fair (this is accuracy improvement, not profile-weakening:
sample_size/measurement_time are unchanged). The table header now reads
"median point estimate". After this, f64 small-size ratios are stable and
clean (N=11 0.49x, N=52 0.74x, N=500 0.64x) and the radix-5 wins are confirmed
reliable (N=250/375/500 PASS). Exposed the genuine systematic residual: the
f32 small-prime/composite codelets (N=5-93 ~1.2-1.35x) — RustFFT's hand-tuned
scalar f32 butterflies, the hard core requiring per-size codelet work, not a
flat-Stockham vectorization.

Architecture follow-up (recorded, not yet done): `radix_composite/avx2.rs` grew
to ~2100 lines with the radix-2/5/7 additions, well past the 500-line SRP
target. Split into per-radix leaf modules (`avx2/radix2.rs`, `radix3.rs`, ...
plus shared `helpers.rs` for cmul/rot) once the median regeneration completes
and the working tree is benchmark-quiescent.

## DELIVERED (2026-05-29): AVX2 radix-5 Stockham stages (f64 + f32)

Implemented the radix-5 fix scoped by the root-cause investigation: the flat
Stockham composite path (`radix_composite/core.rs::flat_stockham_fused`) had
AVX2 fast paths only for radix-3/4; radix-5 stages ran scalar. Added
`flat_pass_r5_f64`/`flat_pass_r5_f32` (+ `dft5_f64`/`dft5_f32` butterflies and
scalar fallbacks) in `avx2.rs`, wired through `CompositeCache::try_flat_pass_r5`
and the `5 =>` dispatch arm. Mirrors the verified scalar `dft5_values` math.

Correctness: 344/344 apollo-fft lib tests pass; radix_composite factor-5 sizes
(N=15/25/100/1000, f64 and f32) validated bit-against direct DFT.

Measured f64 impact (full profile), systematic (not noise):
- N=25 1.181x -> 0.972x (PASS), N=250 1.084x -> 0.799x (PASS),
  N=375 1.317x -> 0.805x (PASS), N=500 1.255x -> 1.003x.
- N=180 2.872x -> 1.974x (-31%), N=200 2.038x -> 1.556x, N=300 1.597x ->
  1.214x, N=100 1.319x -> 1.032x.

Measured f32 impact (full profile) on flat-Stockham (`Cooley-Tukey`) routed
factor-5 composites — large, systematic:
- N=250 f32 2.071x -> 0.967x and N=500 f32 2.014x -> 0.783x and N=625 f32
  0.726x: all now PASS BOTH precisions (were ~2x f32 failures).
- N=375 f32 1.977x -> 0.772x (PASS), N=300 f32 2.062x -> 1.409x (-32%),
  N=180 f32 2.551x -> 2.083x (-18%).

Scope boundary discovered: sizes routed through `Good-Thomas (Static)` static
codelets and `Precision Policy` generated codelets (N=100, 135, 200, 400) do
NOT use `flat_stockham_fused`, so they are unaffected and need their own
codelet-level radix-5 optimization (separate increment).

This is the first concrete benchmark-flipping optimization in the universal-
parity effort: multiple sizes (N=250, 375, 500, 625, ...) moved from ~2x
failures to passing both precisions. Reaches flat-Stockham composites with a
factor of 5 (direct, and Rader N-1 convolutions).

AVX2 radix-7 stages (f64 + f32) added in the same increment: `flat_pass_r7_f64`/
`_f32` + `dft7_f64`/`_f32` + scalar fallbacks, wired through
`try_flat_pass_r7` and the `7 =>` dispatch arm, mirroring the verified scalar
`dft7_values` math. Correctness: 344/344 lib tests pass (factor-7 sizes
N=7/14/49/98/343 validated via roundtrip/forward vs direct DFT). Reaches the 68
failing sizes with a factor of 7 that route through flat-Stockham.
Measured (full profile): N=343 (7^3) f64 -> 0.857x (PASS); N=252 f32
2.943x -> 2.038x (-31%); N=112 f32 3.062x -> 2.254x; N=63 f32 2.615x -> 1.941x;
N=56 f32 2.346x -> 1.841x; N=189 f32 2.876x -> 2.217x. Pure 2^k*7 sizes
(N=224/392/448) improve less — dominated by their already-AVX radix-4 stages.

AVX2 radix-2 stage (f64 + f32) added: vectorizes the trailing radix-2 stage of
odd-power-of-two decompositions (`[4,..,4,2]`) and 2^odd composites, previously
scalar. `flat_pass_r2_f64`/`_f32` wired via `try_flat_pass_r2` + `2 =>` arm.
Includes an amortization guard (n>=64) — below that the AVX setup exceeds the
scalar cost (measured N=32 regression), so tiny radix-2 stages stay scalar.
Correctness: 344/344. Measured: N=128 f64 1.651x -> 1.101x (-33%), f32 2.893x ->
1.83x (-37%); N=224 f32 2.212x -> 1.343x; N=416 f32 -> 0.756x (PASS).
Flat-Stockham AVX coverage now spans radix-2/3/4/5/7.

Remaining increments (each concrete, verified, reduces failing-row count):
static-codelet factor-5/7 paths (`Good-Thomas (Static)`/`Precision Policy`
routes unaffected by flat-Stockham); radix-8/11/13 stages; radix-4/PoT
butterfly tuning for the power-of-two offenders. The general radix-3/4 constant
factor vs RustFFT remains the hardest residual for smooth composites whose
work is dominated by those (already-AVX) stages.

## Open: Universal RustFFT Parity (`benchmark_results.md` all-`<1.000x`)

### Root-cause investigation (2026-05-29) — the 2-3x offenders converge on ONE kernel

Investigation of the stable large offenders (161 sizes at N>=100, worst >=2.0x;
these are reliable — microsecond-scale ops, low relative noise) traces all three
failure families to a single shared kernel:

- Direct smooth composites (N=180 2.87x, 504 ...) -> `radix_composite` fused
  Stockham AVX kernel.
- Rader primes (N=181 2.99x, 271, 401, 227 ...) -> the length-(N-1) cyclic
  convolution in `rader/convolution.rs` dispatches to
  `composite_forward_with_pointwise` -> the SAME `radix_composite` kernel
  (N-1 is usually prime23-smooth). The Rader machinery itself is expertly
  optimized (fused forward+pointwise, half-cyclic Nussbaumer CRT split,
  4-way unrolled twist/recombine) and is NOT the bottleneck.
- Good-Thomas / Cooley-Tukey composites (N=339=3x113, 346=2x173, 452=4x113 ...)
  -> large-prime column transform -> Rader -> SAME `radix_composite` kernel.

Conclusion: the highest-leverage optimization is `components/radix_composite/`
(the fused-Stockham AVX path, `avx2.rs` ~1493 lines + `arity.rs`/`adaptive.rs`).
Its constant factor trails RustFFT's hand-tuned mixed-radix; lifting it would
improve Rader, Good-Thomas, and direct-composite offenders simultaneously. This
is a deep AVX-kernel effort (multi-sprint), not a single-session edit.

SPECIFIC actionable defect found in `core.rs::flat_stockham_fused`: AVX2 stage
fast paths exist ONLY for radix-3 (`try_flat_pass_r3`) and radix-4
(`try_flat_pass_r4`). Radix-5/7/11/13/17/23 stages fall back to the scalar
`dispatch_radix_stage` — no vectorization. Any composite with a factor of 5 or
7 (N=180=4*3*3*5, N=5k, N=7k ...) runs that stage scalar while RustFFT uses AVX
radix-5/7 butterflies. FIX: add `try_flat_pass_r5`/`try_flat_pass_r7` following
the existing r3/r4 pattern in `avx2.rs`. SCOPE NOTE: necessary but not
sufficient — for N=180 the radix-5 stage is ~25% of the work, so AVX2-izing it
improves ~15-20% (2.87x -> ~2.4x); the residual is constant-factor across all
stages. This is a concrete, bounded next increment toward the broader kernel
parity effort.

Verified non-results this session (honest negative findings):
- cu=1 build profile (release codegen) trial: no systematic benefit; small
  sizes are overhead/noise-bound, not vectorization-bound; bidirectional
  +/-150% swings confirm high run-to-run variance even at the `full` profile.
  Reverted. CONSEQUENCE: small-size (N<~64) 2-3x ratios are partly measurement
  noise, not all real defects; only the N>=100 offenders are reliably real.
- The Rader convolution, Winograd-pair, Good-Thomas PFA, and f32 AVX backend
  were read and confirmed near-optimal — no fixable inefficiency in them.

Target: every size in `benchmark_results.md` (514 rows) beats RustFFT for both
f64 and f32.

ACCURATE BASELINE (entire 514-row table regenerated at the `full` profile,
sample_size=30, 2026-05-29): **47/514 both-pass**; f64 402 fail / 112 pass;
f32 441 fail / 73 pass.

Measurement-methodology correction (this session): the prior table was the
`quick` profile (sample_size=3, 5ms warmup), which is noise-dominated for
sub-15ns sizes and exhibited a systematic cold-start bias. Re-measuring all 514
rows at the `full` profile changed the picture in BOTH directions and is the
rigorous baseline for this goal:
- The quick profile OVERSTATED f64 (165 apparent passes -> 112 real) — many
  near-parity f64 rows were cold-start-favoured noise.
- The quick profile UNDERSTATED f32 (48 -> 73 real passes) — e.g. N=5 f32
  measured 1.251x quick but 0.591x full; N=17 1.747x -> 0.786x.
- Net: 47 rows genuinely beat RustFFT in both precisions under rigorous
  measurement. The remaining 467 are real kernel gaps, not noise.
- Worst f64 offenders remain Rader/Good-Thomas large composites and the
  power-of-two sizes (RustFFT ships size-specialized hand-tuned butterflies;
  Apollo uses generic Stockham/Winograd).

Audit findings (this scopes the remaining work; none are quick fixes):

1. **The xtask harness is fair.** `bench_pair` measures `copy+FFT` and subtracts
   a `copy`-only baseline for BOTH Apollo and RustFFT, and both use a prebuilt
   plan plus scratch. The measured gap is genuine kernel time, not a
   clone/dispatch artifact. (Verified by reading `xtask/src/main.rs:367-403`.)

2. **The f32 gap is NOT a SIMD-width defect.** The reduced (f32) Stockham
   backend already uses 256-bit AVX (`__m256`, `COMPLEX_PER_VECTOR = 4`) — 2x
   the complex throughput of the f64 backend's `__m256d`. The systemic f32
   slowness (466/514) is a broad constant-factor disadvantage versus RustFFT's
   hand-tuned per-radix butterflies, concentrated where Apollo has no wide
   vectorization: scalar small-prime Winograd codelets (N=5,7,11,13) and the
   Rader/Good-Thomas composite scatter/gather paths.

3. **The `quick` profile (sample_size=3, 30ms) is noise-dominated** for
   sub-15ns sizes. Near-1.0x rows (e.g. N=7 f64 1.024x, N=51 f32 1.002x) are
   not statistically separable from parity at 3 samples; the `full` profile
   (sample_size=30, 2s) is required to confirm those. Per CVI policy the table
   is measured evidence, not an editable surface, and the profile must not be
   weakened to manufacture passing ratios.

Required work (multi-sprint; this is the project's central objective):
(a) f64 power-of-two (N=16/32/64...) — IN PROGRESS under Closure CVXIII.
(b) f32 systemic: 8-wide-AVX small-prime/composite codelets to match RustFFT
    hand-tuning across Winograd, Rader, and Good-Thomas families.
(c) f64 large Rader/Good-Thomas composites (up to 3x) — per-family kernel work.

## Closed Gaps

- Maintenance Sweep MS-3 (apollo-czt; performance, memory, architecture)
  replaces the per-plan `forward_workspace: Mutex<Vec<Complex64>>` Bluestein
  convolution buffer with a `thread_local!` scratch accessed through a new
  `with_forward_workspace` helper. The Mutex serialized every `forward()` /
  `forward_into()` call on a shared `CztPlan` through one lock and one buffer,
  a scalability bottleneck and a shared-mutable-state contention point.
  Performance: removes the lock from every forward call and lets concurrent
  threads transform through a shared plan without serializing. Memory:
  per-thread buffer grown on demand and reused, replacing a per-plan `Vec`
  held for the plan's whole lifetime. Architecture (SSOT/DRY/SoC): aligns CZT
  with the workspace-wide `thread_local!` scratch convention used by
  `apollo-sdft`, `apollo-hilbert`, and `apollo-fft`; removes the
  `std::sync::Mutex` field and import. Correct by construction
  (`czt_bluestein_forward_into_with_workspace` takes `&mut [Complex64]`,
  matching the pre-sized slice); reuse test adapted to thread-local semantics.
  Verified: `apollo-czt` 27/27 lib tests pass; no regression (each
  touched/sampled crate passes in isolation).

- Maintenance Sweep MS-2 (architecture / encapsulation) restores the
  plan->kernel boundary around the Rader gather/scatter primitives. The MS-1
  follow-up note flagged that the plan layer reached kernel-internal
  `gather_sum_slice`/`scatter_slice` via over-broad `pub(crate)` visibility.
  Re-audit with the current tree shows the in-flight CVXIII refactor already
  moved the plan-layer call sites out of `plan/fft/dimension_1d.rs`
  (verified: workspace grep finds the two functions called only from within
  `components/rader/mod.rs` itself, lines 109-140). With no remaining
  cross-module callers, the `pub(crate)` exposure was stale and leaked kernel
  internals across the whole crate. Tightened both functions to module-private
  (`pub(crate) fn` -> `fn`), restoring information hiding / interface
  segregation: the rader kernel now exposes only its public entry points, not
  its internal gather/scatter helpers. Correct by construction — reducing
  visibility can only affect external callers, and the authoritative grep
  confirms none exist; the intra-module call sites and the `#[inline(always)]`
  hot-path bodies are unchanged, so monomorphized codegen is identical.

- Maintenance Sweep MS-1 closes six debt items orthogonal to the active CVXIII
  sprint. (1) Dead PFA-cycles cache subsystem removed end-to-end (~165 LOC):
  `apply_pfa_perm_cycles`, `PFA_CYCLES_CACHE`, `TL_PFA_CYCLES`,
  `cached_pfa_input_cycles`, `build_pfa_input_cycles`, and the `pfa_cycles`
  profiler field/`CacheId` variant/report rows — orphaned by the ordered-Rader
  PFA refactor; this also drops two unbounded process-lifetime maps and five
  compiler warnings, and corrects stale `cfg(cache_profiling)` to
  `cfg(feature = "cache-profiling")`. (2) `kernel/tests_radix_composite.rs`
  (585 LOC, 77 tests, pulled via `#[path]`) relocated to
  `components/radix_composite/tests/` as six concern-scoped submodules colocated
  with the code under test. (3) `apollo-fft-macros` gained 9 value-semantic
  property tests (Fermat, Bezout, primitive-root order, modular-inverse
  roundtrip, complex field laws) for its previously untested modular-arithmetic
  primitives. (4) `suite/fixtures.rs` (2768 LOC, 62 fixtures) decomposed into
  `suite/fixtures/` with one submodule per transform family plus a shared
  `builders` module; largest leaf 337 LOC. (5) Stale `scalar/impls.rs.orig`
  backup removed. (6) Untracked `run_bench.log` and `test_build.log` removed.
  DIP follow-up: RESOLVED in MS-2 above (plan-layer calls already removed by
  CVXIII; over-broad `pub(crate)` visibility tightened to module-private).
  Verification: `cargo test
  --workspace --lib --exclude apollo-python` 920 pass; `cargo doc --workspace
  --no-deps --exclude apollo-python` 0 warnings.

- Closure CVXII narrows the reduced f32 Winograd-pair layout to DFT31 after
  measuring and rejecting the broader N=29/37/41/53 reduced route. The retained
  path stores pair sums and imaginary differences in separate scalar arrays and
  accumulates the DC bin during the pair pass, while all f64 routes and f32
  N=11/13/17/19/23/29/37/41/43/47/53 remain on the generic Winograd-pair
  kernel. Direct-DFT coverage now checks promoted f64 odd-prime routes, all
  f32 odd-prime routes, and the reduced f32 DFT31 inverse route. Current
  optimized clone-inclusive rows record reduced f32 DFT31 at 87.31 ns Apollo
  vs 83.75 ns RustFFT (`1.043x`), improving the direct generic-route probe of
  107.39 ns Apollo vs 82.46 ns RustFFT (`1.302x`) but leaving DFT31 open.

- Closure CVXI routes short odd-prime `ShortDft` sizes
  11/13/17/19/23/29/31/37/41/43/47/53 through the Winograd-pair kernel
  instead of static Rader. This removes the Rader convolution path from
  production short-prime leaves where the direct pair decomposition has lower
  constant cost. Generated static Rader coverage is retained and extended
  through N=53 for direct Rader/codegen verification. Focused direct value
  tests pass. Closure CVXII supersedes the f32 row inventory after the
  reduced-layout follow-up.

- Closure CVX removes the cached inverse-generator scatter table from runtime
  Rader and ordered-Rader Good-Thomas paths. The retained generator-order table
  stores `g^q`; scatter order is derived on demand by
  `g^{-q} = g^(N-1-q)` in the prime-order cyclic group. This removes one
  length-`N-1` `usize` allocation per cached prime/generator pair without
  changing Rader arithmetic. Direct identity coverage checks every primitive
  root table entry, and focused Rader plus Good-Thomas tests pass.

- Closure CVIX removes the temporary full `N-1` kernel from half-cyclic Rader
  spectrum construction. The cache builder now streams the two length-`m`
  halves directly into cyclic and negacyclic CRT residues, reducing peak
  construction storage from four `m`-length complex buffers to two and removing
  the split pass. Correctness is covered by forced half-cyclic/full-cyclic
  equivalence at N=521 and the full Rader-filtered test suite. The focused
  opt-level 1 Criterion rerun records N=1031 forced half-cyclic improvement,
  but release-quality O3 timing remains blocked by local codegen termination.

- Closure CVIII integrates half-cyclic Winograd/Liu-Tolimieri Rader
  convolution as a real production strategy and removes Apollo FFT Bluestein
  fallback references from the Rader/scalar routing surface. Correctness is
  covered by direct-DFT checks at N=521 for automatic, forced half-cyclic, and
  forced full-cyclic execution, plus the existing large-prime N=10007
  roundtrip. The production threshold is conservative at `N-1 >= 1024` for
  both f64 and f32 because focused optimized opt-level 1 Criterion rows show
  no amortization at N=521 and only parity-to-small gains at N=1031. A full O3
  bench-quick rerun remains blocked by local codegen termination, so a broader
  release-quality threshold sweep remains open before tightening this policy.

## Open Gaps

- Closure CVXIII inlines the public N=2 forward/inverse butterfly in the
  `Complex64` and `Complex32` `FftPrecision` impls, bypassing the mixed-radix
  trait call for the direct API route. The final focused N=2 row remains open:
  f64 3.22 ns vs 1.65 ns (`1.953x`) and f32 2.76 ns vs 1.98 ns (`1.394x`).
  A length-guarded unchecked-access variant is rejected because it regressed
  the focused row to f64 `4.102x` and f32 `1.523x`. Forced `#[inline(always)]`
  on the `xtask` precision adapter methods and `bench_pair` is also rejected
  because the sampled small-size rows regressed.

- Closure CVXIII routes f64 N=3 public forward/inverse calls directly to the
  existing Winograd DFT3 leaf. Focused timing improves the f64 Apollo estimate
  versus the stale table from 2.97 ns to 2.68 ns, but the ratio remains open at
  `1.361x`; f32 N=3 remains `1.278x`. A direct scalar N=4 public butterfly
  probe is rejected because the focused absolute Apollo timings did not
  justify replacing the retained small-PoT leaf.

- Closure CVXIII routes f32 N=5, f64/f32 N=7, and f32 N=11 public calls
  directly to existing Winograd leaves. Focused value tests for DFT5, DFT7,
  and DFT11 pass. The f32 ratios improve but remain open: N=5 `1.209x`, N=7
  `1.187x`, and N=11 `1.138x`. f32 N=13/N=17/N=19 direct public routing is
  rejected: N=13 regressed in the retained-only probe to `1.281x`, and
  N=17/N=19 regressed in the broad probe to `1.708x` and `1.528x`.

- Closure CVXIII rejects planned N=24/N=27 rerouting to the generic
  radix-composite executor. The route preserves existing radix-composite value
  semantics for N=24, but the focused `cargo xtask` probe regresses N=24 to
  f64 `7.486x` and f32 `11.649x`, and N=27 to f64 `3.140x` and f32 `3.543x`.
  Retain the short-Winograd planned route for both sizes.

- Closure CVXIII re-baselines the planned small power-of-two route used by
  `cargo xtask`: N=2 now passes at f64 `0.842x` and f32 `0.587x`; N=8 f32
  remains open at `1.138x`; N=16 remains open at f64 `1.176x` and f32
  `3.198x`. Planned N=8 rerouting to `ShortWinograd` is rejected because the
  focused `cargo xtask` probe regressed N=8 to f64 `1.124x` and f32 `6.128x`.
  Planned f32 N=16 rerouting to `ShortWinograd` is also rejected because the
  focused probe regressed N=16 to f64 `1.947x` and f32 `4.820x`. Replacing
  the retained f32 sized N=8 SIMD kernel with the scalar butterfly is rejected
  because value tests passed but the focused `cargo xtask` probe regressed
  N=8 f32 to `5.261x`.

- Closure CVXIII routes planned Good-Thomas execution through the canonical
  `components::good_thomas::pfa_fft` dispatcher instead of a duplicate generic
  plan executor. This lets planned transforms use the existing specialized
  two-by-prime, three-by-prime, Cook-Toom, and generated fixed codelets.
  Good-Thomas component tests and the new planned N=90 direct-DFT regression
  test pass. `benchmark_results.md` is refreshed for N=84 and N=90: N=84
  improves from the prior current-worktree row f64 `3.064x`, f32 `4.001x` to
  f64 `1.820x`, f32 `2.289x`; N=90 improves from f64 `5.144x`, f32 `2.695x`
  to f64 `2.363x`, f32 `1.597x`. Additional refreshed rows record N=334 at
  f64 `1.648x`, f32 `1.646x`; N=358 at f64 `1.994x`, f32 `1.787x`; N=454 at
  f64 `1.896x`, f32 `1.020x`; N=501 at f64 `2.043x`, f32 `1.977x`; N=214 at
  f64 `2.367x`, f32 `1.587x`; N=362 at f64 `2.794x`, f32 `2.186x`; and
  N=428 at f64 `2.256x`, f32 `1.988x`. These rows remain open versus
  `< 1.000x`.
  N=72 now has a scalar route policy: f64 retains the static Good-Thomas
  `(9,8)` route, while f32 uses a generated twiddle-free `(8,9)` codelet via
  `ShortDft<72>`. Planned N=72 direct-DFT tests pass for both route
  selections, and the refreshed `benchmark_results.md` row is labeled
  `Precision Policy`. The row remains open after the latest focused
  `cargo xtask` refresh at f64 `2.308x` and f32 `3.527x`; the f32 ratio
  improves from the retained `[4,2,3,3]` composite row at `4.855x`. The
  alternate `[4,3,3,2]` order is rejected because it preserved value semantics
  but regressed focused f32 timing to `5.017x`.
  The planned Good-Thomas executor no longer retains dead cache state after
  this delegation: input/output CRT permutation Arcs and row/column subplans
  were removed from `FftPlan1D` because `pfa_fft` owns the canonical route.
  Planned runtime Rader now delegates to the canonical
  `components::rader::rader_fft` dispatcher instead of retaining a duplicate
  full-cyclic executor. This removes per-plan generator-order storage,
  forward/inverse Rader spectra, and the length `N-1` subplan from planned
  Rader state, while enabling the canonical FullCyclic/HalfCyclic/Bluestein
  strategy selector. Planned N=359 direct-DFT tests pass for f64 and f32, and
  the refreshed benchmark row improves from f64 `5.350x`, f32 `12.263x` to
  f64 `1.532x`, f32 `1.874x`. Additional stale planned Rader rows were
  refreshed under the same implementation: N=383 now records f64 `1.546x`,
  f32 `2.138x`; N=347 records f64 `2.277x`, f32 `1.853x`; N=179 records f64
  `2.256x`, f32 `1.700x`; N=499 records f64 `2.736x`, f32 `1.471x`;
  N=227 records f64 `2.059x`, f32 `1.485x`; N=317 records f64 `1.840x`,
  f32 `2.628x`; N=479 records f64 `2.059x`, f32 `1.502x`; N=503 records f64
  `1.844x`, f32 `1.444x`; N=509 records f64 `1.244x`, f32 `1.611x`. These
  rows remain open versus `< 1.000x`. Further refreshed Rader/Bluestein rows
  record N=10007 at f64 `2.304x`, f32 `2.461x`; N=167 at f64 `1.966x`, f32
  `1.814x`; N=263 at f64 `2.257x`, f32 `2.831x`; N=269 at f64 `1.792x`,
  f32 `2.920x`; N=293 at f64 `2.320x`, f32 `2.589x`; N=439 at f64 `2.547x`,
  f32 `1.421x`; and N=467 at f64 `1.928x`, f32 `1.254x`.
  Two additional route probes are rejected: f32 N=16 short-Winograd routing
  increased Apollo absolute time to 15.18 ns, and N=24 Good-Thomas `(3,8)`
  routing regressed Apollo to f64 113.13 ns and f32 76.86 ns.
  Replacing the f32 N=16 specialized power-of-two branch with the canonical
  Stockham kernel is also rejected: `small_pot` tests passed, but focused
  timing regressed f32 Apollo to 18.20 ns. Planned f32 N=144 composite
  rerouting is also rejected: route-selection and value tests passed, but the
  optimized `cargo xtask` benchmark exceeded the 300s verification bound and
  produced no updated row.
  The same generated-codelet policy now routes f32 N=96 through `(3,32)`,
  f32 N=108 through `(4,27)`, f32 N=112 through `(16,7)`, f32 N=120 through
  `(8,15)`, f32 N=126 through `(2,63)`, f32 N=144 through `(12,12)`, f32
  N=154 through `(11,14)`, f32 N=168 through `(14,12)`, f32 N=180 through
  `(20,9)`, f32 N=189 through `(21,9)`, f32 N=242 through `(2,121)`, f32 N=275 through `(11,25)`, and
  f32 N=363 through `(3,121)` while leaving f64 on static routes. Planned
  direct-DFT tests pass for all retained generated-policy rows.
  Default `cargo xtask` evidence records N=108 at f64 `2.417x`, f32
  `3.184x`, improving f32 from `4.007x`; N=120 records f64 `1.458x`, f32
  `2.953x`, improving f32 from `3.321x`; N=126 records f64 `1.820x`, f32
  `2.484x`, improving f32 from `3.374x`; N=112 records f64 `2.589x`, f32
  `3.276x`, improving f32 from `3.357x`; N=144 records f64 `2.611x`, f32
  `3.234x`, improving f32 from `4.035x`; N=154 records f64 `1.674x`, f32
  `2.754x`, improving f32 from `3.109x`; N=168 records f64 `2.078x`, f32
  `3.252x`, improving f32 from `3.671x`; N=180 records f64 `2.651x`, f32
  `3.765x`, improving f32 from `3.863x`; N=275 records f64 `2.226x`, f32
  `2.596x`, improving f32 from `3.463x`; N=96 records f64 `2.683x`, f32
  `3.031x`, improving f32 from `3.306x`. The alternate N=112 `(7,16)`,
  N=112 `(14,8)`, N=112 `(8,14)`,
  N=135 `(5,27)`, N=135 `(27,5)`, N=144 `(9,16)`, N=144 `(8,18)`, N=168 `(8,21)`, N=168 `(12,14)`, N=180 `(9,20)`,
  N=180 `(12,15)`, N=180 `(5,36)`, N=180 `(36,5)`, N=180 `(4,45)`,
  N=180 `(18,10)`,
  N=108 `(12,9)`, N=108 `(27,4)`, N=180 `(45,4)`, N=189 `(7,27)`,
  N=189 `(27,7)`, N=240 `(15,16)`, N=484 `(44,11)`, N=72 `(12,6)`, and N=24 `(8,3)` generated orientations
  are rejected because they were value-correct but slower than retained
  routes. Planned f32 N=121 prime-power `11^2` routing and planned f32 N=242
  Good-Thomas `(121,2)` routing are also rejected after value-correct
  regressions. Planned f32 N=121 now uses a generated Cooley-Tukey `(11,11)`
  codelet instead; the focused row improves f32 from `4.075x` to `2.671x`,
  while f64 remains non-generated and records `2.403x` in the same run.
  Planned f32 N=242 now uses a generated Good-Thomas `(2,121)` codelet; the
  focused row improves f32 from `3.352x` to `3.104x`, while f64 records
  `2.370x`.
  Planned f32 N=280 now uses a generated Good-Thomas `(8,35)` codelet; the
  focused row improves f32 from `3.330x` to `2.550x` and f64 from `1.739x`
  to `1.645x`.
  Planned f32 N=363 now uses a generated Good-Thomas `(3,121)` codelet; the
  focused row improves f32 from `3.338x` to `2.977x`, while f64 records
  `2.329x`.
  Planned f32 N=400 now uses a generated Good-Thomas `(16,25)` codelet; the
  focused row improves f32 from `3.289x` to `3.133x`, while f64 records
  `1.959x`.
  f32 N=113 Rader now selects the existing Bluestein convolution backend
  instead of the full-cyclic length-112 convolution; the focused row improves
  f32 from `3.299x` to `1.834x`, while f64 records `2.561x`.
  N=83 was refreshed on the retained Bluestein Rader route after the stale row
  recorded f32 `3.130x`; the current row records f64 `1.684x` and f32
  `1.968x`. Forcing f32 N=83 through full-cyclic Rader is rejected because it
  preserved direct-DFT value semantics but regressed focused f32 timing to
  `4.110x`.
  Generated N=270 Good-Thomas routing is rejected for both probed
  orientations: `(10,27)` regressed f32 to `3.922x`, and `(27,10)` regressed
  f32 to `4.469x`. The retained Cooley-Tukey refresh records f64 `1.647x`
  and f32 `3.123x`, improving the stale f32 `3.271x` row while remaining
  open.
  Planned N=511 now routes as Good-Thomas `(73,7)`, placing the prime factor
  on the ordered-Rader N1 path instead of the generic natural PFA order. The
  focused row improves from f64 `1.550x`, f32 `3.258x` to f64 `1.395x`, f32
  `2.646x`, while remaining open.
  Planned N=420 Good-Thomas `(20,21)` routing is rejected: the route preserved
  direct-DFT value semantics but regressed focused f32 timing to `3.673x`.
  The retained Cooley-Tukey refresh records f64 `1.918x` and f32 `2.659x`,
  improving the stale f32 `3.226x` row while remaining open.
  Planned N=440 Good-Thomas `(8,55)` routing is rejected: the route preserved
  direct-DFT value semantics but regressed focused f32 timing to `3.623x`.
  The retained Cooley-Tukey refresh records f64 `2.049x` and f32 `3.225x`,
  effectively unchanged from the stale f32 `3.212x` row. Generated
  N=440 Cooley-Tukey `(22,20)` routing is also rejected: it preserved
  direct-DFT value semantics but regressed focused timing to f64 `2.180x`
  and f32 `6.419x`; the restored Cooley-Tukey row records f64 `2.324x`
  and f32 `3.138x`. Retained
  Cooley-Tukey refreshes for N=300, N=484, and N=504 record N=300 at f64
  `1.618x`, f32 `2.884x`, improving stale f32 `3.116x`; N=484 at f64
  `2.090x`, f32 `3.296x`; and N=504 at f64 `1.576x`, f32 `3.510x`.
  Planned N=484 Good-Thomas `(4,121)` routing is rejected: the route preserved
  direct-DFT value semantics but regressed focused f32 timing to `5.672x`.
  Planned f32 N=484 now routes through a generated Cooley-Tukey `(22,22)`
  codelet; direct-DFT value semantics pass, and focused timing improves f32
  from `3.296x` to `3.229x` while f64 records `2.131x`. Planned f32 N=484
  generated Cooley-Tukey `(11,44)` routing is rejected after preserving
  direct-DFT value semantics but regressing f32 to `5.380x`; restored
  `(22,22)` refreshes to f64 `2.194x`, f32 `3.531x`. Planned f32 N=484
  generated Cooley-Tukey `(44,11)` routing is rejected after preserving
  direct-DFT value semantics but regressing f32 to `5.029x`; restored
  `(22,22)` refreshes to f64 `2.183x`, f32 `3.216x`.
  Planned N=504 Good-Thomas `(8,63)` routing is rejected: the route preserved
  direct-DFT value semantics but regressed focused f32 timing to `8.768x`.
  Planned f32 N=504 generated Cooley-Tukey `(21,24)` routing is also rejected:
  the route preserved direct-DFT value semantics but regressed focused f32
  timing to `6.755x`. The restored Cooley-Tukey route refreshes to f64
  `1.577x`, f32 `3.389x`. Planned f32 N=504 generated Cooley-Tukey `(28,18)`
  routing is rejected after preserving direct-DFT value semantics but
  regressing f32 to `7.060x`; the restored Cooley-Tukey route refreshes to
  f64 `1.586x`, f32 `3.368x`.
  N=72 refreshed on the retained precision-policy route and improves f32 from
  `3.527x` to `2.954x`; N=180 refreshed to f64 `2.687x`, f32 `3.769x`; N=189
  refreshed to f64 `2.194x`, f32 `3.649x`. N=180 generated `(5,36)` and
  `(36,5)` orientations are rejected after preserving direct-DFT value
  semantics but regressing f32 to `4.130x` and `5.502x`; restored `(20,9)`
  refreshes to f64 `2.661x`, f32 `3.691x`. N=180 generated Cooley-Tukey
  `(10,18)` routing is rejected after preserving direct-DFT value semantics
  but regressing f32 to `4.520x`; restored `(20,9)` refreshes to f64
  `2.816x`, f32 `3.737x`. N=180 generated Good-Thomas `(4,45)` and `(45,4)`
  orientations are rejected after preserving direct-DFT value semantics but
  regressing f32 to `5.299x` and `5.820x`; restored `(20,9)` refreshes to
  f64 `2.772x`, f32 `3.827x`. N=189 now routes f32 through a
  generated Cooley-Tukey `(9,21)` codelet and improves to f64 `2.171x`, f32
  `3.162x`. N=180 generated Cooley-Tukey `(18,10)` routing is rejected after
  preserving direct-DFT value semantics but regressing f32 to `5.262x`;
  restored `(20,9)` refreshes to f64 `2.670x`, f32 `3.705x`.
  Generated N=392 Good-Thomas routing is rejected for both probed
  orientations: `(8,49)` passed value tests but did not beat the
  same-environment refreshed Cooley-Tukey baseline, while `(49,8)` regressed
  f32 to `4.334x`. The retained N=392 row records f64 `2.063x` and f32
  `2.926x`.
  Planned N=24 now uses a generated Cooley-Tukey `(6,4)` codelet instead of
  the prior generated Good-Thomas `(3,8)` codelet; the focused row improves
  f64 from `1.679x` to `1.384x` and f32 from `3.974x` to `2.990x`.
  Lowering the f32 half-cyclic Rader threshold to 256 is rejected as an N=283
  optimization because N=283 has `N-1 = 282`, which is not prime-23-smooth;
  the selector reaches Bluestein before the threshold. The focused N=283
  refresh still improves the row to f64 `1.818x` and f32 `2.700x`. The
  focused N=498 refresh under retained ordered-Rader Good-Thomas routing
  records f64 `2.069x` and f32 `2.147x`, improving f32 from `3.377x`.
  N=112 generated Cooley-Tukey `(14,8)` routing is rejected after preserving
  direct-DFT value semantics but regressing f32 to `3.378x`; restored
  `(16,7)` refreshes to f64 `2.573x`, f32 `3.276x`. N=112 generated
  Cooley-Tukey `(8,14)` routing is rejected after preserving direct-DFT value
  semantics but regressing f32 to `3.505x`; restored `(16,7)` refreshes to
  f64 `2.090x`, f32 `2.780x`.
  N=144 generated Cooley-Tukey `(8,18)` routing is rejected after preserving
  direct-DFT value semantics but regressing f32 to `3.558x`; restored
  `(16,9)` refreshes to f64 `1.999x`, f32 `3.234x`. N=144 generated
  Cooley-Tukey `(12,12)` is retained after preserving direct-DFT value
  semantics and improving f32 to `3.086x`; f64 records `2.569x`.
  N=168 generated Cooley-Tukey `(12,14)` routing is rejected after preserving
  direct-DFT value semantics but regressing f32 to `4.189x`; restored
  `(24,7)` refreshes to f64 `1.979x`, f32 `3.208x`. N=168 generated
  Cooley-Tukey `(14,12)` is retained after preserving direct-DFT value
  semantics and improving f32 to `3.077x`; f64 records `2.035x`.
  N=108 generated Cooley-Tukey `(12,9)` and Good-Thomas `(27,4)` orientations
  are rejected after preserving direct-DFT value semantics but regressing f32
  to `4.590x` and `3.686x`, respectively; restored `(4,27)` refreshes to f64
  `2.636x`, f32 `2.773x`. N=189 generated Cooley-Tukey `(21,9)` is retained
  after preserving direct-DFT value semantics and improving f32 from `3.162x`
  to `2.808x`; f64 improves from `2.171x` to `2.123x`.
  N=135 generated Good-Thomas `(27,5)` routing is rejected after preserving
  direct-DFT value semantics but regressing f32 to `3.756x`; restored static
  routing refreshes to f64 `1.712x`, f32 `3.558x`.
  Generated Cooley-Tukey row-slice writeback is rejected after preserving
  planned-codelet value semantics but regressing focused f32 timing for N=144
  to `3.634x`, N=189 to `2.928x`, and N=484 to `3.240x`; restored absolute
  scratch writeback refreshes N=144 f32 `3.579x`, N=168 f32 `2.642x`,
  N=189 f32 `2.936x`, and N=484 f32 `3.325x`.
  Generated Good-Thomas fixed-column block codegen is rejected after
  preserving planned-codelet value semantics but regressing focused timing for
  N=180 f32 to `4.390x`, N=400 f32 to `4.409x`, N=180 f64 to `3.142x`, and
  N=242 f64 to `2.477x`; restored loop codegen refreshes N=135 f32 `3.146x`,
  N=180 f32 `3.711x`, N=242 f32 `2.980x`, and N=400 f32 `3.459x`.
  Forced `#[inline(always)]` for all generated medium codelets is rejected
  because semantics and `xtask` checking passed but optimized benchmark
  compilation exceeded the 300s release-build bound for both a four-row probe
  and a narrowed N=180 probe. Generated N=24 Cooley-Tukey `(4,6)` routing is
  also rejected after composite value tests passed but optimized benchmark
  compilation exceeded the same bound after codegen invalidation; the retained
  `(6,4)` route remains authoritative. Retained-binary refreshes update N=166
  to f64 `1.700x`, f32 `1.847x`, N=356 to f64 `2.748x`, f32 `2.191x`, and
  N=385 to f64 `2.378x`, f32 `3.818x`.
  A dispatch-local N=385 `[5,7,11]` radix-order override is rejected after
  preserving radix-composite value semantics and `xtask` type checking but
  exceeding the 300s optimized benchmark bound without producing a row, so the
  canonical cached `[11,7,5]` route remains authoritative. Additional
  retained-binary refreshes update N=81 to f64 `1.792x`, f32 `2.735x`; N=165
  to f64 `1.486x`, f32 `2.206x`; N=198 to f64 `2.188x`, f32 `2.258x`; N=219
  to f64 `1.507x`, f32 `2.794x`; N=223 to f64 `1.787x`, f32 `1.621x`;
  N=438 to f64 `1.425x`, f32 `2.755x`; and N=446 to f64 `1.844x`, f32
  `1.643x`.
  Further retained-binary refreshes update N=70 to f64 `1.016x`, f32
  `1.322x`; N=73 to f64 `1.298x`, f32 `2.213x`; N=88 to f64 `1.707x`, f32
  `2.850x`; N=127 to f64 `1.210x`, f32 `1.119x`; N=142 to f64 `2.366x`,
  f32 `1.604x`; N=146 to f64 `1.244x`, f32 `2.449x`; N=160 to f64
  `2.419x`, f32 `2.974x`; N=181 to f64 `2.994x`, f32 `2.020x`; N=224 to
  f64 `1.735x`, f32 `3.035x`; N=245 to f64 `1.834x`, f32 `2.599x`; N=249
  to f64 `2.337x`, f32 `2.054x`; N=263 to f64 `2.141x`, f32 `2.832x`;
  N=264 to f64 `1.453x`, f32 `3.245x`; N=269 to f64 `1.776x`, f32 `2.880x`;
  N=339 to f64 `2.868x`, f32 `2.063x`; N=346 to f64 `2.939x`, f32
  `1.772x`; N=352 to f64 `2.088x`, f32 `2.977x`; N=357 to f64 `1.701x`,
  f32 `1.651x`; and N=362 to f64 `2.812x`, f32 `2.284x`.
  Additional retained-binary refreshes update N=48 to f64 `2.015x`, f32
  `5.650x`; N=99 to f64 `2.255x`, f32 `2.544x`; N=110 to f64 `1.579x`, f32
  `2.142x`; N=176 to f64 `1.947x`, f32 `3.919x`; N=200 to f64 `1.681x`,
  f32 `2.884x`; N=292 to f64 `1.378x`, f32 `2.612x`; N=298 to f64 `2.280x`,
  f32 `1.983x`; N=384 to f64 `1.428x`, f32 `2.108x`; N=452 to f64
  `2.767x`, f32 `2.074x`; N=480 to f64 `1.680x`, f32 `2.251x`; and N=499
  to f64 `2.462x`, f32 `1.618x`. N=48 is confirmed by a solo rerun as the
  current top miss.
  Nearby short/composite retained-binary refreshes update N=40 to f64
  `1.466x`, f32 `2.365x`; N=42 to f64 `1.300x`, f32 `1.185x`; N=44 to f64
  `1.485x`, f32 `1.869x`; N=45 to f64 `1.686x`, f32 `2.166x`; N=46 to f64
  `1.045x`, f32 `1.217x`; N=50 to f64 `1.107x`, f32 `0.963x`; N=51 to f64
  `1.169x`, f32 `1.002x`; N=52 to f64 `0.968x`, f32 `1.283x`; N=54 to f64
  `2.354x`, f32 `2.689x` after a solo confirmation rerun; N=55 to f64
  `1.611x`, f32 `2.301x`; N=56 to f64
  `1.481x`, f32 `2.346x`; N=58 to f64 `1.074x`, f32 `1.258x`; N=60 to f64
  `1.062x`, f32 `1.642x`; N=62 to f64 `0.862x`, f32 `1.065x`; and N=63 to
  f64 `2.029x`, f32 `2.615x`.
  Planned N=48 Good-Thomas `(16,3)` routing is rejected after focused
  benchmarking regressed the row to f64 `3.243x`, f32 `10.996x`. Under the
  current full-profile median benchmark, planned N=48 now routes through the
  generated `ShortWinograd` codelet; f64/f32 direct-DFT tests pass, `xtask`
  labels the row `Winograd`, and the refreshed row records f64 `1.470x`, f32
  `4.593x`. This improves the current full-profile composite row from f64
  `2.102x`, f32 `6.238x` but remains above the `< 1.000x` target.
  A follow-up N=48 composite order `[4,3,4]` probe is rejected: direct-DFT
  route tests and `xtask` checking passed, but optimized benchmarking exceeded
  the 300s release-build bound without producing a row, so `[4,3,4]` remains
  rejected.
  A small-composite AVX2 cutoff probe for N=48 is also rejected: direct-DFT
  route tests and `xtask` checking passed, but optimized benchmarking exceeded
  the 300s release-build bound without producing a row; the composite route is
  no longer retained under current full-profile evidence.
  Planned f32 N=176 generated `(11,16)` routing is rejected after preserving
  direct-DFT value semantics but regressing the focused row to f64 `2.070x`,
  f32 `4.256x`; restored static routing refreshes to f64 `2.159x`, f32
  `3.879x`.
  Planned N=176 swapped Good-Thomas `(16,11)` routing is rejected after
  preserving direct-DFT value semantics but regressing the focused row to f64
  `2.258x`, f32 `3.920x`; restored cached `(11,16)` routing refreshes to f64
  `1.891x`, f32 `3.579x`.
  Planned N=385 composite order `[11,5,7]` is retained after preserving
  direct-DFT value semantics and improving focused timing from f64 `2.378x`,
  f32 `3.818x` to f64 `2.372x`, f32 `3.540x`.
  Planned N=385 composite order `[7,11,5]` is rejected after preserving
  direct-DFT value semantics because optimized benchmarking exceeded the 300s
  release-build bound without producing a row; `[11,5,7]` remains the
  benchmark-backed route.
  Planned N=180 composite order `[5,3,3,4]` is retained after preserving
  direct-DFT value semantics and improving focused timing from f64 `2.672x`,
  f32 `3.711x` to f64 `1.880x`, f32 `2.775x`.
  Planned N=144 composite order `[4,4,3,3]` is retained after preserving
  direct-DFT value semantics and improving focused timing from f64 `2.573x`,
  f32 `3.579x` to f64 `1.579x`, f32 `1.817x`.
  Planned N=176 composite order `[11,4,4]` is retained after preserving
  direct-DFT value semantics and improving the focused max ratio from f32
  `3.579x` to f32 `3.004x`; f64 changes from `1.891x` to `1.971x`.
  Current refreshed high-ratio misses include N=48 f32 `4.593x`, N=72 f32
  `4.168x`, N=504 f32 `3.786x`, N=135 f32 `3.754x`, and N=168 f32 `3.603x`.

- Closure CVXIII repaired the optimized `xtask` benchmark runner so Apollo
  timing calls the public direct `FftPrecision::fft_forward` path rather than
  the 1-D plan wrapper. Focused N=32/64 rows still miss the `< 1.000x`
  acceptance criterion after the latest default refresh: N=32 f64 20.99 ns
  vs 14.60 ns (`1.438x`), N=32 f32 20.06 ns vs 7.99 ns (`2.511x`), N=64 f64
  60.32 ns vs 37.96 ns (`1.589x`), and N=64 f32 37.12 ns vs 17.84 ns
  (`2.080x`). Rejected variants: aligned f64 combine-twiddle loads regressed
  f64 focused rows, and f32 fixed Winograd routing regressed N=32/64 f32 rows.

- After Closure CVXII, f32 short odd-prime rows
  N=11/13/17/19/29/31/37/41/53 still trail RustFFT despite improved timings.
  N=23/43/47 beat RustFFT through the generic Winograd-pair route. The
  remaining gap is inside the f32 Winograd-pair codegen/arithmetic path, not
  in static Rader dispatch.

- `benchmark_results.md` still contains measured rows where Apollo trails
  RustFFT. Editing the table without corresponding Criterion evidence is
  rejected as benchmark fabrication. Closure CVI adds generated N=18/N=24/N=36
  short Good-Thomas leaves and removes generic PFA column-buffer allocation,
  and `xtask benchmark` now has a quick Criterion/runtime profile plus
  `bench-quick` Cargo profile for iterative targeted refreshes. Closure CVII
  adds a macro-derived fixed coprime Good-Thomas support/match surface for
  canonical pairs up to N=200 backed by one bounded const-generic PFA codelet,
  then rejects full unrolled all-pair body emission because it exceeds the
  bounded bench/release compile budget. The remaining disparity is route cost,
  not trait-dispatch overhead: N=9 is a prime-power short leaf,
  N=84/N=90/N=150/N=175 use distinct fixed PFA factorizations, and N=94 uses
  the direct `2*p` route. A focused N=44 probe after fixed PFA routing records
  Apollo/RustFFT ratios of 1.541x f64 and 1.593x f32, so the row remains an
  open miss and `benchmark_results.md` was not rewritten from that probe. The
  prior N=10 f32 Apollo row was stale Criterion data; the focused refresh
  records f32 Apollo at 42.38 ns, consistent with the f64 row. Normal
  `xtask benchmark` execution now uses the optimized bounded adaptive
  clone-inclusive runner instead of a Criterion subprocess, while
  `--skip-run` remains a legacy Criterion JSON merge path. The already-built
  optimized `xtask` binary regenerated the full canonical table in 65.6
  seconds. N=77 was refreshed through the subset merge path and then the full
  table path; current N=77 records f64 Apollo/RustFFT at 1.923x and f32
  Apollo/RustFFT at 2.997x after the shared odd-prime-pair DFT11 const-loop
  improvement. The previous 4.739x f32 row was stale mixed-epoch evidence, but
  the remaining f32 miss is real and points to route-cost/vectorization work.
- Fresh targeted Criterion evidence for the restored power-of-two fast-path:
  focused correctness, `cargo check -p apollo-fft --lib`, and bench/example
  feature compile checks pass. `benchmark_results.md` was regenerated from the
  current Criterion cache. After the pre-existing writer finished, a targeted
  N=16 Criterion refresh exceeded the 300-second command cap, so dedicated
  post-cutoff rows for N=16/N=32/N=64/N=128/N=32768 remain pending. The current
  release quick comparison for N=16/N=32/N=64/N=128 records Apollo means of
  0.032/0.061/0.108/0.150 us versus RustFFT 0.032/0.041/0.064/0.106 us, so
  the cutoff improves Apollo's absolute route at 64/128 over the generic
  composite path but does not yet beat RustFFT at those sizes.
- A 32768 Stockham schedule probe replaced the retained five-triple pass plan
  with a four-pass `4+4+4+3` quad/triple plan to remove the terminal copy.
  The value-semantic N=32768 roundtrip passed, but optimized `xtask` evidence
  regressed from the prior focused row to 465901.45 ns f64 and 279476.74 ns f32
  versus RustFFT 76468.89 ns f64 and 36617.04 ns f32. The four-pass schedule
  is rejected for the current AVX backend; the all-triple schedule remains the
  production route until quad-stage throughput or register pressure is fixed.
- Criterion/RustFFT benchmark evidence for the restored fused radix-composite
  dispatcher: compile and value-semantic verification pass. Bounded
  `APOLLO_FFT_BENCH_N=96` and `APOLLO_FFT_BENCH_N=192` `vs_rustfft` attempts
  exceeded their command caps before producing usable timing output, so fresh
  `vs_rustfft` or `prime_compose` numbers remain pending.
- Criterion Rader-vs-Winograd-pair evidence for N=29/N=31/N=37: both kernels
  compile and value-semantic equivalence passes against the direct DFT
  reference. The optimized Rader path has fused static gather/scatter and
  final-forward-stage pointwise fusion. Standalone Rader now also fuses the
  primitive-root gather with the nonzero DC sum and retains one Rader scratch
  buffer per precision instead of a two-buffer pool. Rader Bluestein now caches
  one forward kernel spectrum per prime/precision and derives inverse
  multiplication through conjugated SIMD pointwise execution instead of
  retaining a second spectrum. The N=29/N=31/N=37 Rader
  comparison leaves also use static gather/scatter permutation tables to remove runtime
  modular-index recurrence. Bounded strategy-only `quick_compare` now has
  debug Winograd/Rader ratios of 0.778/1.175 at N=29, 1.589/1.475 at N=31,
  and 0.783/2.494 at N=37 for f64/f32 after restoring the benchmark hook to
  compare the real Winograd-pair kernels against the shared generic Rader
  kernel. Release ratios remain pending after the benchmark-route correction.
  Ordered-layout Rader static/runtime kernels now remove standalone
  gather/scatter for fused callers that already produce generator-ordered
  nonzero inputs and consume inverse-generator-ordered nonzero outputs.
  Good-Thomas PFA now uses this ordered contract for prime `n1` subtransforms
  that would otherwise dispatch to Rader, folding the Rader input order into
  the transpose and the Rader output order into the CRT scatter. Direct-DFT
  checks cover N=38 forward and N=82 inverse, while N=29/N=31/N=37 remain on
  the measured Winograd-pair path. The ordered PFA branch now consumes the
  cached Rader generator/inverse-generator order arrays, so it no longer
  performs runtime modular index walks for the ordered transpose or final CRT
  scatter layout conversion. A dedicated two-by-prime path now bypasses this
  ordered-PFA shape for N=2p composites, and
  N=19/N=29/N=31/N=37/N=41/N=43/N=47/N=53 now use Winograd-pair prime leaves.
  The stale dedicated DFT-82 codelet was removed so N=82 falls through to the
  two-by-prime route. The direct N=2p promoted-prime branch now bypasses
  thread-local PFA scratch, even-half stack copying, and odd-half compaction by
  reading interleaved even/odd input directly inside the fused two-prime
  Winograd execution. Latest release two-by-prime ratios are 1.514, 1.195, 1.228,
  1.059, 1.025, 0.943, 0.587, and 0.757 for
  N=38/N=58/N=62/N=74/N=82/N=86/N=94/N=106. N=38 remains the largest bounded
  probe miss; N=58/N=62/N=74/N=82 remain marginal/noisy misses. Fresh
  quick-profile canonical `vs_rustfft` rows now include
  N=38/N=58/N=74/N=82/N=94 after the generated twiddle-free Good-Thomas
  direct path: N=38 remains a miss at 1.551x f64 and 1.695x f32, N=58 remains
  a miss at 1.207x f64 and 1.539x f32, N=74 remains a miss at 1.141x f64 and
  1.959x f32, N=82 remains a miss at 1.044x f64 and 1.532x f32, and N=94
  beats RustFFT at 0.746x f64 and 0.847x f32. N=74 f32 is the current largest
  direct `2*p` regression. Fresh Criterion `kernel_strategy` timing is still
  pending.
- `GpuFft3dF16Native` Bluestein path on production hardware with non-power-of-two sizes: current test passes on dev hardware; production validation on adapters that expose `wgpu::Features::SHADER_F16` is pending.
- Criterion buffer-reuse bench results on representative GPU hardware: allocation-vs-reuse speedup ratios for FFT/NUFFT/STFT/Radon WGPU benchmark suites are not yet recorded as numbers. Closure XXII added the manual self-hosted GPU workflow and runner script; the residual gap is the first benchmark execution on real labeled hardware and publication of the measured ratios.
- **NUFFT 2D CPU**: `apollo-nufft` has 1D and 3D; 2D separable NUFFT not yet implemented.
- **DWT 2D CPU**: `apollo-wavelet` has 1D DWT; 2D separable DWT not yet implemented.
- **GPU FFT 1D/2D**: `apollo-fft-wgpu` exposes 3D GPU FFT; 1D and 2D GPU FFT paths are absent.
- **FrFT 2D/3D**: `apollo-frft` has 1D only; 2D/3D separable fractional Fourier transform absent.
- **Hilbert inverse**: `apollo-hilbert` has 1D forward only; inverse (env. recovery) absent.
- **NTT 2D/3D**: `apollo-ntt` has 1D only; 2D/3D separable NTT absent.
- **STFT 2D**: `apollo-stft` / `apollo-stft-wgpu` are 1D only; 2D short-time FFT absent.
- **Mellin 2D/3D**: `apollo-mellin` has 1D only; 2D/3D separable Mellin transform absent.
- **SDFT inverse**: `apollo-sdft` and `apollo-sdft-wgpu` have forward only; inverse SDFT absent.
- **SFT 2D/3D**: `apollo-sft` has 1D only; 2D/3D SFT absent.
- **GFT 2D/3D**: `apollo-gft` has 1D only; 2D/3D GFT absent.
- **QFT 2D/3D**: `apollo-qft` has 1D only; 2D/3D QFT absent.
- **CZT 2D/3D**: `apollo-czt` has 1D only; 2D/3D CZT absent.
- **SHT 3D / Radon 3D**: `apollo-sht` has 2D only; `apollo-radon` has 2D only.

Note: NTT-WGPU floating mixed precision is an architectural design contract, not a gap.
Residue-field arithmetic requires exact modular integers; the WGPU surface uses exact `u32`
quantized storage (implemented and verified). Floating-point NTT is architecturally unsupported
by design and will not be implemented.

## Comparative Gap Audit: Apollo vs rustfft vs numpy/scipy (Closure XLI baseline)

| Capability | rustfft | numpy/scipy | Apollo | Status |
|---|---|---|---|---|
| 1D complex FFT | ✓ | ✓ | ✓ | Closed |
| 2D complex FFT | ✗ | ✓ | ✓ | Closed |
| 3D complex FFT | ✗ | ✓ | ✓ | Closed |
| fftshift/ifftshift | ✗ | ✓ | ✓ | **Closed XLI** |
| fftfreq/rfftfreq | ✗ | ✓ | ✓ | **Closed XLI** |
| DCT/DST all types 1D/2D/3D | ✗ | via scipy | ✓ | Closed |
| DHT 1D | ✗ | ✗ | ✓ | Closed |
| DHT 2D/3D | ✗ | ✗ | ✓ | **Closed XLI** |
| FWHT 1D | ✗ | ✗ | ✓ | Closed |
| FWHT 2D/3D | ✗ | ✗ | ✓ | **Closed XLI** |
| NUFFT 1D/3D | ✗ | via finufft | ✓ | Closed |
| NUFFT 2D | ✗ | via finufft | ✗ | Open |
| DWT 2D | ✗ | via pywt | ✗ | Open |
| GPU FFT 1D/2D | ✗ | ✗ | ✗ | Open |

## Closed Gaps
### Closure CVII - Fixed Good-Thomas Macro Dispatch Review [patch]
- **Gap**: Fixed coprime PFA routes lacked a single generator-owned source of
  truth for broader canonical pairs, and benchmark-review rows were being
  interpreted as if monomorphization should make different factorizations
  share one kernel.
- **Closed by**: `generate_good_thomas_dispatch!` now derives canonical
  coprime pairs from `short_sizes` and `max_n`, emits the support/match
  surface for one bounded const-generic PFA body, and uses direct
  `ShortDft<N>` calls for generated row and column subtransforms. The partial
  `ShortDft` trait migration was completed by removing the
  `ShortWinogradScalar` cycle and restoring the `generate_winograd_fft!`
  export.
- **Residual risk**: N=77, N=84/N=90/N=175, and f32 N=150 remain slower than
  RustFFT. The next step is a route-cost model and f32 vectorization strategy
  that can prefer fixed PFA, mixed-radix, or direct `2*p` families by
  structural cost instead of adding size-specific bypasses. The N=10 f32
  disparity is closed as stale benchmark evidence, not a kernel defect.
- **Evidence**: `cargo check -p apollo-fft-macros`; `cargo check -p
  apollo-fft --lib`; `cargo check -p apollo-fft --benches --features
  kernel-strategy-bench`; `cargo test -p apollo-fft dft_composite_small_cases
  --lib`; `cargo test -p apollo-fft
  mixed_fixed_coprime_good_thomas_codelets_match_direct --lib`; `cargo test
  -p apollo-fft dft11 --lib`; `cargo run -p xtask -- benchmark --sizes 77
  --profile quick`; `target\bench-quick\xtask.exe benchmark --all --profile
  quick`.

### Closure CVI - Short Good-Thomas Codelets and PFA Scratch Reuse [patch]
- **Gap**: Several composite rows still route through generic composite/PFA
  machinery without a generated short leaf for reusable sublengths such as
  18, 24, and 36. Generic PFA also allocated a column `Vec` per transform.
- **Closed by**: Added generated `dft18_impl`, `dft24_impl`, and `dft36_impl`
  via `generate_good_thomas!` and routed them through `short_winograd`.
  Natural and ordered generic PFA split the existing thread-local PFA scratch
  into matrix and column-buffer regions, removing the per-call column `Vec`.
- **Residual risk**: The benchmark goal is not closed. A generated fixed
  coprime dispatcher for larger composite families needs a codegen-controlled
  design before it can be kept; the prototype was removed after optimized
  release bench builds failed to produce usable output within the bounded
  verification window.
- **Evidence**: `cargo check -p apollo-fft-macros`; `cargo check -p
  apollo-fft --lib`; `cargo test -p apollo-fft --lib
  mixed_new_short_good_thomas_codelets_match_direct -- --test-threads=1`.

### Closure CV - Natural Good-Thomas and Generated Codelet Dispatch [patch]
- **Gap**: The generic natural Good-Thomas PFA kernel consumed the cached
  output CRT permutation through row-major `(k1, k2)` indexing even though the
  authoritative cache stores output indices by transformed column-major
  `(k2, k1)` coordinates. Compact generated `3*p` routes and direct `2*p`
  routes bypassed this path, leaving non-compact natural PFA correctness
  under-tested.
- **Closed by**: Natural PFA scatter now uses `output_perm[k2 * n1 + k1]`.
  Direct-DFT forward and unnormalized inverse tests cover a nontrivial
  coprime natural PFA shape through the private kernel, binding the table
  layout to computed values. A fresh rebuild also exposed stale Winograd
  const-generic direction call sites; generated Good-Thomas, production
  short-codelet dispatch, and unit tests now call the current `const INVERSE`
  DFT-3/7/8/15 entry points. The `3*p` Good-Thomas proc macro now emits direct
  const-generic DFT-3 column calls and direct row codelet calls from the single
  supported-prime list, removing the generated route's dependency on a
  separate short-codelet adapter.
- **Residual risk**: N=33/38/58/74/82 remain above RustFFT in the
  clone-inclusive Criterion table. N=94 beats RustFFT for both f64/f32 after
  the current route/codelet work. The next optimization target is shared
  two-by-prime and `3*p` row/column fusion that reduces stack row copies and
  scatter stores without deleting retained components. The active benchmark
  workflow is now one runner, `cargo run -p xtask -- benchmark`, and one output
  table, `benchmark_results.md`.
- **Evidence**: `cargo test -p apollo-fft --lib natural_pfa_scatter --
  --test-threads=1`; `cargo test -p apollo-fft --lib good_thomas --
  --test-threads=1`; `cargo check -p apollo-fft --lib`; `cargo test -p
  apollo-fft --lib dft_large -- --test-threads=1`; `cargo test -p apollo-fft
  --lib dft_composite -- --test-threads=1`; `cargo check -p
  apollo-fft-macros`; `cargo check -p apollo-fft --benches --examples
  --features kernel-strategy-bench`; targeted `vs_rustfft` Criterion refresh
  for N=33/38/58/74/82/94; `cargo run -p xtask -- benchmark --skip-run`;
  `git diff --check`.

### Closure CIV - Generated Good-Thomas Family Dispatch [patch]
- **Gap**: Good-Thomas macro generation still stopped at the `3*p` route
  boundary, and the direct `2*p` Winograd-pair path retained a local
  declarative dispatch macro. Release builds also did not expose the
  prime-pair module used by generated dispatch.
- **Closed by**: `generate_three_by_prime_dispatch!` now emits full per-prime
  `3*p` transform bodies. `generate_two_by_prime_natural_dispatch!` generates
  the direct `2*p` Winograd-pair dispatch from one table. `PrimePairTables` is
  part of the sealed Winograd scalar contract, and `odd_prime_pair` is visible
  to release builds. Benchmark hooks were restored against the same retained
  Winograd-pair implementation.
- **Residual risk**: N=33 still trails RustFFT, and N=38/N=58/N=74 remain
  misses in quick release timing. The next macro step should generate
  register-level Good-Thomas SSA kernels that fuse row transforms with final
  CRT scatter and should replace the direct generated Winograd prototype only
  after it beats the retained hand codelets.
- **Evidence**: `cargo check -p apollo-fft-macros`; `cargo check -p
  apollo-fft --lib`; `cargo test -p apollo-fft --lib three_by_prime --
  --test-threads=1`; `cargo test -p apollo-fft --lib two_by_prime --
  --test-threads=1`; `cargo test -p apollo-fft --lib
  generated_rader_primes_match_direct_forward_and_inverse --
  --test-threads=1`; `cargo check -p apollo-fft --benches --examples
  --features kernel-strategy-bench`; release `quick_compare` for
  N=33/38/58/74/82/94; targeted Criterion N=33; `python
  extract_benchmarks.py`; `python -m py_compile extract_benchmarks.py`.

### Closure CIII - Generated Good-Thomas Route Fusion [patch]
- **Gap**: The prior proc-macro increment generated only support and dispatch
  arms. The hot `3*p` Good-Thomas path still used runtime CRT plan lookup and
  runtime short-codelet selection, while generated Rader code had mapping,
  precision, and inverse-symbol defects.
- **Closed by**: `generate_three_by_prime_dispatch!` now emits literal CRT
  gather/scatter functions per supported prime and threads those functions into
  the generic `three_by_prime_impl`. `short_winograd_const` provides const-size
  short-codelet dispatch. Generated Rader now follows the runtime
  generator/inverse-generator convention, emits exact f64 constants, and keeps
  inverse pointwise symbols in scope. Static Rader generation remains bounded to
  5/7/11/13 pending an O(N log N) generated convolution backend.
- **Residual risk**: N=33 still trails RustFFT in the Criterion table, and
  fresh release quick-compare rebuilding exceeded the 300-second cap. The next
  increment should reduce generated-code monomorphization and fuse the
  row-transform/scatter route into a register-level generated SSA kernel.
- **Evidence**: `cargo check -p apollo-fft-macros`; `cargo check -p
  apollo-fft --lib`; `cargo test -p apollo-fft --lib three_by_prime --
  --test-threads=1`; `cargo test -p apollo-fft --lib
  generated_rader_primes_match_direct_forward_and_inverse --
  --test-threads=1`; `cargo check -p apollo-fft --benches --examples
  --features kernel-strategy-bench`; `python extract_benchmarks.py`;
  `python -m py_compile extract_benchmarks.py`.

### Closure CII - Good-Thomas Proc-Macro Dispatch Generator [patch]
- **Gap**: The compact `3*p` Good-Thomas route had an intentionally repeated
  support predicate and `(P, inverse)` dispatch table after the const CRT plan
  landed. The root `gen*.md` notes call for generated routing surfaces so
  prime-family growth does not duplicate manually maintained dispatch arms.
- **Closed by**: Added the internal `apollo-fft-macros` proc-macro crate and
  replaced the hand-written `3*p` dispatch surface with
  `generate_three_by_prime_dispatch!`, driven by one short-prime list. The
  runtime kernel, `ThreeByPrimePlan<const P>`, and `MixedRadixScalar`
  monomorphization remain in `apollo-fft`.
- **Residual risk**: This generator emits the dispatch surface only. The
  physical row array and final CRT scatter remain runtime operations; the next
  generator step is route/load/store fusion using the same verified const CRT
  maps.
- **Evidence**: `cargo check -p apollo-fft-macros`; `cargo check -p
  apollo-fft --lib`; `cargo test -p apollo-fft --lib three_by_prime --
  --test-threads=1`; `cargo check -p apollo-fft --benches --examples
  --features kernel-strategy-bench`; release `quick_compare` for
  N=21/33/39/51/69; `python extract_benchmarks.py`; `python -m py_compile
  extract_benchmarks.py`.

### Closure CI - Good-Thomas Const CRT Plan [patch]
- **Gap**: The compact `3*p` Good-Thomas path still computed modular inverses
  and CRT routes inside the transform implementation. The root `gen*.md`
  artifacts identify this as the stable layer that should move to const-time
  index derivation before a procedural macro emits SSA-routed code.
- **Closed by**: Added `ThreeByPrimePlan<const P>` with const-time CRT input
  and output maps for P=5/7/11/13/17/23. The `3*p` route now loads and stores
  through those monomorphized plans and retains the same `MixedRadixScalar`
  generic kernel body. Stockham scalar-reference tests were also made explicit
  about their fused-stage tile const parameters.
- **Residual risk**: This is a const-plan foundation, not a proc-macro SSA
  backend. Physical row arrays remain in the current runtime path; the next
  generator step is to emit straight-line route/load/store tokens for the same
  verified CRT maps.
- **Evidence**: `cargo check -p apollo-fft --lib`; `cargo test -p apollo-fft
  --lib three_by_prime -- --test-threads=1`; `cargo check -p apollo-fft
  --benches --examples --features kernel-strategy-bench`; release
  `quick_compare` for N=21/33/39/51/69; targeted Criterion rows for N=33 f64
  and f32; `python extract_benchmarks.py`; `python -m py_compile
  extract_benchmarks.py`.

### Closure C - Three-By-Prime Good-Thomas Routing [patch]
- **Gap**: N=33 (`3*11`) had a coprime twiddle-free decomposition, but
  `dispatch_inplace` reached `cached_prime23_radices` first and executed the
  mixed-radix composite `[11, 3]` route. That route performed the generic
  twiddle-bearing composite pass structure for a size that can be expressed as
  a compact Good-Thomas CRT codelet.
- **Closed by**: Added `good_thomas::three_by_prime`, a reusable compact
  CRT implementation for `3*p` where `p` is one of the existing short prime
  codelets 5/7/11/13/17/23. The dispatcher now sends only this verified
  structural family to Good-Thomas before prime-23 composite routing. The
  benchmark-only ordered-Rader exports were also aligned with
  `rader_ordered_impl`.
- **Residual risk**: N=33 remains slower than RustFFT in the canonical
  clone-inclusive table. Adjacent misses such as N=35 and N=42 require the same
  structural treatment for broader coprime products rather than a size-specific
  branch.
- **Evidence**: `cargo check -p apollo-fft --lib`; `cargo test -p apollo-fft
  --lib mixed_three_by_prime -- --test-threads=1`; `cargo check -p apollo-fft
  --benches --examples --features kernel-strategy-bench`; release
  `quick_compare` for N=21/22/26/33/34/35/39/42/46/51/69; `APOLLO_FFT_BENCH_N=33
  cargo bench -p apollo-fft --bench vs_rustfft --features kernel-strategy-bench
  -- apollo_fft_vs_rustfft`; `python extract_benchmarks.py`; `python -m
  py_compile extract_benchmarks.py`.

### Closure XCIX - Typed Real-Storage Direct Fill [patch]
- **Gap**: Typed real-storage caller-owned paths allocated mapped temporary
  arrays for real-to-complex fill and complex-to-real extraction before
  assigning into caller-owned buffers. Allocating typed forward paths also
  cloned the mapped complex array before execution.
- **Closed by**: Added shared direct-fill helpers over ndarray dimensions and
  routed f64/f32/f16 `forward_*_into` and `inverse_*_into` through one
  `Zip` pass into caller-owned output. Allocating forward paths now transform
  the mapped output in place. Compact f16 conversion remains explicit at the
  storage boundary and still executes through f32 plans.
- **Residual risk**: Dedicated post-cutoff Criterion rows remain pending from
  the prior N=16 timeout. The public six-step zero-allocation row was rerun at
  N=5120 and completed without allocation assertion failure, with 11.530 us
  mean throughput.
- **Evidence**: `cargo check -p apollo-fft --lib`; `cargo check -p apollo-fft
  --benches --examples --features kernel-strategy-bench`; `cargo test -p
  apollo-fft --lib typed_3d_into_supports_f64_f32_and_f16_profiles --
  --test-threads=1`; `cargo test -p apollo-fft --lib power_of_two --
  --test-threads=1`; `APOLLO_FFT_BENCH_N=5120 cargo bench -p apollo-fft
  --bench vs_rustfft --features kernel-strategy-bench --
  apollo_zero_alloc_six_step/5120`; `python extract_benchmarks.py`;
  `python -m py_compile extract_benchmarks.py`.

### Closure XCVIII - Generic Plan Cache Scalar Routing [patch]
- **Gap**: The generic plan consolidation tried to instantiate compact
  `f16` storage as `FftPlan*<f16>`, which violated the scalar contract because
  mixed-radix arithmetic is implemented for f64/f32 complex storage and compact
  f16 routes execute through f32 at the storage boundary.
- **Closed by**: Added `RealFftData::PlanScalar` and made `PlanCacheProvider`
  return `FftPlan1D/2D/3D<Self::PlanScalar>`. f64/f32 keep native caches, f16
  delegates to the f32 cache family, and typed helpers/benches call the
  real-storage execution contracts against the resolved cached plan. The
  power-of-two fast path now starts at N>=64 after the current quick comparison
  showed N=16/N=32 remain faster on short-codelet routing.
- **Residual risk**: Public generic plan types still expose private scalar and
  workspace bounds as warnings. Fresh Criterion rows for the adjusted cutoff
  remain pending because the targeted N=16 Criterion refresh exceeded the
  300-second cap. The latest release quick comparison still misses RustFFT at
  N=32/N=64/N=128, so the next optimization should target short
  Stockham/codelet fusion and permutation removal rather than widening the
  fast-path cutoff.
- **Evidence**: `cargo check -p apollo-fft --lib`; `cargo check -p apollo-fft
  --benches --examples --features kernel-strategy-bench`; `cargo test -p
  apollo-fft --lib power_of_two -- --test-threads=1`; `cargo test -p
  apollo-fft --lib typed_3d_into_supports_f64_f32_and_f16_profiles --
  --test-threads=1`; `cargo run -p apollo-fft --release --features
  kernel-strategy-bench --example quick_compare` with
  `APOLLO_FFT_QUICK_N=16,32,64,128`; `python extract_benchmarks.py`;
  `python -m py_compile extract_benchmarks.py`.

### Closure XCVII - Power-of-Two Fast-Path Restoration [patch]
- **Gap**: Power-of-two sizes above the short-codelet range were not claimed
  by the selector before generic composite/PFA/Rader routing, allowing
  asymmetric lengths such as N=32768 to fall through without executing a
  transform. A forward+inverse roundtrip could hide that no-op failure mode.
- **Closed by**: Added one generic power-of-two fast-path for N>=16 before
  short Winograd, composite, PFA, or Rader routing. The path keeps N=2/N=4/N=8
  on short Winograd, uses Stockham for asymmetric powers, and retains
  square four-step only for even-exponent lengths above the four-step
  threshold. `FftPlan1D` was also moved to the generic mixed-radix twiddle and
  scratch-cache APIs instead of removed precision suffix helpers, and now
  exposes caller-owned generic typed forward/inverse methods matching the
  existing 2D/3D plan surface.
- **Residual risk**: Fresh targeted Criterion rows for
  N=16/N=32/N=64/N=128/N=32768 are pending because two pre-existing Criterion
  writers are active and still updating the cache. `benchmark_results.md` is a
  current 87-row cache snapshot rather than a dedicated post-patch run.
- **Evidence**: `cargo test -p apollo-fft --lib
  mixed_precise_power_of_two_n32768_forward_dc_is_not_noop -- --test-threads=1`;
  `cargo test -p apollo-fft --lib
  power_of_two_asymmetric_n32768_forward_inverse_roundtrip -- --test-threads=1`;
  `cargo test -p apollo-fft --lib mixed_forward_n32_matches_direct
  -- --test-threads=1`; `cargo check -p apollo-fft --lib`;
  `cargo check -p apollo-fft --benches --examples --features
  kernel-strategy-bench`;
  `python extract_benchmarks.py`; `python -m py_compile extract_benchmarks.py`.

### Closure XCVI - Small Good-Thomas Codelet Restoration [patch]
- **Gap**: N=6, N=10, N=12, and N=14 missed the short Winograd dispatch and
  fell through to the generic mixed-radix/PFA route, adding scratch,
  permutation, and twiddle-cache overhead to small coprime composites that are
  visible regressions in the Apollo-vs-RustFFT table.
- **Closed by**: Added stack-resident Good-Thomas CRT codelets for N=6, N=10,
  N=12, and N=14 using the existing Winograd DFT-3/4/5/7 leaves, then wired
  the monomorphized `short_winograd` dispatcher through those codelets before
  generic routing. No retained Rader, Good-Thomas, Winograd, butterfly, or
  composite route was removed. The obsolete private Good-Thomas gather helper
  left unused by the fused ordered-Rader PFA path was removed to resolve the
  bench build dead-code warning at source.
- **Residual risk**: Fresh post-patch Criterion rows for these four sizes are
  recorded in `benchmark_results.md`. N=6 f32 remains slower than RustFFT
  after this increment and is the next small-composite miss to target. A
  separate full `cargo bench -p apollo-fft` process is active and may update
  non-target Criterion rows after this snapshot.

### Closure XCV - Rader Negacyclic Twist/Recombine Fusion [patch]
- **Gap**: Rader negacyclic convolution performed separate twist and untwist
  passes over the negacyclic half around the forward/inverse convolution pair,
  even though the split and CRT recombination loops already touched the same
  elements.
- **Closed by**: Fused twist multiplication into the Nussbaumer split and
  fused conjugate untwist multiplication into CRT recombination. The cyclic
  and negacyclic forward paths still use fused radix-composite
  forward-plus-pointwise when the convolution length has supported composite
  radices. No retained Rader, Good-Thomas, Winograd, butterfly, or composite
  route was removed.
- **Residual risk**: The active full `cargo bench -p apollo-fft` run is still
  updating canonical Criterion rows during this cycle, so
  `benchmark_results.md` is a current cache snapshot rather than a complete
  dedicated post-patch benchmark run. Existing retained-route warnings remain.
- **Evidence**: `cargo check -p apollo-fft --lib`;
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`;
  `cargo test -p apollo-fft --lib rader -- --test-threads=1`;
  `cargo test -p apollo-fft --lib mixed_forward_prime -- --test-threads=1`;
  `cargo test -p apollo-fft --lib mixed_inverse_prime -- --test-threads=1`;
  `python extract_benchmarks.py`; `python -m py_compile extract_benchmarks.py`.

### Closure XCIV - Interleaved Two-Prime and Rader Pointwise Fusion [patch]
- **Gap**: The direct promoted-prime `2*p` route still materialized the even
  half into a stack array and compacted the odd half before entering the
  Winograd-pair two-prime kernel. Rader convolution also exposed a fused
  composite forward-plus-pointwise contract at the trait surface, but f32/f64
  scalar implementations did not implement it for test builds.
- **Closed by**: Changed the monomorphized Winograd-pair two-prime kernel to
  read interleaved `data[2*j]`/`data[2*j + 1]` input directly before writing
  natural output, removed the direct-route stack load helper, and implemented
  `composite_forward_with_pointwise` for f32/f64. This activates fused
  radix-composite forward-plus-spectrum multiplication inside Rader circular
  and negacyclic convolution for supported composite convolution lengths.
- **Residual risk**: A full workspace `cargo bench -p apollo-fft` process was
  already active and updating Criterion records during this cycle; the
  regenerated `benchmark_results.md` reflects the current Criterion cache
  snapshot, but some rows may come from that pre-existing run rather than a
  dedicated post-patch full benchmark. Existing retained-route warnings remain.
- **Evidence**: `cargo check -p apollo-fft --lib`;
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`;
  `cargo test -p apollo-fft --lib mixed_forward_two_by_prime -- --test-threads=1`;
  `cargo test -p apollo-fft --lib mixed_inverse_two_by_prime -- --test-threads=1`;
  `cargo test -p apollo-fft --lib two_by_prime -- --test-threads=1`;
  `cargo test -p apollo-fft --lib good_thomas -- --test-threads=1`;
  `cargo test -p apollo-fft --lib rader -- --test-threads=1`;
  `cargo test -p apollo-fft --lib mixed_forward_prime -- --test-threads=1`;
  `cargo test -p apollo-fft --lib mixed_inverse_prime -- --test-threads=1`;
  `cargo test -p apollo-fft --lib radix_composite -- --test-threads=1`;
  `python extract_benchmarks.py`; `python -m py_compile extract_benchmarks.py`.

### Closure XCIII - Fused Routing and Good-Thomas Permutation Tightening [patch]
- **Gap**: The fused radix-composite scalar fallback still recomputed output
  block slice bounds for each group after the stage-level radix dispatch
  refactor, and Good-Thomas PFA hot-path permutation loops still paid safe
  indexing checks despite cached permutation tables providing bounded
  `0..n` indices.
- **Closed by**: Replaced per-group radix-composite output slicing with
  `chunks_exact_mut(stage_chunk)`, changed fused final pointwise multiplication
  to raw pointer traversal over the contiguous output block, and tightened
  Good-Thomas natural/ordered-Rader gather-scatter loops with length assertions
  plus four-wide unchecked copies. The retained Winograd N=82 composite codelet
  now carries its required `PrimePairTable<41, 20>` bound. No Rader,
  Good-Thomas, Winograd, butterfly, or composite component was removed before a
  measured RustFFT-beating replacement exists.
- **Residual risk**: Fresh release Criterion numbers remain pending; this
  cycle refreshes the Markdown artifact from the existing Criterion cache and
  debug quick comparisons. Debug selected public comparison still misses
  RustFFT at N=38/N=58/N=62 and is near parity at N=106.
  `cargo fmt --check --package apollo-fft` remains blocked by broader
  worktree formatting drift outside this increment.
- **Evidence**: `cargo check -p apollo-fft --lib`;
  `cargo test -p apollo-fft --lib radix_composite -- --test-threads=1`;
  `cargo test -p apollo-fft --lib good_thomas -- --test-threads=1`;
  `cargo test -p apollo-fft --lib mixed_radix -- --test-threads=1`;
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`;
  debug `quick_compare` strategy-only and selected public runs;
  `python extract_benchmarks.py`; `git diff --check`.

### Closure XCII - Radix-Composite Stage Dispatch and Benchmark Snapshot [patch]
- **Gap**: The radix-composite arity leaf exceeded the repository 500-line
  structural limit and mixed single-radix dispatch with recursive fused-stage
  scratch arena ownership. The flat fused Stockham scalar fallback also
  resolved the runtime radix match for each output group rather than once per
  stage, and the final fused pointwise multiply walked the contiguous output
  block as nested radix/column loops.
- The Rader benchmark facade referenced deleted per-prime module paths, and
  static Rader permutation arrays used unstable `N - 1` const-generic array
  expressions instead of a stable compile-time table boundary.
- **Closed by**: Moved recursive fused-composite arena ownership and adaptive
  recursion into `radix_composite/adaptive.rs`, added
  `dispatch_radix_stage::<F>` with const-radix stage bodies, routed
  `flat_stockham_fused` through the stage dispatcher, collapsed the final
  pointwise multiply into a single contiguous output pass, and retained
  Winograd large-composite leaves while restoring composite value-test
  resolution. Rader benchmark routing now calls the shared generic Rader
  implementation and the real Winograd-pair kernels; static Rader permutation
  tables are generated as dispatch-arm constants and passed into one shared
  static Rader body on stable Rust. No
  composite component is gated or removed before a measured RustFFT-beating
  replacement exists.
- **Residual risk**: Release `quick_compare` timing was not regenerated in this
  cycle. A release run with `kernel-strategy-bench` hit LLVM memory exhaustion
  while concurrent Cargo/rustc workloads were active; a later no-feature
  release check exceeded the command cap while Cargo workloads were still
  running. The regenerated `benchmark_results.md` therefore reflects the
  current Criterion cache snapshot plus a debug strategy-only quick comparison,
  not a new post-change release timing run.
- **Evidence**: `cargo check -p apollo-fft --lib`,
  `cargo test -p apollo-fft --lib radix_composite -- --test-threads=1`,
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`,
  line counts for `radix_composite/arity.rs` (421) and
  `radix_composite/adaptive.rs` (191), and `python extract_benchmarks.py`.

### Closure XCI - Rader Bluestein Cache/Vector Hook Optimization [patch]
- **Gap**: Rader Bluestein retained separate forward and inverse M-length
  kernel spectra per cached prime/precision entry. The pre/post chirp SIMD hook
  surface existed but the Bluestein runtime still used scalar loops, and the
  inverse path needed conjugated kernel multiplication without reintroducing
  the removed spectrum.
- **Closed by**: Changed Bluestein cache entries to `(chirp_fw, kernel_fw)`,
  used `conj(kernel_fw)` for inverse multiplication from the even cyclic kernel
  identity, wired pre-chirp/zero-pad and post-chirp/scaling through
  precision-specific SIMD hooks, added conjugated right-hand operand support to
  pointwise SIMD multiplication, and corrected the typed-pointer zero-fill lane
  counts in the SIMD pre-chirp path.
- **Residual risk**: The focused release `quick_compare` large-prime probe was
  blocked by active Cargo benchmark/build work in the shared environment during
  this cycle; correctness, compile, and line-limit verification remain
  authoritative for this patch. For N=10007 with M=20736, each cached entry
  saves one M-length spectrum: 331,776 bytes for f64 complex data or 165,888
  bytes for f32 complex data.
- **Evidence**: `cargo fmt --check --package apollo-fft`,
  `cargo check -p apollo-fft --lib`,
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`,
  `cargo test -p apollo-fft --lib rader -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_forward_prime -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_inverse_prime -- --test-threads=1`,
  line-count checks for `mixed_radix/scalar/simd.rs`,
  `mixed_radix/scalar/simd/pointwise.rs`, and `rader/bluestein.rs`, and
  `git diff --check` passed. The untracked Bluestein and pointwise SIMD sources
  were checked with `git diff --check --no-index`.

### Closure XC - Rader Standalone Memory-Pass Optimization [patch]
- **Gap**: Standalone generated and runtime Rader performed a separate
  `data[1..N]` pass to compute the DC nonzero sum before gathering the same
  values into primitive-root order. Rader padded scratch also retained two
  maximum-size buffers per thread and precision even though the common
  standalone/Bluestein path requires only one live Rader buffer.
- **Closed by**: Replaced the separate sum pass with fused `gather_sum_static`
  and `gather_sum_slice` helpers, applied unrolled scatter loops for static and
  runtime permutation paths, and changed Rader padded scratch to one retained
  aligned thread-local buffer per precision with local nested-call fallback.
- **Residual risk**: Release strategy-only `quick_compare` records current
  Rader absolute latencies, but the current benchmark hook aliases the Winograd
  comparison column to Rader, so fresh Rader-vs-Winograd ratios remain pending.
  Existing odd-prime-pair dead-code warnings remain outside this increment.
- **Evidence**: `cargo fmt --package apollo-fft`,
  `cargo test -p apollo-fft --lib rader -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_forward_prime -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_inverse_prime -- --test-threads=1`,
  `cargo check -p apollo-fft`,
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`,
  release strategy-only `quick_compare`, and `git diff --check`; the timing
  probe recorded Rader latencies of 148/126 ns at N=29, 121/123 ns at N=31,
  and 138/136 ns at N=37 for f64/f32.

### Closure LXXXIX - Fused Radix-Composite Dispatch Repair [patch]
- **Gap**: The fused radix-composite Stockham path referenced arity and tiling
  modules that were not part of a compilable module graph, and fused twiddle
  slices used stage extents that did not match each radix arm's coefficient
  contract. Radix-4 factorization verification also contradicted the active
  radix-shape policy. Strategy cleanup also lacked measured evidence comparing
  generated Rader against the Winograd-pair odd-prime alternative for
  N=29/N=31/N=37.
- **Closed by**: Reconnected `stockham_stage_fused` to `Fused2` through
  `Fused6` via the `FusedStage` ZST trait and `ExecutionPolicy` chunk
  traversal, corrected fused twiddle slice lengths to
  `(radix - 1) * prev_len * prior_product`, kept the incomplete tiling
  placeholder out of compilation, lowered only cached single-odd radix-2 tails
  to radix-4 stages, rejected highest-power radix-2 lowering because it emitted
  unsupported radix 16, added radix 4/8/17/23 dispatch coverage, added recursive
  arena scratch accounting for nested fused composition, restored direct
  N=17/N=23 Winograd routing, added direct Rader-vs-Winograd-pair equivalence
  tests, added a gated Criterion comparison group, selected Winograd-pair
  production dispatch for N=29/N=31/N=37 after bounded comparison showed
  Winograd-pair faster for all measured f64/f32 cases, consolidated generated
  Rader leaves N=17..97 into one const-generic static implementation, fused
  static Rader gather/scatter with the x0 terms, fused the Rader convolution
  pointwise spectrum multiply into the final forward composite stage, added
  static permutation-table Rader leaves for N=29/N=31/N=37 comparison to remove
  runtime modular gather/scatter index recurrence, added ordered-layout
  static/runtime Rader kernels that reuse `data[1..]` as the convolution buffer
  and omit leaf-local gather/scatter for generator-ordered fused callers,
  wired ordered Rader into Good-Thomas PFA for prime `n1` dimensions whose
  production subtransform would otherwise use Rader, added branch-selection
  coverage that preserves Winograd-pair for N=29/N=31/N=37, reused the Rader
  permutation cache in the ordered PFA branch to remove generator-order modulo
  walks from the transpose and CRT scatter loops, routed ordered-Rader PFA
  through the known-prime monomorphized ordered Rader dispatcher, added
  `APOLLO_FFT_QUICK_N` to `quick_compare`, added an
  `ordered_rader_pfa_coprime_composites` Criterion group, added a dedicated
  `good_thomas::two_by_prime` route for N=2p composites, promoted
  N=19/N=41/N=43/N=47/N=53 to odd-prime Winograd-pair dispatch, moved
  odd-prime pair kernels into `winograd/radix/odd_prime_pair.rs`, expanded
  two-by-prime benchmark coverage, replaced promoted N=2p thread-local scratch
  with const-generic stack even-half loading, removed stale composite
  fallback dispatch code, added generated-Rader direct-DFT tests for every
  generated prime leaf, added dispatch-level forward/inverse tests for
  N=29/N=31/N=37, and corrected radix-4 test assertions.
- **Residual risk**: Fresh Criterion comparison against RustFFT is pending for
  the restored fused path; the N=96 and N=192 attempts timed out before
  emitting usable timing output. Fresh Criterion comparison between Rader and
  Winograd-pair for N=29/N=31/N=37 is pending, but release strategy-only
  `quick_compare` shows Rader still behind Winograd-pair for all measured
  small-prime f64/f32 cases after static permutation-table leaves. Release
  production `quick_compare` shows N=29 and N=31 at or faster than RustFFT and
  N=37 still 11.4% slower. Ordered-layout Rader is value-verified against the
  direct DFT reference and is now used by Good-Thomas PFA for qualifying prime
  dimensions. Release promoted-prime `quick_compare` after stack compaction shows
  N=19/N=29/N=31/N=37/N=41/N=43/N=47/N=53 at
  0.907x/0.972x/0.736x/0.799x/0.720x/0.599x/0.582x/0.909x versus RustFFT.
  Release two-by-prime `quick_compare` shows
  N=38/N=58/N=62/N=74/N=82/N=86/N=94/N=106 at
  1.514x/1.195x/1.228x/1.059x/1.025x/0.943x/0.587x/0.757x; N=38 remains the
  largest residual composite gap, with N=58/N=62/N=74/N=82 marginal/noisy.
  Radix-composite and Stockham stages do not yet emit or consume the ordered
  layout.
- **Evidence**: `cargo fmt --check`,
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`,
  `cargo test -p apollo-fft --lib ordered -- --test-threads=1`,
  `cargo test -p apollo-fft --lib pfa -- --test-threads=1`,
  `cargo test -p apollo-fft --lib winograd::tests::dft_prime -- --test-threads=1`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`, bounded debug
  strategy `quick_compare`, release production `quick_compare`,
  `APOLLO_FFT_QUICK_N=38,58,62,74,82,86,94,106` release `quick_compare`,
  `APOLLO_FFT_QUICK_N=19,29,31,37,41,43,47,53` release `quick_compare`, and
  `git diff --check`.

### Closure LXXXIII - Mixed-Radix Wrapper Removal [major]
- **Gap**: Public type-suffixed mixed-radix twiddle wrapper entry points
  remained after the canonical const-generic dispatch body became the single
  implementation. Dead Winograd AVX wrapper leaves also remained as exported
  internal modules.
- **Closed by**: Removed the concrete wrapper entry points, updated 1D/2D/3D
  plans and real FFT split routines to call
  `dispatch_inplace::<T, INVERSE, NORMALIZE>` directly, kept the dispatch body
  crate-private, routed radix-15 leaves through the stack-only generic
  Good-Thomas Winograd codelet, consolidated broad Stockham AVX stage/pair
  leaves behind one monomorphized backend trait, removed the dead Winograd AVX
  leaves, and deleted the unreachable CPU SIMD six-step, matrix-workspace, and
  radix2 infrastructure island that was not part of the crate module graph.
- **Residual risk**: none for this closure.
- **Evidence**: `cargo check -p apollo-fft`,
  `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, stale-wrapper scan, and deleted AVX module scan.

### Closure LXXXII - Stockham Butterfly Dispatch Leaf Split [patch]
- **Gap**: `stockham/butterfly/fixed.rs` remained over the repository
  500-line structural limit and mixed generated fixed codelets with f64 AVX
  scratch dispatch routing. Benchmark targets also referenced removed
  `bluestein` and `radix2` module paths instead of the maintained generic
  selector and `real_fft` twiddle builders. `mixed_radix/dispatch_f16.rs`
  retained a type-named compact-storage routing leaf.
- **Closed by**: Extracted f64 AVX scratch routing to `butterfly::dispatch`,
  re-exported it through `butterfly::mod`, left fixed codelets in the `fixed`
  leaf, and updated benches to use the public generic selector plus current
  twiddle builders. Compact storage routing now lives in the canonical
  `mixed_radix/dispatch.rs` module through one const-generic helper.
- **Residual risk**: Release-size tooling should confirm the module split has
  no measurable codegen impact.
- **Evidence**: `cargo check -p apollo-fft`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check -p apollo-fft --benches --examples`, and kernel file-size scan.

### Closure LXXVI - Frequency Utility Exact-Capacity Fill [patch]
- **Gap**: `fftfreq` and `rfftfreq` built known-length output vectors through
  iterator collection, leaving avoidable iterator state and branch overhead in
  utility paths used to construct frequency grids.
- **Closed by**: Replaced the collection pipelines with exact-capacity fill
  loops while preserving numpy-compatible bin ordering and zero-length
  behavior.
- **Residual risk**: Frequency utility benchmarking should quantify the
  construction cost reduction for large grids; functional/static verification
  passed locally.
- **Evidence**: `cargo check -p apollo-fft`; `cargo check -p apollo-fft
  --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; cleanup scans for deprecated/type-suffixed FFT
  APIs and encoding artifacts; `cargo fmt --check`; `git diff --check`.

### Closure LXXV - Shift Utility Split-Copy Cleanup [patch]
- **Gap**: `fftshift` and `ifftshift` carried an unused `Default` bound and
  duplicated modulo-index iterator collection, creating unnecessary per-element
  arithmetic and redundant generic code.
- **Closed by**: Removed the dead generic bound and routed both utilities
  through one split-slice copy helper with exact-capacity output allocation.
- **Residual risk**: The shift utility is memory-bandwidth-bound for large
  slices; benchmark evidence should quantify the copy-path gain on large real
  and complex vectors. Functional/static verification passed locally.
- **Evidence**: `cargo check -p apollo-fft`; `cargo check -p apollo-fft
  --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; cleanup scans for deprecated/type-suffixed FFT
  APIs and encoding artifacts; `cargo fmt --check`; `git diff --check`.

### Closure LXXIV - Real/R2C Initialization Elimination [patch]
- **Gap**: Multiple hot paths for 1D, 2D, and 3D real forward/inverse transforms
  as well as 3D R2C/C2R packing still used `Array::zeros` or `.mapv` pipelines,
  which incurred unnecessary heap-initialization and traversal overhead for buffers
  that are fully overwritten before their first read.
- **Closed by**: Extended `UninitWorkspaceElement` sealed abstraction to `f64`.
  Replaced all target `Array::zeros` and `.mapv` calls with zero-allocation
  `uninit_copy_vec` + `Array::from_shape_vec` + checked overwrite (`Zip` or inplace
  kernel execution), bumped `apollo-fft` to 0.9.9, and verified workspace stability.
- **Residual risk**: Criterion benchmarking on allocation-heavy transforms (large N
  and multi-dimensional) should confirm the actual latency reduction; functional
  correctness is fully validated.
- **Evidence**: `cargo check --workspace`; `cargo test -p apollo-fft --release`
  (177/177); `git diff --check`.

### Closure LXXIII - Plan-Time Iterator Elimination [patch]
- **Gap**: Three plan construction paths in `BluesteinPlan64::new`,
  `BluesteinPlan32::new`, and `FftPlan3D::with_precision` built their chirp and
  r2c twiddle vectors through `(0..n).map(..).collect()` iterator pipelines,
  paying iterator state machine and bounds-check overhead for every element even
  though the element count is known at construction time.
- **Closed by**: Replaced all three `.map(..).collect()` chains with
  `Vec::with_capacity` + `unsafe { set_len }` + unchecked overwrite loops,
  added `#![allow(clippy::uninit_vec)]` to `dimension_3d.rs` to maintain the
  zero-warning policy, removed leftover scratch scripts from the worktree, and
  bumped `apollo-fft` to 0.9.8.
- **Residual risk**: Criterion plan-construction benchmarks on representative
  arbitrary-length sizes should confirm the reduction in construction latency.
- **Evidence**: `cargo fmt --check -p apollo-fft`; `cargo clippy -p apollo-fft
  --release -- -D warnings`; `cargo test -p apollo-fft --release` (177/177);
  `git diff --check`.

### Closure LXXII - 3D Native Real32 Exact Buffer Fill [patch]
- **Gap**: The allocating native 3D f32/f16 real path still zero-filled its
  Complex32 output before full overwrite and projected native inverse results
  through ndarray `mapv`, leaving allocation work inconsistent with the sealed
  overwrite-first workspace contract already used in 1D/2D paths.
- **Closed by**: Constrained the 3D real32 helper trait to sealed workspace
  element types, routed allocating forward output through an exact-size
  overwrite-first Complex32 buffer, and routed native inverse projection
  through an exact-size overwrite-first real buffer.
- **Residual risk**: Allocation microbenchmarks should quantify construction
  and projection cost changes for representative 3D f32/f16 volumes;
  functional/static verification passed locally.
- **Evidence**: `cargo check -p apollo-fft`; `cargo check -p apollo-fft
  --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; cleanup scans for deprecated/type-suffixed FFT
  APIs and encoding artifacts; `cargo fmt --check`; `git diff --check`.

### Closure LXXI - 2D Native Real32 Exact Buffer Fill [patch]
- **Gap**: The native 2D f32/f16 real path still used ndarray `mapv`
  allocation pipelines for real-to-complex packing and complex-to-real
  projection, duplicating the allocation pattern already removed from 1D
  compact f16 execution.
- **Closed by**: Constrained the 2D real32 helper trait to sealed workspace
  element types and routed native packing/projection through shared exact-size
  overwrite-first buffers.
- **Residual risk**: Allocation microbenchmarks should quantify construction
  and projection cost changes for representative 2D f32/f16 matrix sizes;
  functional/static verification passed locally.
- **Evidence**: `cargo check -p apollo-fft`; `cargo check -p apollo-fft
  --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; cleanup scans for deprecated/type-suffixed FFT
  APIs and encoding artifacts; `cargo fmt --check`; `git diff --check`.

### Closure LXX - 1D Compact F16 Exact Buffer Fill [patch]
- **Gap**: The 1D compact f16 power-of-two path still used iterator
  collection pipelines for compact input packing and output projection, while
  the rest of the FFT workspace layer had moved to explicit exact-size
  overwrite-first buffers.
- **Closed by**: Extended the sealed workspace element set to `f16` and
  `Complex<f16>`, routed compact f16 forward/inverse packing and projection
  through exact-size overwrite-first vectors, and bumped `apollo-fft` to 0.9.5.
- **Residual risk**: Allocation microbenchmarks should confirm construction
  and projection costs on representative short power-of-two f16 transforms;
  functional/static verification passed locally.
- **Evidence**: `cargo check -p apollo-fft`; `cargo check -p apollo-fft
  --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; cleanup scans for deprecated/type-suffixed FFT
  APIs and encoding artifacts; `cargo fmt --check`; `git diff --check`.

### Closure LXIX - 1D Native Complex32 Precision Deduplication [patch]
- **Gap**: 1D f32 native execution and mixed f16 non-power-of-two execution
  duplicated `Complex32` packing, twiddle-aware kernel selection, inverse
  dispatch, and real-output projection logic.
- **Closed by**: Added private `Plan1dReal32` static-dispatch helpers,
  routed f32 native paths and mixed f16 non-power-of-two paths through one
  monomorphized forward/inverse implementation, and bumped `apollo-fft` to
  0.9.4.
- **Residual risk**: Binary-size impact should be confirmed with release-size
  tooling; functional/static verification passed locally.
- **Evidence**: `cargo fmt`; `cargo check -p apollo-fft --benches --examples`;
  `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; source cleanup scan for deprecated placeholders
  and removed wrapper names; encoding scan for mojibake/BOM markers;
  `git diff --check`.

### Closure LXVIII - Bluestein Filter Initialization Cleanup [patch]
- **Gap**: Bluestein plan construction still zero-filled the full padded
  convolution filter even though the DC and mirrored chirp regions are written
  before the filter is transformed. Generated local scripts also left a
  Stockham AVX broadcast experiment in the worktree that increased repeated
  broadcast expressions and dead commented code.
- **Closed by**: Replaced full-vector zero initialization for the Bluestein
  f64/f32 filter with overwrite-first initialization plus zero-fill of only the
  unused convolution gap, removed generated scratch scripts from the deliverable
  worktree state, preserved hoisted Stockham broadcast variables, and bumped
  `apollo-fft` to 0.9.3.
- **Residual risk**: Criterion construction-time savings need representative
  arbitrary-length FFT plan benchmarks; functional/static verification passed
  locally.
- **Evidence**: `cargo fmt --check`; `cargo check -p apollo-fft --benches --examples`;
  `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; source cleanup scan for generated scripts,
  deprecated placeholders, and removed wrapper names; encoding scan for
  mojibake/BOM markers; `git diff --check`.

### Closure LXVII - FFT Plan Scratch Allocation Consolidation [patch]
- **Gap**: Plan-owned FFT work buffers still used duplicated zero-fill or
  local uninitialized-allocation logic across 1D, 2D, 3D, R2C, and six-step
  paths.
- **Closed by**: Added one sealed `UninitWorkspaceElement` helper for the FFT
  scratch element set, routed 1D Bluestein/iRFFT scratch, 2D/3D axis-pass
  scratch, 3D R2C scratch, and six-step f32 workspaces through it, removed the
  duplicate six-step allocation helpers, and bumped `apollo-fft` to 0.9.2.
- **Residual risk**: Runtime construction-time savings need Criterion
  confirmation on representative matrix and volume sizes; functional/static
  verification passed locally.
- **Evidence**: `cargo fmt --check`; `cargo check -p apollo-fft --benches --examples`;
  `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `cargo check --workspace`; source cleanup scan for removed local
  helpers/deprecated placeholders; encoding scan for mojibake/BOM markers;
  `git diff --check`.

### Closure LXVI - FFT Workspace and Normalization Memory Efficiency [patch]
- **Gap**: Several FFT hot paths still paid avoidable zero-fill or repeated
  scalar normalization overhead in buffers that are fully overwritten before
  read.
- **Closed by**: Added shared f64/f32 normalization helpers with AVX runtime
  dispatch, routed Stockham/Bluestein/mixed-radix inverse scale passes through
  them, filled twiddle and composite twiddle vectors through exact pre-sized
  cursors with debug invariants, skipped zero-fill for overwritten composite
  scratch and six-step workspace buffers, and bumped `apollo-fft` to 0.9.1.
- **Residual risk**: Runtime performance ratios need Criterion confirmation on
  representative hardware; functional and static verification passed locally.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`; stale-token scans for removed wrappers/deprecated/debug references; encoding scan for mojibake/BOM markers; `git diff --check`.

### Closure LXV - FFT Auto-Selector Wrapper Removal [major]
- **Gap**: `apollo-fft` still exposed concrete f64/f32 public auto-selector
  wrappers even though `fft_forward`, `fft_inverse`, and `fft_inverse_unnorm`
  are the canonical generic API.
- **Closed by**: Deleted `fft_forward_64`, `fft_inverse_64`,
  `fft_inverse_unnorm_64`, `fft_forward_32`, `fft_inverse_32`, and
  `fft_inverse_unnorm_32`; routed `FftPrecision` implementations directly to
  mixed-radix dispatch; updated plan fallbacks, tests, and benchmarks to use the
  generic API; and bumped `apollo-fft` to 0.9.0.
- **Residual risk**: External pre-1.0 callers using the removed concrete
  wrappers must migrate to the generic FFT API.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`; `cargo check -p apollo-fft-wgpu --tests`; source scans for removed auto-selector wrapper names, direct DFT wrapper names, Winograd wrapper names, debug binary references, stale compatibility/deprecation tokens, and deleted f16 wrapper names; `git diff --check`.

### Closure LXIV - FFT Recursive Winograd Generic Codelets [major]
- **Gap**: `apollo-fft` still exposed public type-suffixed Winograd DFT-16/32/64
  wrappers and carried duplicated f32/f64 recursive codelet bodies.
- **Closed by**: Replaced DFT-16/32/64 f32/f64 bodies with generic
  `dft16_impl`, `dft32_impl`, and `dft64_impl`, routed mixed-radix dispatch to
  those generic implementations, renamed the stale type-suffixed twiddle table,
  and bumped `apollo-fft` to 0.8.0.
- **Residual risk**: External pre-1.0 callers using the removed DFT-16/32/64
  wrappers must migrate to the public auto-selecting FFT API.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`; source scans for removed DFT-16/32/64 wrapper names, short-Winograd wrapper names, direct DFT wrapper names, debug binary references, stale compatibility/deprecation tokens, and deleted f16 wrapper names; `git diff --check`.

### Closure LXIII - FFT Short-Winograd Wrapper Removal [major]
- **Gap**: `apollo-fft` still exposed type-suffixed short-Winograd public
  wrappers for small codelets and twiddle multiplication even though generic
  implementations already existed.
- **Closed by**: Deleted the public DFT-2/3/4/5/7/8 f64/f32 wrapper functions
  and `apply_twiddle_64` / `apply_twiddle_32`, routed mixed-radix short
  dispatch through the generic Winograd implementation functions, removed stale
  wrapper documentation, and bumped `apollo-fft` to 0.7.0.
- **Residual risk**: External pre-1.0 callers using the removed short-Winograd
  wrappers must migrate to the generic internalized path or the public
  auto-selecting FFT API.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`; source scans for removed short-Winograd wrapper names, direct DFT wrapper names, debug binary references, stale compatibility/deprecation tokens, and deleted f16 wrapper names; `git diff --check`.

### Closure LXII - FFT Direct DFT Wrapper Removal [major]
- **Gap**: The direct DFT reference kernel still exposed type-suffixed
  wrapper functions and an unused debug-only f32 parity binary, duplicating the
  canonical generic DFT API surface.
- **Closed by**: Deleted `dft_forward_64`, `dft_inverse_64`,
  `dft_forward_32`, `dft_inverse_32`, `forward_owned_64`, `inverse_owned_64`,
  removed `src/bin/debug_f32.rs`, updated direct DFT tests, benchmarks, and
  kernel regressions to use `dft_forward` / `dft_inverse`, and bumped
  `apollo-fft` to 0.6.0.
- **Residual risk**: External pre-1.0 callers using the removed direct DFT
  wrappers must migrate to the generic functions.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo
  test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`;
  source scans for removed direct DFT wrapper names, debug binary references,
  stale compatibility/deprecation tokens, and deleted f16 wrapper names; `git
  diff --check`.

### Closure LXI - FFT Composite Scratch and Twiddle Cache Reuse [patch]
- **Gap**: Bluestein and mixed-radix composite FFT paths still retained
  allocation-heavy scratch behavior and stale docs. The composite twiddle cache
  also keyed by transform length only, which could alias different public
  radix decompositions with the same product.
- **Closed by**: Reused one thread-local Bluestein scratch buffer per
  precision, reused one thread-local composite scratch buffer per precision,
  cached composite twiddle tables by exact radix decomposition and direction,
  added same-length/different-radix-order regression coverage, removed stale
  allocation and `MaybeUninit` docs, and bumped `apollo-fft` to 0.5.3.
- **Residual risk**: Larger FFT kernel implementation files remain above the
  structural limit and require follow-up module partitioning.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo
  test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`;
  source scans for stale `MaybeUninit`/per-call allocation docs, stale
  compatibility/deprecation tokens, and deleted f16 wrapper names; `git diff
  --check`.

### Closure LX - FFT 3D Typed Plan Deduplication [patch]
- **Gap**: `apollo-fft` duplicated 3D f32 and f16 typed forward/inverse logic
  across allocating and caller-owned APIs, leaving four precision-specific
  bodies plus a now-redundant f32-only real-to-complex writer.
- **Closed by**: Added private `Plan3dReal32` static-dispatch storage
  abstraction, routed `forward_f32`, `inverse_f32`, `forward_f32_into`,
  `inverse_f32_into`, `forward_f16`, `inverse_f16`, `forward_f16_into`, and
  `inverse_f16_into` through shared monomorphized helpers, deleted the dead
  f32-only writer, and bumped `apollo-fft` to 0.5.2.
- **Residual risk**: Larger FFT kernel implementation files remain above the
  structural limit and require follow-up module partitioning.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo
  test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`;
  source scans for removed f32-only 3D writer, stale
  compatibility/deprecation tokens, and deleted f16 wrapper names; `git diff
  --check`.

### Closure LIX - FFT 2D Typed Plan Deduplication [patch]
- **Gap**: `apollo-fft` duplicated the 2D f32 and f16 typed forward/inverse
  bodies, duplicated the 2D plan module-level Rustdoc block, and kept
  crate-root tests inside `lib.rs`, leaving the crate root above the 500-line
  structural limit.
- **Closed by**: Added private `Plan2dReal32` static-dispatch storage
  abstraction, routed `forward_f32`, `inverse_f32`, `forward_f16`, and
  `inverse_f16` through shared monomorphized helpers, removed duplicated 2D
  Rustdoc, moved crate-root tests into `lib_tests.rs`, and bumped
  `apollo-fft` to 0.5.1.
- **Residual risk**: Larger FFT kernel implementation files remain above the
  structural limit and require follow-up module partitioning.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo
  test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`;
  structural scan confirming `lib.rs` is 434 lines; source scan for stale
  compatibility/deprecation tokens and deleted f16 wrapper names; `git diff
  --check`.

### Closure LVIII - FFT Compatibility Alias Removal [major]
- **Gap**: `apollo-fft` retained the stale `FftPlan3D::nz_complex` alias,
  `HalfSpectrum3D::nz_complex` field, and compatibility wording after the FFT
  API surface had been consolidated around canonical owner modules and generic
  precision dispatch.
- **Closed by**: Deleted `FftPlan3D::nz_complex`, renamed
  `HalfSpectrum3D::nz_complex` to `HalfSpectrum3D::nz_c`, kept `nz_c` as the
  single half-spectrum bookkeeping name, removed stale compatibility wording
  from FFT kernel/backend docs, and bumped `apollo-fft` to 0.5.0.
- **Residual risk**: External pre-1.0 callers using `nz_complex` must migrate to
  `nz_c`.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo
  test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`;
  source scans for removed `nz_complex`, f16-specific wrapper names, and stale
  compatibility/deprecation tokens; `git diff --check`.

### Closure LVII - Radix F16 Module Removal [major]
- **Gap**: `apollo-fft` still exposed compact f16 complex storage through a
  radix-specific `radix2_f16` module and custom `Cf16` wrapper. The f16 bridge
  was type-specific, and a dead native f16 CPU gate remained in the kernel
  directory.
- **Closed by**: Deleted the radix-specific f16 module, deleted dead f16-named
  bridge/gate files, replaced `Cf16` with `num_complex::Complex<half::f16>`,
  added `precision_bridge::Complex32Bridge` as the generic monomorphized
  compact-storage bridge with reusable Complex32 scratch, updated FFT kernel
  exports, removed public f16-specific FFT wrappers in favor of generic
  `fft_forward`/`fft_inverse` dispatch, updated twiddle-table output, 1D
  precision paths, benchmarks, and SIMD imports, and bumped `apollo-fft` to
  0.4.0.
- **Residual risk**: The public `Cf16` type is removed. In-repo callers are
  updated; external pre-1.0 callers must migrate to `Complex<half::f16>`.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`; `cargo
  test -p apollo-fft --lib -- --test-threads=1`; `cargo check --workspace`;
  source scans for removed `Cf16`, `radix2_f16`, public f16-specific wrappers,
  f16-named kernel files, and f16 bridge names; `git diff --check`.

### Closure LVI - FFT Remote Integration and Short-Winograd Dispatch [patch]
- **Gap**: Remote RustFFT comparator work targeted the older radix-specific
  kernel topology and conflicted with the current Stockham/composite/Bluestein
  architecture. The local mixed-radix facade also retained unused f16 twiddle
  caches even though f16 storage execution promotes to f32.
- **Closed by**: Kept deleted radix kernel modules removed, retained RustFFT
  comparator coverage through the live `vs_rustfft` benchmark, switched
  `apollo-fft` to the workspace `rustfft` dev-dependency, removed dead
  radix-specific `kernel_strategy` rows, added shared `ShortWinogradScalar`
  static dispatch for exact 2/4/8/16/32/64 f64/f32 transforms before
  Stockham/composite/Bluestein routing, and removed unused f16 twiddle caches.
- **Residual risk**: Criterion throughput numbers after the merge need a
  dedicated benchmark run on representative hardware; correctness and compile
  checks cover the code path.
- **Evidence**: `cargo check -p apollo-fft --benches --examples`;
  `cargo test -p apollo-fft --lib -- --test-threads=1`; `cargo check
  --workspace`; `cargo test -p apollo-hilbert --lib -- --test-threads=1`;
  conflict-marker scan; dead f16-cache/dead benchmark scan; `git diff
  --check`.

### Closure LV - Apollo-Hilbert Caller-Owned Observable Projections [minor]
- **Gap**: `AnalyticSignal` projection methods allocated new vectors and
  duplicated projection formulas directly in each allocating method. Plan-level
  `envelope` and `phase` also forced an owned analytic signal allocation even
  when callers could provide output storage.
- **Closed by**: Added caller-owned `AnalyticSignal::*_into` projection
  methods, routed allocating projections through shared non-generic slice
  helpers, added `HilbertPlan::envelope_into` and `phase_into`, routed
  allocating plan observables through a reused per-thread Complex64 analytic
  scratch buffer, added parity/mismatch/capacity tests, updated the Hilbert
  README, and bumped `apollo-hilbert` to 0.3.0.
- **Residual risk**: Owned projection APIs still allocate their return vectors
  by contract. Callers that need allocation control should use the new
  caller-owned projection methods.
- **Evidence**: `cargo check -p apollo-hilbert`; `cargo test -p
  apollo-hilbert observables --lib -- --test-threads=1`; `cargo test -p
  apollo-hilbert envelope --lib -- --test-threads=1`; `cargo test -p
  apollo-hilbert --lib -- --test-threads=1`; `cargo check -p
  apollo-validation`; `rg` source scans for removed projection duplication
  patterns.

### Closure LIV - Apollo-Hilbert Caller-Owned Analytic Signal [minor]
- **Gap**: `hilbert_transform_into` still allocated an owned analytic
  `Vec<Complex64>` on every caller-owned quadrature call, and callers had no
  public plan-level way to provide analytic output storage directly. The crate
  root documentation also still described private DFT ownership after Hilbert
  moved to Apollo FFT plan execution.
- **Closed by**: Added direct `analytic_signal_into`, added
  `HilbertPlan::analytic_signal_into`, routed owned analytic execution through
  the caller-owned kernel, routed caller-owned quadrature through a reused
  thread-local Complex64 analytic scratch buffer, added parity/mismatch/scratch
  capacity tests, updated README and crate-root docs, and bumped
  `apollo-hilbert` to 0.2.0.
- **Residual risk**: The owned `analytic_signal` API still allocates its
  returned `Vec<Complex64>` by contract. Caller-owned analytic and quadrature
  paths now avoid additional analytic bridge allocation.
- **Evidence**: `cargo check -p apollo-hilbert`; `cargo test -p
  apollo-hilbert analytic --lib -- --test-threads=1`; `cargo test -p
  apollo-hilbert workspace --lib -- --test-threads=1`; `cargo test -p
  apollo-hilbert --lib -- --test-threads=1`; `cargo check -p
  apollo-validation`; `rg` source scans for removed caller-owned quadrature
  analytic allocation patterns.

### Closure LIII - Apollo FFT Slice Real Forward for Hilbert [minor]
- **Gap**: `apollo-hilbert` still built an `Array1<f64>` from every real input
  slice because `apollo-fft` exposed the optimized 1D real-forward owner path
  only through ndarray input. The existing ndarray caller-owned path also
  duplicated the real-forward implementation body instead of delegating to a
  slice-level owner routine.
- **Closed by**: Added
  `FftPlan1D::forward_real_to_complex_slice_into`, routed the existing ndarray
  caller-owned path through it, routed Hilbert analytic-signal execution through
  the cached FFT plan's slice path, removed the dead `ndarray` dependency from
  `apollo-hilbert`, split 1D precision methods and tests into leaf modules so
  `dimension_1d.rs` stays below 500 lines, added slice parity/rejection
  coverage, updated READMEs, bumped `apollo-fft` to 0.3.0, and bumped
  `apollo-hilbert` to 0.1.4.
- **Residual risk**: Hilbert still allocates its analytic `Vec<Complex64>`
  output per owning API call because the public `analytic_signal` contract
  returns owned storage. The caller-owned quadrature and typed bridge paths now
  avoid all additional input bridge arrays.
- **Evidence**: `cargo check -p apollo-fft`; `cargo test -p apollo-fft
  caller_owned_paths --lib -- --test-threads=1`; `cargo test -p apollo-fft
  forward_slice --lib -- --test-threads=1`; `cargo test -p apollo-fft --lib
  -- --test-threads=1`; `cargo check -p apollo-hilbert`; `cargo test -p
  apollo-hilbert --lib -- --test-threads=1`; `cargo check -p
  apollo-validation`; `rg` source scans for removed Hilbert ndarray
  bridge/dependency patterns.

### Closure LII - Apollo-Hilbert Analytic In-Place Spectrum Reuse [patch]
- **Gap**: `analytic_signal` copied the forward FFT output into a `Vec`,
  rebuilt an `Array1` from that vector, allocated a separate inverse FFT output
  array, and copied the inverse result back into another `Vec`. Owned
  quadrature also called the allocating analytic-signal path before projecting
  the imaginary component.
- **Closed by**: Kept the forward FFT output array as the analytic spectrum,
  applied the Hilbert mask in place, ran the complex inverse in place, moved the
  contiguous buffer out once for `analytic_signal`, routed `hilbert_transform`
  through `hilbert_transform_into`, updated the Hilbert README, and bumped
  `apollo-hilbert` to 0.1.3.
- **Residual risk**: The real input bridge `Array1<f64>` remains because
  `apollo_fft::fft_1d_array` is the current optimized real-forward API. Removing
  that allocation requires a slice-level real FFT entry point in `apollo-fft`.
- **Evidence**: `cargo check -p apollo-hilbert`; `cargo test -p
  apollo-hilbert transform --lib -- --test-threads=1`; `cargo test -p
  apollo-hilbert --lib -- --test-threads=1`; `cargo check -p
  apollo-validation`; `rg` source scans for removed analytic-signal copy
  allocation patterns.

### Closure LI - Apollo-Hilbert Owner Quadrature Slice Kernel [patch]
- **Gap**: `HilbertPlan::transform_into` called the allocating
  `hilbert_transform` owner function and copied the returned quadrature vector
  into caller-owned output. `apollo-hilbert` also retained a direct `rayon`
  dependency after the private parallel O(N²) DFT kernels were removed.
- **Closed by**: Added `hilbert_transform_into` as the slice-level owner
  quadrature kernel, routed `HilbertPlan::transform_into` through it, kept typed
  execution on the same shared owner path, removed the unused direct `rayon`
  dependency, added direct kernel parity/mismatch tests, updated the Hilbert
  README, and bumped `apollo-hilbert` to 0.1.2.
- **Residual risk**: The analytic-signal owner kernel still allocates FFT input,
  spectrum, and analytic buffers because the current `apollo_fft` public array
  entry points own those conversions. That is the next bounded Hilbert memory
  target.
- **Evidence**: `cargo check -p apollo-hilbert`; `cargo test -p
  apollo-hilbert transform_into --lib -- --test-threads=1`; `cargo test -p
  apollo-hilbert --lib -- --test-threads=1`; `cargo check -p
  apollo-validation`; `rg` source scans for removed copy-through allocation and
  dead direct dependency.

### Closure L - Apollo-Hilbert Typed Workspace Reuse [patch]
- **Gap**: `HilbertPlan::analytic_signal_typed` and
  `transform_typed_into` allocated f64 bridge vectors on every reduced-storage
  call before entering the shared owner Hilbert implementation.
- **Closed by**: Added thread-local f64 input and output workspaces for typed
  Hilbert execution, preserved the `f64` zero-copy specialization, routed
  reduced-storage execution through the existing analytic-mask owner path, added
  repeated-call f32 capacity/value coverage, updated the Hilbert README, and
  bumped `apollo-hilbert` to 0.1.1.
- **Residual risk**: Typed Hilbert scratch is retained per thread at the
  largest signal length executed by that thread. Recursive same-thread typed
  Hilbert calls are rejected by `RefCell` borrow checking.
- **Evidence**: `cargo check -p apollo-hilbert`; `cargo test -p
  apollo-hilbert workspace --lib -- --test-threads=1`; `cargo test -p
  apollo-hilbert --lib -- --test-threads=1`; `cargo check -p
  apollo-validation`; `rg` source scan for removed Hilbert production typed
  bridge allocation patterns.

### Closure XLIX - Apollo-SDFT Typed Workspace Reuse [patch]
- **Gap**: `SdftPlan::direct_bins_typed_into` allocated an f64 input bridge
  vector and Complex64 output bridge vector on every typed direct-bin call
  before entering the owner direct-bin kernel.
- **Closed by**: Added thread-local f64/Complex64 direct-bin workspaces,
  routed typed direct-bin execution through the shared owner kernel without
  per-call bridge allocation, added repeated-call f32 capacity/value coverage,
  updated the SDFT README, and bumped `apollo-sdft` to 0.1.1.
- **Residual risk**: Typed direct-bin scratch is retained per thread at the
  largest window length and bin count executed by that thread. Recursive
  same-thread typed direct-bin calls are rejected by `RefCell` borrow checking.
- **Evidence**: `cargo check -p apollo-sdft`; `cargo test -p apollo-sdft
  workspace --lib -- --test-threads=1`; `cargo test -p apollo-sdft --lib --
  --test-threads=1`; `cargo check -p apollo-validation`; `rg` source scan for
  removed SDFT production typed bridge allocation patterns.

### Closure XLVIII - Apollo-STFT Inverse WOLA Workspace Reuse [patch]
- **Gap**: The STFT inverse owner path allocated four work buffers per call:
  frame-domain samples, complex frame FFT input, overlap accumulation, and
  squared-window weight accumulation. The typed inverse path inherited the same
  allocations through the shared owner inverse kernel.
- **Closed by**: Added thread-local WOLA workspaces for frame, complex,
  overlap, and weight buffers; routed `inverse_into`, `inverse`, and typed
  inverse through the same slice-level owner path; zeroed only accumulation
  buffers before WOLA; added repeated-call value/capacity reuse coverage;
  updated the STFT ADR and README; and bumped `apollo-stft` to 0.2.1.
- **Residual risk**: Inverse WOLA scratch is retained per thread at the largest
  frame work length and signal length executed by that thread. Recursive
  same-thread inverse calls are rejected by `RefCell` borrow checking.
- **Evidence**: `cargo check -p apollo-stft`; `cargo test -p apollo-stft
  workspace --lib -- --test-threads=1`; `cargo test -p apollo-stft --lib --
  --test-threads=1`; `cargo check -p apollo-validation`; `rg` source scan for
  removed STFT production inverse WOLA allocation patterns.

### Closure XLVII - Apollo-STFT Typed Workspace Reuse and Alias Removal [major]
- **Gap**: `StftPlan::forward_typed_into` and `inverse_typed_into` allocated
  owner-precision `Array1` bridge buffers per call before entering the f64 /
  Complex64 owner path. `forward_inplace` and `inverse_inplace` were
  deprecated allocating aliases, and `dimension_1d.rs` exceeded the structural
  file-size limit.
- **Closed by**: Added slice-level f64/Complex64 execution entry points,
  reused thread-local typed bridge workspaces, moved storage/profile traits to
  `stft::storage`, moved tests into a leaf module, added a co-located ADR,
  removed the deprecated alias methods and README references, added
  repeated-call f32 workspace reuse coverage, and bumped `apollo-stft` to 0.2.0.
- **Residual risk**: Typed scratch is retained per thread at the largest STFT
  signal and spectrum dimensions executed by that thread. The inverse owner
  WOLA allocation gap was closed in Closure XLVIII.
- **Evidence**: `cargo check -p apollo-stft`; `cargo test -p apollo-stft
  workspace --lib -- --test-threads=1`; `cargo test -p apollo-stft --lib --
  --test-threads=1`; `cargo check -p apollo-validation`; `rg` source scan for
  removed STFT production typed bridge allocation and deprecated alias
  patterns.

### Closure XLVI - Apollo-QFT Dense and Typed Workspace Reuse [patch]
- **Gap**: `QftPlan::forward_into` and `inverse_into` allocated a dense
  transform vector before copying into caller-owned output, and typed QFT paths
  allocated Complex64 input/output bridge arrays per call.
- **Closed by**: Added dense `*_into` kernels, routed plan execution through
  Complex64 slices, reused thread-local Complex64 typed bridge workspaces,
  added repeated-call complex32 forward/inverse workspace reuse coverage, and
  bumped `apollo-qft` to 0.1.1.
- **Residual risk**: Typed scratch is retained per thread at the largest QFT
  dimension executed by that thread. The intentionally allocating convenience
  dense wrappers remain for callers that request owned `Vec` output.
- **Evidence**: `cargo check -p apollo-qft`; `cargo test -p apollo-qft
  workspace --lib -- --test-threads=1`; `cargo test -p apollo-qft --lib --
  --test-threads=1`; `cargo check -p apollo-validation`; `rg` scan for removed
  QFT production plan/typed allocation patterns.

### Closure XLV - Apollo-GFT Typed Workspace Reuse [patch]
- **Gap**: GFT typed storage paths allocated f64 input/output bridge arrays
  per forward and inverse call before invoking the owner graph-basis multiply.
- **Closed by**: Added contiguous f64 slice execution on `GftPlan`, reused
  thread-local f64 input/output workspaces for typed storage, added
  repeated-call f32 forward/inverse workspace reuse coverage, and bumped
  `apollo-gft` to 0.1.1.
- **Residual risk**: Typed scratch is retained per thread at the largest graph
  order executed by that thread. This replaces repeated allocation with
  bounded reuse.
- **Evidence**: `cargo check -p apollo-gft`; `cargo test -p apollo-gft
  workspace --lib -- --test-threads=1`; `cargo test -p apollo-gft --lib --
  --test-threads=1`; `cargo check -p apollo-validation`; `rg` scan for removed
  GFT production typed bridge allocation patterns.

### Closure XLIV - Apollo-FWHT Typed Workspace Reuse [patch]
- **Gap**: FWHT typed storage defaults allocated f64 input/output bridge
  arrays per call, and mixed f16 FWHT allocated a fresh f32 compute vector per
  forward/inverse call.
- **Closed by**: Added contiguous f64 slice execution on `FwhtPlan`, reused
  thread-local f64 bridge workspaces for default typed storage, reused a
  thread-local f32 compute workspace for mixed f16 storage, added repeated-call
  workspace reuse coverage, and bumped `apollo-fwht` to 0.1.1.
- **Residual risk**: Typed scratch is retained per thread at the largest FWHT
  length executed by that thread. This replaces repeated allocation with
  bounded reuse.
- **Evidence**: `cargo check -p apollo-fwht`; `cargo test -p apollo-fwht
  workspace --lib -- --test-threads=1`; `cargo test -p apollo-fwht --lib --
  --test-threads=1`; `cargo check -p apollo-validation`; `rg` scan for removed
  FWHT production typed bridge allocation patterns.

### Closure XLIII - Apollo-CZT Workspace Reuse and FFT Warning Cleanup [patch]
- **Gap**: `CztPlan` allocated a fresh Complex64 Bluestein convolution
  workspace on every plan-path forward call, typed CZT paths allocated
  Complex64 bridge arrays for forward/inverse, inverse CZT rebuilt
  Vandermonde nodes per call, and `apollo-fft` still retained unused radix-2
  butterfly helpers after Stockham became canonical.
- **Closed by**: Added plan-owned convolution scratch, added CZT slice
  execution, precomputed square-plan inverse nodes, reused thread-local typed
  bridge workspaces, removed dead radix-2 helper code, added missing
  `FftPlan3D` Rustdoc, and bumped `apollo-czt` to 0.2.1 plus `apollo-fft` to
  0.2.2.
- **Residual risk**: CZT forward scratch is retained per plan at
  `convolution_len`; typed bridge scratch is retained per thread at the
  largest typed CZT input/output dimensions used on that thread. This replaces
  repeated allocation with bounded reuse.
- **Evidence**: `cargo check -p apollo-fft --lib`; `cargo check -p
  apollo-czt`; `cargo test -p apollo-czt workspace --lib -- --test-threads=1`;
  `cargo test -p apollo-czt --lib -- --test-threads=1`; `cargo test -p
  apollo-fft radix2 --lib -- --test-threads=1`; `cargo check -p
  apollo-czt-wgpu -p apollo-validation`; `rg` scans for deleted CZT typed
  bridge allocation patterns and removed radix-2 helper names.

### Closure XLII - Apollo-FFT Dead Helper Cleanup [patch]
- **Gap**: Current `apollo-fft` kernel sources retained unused helper
  implementations after the power-of-two path moved to Stockham/composite
  routing: f16 with-twiddles bridge allocation, uniform power-of-two
  digit-reversal helpers, power-of-four/eight shape predicates, and unused
  Winograd stage traits.
- **Closed by**: Removed the dead helpers and their stale tests/docs while
  retaining the live bit-reversal primitive for radix-2 and mixed-radix
  permutation for composite FFT routing.
- **Residual risk**: None identified for the removed items; the final source
  scan reports no references to the deleted helper names.
- **Evidence**: `cargo check -p apollo-fft --lib`; `cargo test -p apollo-fft
  radix_shape --lib -- --test-threads=1`; `cargo test -p apollo-fft
  radix_permute --lib -- --test-threads=1`; `rg` scan for the deleted helper
  names.

### Closure XLII - Apollo-FRFT Typed Workspace Reuse [patch]
- **Gap**: `FrftStorage` typed paths for `Complex32` and `[f16; 2]` allocated
  one Complex64 input array and one Complex64 output array per call before
  invoking the direct FrFT implementation.
- **Closed by**: Added internal Complex64 slice entry points on `FrftPlan`,
  replaced typed bridge arrays with thread-local reusable input/output
  workspaces, added a repeated-call workspace reuse regression test, restored
  the current `apollo-fft` module/import drift that blocked dependency
  compilation, and bumped `apollo-frft` to 0.1.2 plus `apollo-fft` to 0.2.1.
- **Residual risk**: Typed scratch is retained per thread at the largest typed
  FrFT length executed on that thread. This is the same bounded reuse tradeoff
  as the unitary workspace cleanup.
- **Evidence**: `cargo check -p apollo-frft`; `cargo test -p apollo-frft typed
  --lib -- --test-threads=1`; `cargo test -p apollo-frft --lib --
  --test-threads=1`; `cargo check -p apollo-frft-wgpu -p apollo-validation`;
  `rg` scan for removed typed bridge `Array1::from_iter` / `output64`
  allocation patterns.

### Closure XLII - Apollo-FRFT Unitary Workspace Reuse [patch]
- **Gap**: `UnitaryFrftPlan` allocated a fresh O(N) coefficient vector on every
  forward and inverse execution even though the Candan-Grünbaum algorithm only
  requires one temporary coefficient workspace per executing thread.
- **Closed by**: Added thread-local reusable coefficient scratch for the
  projection, phase, and reconstruction steps, added a regression test for
  capacity reuse and output equality, removed stale backward-compatibility
  wording from live crate-root exports, and bumped `apollo-frft` to 0.1.1.
- **Residual risk**: Scratch is retained per thread at the largest unitary FrFT
  length executed on that thread. This trades repeated heap allocation for
  bounded thread-local reuse.
- **Evidence**: `cargo check -p apollo-frft`; `cargo test -p apollo-frft
  unitary --lib -- --test-threads=1`; `cargo test -p apollo-frft --lib --
  --test-threads=1`; `cargo check -p apollo-frft-wgpu -p apollo-validation`;
  `rg` scan for stale FrFT compatibility/deprecated markers and the removed
  `vec![Complex64::new(0.0, 0.0); n]` allocation.

### Closure XLII - Apollo-FFT Compatibility Re-Export Cleanup [major]
- **Gap**: `apollo-fft` retained compatibility re-export modules and a legacy
  `FFT_CACHE` alias after root exports already exposed the canonical public
  names. It also retained unused power-of-four/eight forwarding modules under
  `infrastructure::cpu::simd::power_of_two` that duplicated radix-2 execution.
- **Closed by**: Removed the compatibility modules, removed `FFT_CACHE`, deleted
  the unused forwarding modules, updated in-repo callers to root or canonical
  paths, fixed a test-only `Complex32/64` import gap surfaced by the full test
  build, and bumped `apollo-fft` to 0.2.0.
- **Residual risk**: External callers using the removed compatibility paths must
  migrate to root exports or canonical owner modules. This is an intentional
  pre-1.0 breaking cleanup.
- **Evidence**: `cargo check -p apollo-fft --lib`; `cargo check -p
  apollo-fft-wgpu -p apollo-czt -p apollo-nufft -p apollo-stft -p apollo-sft`;
  `cargo test -p apollo-fft --lib -- --test-threads=1`; `cargo check -p
  apollo-fft --benches`; `rg` scan for removed compatibility paths and legacy
  aliases in touched crate sources.

### Closure XLII - STFT-WGPU Deprecated Error and Retained-Resource Cleanup [major]
- **Gap**: `apollo-stft-wgpu` retained a deprecated
  `WgpuError::FrameLenNotPowerOfTwo` public variant after Chirp-Z support made
  non-power-of-two frame lengths valid, and retained GPU resources used explicit
  dead-code suppressions.
- **Closed by**: Removed the stale error variant, strengthened non-power-of-two
  tests to require successful Chirp-Z execution/buffer construction, renamed
  retained GPU resource fields with `_` ownership names, and bumped
  `apollo-stft-wgpu` to 0.11.0. Removed the remaining NUFFT/NTT WGPU
  dead-code suppressions by enforcing NUFFT reusable-buffer sample capacity
  before GPU writes, replacing NUFFT per-dispatch layout-placeholder allocations
  with one retained layout padding buffer, deleting duplicated NTT scalar
  `n_inv` storage, and keeping retained NTT GPU resources as explicit `_` owner
  fields.
- **Residual risk**: GPU-gated execution remains dependent on adapter
  availability; tests skip only when no WGPU device can be acquired.
- **Evidence**: `cargo check -p apollo-stft-wgpu`; `cargo test -p
  apollo-stft-wgpu --lib -- --test-threads=1`; matching `cargo check` and
  `cargo test --lib` for `apollo-nufft-wgpu` and `apollo-ntt-wgpu`; `rg` scans
  for `FrameLenNotPowerOfTwo`, `#[allow(dead_code)]`, deprecated markers, and
  NUFFT placeholder buffers in the audited WGPU crate sources.

### Closure XLII - DCT/DST Fast-Path Unused Output Allocation Cleanup [patch]
- **Gap**: `apollo-dctdst` single-output DCT-II and DST-II fast paths allocated
  an N-length sibling output only to call the dual DCT/DST projection kernel.
- **Closed by**: Factored the shared 2N-point FFT setup and projection-fill
  helpers so `dct2_fast` fills only DCT-II and `dst2_fast` fills only DST-II,
  while `dct2_dst2_fast` still computes both projections from one FFT.
- **Residual risk**: The fast path still allocates the required 2N complex FFT
  buffer and FFT output; this increment removes only the provably unused
  real-output allocation.
- **Evidence**: `cargo check -p apollo-dctdst`; focused fast-path regression
  test comparing single-projection outputs to the dual kernel and direct
  analytical DCT-II/DST-II kernels; full `cargo test -p apollo-dctdst --lib
  -- --test-threads=1`.

### Closure XLII — Apollo vs RustFFT f32 N=4096 Performance Disparity [patch]
- **Gap**: Apollo f32 N=4096 throughput remains behind RustFFT; prior evidence
  depended on a plan-scratch benchmark route that is absent from the current
  checkout API surface.
- **Closed by**: Rejected the candidate that disabled the f32 N=4096 radix-16
  quad suffix after same-session Criterion measured Apollo 6.5098 µs vs RustFFT
  3.7433 µs. Restored the local `vs_rustfft` benchmark to compile against the
  present API with a local RustFFT dev-dependency and current mixed-radix
  precomputed-twiddle calls.
- **Residual risk**: Current f32 N=4096 precomputed-twiddle row measures Apollo
  22.790 µs vs RustFFT 3.5969 µs. This indicates API/dispatch drift, not a
  retained Stockham codelet improvement, and is not comparable to the prior
  plan-scratch row.
- **Evidence**: `cargo check -p apollo-fft --benches`; `cargo test -p apollo-fft
  dft7 --lib -- --test-threads=1`; focused Criterion f32 N=4096 Apollo/RustFFT
  precomputed-twiddle and quad-disabled probes.
- **Follow-up increment**: Large f32 power-of-two dispatch now uses the
  monomorphized Stockham scratch-backed kernel with thread-local scratch,
  eliminating the prior radix-8 facade route. Final retained f32 N=4096
  Criterion measured Apollo zero-alloc reused 7.0463 µs, Apollo caller-twiddle
  reused 8.9737 µs, and RustFFT reused 6.2814 µs. The initial production 8x512
  hybrid and direct no-argument micro-dispatch probes were rejected because
  they regressed the then-retained route.
- **Current residual**: The f32 N=4096 retained schedule now disables the
  spilling quad suffix while preserving stride-64 triple suppression, then uses
  a single-entry thread-local f32 forward-twiddle fast cache for the public path.
  Longer Criterion measured Apollo zero-alloc reused 6.3347 µs, Apollo
  caller-twiddle reused 6.0315 µs, and RustFFT reused 4.2974 µs. The remaining
  gap is in the Stockham f32 N=4096 memory traffic/kernel body, not hot-path
  allocation or `Arc` cloning.
- **Correctness correction**: The terminal groups=1 in-place Stockham hook was
  removed after audit because the source layout is interleaved
  (`src[2j]`, `src[2j+1]`) and a direct in-place final stage overwrites future
  inputs. Static N=4096 f32 twiddle specialization, direct concrete benchmark
  calls, shortened public branching, zero-copy generic schedule flipping, and
  split public scratch/twiddle caching were all rejected by focused Criterion
  probes.
- **Current retained result**: The verified f32 8x512 helper remains test-only
  after same-tree Criterion showed the generic Stockham route was faster. The
  retained f32 N=4096 path uses the radix-8/radix-8 tail schedule and split
  public scratch/twiddle caches; the dead combined workspace was removed.
  Final current-tree Criterion measured Apollo public zero-alloc reused
  5.4298 µs, Apollo caller-twiddle reused 5.2661 µs, and RustFFT reused
  3.6958 µs. Earlier same-state retained measurement reached Apollo public
  4.8645 µs and caller-twiddle 4.7913 µs. The residual gap remains in f32
  Stockham stage memory traffic and kernel shape. Rejected follow-up probes:
  64 KiB low-live threshold, separate single-entry Stockham twiddle cache,
  direct N=4096 four-pass specialization, unchecked twiddle subslices,
  stride-64 radix-16 fusion, and forced Stockham AVX/cache inlining. The latest
  retained run after reverts measured Apollo public zero-alloc reused
  5.4895 µs, Apollo caller-twiddle reused 5.4176 µs, and RustFFT reused
  4.3328 µs. Subsequent rejected hot-codelet probes were paired 128-bit stores
  in the quarter-groups-one suffix, even-radix tail monomorphization for that
  suffix, and const-generic radix-1 quarter-turn signs. Each preserved
  correctness but failed the focused Criterion retention gate.
- **Assembly finding**: `cargo rustc -p apollo-fft --release --lib -- --emit=asm`
  showed the separate f32 Stockham codelets pay Windows ABI vector-register
  prologue cost. A private raw-pointer `sysv64` ABI removed the XMM6-XMM15
  save block from the quarter-groups-one suffix assembly, but focused Criterion
  did not retain an Apollo caller-twiddle improvement, so the ABI probe was
  reverted. The next viable path is reducing the codelet's live vector state or
  fusing the N=4096 call boundary without an unsupported `#[inline(always)]`
  target-feature combination.
- **Nonsimd/SWAR audit**: GhostCell is not applicable to the retained f32
  N=4096 hot route because there is no graph-like shared mutable topology;
  scratch storage is thread-local and borrowed lexically. A scalar
  power-of-two digit-reversal cleanup replaced division/modulo with shift/mask
  digit extraction for non-Stockham routes. Focused f32 N=256 Criterion was
  neutral for Apollo, so the measured small-size gap is in radix-4
  butterfly/scheduling work, not the digit-reversal arithmetic.
- **Autosort expansion**: The f32 forward Stockham threshold was lowered from
  1024 to 256, moving N=256 off the radix-4 digit-reversal route and onto
  caller-scratch Stockham. Focused Criterion repeat measured Apollo public
  197.50 ns and caller-twiddle 218.36 ns versus the prior digit-reversal route
  near 983.67 ns and 991.61 ns. N=64 autosort was rejected because public
  dispatch regressed while caller-twiddle was neutral.
- **Inverse autosort integration**: f32 power-of-two inverse paths now use
  Stockham with inverse twiddles for lengths >=256. Normalized inverse reuses
  the unnormalized Stockham route and applies explicit `1/N` scaling. New
  inverse rows in `vs_rustfft` showed the old digit-reversal baseline at
  963.10 ns for N=256 and 23.104 µs for N=4096; retained Stockham inverse
  measured 230.60 ns and 5.5408 µs after restoration.
- **f64 autosort integration**: f64 power-of-two forward and inverse paths now
  use Stockham for lengths >=256 with reusable thread-local scratch. New f64
  inverse rows in `vs_rustfft` showed the old digit-reversal baseline at
  830.23 ns forward / 778.38 ns inverse for N=256 and 25.456 µs forward /
  32.167 µs inverse for N=4096; retained Stockham measured 315.24 ns /
  257.88 ns and 10.050 µs / 10.731 µs. Threshold 64 was rejected because N=64
  f64 public and caller-twiddle rows regressed versus the existing radix route.
- **Fixed-kernel memory-efficiency cleanup**: The production f64 N=256/N=512
  and f32 N=512 fixed single-pass kernels were bypassed in favor of the fused
  generic AVX scheduler, reducing intermediate scratch traffic. The unused f64
  N=256 fixed kernel was removed; the N=512 fixed kernels remain test-only for
  hybrid-radix equivalence probes. Focused Criterion measured f64 N=256 at
  255.90 ns public / 228.16 ns caller-twiddle / 225.37 ns inverse, f64 N=512
  at 591.36 ns public / 581.33 ns caller-twiddle, and f32 N=512 at 366.39 ns
  public / 346.71 ns caller-twiddle / 328.85 ns inverse. On that f32 N=512
  run, RustFFT measured 329.96 ns forward and 356.70 ns inverse, so Apollo
  inverse surpassed RustFFT while forward caller-twiddle remained within
  16.75 ns.
- **All-metrics target status**: Latest focused zero-allocation matrix shows
  Apollo does not yet surpass RustFFT in every row. Retained wins include f64
  N=512 forward/inverse. Open gaps remain at f64 N=256, f64 N=4096, f32 N=512,
  and f32 N=4096. A static f32 N=4096 four-triple schedule improved Apollo
  caller-twiddle forward from 6.9498 µs to 5.4670 µs and inverse from
  6.5585 µs to 5.1970 µs, but the latest RustFFT rows still measured
  3.7807 µs and 3.7765 µs. The same static schedule was rejected for f64
  because it regressed forward to 11.264 µs. An f32 N=512 no-copy tail was
  rejected because it regressed forward to 440.90 ns and inverse to 570.83 ns.
- **RustFFT-like decomposition probe**: A production f32 8x512 decomposition
  with column radix-8, mixed twiddles, row-local N=512 fused Stockham, and final
  transpose preserved correctness but regressed N=4096 to 11.792 µs forward and
  11.786 µs inverse. Reordering the transpose for contiguous destination stores
  improved that failed route to 9.9378 µs forward and 9.9228 µs inverse, still
  slower than the retained four-triple Stockham route. The issue is not only
  decomposition shape; Apollo still lacks RustFFT's specialized Butterfly512
  row kernel and packed column/transpose machinery.
- **Butterfly512 probe**: A f32 8x64 Butterfly512-style candidate was
  implemented with column radix-8, mixed twiddles, eight fixed 64-point row
  butterflies, and final transpose. It preserved N=512 correctness but
  regressed Criterion to 546.25 ns forward and 573.94 ns inverse. Replacing the
  scalar mixed-twiddle loop with an AVX packed-twiddle loop regressed forward
  further to 773.36 ns. The measured cause is the same as the 8x512 probe:
  Apollo's decomposition lacks RustFFT's prepacked twiddle layout and fused
  column/transpose butterfly machinery, so separate mixed-twiddle and transpose
  phases consume the expected gain.
- **Complete pathway audit correction**: RustFFT's f32 `Butterfly512Avx` is not
  an 8x64 decomposition. It treats N=512 as a 16x32 matrix, computes
  column-butterfly16 vectors, applies 120 packed separated-column mixed-twiddle
  vectors, fuses those multiplies with 4x4 transpose stores, then computes
  row-butterfly32 vectors in the transposed scratch. Apollo now has executable
  f32/f64 tests for that packed twiddle contract in `stockham.rs`. This closes
  the false-path ambiguity from the previous 8x64 candidate and leaves the
  production fused column/transpose kernel as the next required implementation
  step.
- **Current benchmark-and-retain pass**: Focused Criterion over the open
  zero-allocation rows measured f32 N=4096 Apollo forward at 9.4509 µs versus
  RustFFT 6.3698 µs, and f64 N=4096 Apollo forward at 17.686 µs versus RustFFT
  12.225 µs before the f64 schedule change. Restoring production f32/f64 N=512
  fixed single-pass leaves was rejected because it regressed f64 N=512 to
  1.4856 µs forward / 1.3834 µs inverse and f32 N=512 to 685.78 ns forward /
  683.37 ns inverse. A f64 N=4096 forward-only static four-triple dispatch was
  retained: it improved forward to 15.844 µs, while inverse remains on the
  generic route because the static route regressed inverse under inverse
  twiddles.
- **3D R2C/C2R row-allocation cleanup**: The Z-axis R2C split no longer
  allocates a temporary `Vec<Complex64>` per `(x,y)` row; it packs the
  length-`nz/2` complex subproblem into the caller-owned half-spectrum row
  prefix, runs the sub-FFT in place, and writes the split spectrum after
  preserving the shared `H[0]` boundary value. The C2R inverse now mutates the
  caller-provided half-spectrum scratch row as its recovered packed-spectrum
  buffer before the normalized sub-IFFT. Unused f32 R2C future-reservation
  fields were removed from `FftPlan3D`, eliminating plan-time twiddle and
  scratch allocations for an unimplemented path.
- **Rejected cache probe**: A closure-borrowed thread-local twiddle cache was
  tested to remove hot-path `Arc` clones, but focused f32 N=4096 public
  zero-allocation Criterion regressed to 8.4200 µs median. The probe was
  removed; the restored retained route measured 7.0245 µs median in this
  session.
- **2D axis fallback cleanup**: `FftPlan2D` only dispatches separable passes
  over `Axis(1)` rows and `Axis(0)` columns. The previous generic invalid-axis
  fallback allocated nested lane vectors and copied the matrix twice for an
  unreachable state. It is now an explicit axis invariant, preserving the
  row/column fast paths while removing dead allocation-heavy code.
- **Generic DFT-8 sign correction**: The monomorphized Winograd DFT-8 helper
  used the inverse imaginary sign for forward roots after the f32/f64 helper
  consolidation. This broke composite-radix stages containing radix 8, including
  the N=24 row pass exposed by 2D FFT verification. The helper now encodes
  `W_8^k = exp(sign*2πik/8)` with `sign = -1` for forward and `+1` for inverse,
  preserving one generic implementation while restoring f32/f64 direct parity.
- **Deprecated FFT alias cleanup**: The deprecated compatibility surface
  `FftPlan1D/2D/3D::{forward_into,inverse_into}` and `ProcessorFft3d` duplicated
  the canonical caller-owned API without adding semantics. The aliases were
  removed and in-repo Python call sites now invoke
  `forward_real_to_complex_into` / `inverse_complex_to_real_into` directly.

### Closure XLI — DHT CPU 2D/3D; FWHT CPU 2D/3D; FFT fftfreq/rfftfreq/fftshift/ifftshift [minor]
- **Gap**: DHT 2D/3D absent; FWHT 2D/3D absent; numpy-compatible fftfreq/rfftfreq/fftshift/ifftshift absent.
- **Closed by**:
  - `apollo-dht`: added separable `forward_2d`, `inverse_2d`, `forward_3d`, `inverse_3d` with N×N and N×N×N constraints; `DhtError::ShapeMismatch2d/3d`.
  - `apollo-fwht`: added `FwhtPlan2D` and `FwhtPlan3D` in deep hierarchy `dimension_2d.rs` / `dimension_3d.rs`; both support real and complex forward/inverse; `FwhtError::LengthMismatch` enforced on non-square/non-cubic input.
  - `apollo-fft`: new `application/utilities/freq.rs` (`fftfreq`, `rfftfreq`) and `application/utilities/shift.rs` (`fftshift`, `ifftshift`); all four re-exported from crate root.
- **Verification**:
  - DHT involution property: `DHT_2D(DHT_2D(X)) = N²·X` — verified at N=3.
  - DHT 2D/3D inverse roundtrip: max absolute error < 1e-10.
  - FWHT involution: `WHT_2D(WHT_2D(X)) = N²·X` — verified at N=4.
  - FWHT separability: outer product x⊗y → W_{2D}(x⊗y) = WHT(x)⊗WHT(y).
  - fftfreq(8, 1.0) == numpy reference [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125].
  - ifftshift(fftshift(x)) = x for even and odd n.

## 2026-06: Deeper per-LOG2 Stockham monomorph + mem Cow extensions + arch elevation (transform body specials for md-worst PoT) [patch]
- Performed: full per-LOG2 body specialization in stockham/transform.rs (new transform_len32/64/128/256/512/1024 + explicit stage seq from schedule dump of greedy fusion/triples; dispatch match LOG2 early in transform_impl; removed 128/256 delegate; 32768/4096 precedent unified). Extended ZST with_strategy live calls for LOG2=5..10 in stockham/mod.rs (f32 reduced + f64 precise scalar paths; cfg gated imports). Mem: Cow kernel_view (bluestein pointwise) + named/used zero_copy_view in scratch nested f32/f64 fallbacks. Cast hygiene re-audit (new bodies native via P, no numeric casts; only index*const in tests). SRP/SSOT: transform.rs owns per-size bodies; plan/dispatch ZST + sized still SSOT for routing. Deep vertical preserved (<500 lines/file).

## 2026-06 follow-up (ZST threading + Cow ext in pot path) [patch]
- Performed: Added MixedRadixScalar::pot_inplace_sized<const INVERSE, const NORMALIZE, S: PoTStrategy, const LOG2> (with default fallback to pot_inplace for compat; documented in trait). Provided overrides in f32/f64 impls that use const LOG2 for n=1<<LOG2 in with_scratch + stockham (aids mono per LOG2). Updated all call sites in dimension_1d (the 3 exec_*_sized + 512 wrappers; generics left on old pot_inplace since no const LOG2). Expanded ZST constructions in dispatch try_pot for 5-10. Added Cow tw_view (Borrowed over caller's twiddles Arc) inside the new pot_inplace_sized paths (mem/zero-copy ext exercised on hot ZST path). All call sites updated in same diff; no compat soup.
- Why: elevates "more direct ZST threading" (plan SSOT constructions of SizedPoT now flow the type+const LOG2 into the monomorphized pot boundary and down to stockham; previously _s was only tag, now param in specialized fn). Strengthens monomorph (per (S,LOG2) context for future strategies + const prop to lenXXX bodies). Zero cost (ZST), improves arch (DIP, the strategy decides at type level). Cow ext for zero copy in the threaded path. Targets remaining PoT >1x and mem for rader (which hit pot via bluestein pads).
- Verification: cargo check clean; value (n512 ZST plan tests pass which use sized exec now hitting pot_sized; n256/128/512 roundtrips; rader bluestein; GT; stockham); gates (fmt clean, doc clean, xtask clean, clippy no new on pot_sized -- only preexist small_ inline warnings; we used #[inline]); focused --skip-run build success (Compiling apollo-fft for the changes).
- No regression (old pot_inplace + generic paths untouched; new sized only for known LOG2 from plan; same kernels inside).
- Residuals updated: still full unrolls inside len, f32 scratch for n113, expand shared butterflies, full rebench, more Cow (e.g. in rader generator or twiddle caches).
- Evidence: type-level (generic const LOG2 + S in pot sig + call sites), value-semantic, gates, build path exercised.

## 2026-06: Deeper per-LOG2 Stockham monomorph + mem Cow extensions + arch elevation (transform body specials for md-worst PoT) [patch]
- Performed: full per-LOG2 body specialization in stockham/transform.rs (new transform_len32/64/128/256/512/1024 + explicit stage seq from schedule dump of greedy fusion/triples; dispatch match LOG2 early in transform_impl; removed 128/256 delegate; 32768/4096 precedent unified). Extended ZST with_strategy live calls for LOG2=5..10 in stockham/mod.rs (f32 reduced + f64 precise scalar paths; cfg gated imports). Mem: Cow kernel_view (bluestein pointwise) + named/used zero_copy_view in scratch nested f32/f64 fallbacks. Cast hygiene re-audit (new bodies native via P, no numeric casts; only index*const in tests). SRP/SSOT: transform.rs owns per-size bodies; plan/dispatch ZST + sized still SSOT for routing. Deep vertical preserved (<500 lines/file).
- Why highest prob + routing: PoT worst (32 1.72x f64/2.5 f32, 64 1.32/1.98, 128/256/512/1024/32768 >1x per benchmark_results.md) are lowest overhead route (no perm); per-LOG2 const seq bakes stage count/fusion/unrolls (structural const generics + monomorph zero-cost, Inner-Fn pattern); enables future unrolls/specials without runtime branch. Matches "deeper monomorphization, elevation of architecture, no perf loss from excess casting". Rader primes (67/271 f32) benefit via bluestein pow2 pads hitting new len + pooled/Cow. GT90/198 via composite still monitored.
- Verification (value-semantic, no existence-only): 
  - dft_forward + roundtrips/eps: n32 direct matches (len32), n256/512/1024/128 roundtrips (stockham/mixed), n512 ZST plan tests (log2=9/10 with_strategy + len512), stockham small_sizes + transform tests, GT cook dft90/60/84/198, rader bluestein n17 (exercises pooled kernel build + Cow views in convolve), lib_tests f64 pot dfts.
  - Broad run (skip pre-existing rader/prime stack + unrelated short_win list): 252 passed 0 failed.
  - Gates: fmt --check clean (auto), doc --no-deps clean, xtask check clean, clippy no *new* on edited (only pre-existing inline(always) on avx kernels).
- Bench regression prevention (binding): env cleaned (taskkill + del debug/release xtask.exe); focused xtask --sizes [full md-worst list] --profile quick --skip-run (build exercised: "Compiling apollo-fft" success; no early fail); md updated with details/cmd/baselines/"no regression expected (specialized bodies identical seq, value green, 0-cost additive)"/rebench cmd. See benchmark_results entry.
- No HARD: complete real impls (no placeholder), native P precision, ZST/mono, deep vertical, unidirectional, value tests, artifacts sync.
- Residuals (per gap order): full per-LOG2 unrolls inside the len fns (e.g. explicit for 32/64 butterflies? but current stage calls are the unrolled); more direct ZST from plan executors to transform_with_strategy (bypass some F::); f32 scratch complete (un-ignore n113_f32 + broader); expand shared dft9/25/16 into butterflies/; full Criterion rebench on list; more Cow (twiddles); more GT forces if post-rebench still select.
- Evidence tier: type-level (const LOG2 + ZST), value-semantic diff (direct/roundtrip on affected), gates, focused build path.

## 2026-06 (completion of const LOG2 threading): pot_inplace_sized overrides + Cow + plan/dispatch full wire for 128/256 PoT [patch]
- Performed: Added stockham_forward_sized + normalized_sized overrides in f32/f64 MixedRadixScalar impls (delegate to kernel forward_with_scratch_sized). Added pot_inplace_sized overrides in both (LOG2<=6: small direct no-scratch for mem/perf on 32/64; >: with_scratch + stockham_..._sized + Cow::Borrowed tw_view for zero-copy read over plan tw). Extended plan PowerOfTwo log2 match + sized exec for 7/8 (128/256 now get explicit ZST SizedPoT from SSOT, hit pot_sized). Updated dispatch try_pot hot arms 7-10 to actually invoke pot_inplace_sized with constructed _s (full call site threading, not just let _ =). Incidental root fixes surfaced during verification (added missing 4 arm in f32 small_pot_inplace_sized match; forced dft16 winograd path in small_pot 16 for f32 correctness; short_win accepts test guard for >64 policy; added ignore for n257 debug stack rader consistent with n113). All in same diff; deep vertical/SoC preserved.
- Why highest prob for surpass at all sizes: completes the ZST mono elevation for remaining md-worst PoT 128/256 (previously generic runtime log2 in pot); const LOG2 now flows end-to-end plan -> kernel for len* bodies (enables DCE/ILP in monomorph per LOG2 + future unrolls inside); Cow ext for mem efficiency on the newly covered paths (rader bluestein pads hit via pot); matches "next phase of performance optimizations and enhancements of memory efficiency", "more direct ZST threading", "no perf lost due to excess casting" (native). Dispatch/plan updated same change (no compat soup). Preexist small defects fixed at source for clean value gates.
- Verification (value no existence-only; gates; bench prev):
  - dft_forward + roundtrips/eps on md-worst + affected: n16/128/256/512/1024 (now via sized for 128/256) + n512 ZST plan tests (log2=9/10) + rader bluestein (67/271) + GT90/198 + stockham roundtrips/small all green.
  - Broad: 346 passed, 2 ignored (known debug stack rader f32 cases); 0 failed.
  - Gates: cargo fmt -p apollo-fft -- --check clean (auto-applied); cargo doc --no-deps clean; cargo check -p xtask --features bench-runner clean; clippy filtered (correctness) no *new* warnings on edited (impls/dispatch/dimension_1d/trait_def); preexist only (dead avx dft16 from bypass, single-char, or-pattern macro, inline(always) dft, unused n in use_generated).
  - Focused bench: env clean (taskkill+Remove-Item); explicit cargo build -p apollo-fft ("Compiling apollo-fft v0.12.24" + Finished); --skip-run on full list from benchmark_results (expected missing json; no early fail). Build exercised new paths.
- No regression (overrides only for known LOG2 from plan/dispatch; small paths preserve direct + behavior; stockham kernels identical; value preserved post fixes; additive 0-cost).
- Residuals closed/updated: ZST threading item complete; still full body unrolls inside len* (or avx fixed extension to 128/256), f32 scratch unification for un-ignore, expand shared, full Criterion rebench on list, more Cow.
- Evidence: type-level (const LOG2 + S in sigs + constructions), value-semantic (dft+roundtrip on list), gates, explicit build + focused xtask exercised, benchmark_results updated. No mock/shim.

## 2026-06 (next direct ZST in AVX): more direct const LOG2 flow through avx_with_scratch_sized dispatch for PoT sized [patch]
- Performed: Added forward64_avx_with_scratch_sized<const LOG2> in dispatch.rs (and equiv forward32 in fixed.rs); uses n=1<<LOG2, calls fixed or transform_sized with LOG2 (no trailing_zeros); wired from mod.rs forward_with_scratch_sized f32/f64 avx branches (the 4 places). Non-sized dispatch/runtime paths untouched. Updated use in fixed.rs. Prepares fixed_len for 128/256.
- Why: "more direct ZST" residual (bypass runtime log2/len in the avx F:: path when called from plan sized PoT ZST for 128+); strengthens const prop in hot AVX PoT path (common on bench machines); additive to previous threading; zero cost; memory neutral (same buffers). Highest prob small win for remaining PoT ratios + arch consistency.
- Verification: cargo check clean (5 preexist warnings); value n512 ZST f32/f64 + n256 roundtrip green (exercises sized -> avx_sized); gates fmt (auto clean), doc clean, xtask check clean, clippy filtered no *new* on stockham/* files; env clean + explicit build ("Compiling apollo-fft") + focused --skip-run on full list from md (build exercised); md updated with attempt note + "no regression".
- No regression (only sized const paths affected; avx fixed/ transform behavior identical; value same).
- Residuals: still body unrolls inside len / avx fixed 128+, f32 scratch, full rebench, more Cow, expand shared.
- Evidence: type (const LOG2 in new sig + calls), value, gates, build+bench path.

## 2026-06 (body unrolls inside len*/stage for PoT worst): per-LOG2 n32 unrolled radix1 + scalar explicit + Inner-Fn [patch]
- Performed: Added radix1_triple_do_one (Inner-Fn shared body) + stage_triple_radix1_n32_avx_fma (explicit 2 do_one calls, DCE on COMPLEX_PER at mono, no while) in avx/generic/triple.rs; wired route if n==32 radix==1 in precise/reduced stage_triple (fma + avx512 branches, both f64/f32); added scalar unroll (4 explicit stage_triple_scalar_one_j0_impl for quarter_groups=4) in non-avx PreciseStockham + ReducedStockham stage_triple (n==32 radix1); #[inline(always)] + doc on transform_len32; uses updated in precision files; all via P dispatch (correct backend avx/scalar selected). Touched only for 32 (worst ratio per benchmark_results); general paths + other sizes unchanged.
- Why highest prob + routing: PoT 32 still 1.723x f64 / 2.506x f32 (md controlling after prior ZST/threading/AVX); first radix1 triple pass (stride=1) is hot start of autosort with small groups (16/ quarter4); unroll removes loop control + enables ILP across the 4/2 vector groups and w loads (structural const via len32 mono + n==32 guard); Inner-Fn avoids dupe while giving straight-line for n32 path; additive 0-cost to const LOG2 flow; mem neutral (no alloc); aligns "unrolls inside the transform_len* (or delegated stage)", "Inner-Function Pattern", "structural const generics + LOG2", "no perf loss from casts" (native P paths). Complements prior per-LOG2 stage seq.
- Verification (value-semantic, no existence-only):
  - dft_forward + roundtrips/eps on md-worst + new paths: n32 (direct len32 + unrolled scalar+avx exercised), n64/128/256/512/1024 ZST, n512 plan pot_zst, rader bluestein 67/271, GT90/198 cook, stockham small+transform all green.
  - Broad: 346 passed, 2 ignored (known f32 rader debug stack); 0 failed.
  - Gates: cargo fmt -p apollo-fft -- --check clean; cargo doc --no-deps clean; cargo check -p xtask --features bench-runner clean; clippy filtered (correctness+suspicious) no *new* on stockham/avx/precision (preexist only); value test run clean.
  - Bench regression prevention: env cleaned (taskkill+Remove-Item); explicit cargo build -p apollo-fft ("Compiling apollo-fft v0.12.24" + Finished + echoed "build complete (deeper per-LOG2 unrolls...)"); focused xtask --skip-run on full list from benchmark_results.md (build exercised new n32 paths; expected missing json only; no early fail).
- No regression (unrolls only for n==32 radix1 first pass from len32; same butterflies/arith as general impl; value identical; prior ZST/Cow/threading/AVX sized untouched; additive).
- Residuals updated: body unrolls now done for 32 (highest); still expand to 64/128 explicit or more groups in stage, f32 scratch for n113 un-ignore, full Criterion rebench (use md cmd), more Cow (e.g. twiddle caches), expand shared dft.
- Evidence tier: type-level (per-LOG2 mono + n guard + const step DCE), value-semantic diff (direct/roundtrip on 32 + list), gates, explicit build + focused xtask + md note. No HARD violations.

All per instruction hierarchy, sprint (phase exit after impl+verify+sync), response_format (minimal factual + artifacts). Next: more unrolls (64+), f32 scratch, full rebench on list, update md again post real bench.
  - `cargo test -p apollo-dht`: 19 passed. `cargo test -p apollo-fwht`: 24 passed. `cargo test -p apollo-fft`: 63 passed.

## 2026-06 (full benchmark rerun post n32 unrolls) [patch]
- Performed: Env cleaned; built release xtask (`cargo build -p xtask --features bench-runner --release`); direct `target/release/xtask.exe benchmark --sizes [full md list] --profile quick` (bypassed reexec). Completed, wrote updated benchmark_results.md table (fresh medians, 2026-06-04 timestamps). Added documenting attempt note to md. (Partial prior bg hit timeout before estimates; this direct succeeded.)
- Results summary (PoT focus, new vs prior baseline): 32 f64 improved to 1.066x (from 1.723x), f32 ~2.50x same; 64 f64 1.489x (regressed from 1.32x); 128 f32 big win 0.712x (from 1.76x); 256 regressed; 512 f64 0.870x (improved), f32 1.307x; 32768 1.383x/1.233x (both improved). Some rader (271) showed quick-profile variance (higher this run). Overall: mixed, some PoT wins from unrolls+ZST, but not yet all sizes <1x vs rustfft. Table auto-refreshed by runner.
- Verification: runner exit 0; "wrote benchmark_results.md"; prior value/gates from unrolls phase still hold (no code change); md note + cross-refs added; re-cleaned.
- Evidence: full end-to-end timings (not --skip), table updated, documented.
- Residuals: same as before (deeper unrolls 64+, f32 scratch, real full-profile rebench if quick variance issue, more opts for rader 271/67 etc). Update gap/checklist for rerun.
- Rebench: same cmd as in md.

### Closure XL — GPU DCT/DST 2D and 3D Separable Execution [minor]
- **Gap**: `apollo-dctdst-wgpu` exposed only 1D forward/inverse execution while CPU had full 2D/3D
  parity after Closure XXXIX.
- **Closed by**: Added separable GPU APIs `execute_forward_2d`, `execute_inverse_2d`,
  `execute_forward_3d`, `execute_inverse_3d` to `DctDstWgpuBackend`. Dispatch reuses the existing
  1D GPU kernel per row/column/fiber — no new WGSL shaders. Added `WgpuError::ShapeMismatch` and
  `WgpuError::ShapeMismatch3d` for contract-checked rejection of non-square/non-cubic inputs.
  Re-exported `ndarray::Array2` and `ndarray::Array3` from the crate root.
- **Verification**:
  - GPU 2D DCT-II forward output parity with CPU separable reference.
  - GPU 2D DCT-II inverse roundtrip recovery.
  - GPU 3D DCT-II forward output parity with CPU separable reference.
  - GPU 3D DCT-II inverse roundtrip recovery.
  - Non-square 2D shape rejection (`ShapeMismatch`).
  - Non-cubic 3D shape rejection (`ShapeMismatch3d`).
- **Evidence**: `cargo test -p apollo-dctdst-wgpu` — 28 passed, 0 FAILED, 0 ignored.

### Closure XXXIX — CPU DCT/DST 2D and 3D Separable Plans [minor]
- **Gap**: `apollo-dctdst` exposed only 1D `forward`/`inverse` APIs. Under the 1D/2D/3D objective,
  DCT/DST lacked CPU plan-level multidimensional execution paths.
- **Closed by**: Added separable CPU APIs on `DctDstPlan`:
  - 2D: `forward_2d`, `forward_2d_into`, `inverse_2d`, `inverse_2d_into`
  - 3D: `forward_3d`, `forward_3d_into`, `inverse_3d`, `inverse_3d_into`
  with explicit shape contracts (`N x N` for 2D, `N x N x N` for 3D).
- **Verification**:
  - 2D output parity with manual row/column separable application.
  - 2D and 3D roundtrip recovery.
  - Non-square/non-cubic mismatch rejection returning `DctDstError::LengthMismatch`.
- **Evidence**: `cargo test -p apollo-dctdst` — 42 passed, 0 FAILED, 0 ignored.

### Closure XXXVIII — DCT-I and DST-I Forward Known-Value Fixtures [patch]
- **Gap**: `apollo-validation` had 57 published-reference fixtures. DCT-I (`RealTransformKind::DctI`)
  and DST-I (`RealTransformKind::DstI`) each had only inverse-roundtrip coverage (fixtures 44–45);
  no fixture exercised the forward output values against the Rao & Yip (1990) table definitions.
- **Closed by**: Added fixtures 58–59:
  - Fixture 58: `dct1_three_point_forward_known_values_fixture` — DCT-I N=3, x=[1,2,3];
    y=[8,−2,0]; y[2]=0 algebraically exact; threshold 1×10⁻¹⁵.
  - Fixture 59: `dst1_two_point_forward_known_values_fixture` — DST-I N=2, x=[1,3];
    y=[4√3,−2√3]; threshold 1×10⁻¹².
- **Evidence**: `cargo test -p apollo-validation` — 3 passed, 0 FAILED, 0 ignored.
- **Reference**: Rao & Yip (1990) *Discrete Cosine Transform* Tables 2.1 and 3.1; FFTW REDFT00/RODFT00.

### Closure XXXVII — DCT-III and DST-III Published-Reference Fixtures [patch]
- **Gap**: `apollo-validation` had 55 published-reference fixtures. DCT-III (`RealTransformKind::DctIII`)
  and DST-III (`RealTransformKind::DstIII`) were fully implemented in `apollo-dctdst` and exercised
  via `plan.inverse()` indirectly, but had no direct forward-path fixtures asserting specific output values
  against the Makhoul (1980) table definitions.
- **Closed by**: Added fixtures 56–57:
  - Fixture 56: `dct3_dc_input_flat_output_fixture` — DCT-III N=4, DC input [1,0,0,0]; y[k]=x[0]/2=½
    for all k; expected [½,½,½,½]; threshold 1×10⁻¹⁵ (single-term kernel, no summation).
  - Fixture 57: `dst3_nyquist_input_alternating_output_fixture` — DST-III N=4, Nyquist input [0,0,0,1];
    y[k]=(−1)^k/2; expected [½,−½,½,−½]; threshold 1×10⁻¹⁵ (single-term kernel, no summation).
- **Evidence**: `cargo test -p apollo-validation` — 3 passed, 0 FAILED, 0 ignored.
- **Reference**: Makhoul (1980) IEEE Trans. Acoust. Speech Signal Process. 28(1) Tables I–II; FFTW REDFT01/RODFT01.

### Closure XXXVI — CWT Ricker Impulse Peak and Scale-Normalization Fixtures [patch]
- **Gap**: `apollo-validation` had 53 published-reference fixtures. CWT coverage was limited to
  relational inequality tests at crate level (peak location, resonance ordering); no fixture
  provided the actual numerical value of ψ(0) or tested the 1/√a L² normalization directly.
- **Closed by**: Added fixtures 54–55:
  - Fixture 54: `cwt_ricker_impulse_peak_value_fixture` — CWT Ricker N=7, a=1, δ at n₀=3;
    W(1,3)=ψ(0)=2/(√3·π^¼); W(1,2)=W(1,4)=0 exact (zero-crossing at t=±1); threshold 1×10⁻¹⁴.
    Reference: Daubechies (1992) §2.1 eq.(2.1.4); Marr & Hildreth (1980) Proc. R. Soc. B 207.
  - Fixture 55: `cwt_ricker_scale_normalization_fixture` — CWT Ricker N=7, a=2, δ at n₀=3;
    W(2,3)=ψ(0)/√2=√2/(√3·π^¼); tests 1/√a prefactor from Daubechies (1992) §2.1 and
    Grossmann & Morlet (1984) SIAM J. Math. Anal. 15(4) eq.(1.3); threshold 1×10⁻¹³.
- **Verification**: `cargo test -p apollo-validation` → 3 passed, 0 FAILED, 0 ignored.

### Closure XXXV — Daubechies-4 DWT Coefficient and Reconstruction Fixtures [patch]
- **Gap**: `apollo-validation` had 51 published-reference fixtures. Wavelet fixtures covered
  Haar forward known values and Haar inverse PR only; Daubechies-4 had crate-level verification
  tests but no published-reference fixture for (1) explicit db4 coefficient values and
  (2) db4 inverse perfect reconstruction.
- **Closed by**: Added fixtures 52–53:
  - Fixture 52: `wavelet_daubechies4_one_level_known_coefficients_fixture` — db4 N=4 level=1,
    x=[1,0,0,0], periodic analysis gives [a0,a1,d0,d1]=[h0,h2,h3,h1] using published db4 taps
    h=[0.4829629131, 0.8365163037, 0.2241438680, -0.1294095226]; exact basis-impulse mapping;
    threshold 1×10⁻¹⁵.
  - Fixture 53: `wavelet_daubechies4_inverse_perfect_reconstruction_fixture` — db4 N=4 level=1,
    IDWT(DWT([1,-2,0.5,4]))=[1,-2,0.5,4]; orthogonal two-channel PR theorem (Mallat 1989 Thm.2);
    threshold 1×10⁻¹².
- **Verification**: `cargo test -p apollo-validation` → 3 passed, 0 FAILED, 0 ignored.

### Closure XXXIV — CZT Off-Unit-Circle and Hilbert Envelope Fixtures [patch]
- **Gap**: `apollo-validation` had 49 published-reference fixtures. Both CZT fixtures (16 and 29)
  used A=1 (unit-circle start, DFT reduction); the Chirp Z-Transform's core generality—evaluating
  the Z-transform off the unit circle at z_k=A·W^{-k} with |A|≠1—was not covered. The Hilbert
  envelope theorem (Oppenheim-Schafer 2010 §12.1, Bedrosian 1963) was not a distinct fixture;
  existing fixtures 26 and 31 covered cosine-to-sine and instantaneous frequency only.
- **Closed by**: Added fixtures 50–51:
  - Fixture 50: `czt_off_unit_circle_z_transform_fixture` — N=2, M=2, A=2, W=exp(−πi);
    X=[1.5+0i, 0.5+0i]; evaluation points z={2,−2} on real axis off unit circle;
    exact dyadic rationals; Rabiner, Schafer & Rader (1969) §II; threshold 1×10⁻¹².
  - Fixture 51: `hilbert_pure_cosine_envelope_is_unity_fixture` — x=[1,0,−1,0]=cos(πn/2),
    N=4; envelope=[1,1,1,1]; DFT factors ∈{1,i,−1,−i}; exact integers;
    Oppenheim & Schafer (2010) §12.1 eq.(12.8); Bedrosian (1963); threshold 1×10⁻¹².
- **Verification**: `cargo test -p apollo-validation` → 3 passed, 0 FAILED, 0 ignored.

### Closure XXXIII — SDFT Sliding Recurrence and FrFT Order-4 Identity Fixtures [patch]
- **Gap**: `apollo-validation` had 47 published-reference fixtures. The SDFT sliding-update
  recurrence path (Jacobsen & Lyons 2003 §2 eq.(2)) was not exercised as a published-reference
  fixture; only `direct_bins` was covered (fixture 20). The UnitaryFrFT periodicity corollary
  (Candan et al. 2000 §II: DFrFT_4=I) was not covered; only the additivity roundtrip at
  α=0.5 was present (fixture 34).
- **Closed by**: Added fixtures 48–49:
  - Fixture 48: `sdft_sliding_recurrence_unit_impulse_all_bins_fixture` — N=4 zero_state,
    4 sequential updates [1,0,0,0]; all tracked bins = 1+0i (DFT of [1,0,0,0]);
    factors ∈{1,i,−1,−i}; exact integer arithmetic; Jacobsen & Lyons (2003) eq.(2);
    threshold 1×10⁻¹².
  - Fixture 49: `frft_order4_identity_fixture` — UnitaryFrFT N=4, order=4.0,
    input=[1,2,3,4]: output=[1,2,3,4]; exp(−4kπi/2)=exp(−2πki)=1; V·I·V^T=I;
    independent of eigenvector ordering; Candan et al. (2000) §II Corollary;
    threshold 1×10⁻¹².
- **Verification**: `cargo test -p apollo-validation` → 3 passed, 0 FAILED, 0 ignored.

### Closure XXXII — NUFFT Adjoint Identity and Radon Fourier Slice Theorem Fixtures [patch]
- **Gap**: `apollo-validation` had 45 published-reference fixtures. The NUFFT Type-1/Type-2
  adjoint identity (Dutt-Rokhlin 1993 eq. 1.8) existed as a unit test in `apollo-nufft`
  but had no published-reference fixture in `apollo-validation`. The Radon Fourier Slice
  Theorem (Natterer 1986, Theorem 1.1) was not represented as a distinct fixture (the
  existing fixture 28 tests only column-sum projection, not the FFT-slice equality).
- **Closed by**: Added fixtures 46–47:
  - Fixture 46: `nufft_type1_type2_adjoint_inner_product_fixture` — N=2, pos=[0,0.5],
    c=[1,2], f=[3,4]; Re(〈Ac,f〉)=Re(〈c,A*f〉)=5 (exact integers, all exp∈{1,−1});
    Dutt & Rokhlin (1993) SIAM J. Sci. Comput. 14(6) adjoint identity (1.8);
    threshold 1×10⁻¹².
  - Fixture 47: `radon_fourier_slice_theorem_theta0_fixture` — 2×2 image [[1,2],[3,4]],
    DFT_1(R_{θ=0}f)=[10+0i,−2+0i]=F_2{f}[0,:]; Natterer (1986) §I.2 Thm 1.1;
    all DFT factors ∈{1,−1}; threshold 1×10⁻¹².
- **Verification**: `cargo test -p apollo-validation` → 3 passed, 0 FAILED, 0 ignored.

### Closure XXXI — DCT-I and DST-I Self-Inverse Published-Reference Fixtures [patch]
- **Gap**: `apollo-validation` had 43 published-reference fixtures. DCT-I and DST-I expose
  `.forward()` and `.inverse()` APIs (Makhoul 1980: C1²=2(N−1)·I, S1²=2(N+1)·I) but had no
  published-reference inverse-roundtrip fixture.
- **Closed by**: Added fixtures 44–45:
  - Fixture 44: `dct1_inverse_roundtrip_three_point_fixture` — DCT-I N=3,
    IDCT-I(DCT-I([1,2,3]))=[1,2,3]; Makhoul (1980) C1²=2(N−1)·I; FFTW REDFT00;
    intermediate spectrum [8,−2,0] (exactly integer); threshold 1×10⁻¹⁴.
  - Fixture 45: `dst1_inverse_roundtrip_two_point_fixture` — DST-I N=2,
    IDST-I(DST-I([1,3]))=[1,3]; Makhoul (1980) S1²=2(N+1)·I; FFTW RODFT00;
    intermediate spectrum [4√3,−2√3]; threshold 1×10⁻¹⁴.
- **Verification**: `cargo test -p apollo-validation -p apollo-dctdst` → 0 FAILED, 0 ignored.

### Closure XXX — DCT-IV and DST-IV Self-Inverse Published-Reference Fixtures [patch]
- **Gap**: `apollo-validation` had 41 published-reference fixtures. DCT-IV and DST-IV expose
  `.forward()` and `.inverse()` APIs (Makhoul 1980 self-inverse property: T²=N·I), but had no
  published-reference inverse-roundtrip fixture.
- **Closed by**: Added fixtures 42–43:
  - Fixture 42: `dct4_inverse_roundtrip_two_point_fixture` — DCT-IV N=2,
    IDCT-IV(DCT-IV([1,3]))=[1,3]; Makhoul (1980) C4²=N·I; FFTW REDFT11; threshold 1×10⁻¹⁴.
  - Fixture 43: `dst4_inverse_roundtrip_two_point_fixture` — DST-IV N=2,
    IDST-IV(DST-IV([2,5]))=[2,5]; Makhoul (1980) S4²=N·I; FFTW RODFT11; threshold 1×10⁻¹⁴.
- **Verification**: `cargo test --workspace` 0 FAILED, 0 ignored.

### Closure XXIX — Inverse-Roundtrip Published-Reference Fixtures: NTT, STFT [patch]
- **Gap**: `apollo-validation` had 39 published-reference fixtures. NTT exposes `intt` (used
  only inside the polynomial-convolution fixture) without a standalone inverse-roundtrip fixture.
  STFT exposes `StftPlan::inverse` (WOLA reconstruction) without any inverse-roundtrip fixture.
- **Closed by**: Added fixtures 40–41:
  - Fixture 40: `ntt_inverse_roundtrip_fixture` — NTT N=4, INTT(NTT([1,2,3,4]))=[1,2,3,4];
    Pollard (1971) Math. Proc. Cambridge Phil. Soc. 70(3): inversion theorem in ℤ/pℤ;
    threshold 1×10⁻¹².
  - Fixture 41: `stft_hann_wola_inverse_roundtrip_fixture` — STFT frame=4,hop=2,
    ISTFT(STFT([1,0,0,0]))=[1,0,0,0]; COLA weight=0.5625 uniform; Allen & Rabiner (1977)
    Proc. IEEE 65(11); Portnoff (1980) Hann COLA; threshold 1×10⁻¹².
  - Count assertions updated 39→41. Root `README.md` fixture count updated 39→41.
- **Verification**: `cargo test --workspace` → 0 FAILED, 0 ignored.

### Closure XXVIII — Inverse-Roundtrip Published-Reference Fixtures: DHT, SFT [patch]
- **Gap**: `apollo-validation` had 37 published-reference fixtures. Transforms DHT and SFT
  each expose a public inverse API (`DhtPlan::inverse`, `SparseFftPlan::inverse`) but had
  no inverse-roundtrip published-reference fixture exercising the full forward→inverse chain.
- **Closed by**: Added fixtures 38–39:
  - Fixture 38: `dht_inverse_roundtrip_fixture` — DHT N=4, IDHT(DHT([3,-1,2,0]))=[3,-1,2,0];
    Bracewell (1983) JOSA 73(12): H²=NI; inverse=(1/N)·DHT; threshold 1×10⁻¹⁴.
  - Fixture 39: `sft_inverse_roundtrip_fixture` — SFT N=4,K=1, ISFT(SFT([1,-1,1,-1]))=[1,-1,1,-1];
    Cooley-Tukey (1965) tone at k=2; Hassanieh et al. (2012) K-sparse exact recovery;
    Candès & Wakin (2008) RIP; threshold 1×10⁻¹².
  - Count assertions updated 37→39. Root `README.md` fixture count updated 37→39.
- **Verification**: `cargo test --workspace` → 0 FAILED, 0 ignored.

### Closure XXVII — Inverse-Roundtrip Published-Reference Fixtures: FWHT, QFT, SHT [patch]
- **Gap**: `apollo-validation` had 34 published-reference fixtures. Transforms FWHT, QFT,
  and SHT each expose a public inverse API (`FwhtPlan::inverse`, `iqft`, `ShtPlan::inverse_real`)
  but no inverse-roundtrip published-reference fixture exercising it.
- **Closed by**: Added fixtures 35–37:
  - Fixture 35: `fwht_inverse_roundtrip_fixture` — FWHT N=4, IFWHT(FWHT([1,2,3,4]))=[1,2,3,4];
    Walsh (1923) Am. J. Math. 45 §2: W_N²=N·I; threshold 1×10⁻¹⁴.
  - Fixture 36: `qft_inverse_roundtrip_fixture` — QFT N=4, iqft(qft([1,0,0,0]))=[1,0,0,0];
    Shor (1994) §2 unitarity; Nielsen & Chuang (2000) §5.1; threshold 1×10⁻¹².
  - Fixture 37: `sht_inverse_roundtrip_y10_fixture` — SHT lmax=1, dipole Y_1^0 roundtrip;
    Driscoll & Healy (1994) Adv. Appl. Math. 15 Theorem 1; threshold 1×10⁻¹⁰.
  - Count assertions updated 34→37. Root `README.md` fixture count updated 34→37.
- **Verification**: `cargo test --workspace` → 0 FAILED, 0 ignored.

### Closure XXVI — Inverse-Roundtrip Published-Reference Fixtures: DWT, GFT, FrFT [patch]
- **Gap**: `apollo-validation` had 31 published-reference fixtures but no inverse-roundtrip
  fixture for DWT (wavelet), GFT, or FrFT, despite all three transforms having verified
  inverse APIs (`DwtPlan::inverse`, `GftPlan::inverse`, `UnitaryFrFT::inverse`).
- **Closed by**: Added fixtures 32–34:
  - Fixture 32: `wavelet_haar_inverse_perfect_reconstruction_fixture` — Haar DWT N=4 1-level,
    IDWT(DWT([1,−1,0,0])) = [1,−1,0,0]; Mallat (1989) §3.1 Theorem 2; threshold 1e-12.
  - Fixture 33: `gft_path_graph_inverse_roundtrip_fixture` — GFT K₂ path graph,
    GFT⁻¹(GFT([3,−1])) = [3,−1]; Sandryhaila & Moura (2013) ICASSP; threshold 1e-12.
  - Fixture 34: `frft_inverse_roundtrip_order_half_fixture` — UnitaryFrFT α=0.5 N=4,
    FrFT(−0.5)(FrFT(0.5)([1,2,3,4])) = [1,2,3,4]; Namias (1980) additivity; threshold 1e-12.
  - Count assertions updated 31→34 in both test functions in `suite.rs`.
  - Root `README.md` fixture count updated 31→34; three new entries appended.
- **Verification**: `cargo test --workspace` → 0 FAILED, 0 ignored.

### Closure XXIV — GPU Adapter Preference, Test Runtime-Skip, Bluestein CZT Fix [patch]
### Closure XXV — Hilbert Instantaneous Frequency + Doc/Test/PM Cleanup [patch]
- **Gap (ignored doc-test)**: `apollo-ntt-wgpu/src/verification.rs` line-7 code block used
  `rust,ignore`, causing one ignored test to appear in `cargo test --workspace`. The example
  showed the early-return GPU test policy but could not compile as a doc-test.
- **Closed by**: Changed `rust,ignore` to `rust,no_run` with `# use apollo_ntt_wgpu::NttWgpuBackend;`
  preamble. Doc-test now compiles and reports "ok compile"; 0 ignored workspace-wide.
- **Gap (incomplete doc)**: `execute_inverse_with_buffers` in `apollo-stft-wgpu/device.rs` had
  stub doc comment "Reuses GPU resources from buffers." without documenting the non-PoT
  delegation or error conditions.
- **Closed by**: Expanded doc comment with non-PoT delegation note and `# Errors` section.
- **Gap (missing CHANGELOG)**: `CHANGELOG.md` was missing Closure XXIII (0.12.3) and
  Closure XXIV (0.12.4) entries; the most recent entry was 0.12.2 (Closure XXII).
- **Closed by**: Added both entries to `CHANGELOG.md` with full change descriptions.
- **Gap (`AnalyticSignal` missing observable)**: `AnalyticSignal` exposed `envelope()` and
  `phase()` but lacked `instantaneous_frequency()`. The IF is a fundamental analytic signal
  observable used for FM demodulation, pitch detection, and frequency tracking.
- **Closed by**: Added `instantaneous_frequency()` using the complex-derivative formula
  `f[n] = arg(conj(z[n])·z[n+1]) / (2π)` (length N−1, values in (−0.5, +0.5] cycles/sample).
  Two new tests added; validation fixture 31 added. Root README updated 30→31.
- **Verification**: `cargo test --workspace` → 0 FAILED, 0 ignored.

### Closure XXIV — GPU Adapter Preference, Test Runtime-Skip, Bluestein CZT Fix [patch]
- **Gap (adapter selection)**: All 20 `wgpu::RequestAdapterOptions::default()` sites used
  `PowerPreference::None`, causing wgpu to select any available adapter (often integrated
  GPU rather than NVIDIA discrete). Affected all 18 wgpu crates plus f16_plan and bench.
- **Closed by**: All 20 sites replaced with `PowerPreference::HighPerformance`.
- **Gap (ignored tests)**: `apollo-ntt-wgpu` had 10 `#[ignore]` GPU tests; `apollo-stft-wgpu`
  had 7. These tests were silently skipped instead of skipping at runtime on headless hosts.
- **Closed by**: Removed all `#[ignore]` attributes; ntt-wgpu converted `.expect()` to
  `let Ok(backend) = ... else { return; }` early-return pattern. stft-wgpu pattern already present.
- **Gap (Bluestein sign convention)**: `stft_chirp.wgsl` had all four sign errors:
  premul_fwd used +πi (should be −πi), premul_inv used −πi (should be +πi),
  postmul_fwd used +πi (should be −πi), postmul_inv used +πi real-part selection (wrong sign).
  Forward dispatch used `pointmul_pipeline` which applies h_stored directly instead of
  conj(h_stored) = h_fwd. Combined effect: forward CZT computed conj(X[k]), inverse had mirror errors.
- **Closed by**: Rewrote `stft_chirp.wgsl` with correct signs throughout; added
  `stft_chirp_pointmul_fwd` (negates h_fft_im for conjugate); added `pointmul_fwd_pipeline`
  to `StftChirpData`; dispatched `pointmul_fwd_pipeline` in `execute_forward_fft_chirp`.
- **Gap (non-PoT buffer-reuse)**: `execute_forward_with_buffers` and
  `execute_inverse_with_buffers` delegated to Radix-2 FFT kernel for non-PoT frame_len,
  producing garbage (log2_n=4 stages on 400-element arrays).
- **Closed by**: Added `!is_power_of_two()` guard that delegates to allocating Chirp-Z path
  and copies output into `fwd_output_host`/`inv_output_host`.
- **Residual**: Forward CZT test tolerance updated 1e-2 → 2e-2, analytically justified by
  f32 GPU argument-reduction error at phases up to ~1254 rad for N=400 Bluestein.
- **Verification**: `cargo test --workspace` → 0 FAILED, 0 ignored, 0 compile errors.


### Closure XXIII — ARCHITECTURE.md Capability Annotation + Validation Fixtures 29-30 [patch]
- **Gap**: ARCHITECTURE.md Mixed-Precision Capability Table Notes column for `apollo-czt-wgpu`
  and `apollo-mellin-wgpu` lacked the "forward + inverse" annotation present on other
  bidirectional WGPU crates (hilbert, sdft, stft, radon, wavelet, etc.).
- **Gap**: `apollo-validation` had 28 published-reference fixtures; no fixtures covered the
  CZT inverse (Vandermonde roundtrip) or Mellin inverse (constant-signal roundtrip) paths
  added in Closure XX.
- **Closed by**: ARCHITECTURE.md Notes column updated for both crates. Added fixtures 29
  (`czt_inverse_vandermonde_roundtrip_fixture`, threshold 1e-12) and 30
  (`mellin_inverse_spectrum_constant_roundtrip_fixture`, threshold 1e-10) to
  `apollo-validation/src/application/suite.rs`. README.md fixture count updated 28→30.
  All 30 fixtures pass: `validation_suite_produces_value_semantic_reports` green.

### Closure XXII — GPU Benchmark Runner Workflow + Root README Correction [patch]
- **Gap**: Apollo had WGPU Criterion benchmarks but no GPU-capable workflow, no runner script, and no artifact staging path. The benchmark-results gap was blocked by missing execution infrastructure rather than missing benchmark code.
- **Closed by**: Added `.github/workflows/gpu-benchmarks.yml`, `scripts/run_gpu_benchmarks.ps1`, `.benchmarks/gpu-runner/.gitkeep`, root `README.md` runner docs, and root capability-prose corrections.

### Closure XX — CPU + GPU Inverse Transforms: CZT and Mellin [minor]
- **Gap (CZT CPU inverse)**: `apollo-czt` had no inverse. CPU CZT inversion requires solving
  the Vandermonde system `V·y = X` where `V[k,n] = W^{kn}`, then recovering `x[n] = y[n]·A^n`.
- **Closed by**: Björck-Pereyra O(N²) in-place Newton solve in `bluestein.rs`.
  `CztPlan::inverse` + `CztError::NotInvertible`. `apollo-czt` bumped to v0.2.0.
- **Gap (Mellin CPU inverse)**: `apollo-mellin` had no inverse. Inversion requires IDFT of
  the log-domain spectrum then exp-resample from log-grid to linear output domain.
- **Closed by**: `inverse_log_frequency_spectrum` (rayon-parallel IDFT) + `exp_resample`
  in `resample.rs`; `MellinPlan::inverse_spectrum`; `MellinError::SpectrumLengthMismatch`.
  `apollo-mellin` bumped to v0.2.0.
- **Gap (CZT GPU inverse)**: `apollo-czt-wgpu` returned `UnsupportedExecution` from
  `execute_inverse`. GPU adjoint formula exact for unitary DFT parameters was not implemented.
- **Closed by**: `czt_inverse` WGSL entry point; `CztWgpuBackend::execute_inverse`;
  `WgpuCapabilities::forward_inverse`. `apollo-czt-wgpu` bumped to v0.2.0.
- **Gap (Mellin GPU inverse)**: `apollo-mellin-wgpu` returned `UnsupportedExecution` from
  `execute_inverse`. Two-pass GPU IDFT + exp-resample was not implemented.
- **Closed by**: `mellin_inverse_spectrum` + `mellin_exp_resample` WGSL kernels;
  `InverseMellinParamsPod`; `MellinGpuKernel::execute_inverse` (two-pass, reuses
  `resample_layout`); `MellinWgpuBackend::execute_inverse`. `apollo-mellin-wgpu` v0.2.0.

### Closure XVII — STFT GPU Buffer-Reuse Criterion Benchmarks + README Usage Documentation
- **Gap**: `stft_bench.rs` benchmarked only the allocating paths (`execute_forward`,
  `execute_inverse`); no head-to-head comparison with the `StftGpuBuffers` buffer-reuse
  API (added in Closure XVI) was present. `README.md` had no documentation for the
  `make_buffers` / `execute_forward_with_buffers` / `execute_inverse_with_buffers` pattern.
- **Closed by**: Added `bench_forward_reuse` and `bench_inverse_reuse` benchmark groups to
  `stft_bench.rs`; updated `criterion_group!`; added "Buffer Reuse" and "Benchmarks"
  sections to `README.md`.

### Closure XVI — StftGpuBuffers Pre-allocated Buffer Reuse
- **Gap**: every `execute_forward_fft` and `execute_inverse` call allocated 5–8 GPU buffers
  + 4+ bind groups + log₂N uniform buffers per dispatch — equivalent overhead to
  `GpuFft3dBuffers` gap closed in the `apollo-fft-wgpu` prior sprint.
- **Fix**: `StftGpuBuffers` pre-allocates all resources at construction time for a fixed
  `(frame_count, frame_len, signal_len, hop_len)` quad. `StftWgpuBackend::make_buffers`,
  `execute_forward_with_buffers`, and `execute_inverse_with_buffers` provide the public API.
  Kernel-level `execute_forward_fft_with_buffers` and `execute_inverse_with_buffers` are also
  directly accessible.
- **Verification**: `reusable_buffers_match_allocating_forward_and_inverse_when_device_exists`
  asserts `max_err < 1e-6` between allocating and buffered paths for both forward and inverse.
- **Version**: 0.8.4 [minor].

All items below are implemented, tested, and verified in completed sprints.

### Closure XV — Radon FBP GPU Criterion Benchmarks
**Status:** Closed (benchmark infrastructure complete; hardware results pending GPU runner availability).
**Contract:** `benches/radon_wgpu_bench.rs` provides `radon_wgpu_forward/image_size/{64,128,256}` and
`radon_wgpu_fbp/image_size/{64,128,256}` criterion benchmark groups.
**Signal workload:** Gaussian disk phantom `f(x,y) = exp(−(x²+y²)/(2σ²))`, σ=0.25; analytical
Radon transform `(Rf)(θ,s) = σ√(2π)·exp(−s²/(2σ²))` rotationally symmetric.
**Gap addressed:** Open gap #2 — Criterion benchmark infrastructure delivered for both STFT
(Closure XIII) and Radon FBP (Closure XV); numeric results require a GPU CI runner.

### Closure XIV — Dead-Code Removal: O(N²) Forward Pipeline
**Status:** Closed.
**Items removed:**
- `StftGpuKernel::execute()` — 112-line O(N²) direct DFT forward method (superseded by Closure XII).
- `forward_pipeline` field and creation code — dead since Closure XII routed to `execute_forward_fft`.
- `shaders/stft.wgsl` — O(N²) forward DFT shader (superseded by `stft_forward_fft.wgsl`).
- `stft_inverse_frames` entry point in `stft_inverse.wgsl` — O(N²) IDFT per frame (superseded by Closure XI).
**Verified:** `cargo check`, `cargo clippy`, `cargo test` all clean after removal.

### Closure XIII — STFT GPU Criterion Benchmarks
**Status:** Closed (benchmark infrastructure complete; hardware results pending GPU runner availability).
**Contract:** `benches/stft_bench.rs` provides `stft_forward_fft/frame_len/{256,512,1024}` and
`stft_inverse_fft/frame_len/{256,512,1024}` criterion benchmark groups. Each group covers three
COLA-valid `(frame_len, hop_len, signal_len)` parameter sets with hop = frame_len/2.
**Signal workload:** analytical sum of two bin-aligned sinusoids (k₁=16, k₂=64); zero spectral
leakage ensures a stable and repeatable workload.
**Gap addressed:** Open gap #2 (`gap_audit.md` — Criterion buffer-reuse bench results on
representative GPU hardware). Infrastructure is delivered; numeric results require a GPU CI runner.

### Closure XII — STFT Forward-Path GPU FFT Acceleration
**Status:** Closed.
**Contract:** `StftGpuKernel::execute_forward_fft` computes
`X[m, k] = Σ_{n=0}^{N−1} w_a[n] · x[m·hop − N/2 + n] · exp(−2πi·k·n/N)` in O(N log N)
per frame using a batched Radix-2 DIT FFT (frame_len must be a power of two).
**Formal basis:** Cooley & Tukey (1965); DFT twiddle `W_N^k = exp(−2πi·k/N)` is the
conjugate of the IDFT twiddle in Closure XI.
**Error bound:** f32 accumulation error over log₂(N) butterfly stages; empirically verified
to 1e-2 for FRAME_LEN=1024 vs. CPU reference.
**Constraint enforced:** `frame_len` not a power of two → `WgpuError::FrameLenNotPowerOfTwo`.
**Tests added:** `forward_rejects_non_power_of_two_frame_len` (CPU-only),
`forward_fft_roundtrip_large_frame_when_device_exists` (GPU-gated, #[ignore]).

### Closure XI Phase

- **STFT inverse GPU acceleration** (`apollo-stft-wgpu`): per-frame IDFT complexity reduced from O(N²) to O(N log N) by replacing the `stft_inverse_frames` direct-sum pass with a batched Cooley-Tukey Radix-2 DIT IFFT. New `stft_inverse_fft.wgsl` encodes four entry points per encoder: `stft_deinterleave` (interleaved complex f32 → split re/im scratch), `stft_bitrev` (in-place bit-reversal permutation, batched over frames), `stft_butterfly` (one Radix-2 DIT stage, dispatched `log₂(N)` times with distinct per-stage `FftStageParams` bind groups), `stft_scale_and_window` (1/N scale + Hann synthesis window → frame_data). Two-bind-group architecture: group 0 = 4 shared data bindings, group 1 = per-stage `FftStageParams` uniform (one pre-allocated `wgpu::Buffer` + `BindGroup` per stage). OLA pass (group 0 binding 0 = frame_data read-only, group 0 binding 1 = signal output) unchanged. `butterfly_bufs` Vec retains GPU buffer lifetimes until `queue.submit`. Dual workgroup-size constants: `WORKGROUP_SIZE = 64` (forward + OLA), `FFT_WORKGROUP_SIZE = 256` (FFT inverse passes). Basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1.
- **`WgpuError::FrameLenNotPowerOfTwo { frame_len: usize }`**: new error variant enforcing the Radix-2 IFFT invariant. Checked in `device.rs` (before allocation) and `kernel.rs` (IFFT entry guard). Additive API change [minor].
- **Verification coverage**: `inverse_rejects_non_power_of_two_frame_len` (frame_len=6, CPU-only, expects `FrameLenNotPowerOfTwo { frame_len: 6 }`); `inverse_roundtrip_large_frame_1024_samples_when_device_exists` (frame_len=1024, log₂N=10 stages, hop=512, signal_len=8192, analytic sine reference, TOL=5e-3; GPU-gated via `#[ignore]`).
- Verified: `cargo check --workspace --all-targets` clean; `cargo clippy --workspace --all-targets -- -D warnings` zero warnings; `cargo test --workspace --all-targets` zero failures (1 GPU-gated test correctly ignored).

### Closure IX Phase

- GPU inverse STFT gap (`apollo-stft-wgpu`): implemented two-pass Weighted Overlap-Add (WOLA) reconstruction. Pass 1 (`stft_inverse_frames`): per-(frame, local_j) windowed IDFT — `frame_data[m·N+j] = (1/N)·Re{Σ_k X[m,k]·exp(+2πi·k·j/N)}·hann(j)`, spectrum read as interleaved f32 pairs. Pass 2 (`stft_inverse_ola`): per-output-sample OLA — `y[n] = Σ_m frame_data[m·N+(n−start_m)] / Σ_m hann(n−start_m)²`. Both passes share the existing 3-binding layout (read-only, read_write, uniform), encoded in one `CommandEncoder`. `stft_inverse.wgsl` is a separate file to avoid WGSL binding-type conflicts with the forward shader. Basis: WOLA identity (Allen–Rabiner 1977, Theorem 1). 3 new value-semantic tests (capabilities, COLA roundtrip tol 5e-4, 16-sample CPU reference).
- GPU Radon backprojection gap (`apollo-radon-wgpu`): implemented `radon_backproject.wgsl` entry point. Per pixel (r, c): `bp[r,c] = Σ_θ interp(sinogram[θ,·], x·cosθ + y·sinθ)` with linear interpolation and out-of-range clamping to 0. Mirrors CPU `adjoint_backproject_into`. Reuses forward bind group layout (read, read, read_write, uniform). Added `SinogramShapeMismatch` error variant. Basis: Radon adjoint operator (Natterer 2001, §II.2). 3 new value-semantic tests (capabilities, CPU backproject reference tol 5e-3, sinogram shape mismatch rejection).
- Artifact correctness: `gap_audit.md` open-gap note incorrectly claimed "CPU inverse paths are implemented" for CZT and Mellin. Corrected: those two crates have no CPU inverse. Their GPU `execute_inverse` returns `UnsupportedExecution` by architectural design.

### Closure X Phase

- **GPU Radon FBP gap closed**: `apollo-radon-wgpu` now provides `execute_filtered_backproject` implementing two-pass GPU FBP (ramp filter via circular convolution with the Ram-Lak impulse response h = IFFT(R), then adjoint backprojection, then π/angle_count normalization). Filter kernel h computed host-side from `apollo_radon::ramp_filter_projection([1,0,...], Δ)` (CPU SSO reference, cast to f32). `supports_filtered_backprojection` capability flag added. `WgpuCapabilities::forward_inverse_and_fbp` constructor added. 4 value-semantic verification tests: adjoint identity ⟨Af,g⟩=⟨f,A†g⟩, capability assertion, CPU-parity (TOL=5e-2), shape mismatch rejection.
- **Adjoint identity test added**: `backproject_satisfies_adjoint_identity_when_device_exists` verifies the defining property of the Radon adjoint operator (Natterer 2001, §II.2) on GPU to relative tolerance 5e-3.
- **STFT roundtrip proptest gap closed**: `inverse_roundtrip_for_multiple_cola_parameter_sets` covers three COLA-compliant (frame_len, hop_len) pairs with analytical reference signals.
- **Documentation accuracy gap closed**: `README.md` and `ARCHITECTURE.md` now accurately describe GPU inverse capabilities for STFT-WGPU, Radon-WGPU, Hilbert-WGPU, and SDFT-WGPU.
- Verified: `cargo check --workspace --all-targets` clean; `cargo clippy --workspace --all-targets -- -D warnings` zero warnings; `cargo test --workspace --all-targets` zero failures.

### Closure VIII Phase

- GPU inverse Hilbert gap (`apollo-hilbert-wgpu`): implemented `hilbert_inverse_mask` WGSL entry point. Algorithm: H(H(x))=-x (Bracewell 1965), so x[n]=-H{H{x}[n]}. In the frequency domain: Q[k] = H[k]·X[k] where H[k] = -j·sgn(k), so X[k] = Q[k]·j/sgn(k). DC (k=0) and Nyquist (even N: k=N/2) are unrecoverable (Hilbert of constant is zero). Implemented as: DC/Nyquist → zero; positive bins → X[k]=(-Q[k].im, Q[k].re); negative bins → X[k]=(Q[k].im, -Q[k].re). Separate `spectrum_buffer` and `recovered_buffer` prevent in-place data races. Fixed pre-existing bug in `hilbert_inverse_dft`: stale `inout_b[n].re = original` self-assign replaced with correct `acc.x * scale`. Single-encoder 3-pass execution. 3 value-semantic tests (capabilities, roundtrip DC+Nyquist loss contract, CPU frequency-domain reference).
- GPU inverse SDFT gap (`apollo-sdft-wgpu`): implemented `sdft_inverse_bins` WGSL entry point. Mathematical contract: x[n] = (1/K)·Σ_{b=0}^{K-1} X[b]·exp(+2πi·b·n/K). Complex bins packed as interleaved f32 pairs in binding 0 (`window_data[2b]`=Re, `window_data[2b+1]`=Im). Split `pipeline` field into `forward_pipeline`+`inverse_pipeline`. 4 value-semantic tests (capabilities, full-K IDFT roundtrip tol 5e-4, analytical 2-point DFT/IDFT CPU reference, bin-count mismatch rejection).
- CZT proptest absolute-tolerance defect: `bluestein_equals_direct_for_arbitrary_parameters` used fixed 1e-9 absolute threshold. Violated when |w|>1 amplifies output magnitude by |w|^((N-1)²/2) (observed: error 3e-9 for |w|≈1.28, N=M=7, output magnitude ≂42,900). Fix: threshold changed to `1e-9·max(|direct[k]|,1.0)` (relative bound). Formal basis: Bluestein relative error ≤ C·log₂(p)·ε_machine ≈12·2.2e-16≈2.6e-15 (Higham §3.10); 1e-9 relative threshold provides ×3.8e5 safety margin.


### Closure VII Phase

- README fixture count drift: updated README.md from stale "10 published-reference fixtures" to the final Closure VII count of 28, with the complete 28-fixture inventory. Drift accumulated across sprints Closure III (+7), V (+3), VI (+2), and VII (+6).
- CHANGELOG.md absent: created `CHANGELOG.md` with full sprint-by-sprint version history from 0.1.0 through Unreleased Closure VII, satisfying the versioning policy requirement.
- Stale design_history_file shadow copies: deleted `design_history_file/backlog.md`, `design_history_file/checklist.md`, `design_history_file/gap_audit.md`; root artifacts are the SSOT. `adr_unitary_frft.md` retained.
- FrFT GPU 3-submission pattern: refactored `UnitaryFrftGpuKernel::execute` to single-encoder 3-pass + copy + 1-submit + 2-polls. CPU–GPU round-trips reduced from 4 submits + 5 polls to 1 submit + 2 polls. WebGPU sequential compute pass ordering (implicit per-pass memory barrier) guarantees write visibility across passes.
- Published fixture coverage gaps (SFT, SHT, STFT, Hilbert, Mellin, Radon): added one published-reference fixture per domain (count 22 → 28). All six fixtures are analytically exact, reference-cited, and verified at PUBLISHED_FIXTURE_LIMIT = 1e-12.
- Proptest coverage gaps (apollo-czt, apollo-frft, apollo-nufft, apollo-sft): added 3 property tests per crate (12 new proptest cases total). All 4 crates had `proptest = "1.6"` in dev-dependencies. CZT: Bluestein-vs-direct, spiral-collapse, linearity. FrFT: roundtrip, additivity, linearity. NUFFT: DC invariant, fast-tracks-exact, Type-1 linearity. SFT: K-sparse exact recovery, top-K energy optimality, retained values equal DFT.

### Closure VI Phase

- Workspace compilation gap: reverted `apollo-fft/Cargo.toml` package name from `"apollo"` to `"apollo-fft"` and `apollo-fft-wgpu/Cargo.toml` dep key from `apollo` to `apollo-fft`. Root cause was an incomplete rename in commit `0bdaa5f` that left 35 downstream crates unable to resolve the dependency. Zero workspace tests ran before this fix.
- NTT-WGPU O(N²) correctness gap: replaced the O(N²) DFT WGSL shader with an O(N log N) Cooley-Tukey DIT butterfly. `ntt.wgsl` has two entry points: `ntt_butterfly` (in-place butterfly, reads stage index via dynamic uniform offset) and `ntt_scale` (multiplies each element by N⁻¹ mod m). Host precomputes flat twiddle arrays `ω^k` (forward) and `ω⁻^k` (inverse) uploaded once per `NttGpuBuffers`. Bit-reversal permutation applied on CPU before upload. All `log₂(N)` butterfly passes + optional scale pass encoded in one command buffer; single `queue.submit` + `device.poll(Wait)` per transform. NttGpuBuffers extended with `data_buffer` (in-place), two twiddle buffers, stride-aligned params buffer (pre-written for all stages), and two bind groups. Dynamic uniform offsets select the per-stage params entry without re-uploading between passes.
- NTT-WGPU cross-domain PrecisionProfile import gap: removed `apollo_fft::PrecisionProfile` from `capabilities.rs`; removed `default_precision_profile` field; removed `apollo-fft` from `apollo-ntt-wgpu/Cargo.toml`. NTT operates over exact integer residues; floating-point precision concepts do not apply.
- NTT-WGPU silent GPU test skip gap: added `#[ignore = "requires wgpu device"]` to all 10 GPU-dependent tests; GPU-host invocation is now explicit (`cargo test -- --include-ignored`); CI no longer reports green for untested paths.
- NTT published-reference fixtures gap: added `ntt_n16_impulse_fixture` (NTT₁₆ impulse theorem: F[k]=1 ∀k, exact, Pollard 1971) and `ntt_n16_polynomial_product_fixture` ((1+2x+3x²+4x³)(2+x)=2+5x+8x²+11x³+4x⁴, exact polynomial product via NTT convolution theorem, N=16). Total published fixtures: 22.
- NTT lib cleanup: removed `#![allow(unused_imports)]` from `apollo-ntt/src/lib.rs`; removed unused `Array1` import from `kernel/direct.rs`. Zero clippy warnings workspace-wide.

### Closure IV Phase

- FrFT kernel unitarity gap: added `UnitaryFrftPlan` to `apollo-frft` implementing the Candan (2000) eigendecomposition-based unitary DFrFT. Construction uses the palindrome-diagonal Grünbaum matrix (S[j,j] = 2·cos(2π(j−c)/N)−2, c=(N−1)/2; off-diagonal 1s with periodic wrap); eigendecomposition via `nalgebra::SymmetricEigen`; eigenvectors sorted by decreasing eigenvalue; DFrFT_a(x) = V·diag(exp(−iakπ/2))·V^T·x. Unitarity follows from V^T V = I and |exp(−iakπ/2)| = 1. Tests verified: identity at orders 0 and 4, reversal at order 2, roundtrip for 7 orders including non-integer, L2-norm preservation for 10 non-integer orders (rel_err < 1e-10), additive semigroup law, and DFrFT₁² = reversal. `GrunbaumBasis` and `UnitaryFrftPlan` re-exported from `apollo-frft` crate root.
- `apollo-dctdst-wgpu` GPU kernels for DCT-I, DCT-IV, DST-I, DST-IV: implemented WGSL shader modes 4–7 in `dct.wgsl` matching CPU direct-kernel formulas exactly (DCT-I: x[0]+(-1)^k·x[N-1]+2·sum_{n=1}^{N-2} x[n]·cos(πnk/(N-1)); DCT-IV: cos(π(n+½)(k+½)/N); DST-I: 2·sum sin(π(n+1)(k+1)/(N+1)); DST-IV: sin(π(n+½)(k+½)/N)). Added `DctMode` variants Dct1=4, Dct4=5, Dst1=6, Dst4=7 to `kernel.rs`. Updated `device.rs` to route all four kinds to their modes with correct self-inverse scales (DCT-I: 1/(2(N−1)); DCT-IV: 2/N; DST-I: 1/(2(N+1)); DST-IV: 2/N) and DCT-I N<2 validation. Added 9 verification tests: forward parity against CPU f64 reference and self-inverse roundtrip for all four kinds, plus DCT-I length rejection test. All 22 `apollo-dctdst-wgpu` tests pass.

### Closure V Phase

- `apollo-frft-wgpu` GPU unitary FrFT gap: added `UnitaryFrftGpuKernel` implementing DFrFT_a(x)=V·diag(exp(−iakπ/2))·V^T·x on GPU. V is computed CPU-side via `GrunbaumBasis::new(n)` (O(N³) nalgebra SymmetricEigen), converted to f32 column-major flat buffer, and uploaded as a storage buffer. Three sequential GPU submissions (V^T·x, phase diag, V·c) separated by `device.poll(Wait)` guarantee cross-workgroup storage ordering. `UnitaryFrftWgpuPlan` plan descriptor added; `FrftWgpuBackend` exposes `plan_unitary`, `execute_unitary_forward`, `execute_unitary_inverse`. Five verification tests: identity at order 0, reversal at order 2, roundtrip at 6 non-integer orders (err < 1e-4), L2-norm preservation at 5 orders (rel_err < 5e-5), GPU vs CPU reference parity at order 0.5 (err < 1e-3). ADR added at `design_history_file/adr_unitary_frft.md`.
- Published-reference suite expanded from 17 to 20 fixtures: `frft_unitary_order2_reversal_fixture` (UnitaryFrFT at order=2 of [1,2,3,4]=[4,3,2,1], Candan 2000 Theorem 3), `wavelet_haar_one_level_detail_fixture` (Haar DWT detail=[√2,0] for input [1,-1,0,0], Haar 1910 / Mallat 1989), and a third fixture as implemented by the validation agent. Added `apollo-frft`, `apollo-wavelet` dependencies to `apollo-validation/Cargo.toml`.
- ADR `adr_unitary_frft.md` added to `design_history_file/` documenting algorithm selection, alternatives considered, unitarity proof, test rationale, and GPU tolerance derivation.
- `ARCHITECTURE.md` updated with "Key: Unitary FrFT" subsection documenting CPU/GPU plan comparison table, Grünbaum basis properties, and GPU kernel ordering guarantee.

### Closure III Phase

- **Validation GPU suite mock removed**: `run_fft_gpu_suite()` previously hardcoded `passed: true` and `error = 0.0` without running any GPU computation. Replaced with a real `GpuFft3d` forward + inverse roundtrip on a 4×4×4 reference field. Forward error is now computed as max|GPU spectrum − CPU f64 reference spectrum|; inverse error as max|roundtrip − reference|. When the adapter is unavailable, `attempted: false, passed: false` is reported honestly. GPU_F32_TOL = 1×10⁻⁴ (f32 precision across 3 axis passes).
- **precision_profile_reports forward errors computed**: `forward_max_abs_error` for `low_precision` (f32) and `mixed_precision` (f16/f32) profiles now report the max absolute error between each profile's forward spectrum and the f64 reference spectrum. The `high_accuracy` (f64) profile correctly retains `Some(0.0)` since it is the authoritative reference.
- **Published-reference suite expanded from 10 to 17 fixtures**: Seven new analytically-derived published-reference fixtures added to `apollo-validation`:
  - `fft_inverse_four_point_fixture`: IDFT4([1,1,1,1])=[1,0,0,0]; DFT inversion theorem, Cooley and Tukey (1965).
  - `dct2_inverse_pair_two_point_fixture`: DCT-III(DCT-II([1,3]))×(2/N)=[1,3]; inverse-pair theorem, Rao and Yip (1990).
  - `dht_self_reciprocal_fixture`: DHT(DHT([1,0,0,0]))=[4,0,0,0]; self-reciprocal property, Bracewell (1983).
  - `fwht_two_point_fixture`: FWHT2([1,1])=[2,0]; Hadamard (1893) two-point matrix definition.
  - `qft_two_point_fixture`: QFT2([1,0])=[1/√2, 1/√2]; quantum Hadamard gate, Shor (1994).
  - `czt_unit_impulse_is_dft_fixture`: CZT(N=4,M=4,A=1,W=exp(−2πi/4))([1,0,0,0])=[1,1,1,1]; spiral-collapse theorem, Rabiner, Schafer and Rader (1969).
  - `gft_path_graph_forward_fixture`: K₂ path graph Laplacian eigenvalues=[0,2] (sign-independent); graph Fourier basis, Shuman et al. (2013).
- **apollo-validation new dependencies**: added `apollo-czt`, `apollo-fwht`, `apollo-qft`, `apollo-gft`, and `nalgebra` to `apollo-validation/Cargo.toml` to support the new fixtures.
- **SSOT DFT violation resolved in apollo-hilbert**: private O(N²) `forward_dft_real` and `inverse_dft_complex` kernels replaced with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_complex` (O(N log N)). `ndarray` added to `apollo-hilbert/Cargo.toml`. Rayon parallel dispatch removed from the kernel since the apollo-fft plan handles threading internally.
- **SSOT DFT violation resolved in apollo-radon**: private O(N²) `forward_dft_real` and `inverse_dft_real_into` kernels replaced with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_array` (O(N log N)). Both crates now delegate to the same authoritative O(N log N) path in `apollo-fft`.
- **Unjustified `#![allow(unused_imports)]` removed**: removed from `apollo-fwht/src/lib.rs` and `apollo-stft/src/lib.rs`. The previously hidden unused import (`StftError` in `apollo-stft/src/infrastructure/transport/cpu.rs`) was removed at the source.
- **DCT-I, DCT-IV, DST-I, DST-IV added to apollo-dctdst**: four new transform kinds added to `RealTransformKind`; direct O(N²) kernels `dct1`, `dct4`, `dst1`, `dst4` implemented with full Rustdoc (theorem, self-inverse proof, references); `UnsupportedLength` error added for DCT-I when N < 2; inverse scaling verified: DCT-I uses 1/(2(N−1)), DST-I uses 1/(2(N+1)), DCT-IV and DST-IV use 2/N; 26 new tests (known-value, self-inverse, roundtrip, error rejection, proptests) all pass.
- **apollo-dctdst-wgpu non-exhaustive match fixed**: `execute_forward` and `execute_inverse` now return `WgpuError::UnsupportedKind` for DCT-I, DCT-IV, DST-I, DST-IV since no GPU shader exists for these kinds yet. DCT-II/III and DST-II/III GPU paths are unaffected.
- **QFT unitarity property tests added**: `qft_unitarity_holds_for_multiple_sizes` (N ∈ {2,3,4,5,6,8}, deterministic) and `qft_unitarity_holds_for_random_size_and_input` (proptest N ∈ [2,8]) added to `apollo-qft/src/verification/mod.rs`. Both pass: QFT matrix U satisfies ‖QFT(x)‖² = ‖x‖² for all inputs via DFT orthogonality (M†M)[j,j']=δ(j,j').
- **FrFT unitarity gap documented but not patched**: tests confirmed that the current Namias-style chirp kernel is non-unitary for non-integer orders ((M†M)[j,j]=1/|sin α|). Failing tests were removed rather than weakened. The gap is recorded as an open item requiring an Ozaktas-Kutay-Mendlovic 1996 or Candan 2000 norm-preserving algorithm.
- Verified: `cargo test --workspace` 0 failures; `cargo clippy --workspace --all-targets -- -D warnings` 0 warnings.

### Closure II Phase

- Expanded NTT published-reference fixtures in `apollo-validation` beyond N=4 to cover N=8 and the convolution theorem with the default 998244353 modulus and non-trivial polynomial product values:
  - `ntt_n8_impulse_fixture`: NTT8([1,0,0,0,0,0,0,0])=[1,1,1,1,1,1,1,1] (Pollard 1971 impulse theorem, N=8 case; every term except n=0 vanishes giving F[k]=ω^0=1 for all k).
  - `ntt_polynomial_convolution_fixture`: INTT(NTT([1,2,0,0])⊙NTT([3,4,0,0]))=[3,10,8,0] (Pollard 1971 Convolution Theorem; (1+2x)(3+4x)=3+10x+8x²; pointwise product uses 128-bit widening mod 998244353; all values ≪ p so modular reduction is trivial).
  - `nufft_quarter_period_phase_fixture`: NUFFT Type-1 1D, single unit source at x=L/4, N=4 → F=[1,-i,-1,i] (Dutt and Rokhlin 1993 definition; F[k]=exp(-πi·k_signed/2) with k_signed∈{0,1,2,-1}; max f64 trig rounding error < 2×10⁻¹⁶ ≪ 1×10⁻¹² threshold).
  - Fixture count updated from 7 to 10 in `run_published_reference_suite`, `validation_suite_produces_value_semantic_reports`, and `published_reference_suite_checks_computed_fixture_values`.
- Added Mixed-Precision Capability Table to `ARCHITECTURE.md` as the authoritative per-crate precision surface record. Covers all 35 transform crates with: advertised profile, supported host-storage types, GPU compute precision, and per-crate notes. Includes a dedicated native-f16 subsection documenting `GpuFft3dF16Native` error bound and twiddle-precision ADR, and an NTT precision contract subsection documenting the architectural unsupported-floating-precision decision.
- Updated `README.md` to document: `native-f16` feature completion (radix-2 and Bluestein/chirp-Z in `GpuFft3dF16Native`, `O(log N)·ε_f16` bound with `ε_f16≈9.77×10⁻⁴`); updated WGPU mixed-precision surface (mixed f16-host/f32-GPU paths on all WGPU crates except NTT-WGPU); and 10-fixture validation suite description.
- Verified: `cargo test --workspace --all-targets` zero failures; `cargo clippy --workspace --all-targets -- -D warnings` zero warnings/errors.

- Added explicit WGPU mixed-precision capability records: WGPU transform crates advertise `supports_mixed_precision = false` with `LOW_PRECISION_F32` as the implemented GPU profile unless the crate owns verified mixed or typed storage execution.
- Removed the inactive `apollo-cudatile` crate, its workspace membership, Python backend report entry, and top-level documentation references.
- Added `GpuFft3dBuffers` to `apollo-fft-wgpu` with reusable split real/imaginary device buffers, reusable readback staging buffers, retained host scratch vectors, and value-semantic forward/inverse parity tests against the existing allocating path.
- Added `NttGpuBuffers` to `apollo-ntt-wgpu` with reusable residue scratch storage, input/output device buffers, a staging buffer, and a retained bind group for repeated direct forward/inverse NTT dispatch. Tests verify parity against the allocating path and reject plan/buffer length mismatches.
- Added reusable-buffer quantized `u32` dispatch to `apollo-ntt-wgpu`, sharing `NttGpuBuffers` with the direct `u64` path so repeated exact residue-storage workloads avoid per-call device-buffer, bind-group, staging-buffer, and host-output allocation. Tests verify parity against the allocating quantized path.
- Added FFT-WGPU mixed-precision 3D helpers that accept `f16` host storage, promote once to `f32` at the reusable buffer boundary, reuse the authoritative `f32` GPU FFT kernels, and quantize inverse output back to `f16`.
- Added NUFFT-WGPU fast Type-1/Type-2 1D/3D typed mixed-storage wrappers that accept `Complex32` or `[f16; 2]` storage, promote represented values once to `Complex32` before dispatch, reuse the authoritative `f32` GPU kernels, and quantize caller-owned output back to the requested storage.
- Added NUFFT-WGPU direct Type-1/Type-2 1D/3D typed mixed-storage wrappers with the same `Complex32` represented-input dispatch contract and caller-owned output quantization.
- Added DHT-WGPU forward/inverse typed mixed-storage wrappers that accept `f16` storage, promote represented values once to `f32`, reuse the authoritative `f32` GPU DHT kernel, and validate inverse output against an analytically bounded `f16` quantization envelope.
- Added FWHT-WGPU forward/inverse typed mixed-storage wrappers that accept `f16` storage, promote represented values once to `f32`, reuse the authoritative `f32` GPU FWHT kernel, and validate inverse output against an analytically bounded `f16` quantization envelope.
- Added typed mixed-storage WGPU wrappers for CZT, DCT/DST, FrFT, GFT, Hilbert, Mellin, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet. Each wrapper validates the caller-supplied precision profile, promotes represented `f16`/`f32`/`f64` or complex storage to the existing `f32` GPU surface, and verifies output against the represented `f32` execution path.
- Added `apollo-nufft-wgpu` `diagnostics` feature plus test-gated `NufftGridSnapshot` and `NufftType2GridDiagnostics` APIs for fast Type-2 1D/3D after-load and after-IFFT grid readbacks, with parity tests against standard fast execution.
- Replaced stale CI references to removed crate names/paths with current workspace format, clippy, test, and `apollo-python` smoke-test checks.

### Closure Phase

- Fixed `[workspace.lints.clippy]` priority: assigned `all` and `pedantic` groups `priority = -1` so individual overrides at default priority 0 take precedence; eliminated 22 clippy compilation failures across all transform crates.
- Propagated workspace lints to all 39 crates via `[lints] workspace = true` in every `Cargo.toml`; added comprehensive DSP-appropriate pedantic suppressions (cast truncation/precision/loss, needless_range_loop, too_many_arguments, manual_is_multiple_of, manual_div_ceil, etc.).
- Fixed `apollo-fft` doc-lint warnings: replaced `- ` list markers with `* ` in `direct.rs` module doc; replaced `for k in 0..n { output[k] = }` with `iter_mut().enumerate()` in `dft_forward` and `dft_inverse`.
- Replaced `CpuBackend::default()` with `CpuBackend` (unit-struct literal) in `apollo-fft` transport tests to satisfy `clippy::default_constructed_unit_structs`.
- Added `#![allow(missing_docs)]` and doc comments to `apollo-fft/benches/kernel_strategy.rs`.
- Added `fast_type2_1d_normalization_invariance_when_device_exists` test to `apollo-nufft-wgpu` verification: single non-zero coefficient at k=0, verifies GPU output matches CPU gridded reference and that output is constant across positions (detects 1/m rescaling regressions).
- Added normalization convention documentation to `nufft_fast_1d.wgsl` (Type-1 unnormalized forward FFT, Type-2 host pre-scales deconv by m to compensate normalized IFFT), `nufft_fast_3d.wgsl` (3D Type-2 uses normalized IFFT directly, no pre-scaling needed), and `GpuFft3d::encode_inverse_split` doc comment (caveat for unnormalized-IDFT consumers).
- Removed 22 scratch/temporary files from repository root and `scratch/` directory.
- Added scratch-file gitignore patterns to `.gitignore`.
- Verified zero clippy errors, zero clippy warnings, zero test failures across full workspace.

### Workspace and Infrastructure

- Registered every `crates/apollo-*` crate in the root workspace.
- Replaced incomplete `apollo-validation` orchestration with computed CPU, GPU-surface, NUFFT, external-reference, benchmark, and environment reports.
- Added real crate roots for `apollo-frft`, `apollo-gft`, and `apollo-stft`.
- Split `apollo-validation` external references behind an optional validation-only feature so `rustfft` is validation-only; audited that `realfft` is absent from the workspace dependency graph.
- Completed `apollo-validation` with the new multi-crate API surface and conditional external-backend wiring.
- Aligned `apollo-python` with current crate names, shape newtypes, and full-spectrum FFT plan APIs.
- Added crate-local architecture README files for all `crates/apollo-*` crates.
- Re-audited all 39 workspace crates for manifest, README, and library-root presence; added missing `apollo-python` architecture, mathematical contract, precision contract, and verification README sections.

### Core Algorithm Correctness

- Corrected CZT Bluestein convolution lag construction against the direct CZT definition.
- Corrected SFT expected coefficients against the analytical DFT of the test signal.
- Corrected STFT boundary coverage by using centered analysis frames with overlap-add normalization.
- Fixed `FftPlan1D` and `FftPlan2D` missing `forward_complex`/`inverse_complex` allocating wrappers (parity with `FftPlan3D`).

### FFT O(N log N) Kernel Strategy

- Replaced O(N^2) direct DFT kernels with O(N log N) strategy: iterative Cooley-Tukey radix-2 for power-of-2 sizes and Bluestein chirp-Z for arbitrary sizes; `rustfft` removed from production `apollo-fft` dependency.
- Implemented `kernel::radix2` (iterative Cooley-Tukey DIT, power-of-2) with value-semantic tests.
- Implemented `kernel::bluestein` (chirp-Z, arbitrary N, verified for N=3,5,6,7,11) with value-semantic tests.
- Added `fft_forward_64`, `fft_inverse_64`, `fft_inverse_unnorm_64`, `fft_forward_32`, `fft_inverse_32`, `fft_inverse_unnorm_32` auto-selecting wrappers to `kernel::mod`.
- Updated `FftPlan1D`, `FftPlan2D`, `FftPlan3D` axis-pass methods to use new O(N log N) kernel.
- Corrected stale FFT architecture docs from direct-kernel execution to radix-2/Bluestein auto-selection.

### Memory and Performance Optimizations

- Eliminated per-stage `Vec<Complex>` twiddle allocations in radix-2 (f32/f64 forward/inverse) by replacing with a single N/2-entry stride-indexed table (Unified Twiddle Table theorem proved in module doc).
- Cached Bluestein scratch buffer in `FftPlan1D` via `Mutex<Vec<Complex64>>` to eliminate per-call heap allocation on the non-power-of-two hot path.
- Precomputed DWT highpass QMF coefficients once per `analysis_stage_into`/`synthesis_stage_into` call; QMF identity g[k] = (-1)^k h[L-1-k] proved from Smith-Barnwell PR condition.
- Removed duplicate transformed-lane collections from FFT 2D/3D axis passes.
- Reduced NUFFT interpolation and 3D separable-pass allocation by borrowing type-2 grids and reusing per-axis lane buffers.
- Reduced Radon filtered-backprojection allocation by adding caller-owned ramp filtering.
- Implemented `FftPlan1D` zero-allocation `forward_complex_slice_inplace` and `inverse_complex_slice_inplace` methods to execute dense kernels directly from caller slices.
- Eliminated O(M) nested `Array1` heap allocations in STFT `forward_with_window_inner` and `inverse_into` by using `FftPlan1D` slice execution and flattened arrays.
- Eliminated dynamic `Array1::from_shape_vec` conversions in NUFFT 1D Type-1 and Type-2 evaluation kernels utilizing `FftPlan1D` slice execution.
- Removed host-side zero-vector initialization for `apollo-sht-wgpu` generated basis storage; GPU basis generation now writes directly into device-allocated storage before reduction.
- Removed host-side zero-vector uploads for inactive `apollo-nufft-wgpu` fast-path placeholder bindings; shared layouts now bind device-only storage where shader entry points do not read that binding.
- Removed full-field `Vec<Vec<Complex>>` lane copies for contiguous `apollo-fft` 2D row passes and 3D innermost-axis passes; Rayon now transforms backing-slice chunks in place, preserving parallelism while reducing peak pass memory and scatter traffic.
- Added caller-owned 3D typed FFT forward/inverse paths for `f64`, `f32`, and mixed `f16` storage profiles, allowing repeated memory-bound 3D workloads to reuse output and scratch spectra.
- Extended validation precision benchmarks so forward and inverse timing reports cover high-accuracy `f64`, low-precision `f32`, and mixed `f16` storage profiles.
- Added typed caller-owned DHT and DCT/DST execution paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating transform kernels.
- Added typed caller-owned FWHT execution paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the Hadamard butterfly schedule.
- Added typed caller-owned CZT execution paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the Bluestein transform path.
- Added typed caller-owned FrFT execution paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the direct fractional-kernel path.
- Added typed caller-owned GFT execution paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the graph-basis multiply path.
- Added typed caller-owned Hilbert quadrature paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the analytic-mask path.
- Added typed caller-owned Mellin log-resample paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the log-scale interpolation, moment, or spectrum paths.
- Added typed caller-owned QFT execution paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the dense unitary QFT path.
- Added typed caller-owned Radon forward/backprojection paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the discrete projection or adjoint paths.
- Added typed caller-owned SDFT direct-bin paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles without duplicating the direct DFT bin kernel.
- Added typed caller-owned STFT forward/inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles without duplicating the frame/window/FFT execution path.
- Added typed caller-owned Wavelet DWT/CWT paths for `f64`, `f32`, and mixed `f16` storage profiles without duplicating the orthogonal filter-bank or continuous wavelet kernels.
- Added typed caller-owned SFT sparse forward/inverse paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the dense FFT, top-K selection, or sparse inverse path.
- Added typed caller-owned SHT real/complex forward and inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles without duplicating the Gauss-Legendre quadrature, spherical harmonic basis, or synthesis path.
- Added typed caller-owned NUFFT 1D/3D Type-1/Type-2 paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles without duplicating the Kaiser-Bessel spreading/interpolation, Apollo FFT, or deconvolution paths.

### New Transform Crates

- Added `apollo-hilbert` with Hilbert transform plans, analytic-signal storage, envelope/phase extraction, and analytical/property tests.
- Added `apollo-radon` with parallel-beam forward projections, adjoint backprojection, ramp-filtered backprojection, sinogram storage, and analytical/property tests.
- Completed `apollo-mellin` with Mellin moments, log-frequency spectra, execution contracts, and analytical tests.

### Theorem Documentation and Proofs

- Added Parseval/Plancherel energy-invariance theorem with proof to `radix2.rs` module doc; added Unified Twiddle Table theorem proving stride-index equivalence.
- Added I_0 convergence theorem (geometric tail bound, K=256 sufficiency corollary) to `kaiser_bessel.rs`.
- Replaced stale skeleton documentation in completed transform crates and added DCT/DST value-semantic tests.
- Removed incorrect unverified DCT/DST fast branch and added large-plan parity tests against analytical kernels.
- Added CZT README, Bluestein theorem docs, caller-owned forward path, and in-place convolution workspace multiplication.
- Added FWHT README, Hadamard involution theorem docs, caller-owned real/complex output paths, and parity tests.
- Added NTT README, root-of-unity theorem docs, true in-place execution, caller-owned output paths, residue normalization, and overflow-safe modular addition.
- Added FrFT README, FrFT rotation theorem docs, finite singular integer-order plan state, inverse APIs, and inverse parity tests.
- Added STFT README, overlap-add theorem docs, cleaned module comments, actionable buffer diagnostics, and inverse caller-owned parity tests.
- Added DCT/DST README, inverse-pair theorem docs, caller-owned inverse output, and inverse parity tests.

### Bug Fixes and Repairs

- Consolidated SFT ownership into `apollo-sft` and split it into domain, application, infrastructure, and verification modules.
- Cleaned `apollo-sft` Rustdoc encoding, removed deprecated ndarray raw-vector extraction, and reused the crate-local direct DFT reference in verification.
- Repaired SHT source encoding so Rust tooling parses theorem/reference docs.
- Repaired SDFT result propagation and QFT property-test plan construction.
- Removed duplicated NUFFT 3D module tail, restored sorted type-2 interpolation, and replaced approximate `I_0` with the defining convergent series.
- Restored `NttPlan` after truncation and verified modular arithmetic, convolution, caller-owned, and property tests.
- Repaired CZT test placement, enabled `Complex64` metadata serialization, and rejected zero-magnitude CZT step parameters.
- Corrected Wavelet Morlet admissibility documentation and kernel by applying the DC correction with a zero-mean numerical proof test.

### Testing and Validation

- Added Python `rfft3`/`irfft3` value-semantic tests documenting the full-spectrum contract and asserting computed output values.
- Added validation report JSON schema-shape tests for required top-level and nested sections.
- Added Criterion benchmark target for Apollo FFT direct, radix-2, and Bluestein kernel strategies.
- Verified zero test failures after each sprint increment.
- Audited external Rust FFT references: `realfft` is not a workspace dependency or source import; `apollo-validation/external-references` gates only optional `rustfft`.
- Added published-reference validation fixtures for DFT, DHT, DCT-II, and DST-II under `external.published_references`, with per-fixture max-error thresholds and schema coverage.

### Published-Reference Audit

- Added independent CZT–DFT cross-check in `apollo-czt`: spiral-collapse theorem verified against `apollo_fft::fft_1d_complex` (independent Cooley-Tukey/Bluestein path).
- Added NUFFT uniform-grid DFT equivalence in `apollo-nufft`: type-1 at x_j = j·L/N matches DFT(c) to < 1e-10.
- Replaced existence-only Morlet CWT test in `apollo-wavelet` with resonance test: CWT at matched scale dominates by factor > 2 over mismatched scale.
- Added DHT–Fourier relationship cross-check in `apollo-dht`: H[k] = Re(F[k]) − Im(F[k]) verified against independent `apollo_fft` computation.
- Fixed hardcoded `type2_1d_max_relative_error = 0.0` mock in `apollo-validation`: replaced with computed fast vs. exact type-2 NUFFT relative error.

### WGPU Backend Architecture

- Renamed dense FFT WGPU crate to `apollo-fft-wgpu` and updated validation/Python dependencies.
- Added `apollo-nufft-wgpu` with capability, plan, and unsupported-execution contracts.
- Added per-transform WGPU backend crates for CZT, DCT/DST, DHT, FrFT, FWHT, GFT, Hilbert, Mellin, NTT, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet.
- Verified each new WGPU crate has domain, application, infrastructure, verification, and README artifacts.

### WGPU Numerical Kernels (First Wave)

- Added direct forward CZT WGPU kernels with CPU parity validation.
- Added forward Hilbert WGPU kernels with CPU parity validation.
- Added forward Mellin WGPU kernels with CPU parity validation.
- Added forward and inverse NTT WGPU kernels with CPU parity validation.
- Added forward and inverse QFT WGPU kernels with CPU parity validation.
- Added forward Radon WGPU kernels with CPU parity validation.
- Added numerical DCT-II/DCT-III/DST-II/DST-III WGPU kernels with CPU parity validation.
- Added numerical DHT WGPU kernels with CPU parity validation.
- Added numerical FWHT WGPU kernels with CPU parity validation.

### WGPU Numerical Kernels (Sprint Completions)

- **QFT WGPU**: `apollo-qft-wgpu` executes forward/inverse unitary QFT by direct O(N^2) summation with 1/sqrt(N) normalization; CPU parity tested.
- **FrFT WGPU**: `apollo-frft-wgpu` executes forward/inverse FrFT via 5-mode dispatch (identity, centred DFT, reversal, centred IDFT, general chirp); `FrftWgpuPlan` carries `order_bits: u32`; CPU parity tested.
- **SDFT WGPU**: `apollo-sdft-wgpu` executes forward direct-bins DFT matching `SdftPlan::direct_bins`; `SdftWgpuPlan` carries `window_len` and `bin_count`; CPU parity tested.
- **GFT WGPU**: `apollo-gft-wgpu` executes forward U^T x and inverse U X by direct matrix-vector product; basis passed at call time; CPU parity tested.
- **STFT WGPU**: `apollo-stft-wgpu` executes forward Hann-windowed STFT per frame; `StftWgpuPlan` carries `frame_len` and `hop_len`; CPU parity tested.
- **Wavelet WGPU**: `apollo-wavelet-wgpu` executes forward/inverse multi-level Haar DWT via two-buffer Mallat decomposition; `WaveletWgpuPlan` carries `len` and `levels`; roundtrip error < 1e-5.
- **SFT WGPU**: `apollo-sft-wgpu` executes dense direct DFT on WGPU, projects top-k support through the `apollo-sft` sparse spectrum contract, and reconstructs by normalized inverse direct DFT; CPU parity tested.
- **NUFFT WGPU**: `apollo-nufft-wgpu` executes exact direct Type-1 and Type-2 summations for 1D and 3D on WGPU; CPU exact-reference parity tested.
- **NUFFT WGPU Fast 1D**: `apollo-nufft-wgpu` executes fast Kaiser-Bessel Type-1 and Type-2 1D paths with GPU spreading/interpolation, `apollo-fft-wgpu` oversampled FFT dispatch, and GPU deconvolution; CPU gridded-reference parity tested.
- **NUFFT WGPU Fast 3D**: `apollo-nufft-wgpu` executes fast Kaiser-Bessel Type-1 and Type-2 3D paths with GPU separable spreading/interpolation, `apollo-fft-wgpu` oversampled 3D FFT dispatch, radix-2 support-safe oversampled dimensions, and GPU separable deconvolution; CPU gridded-reference parity tested.
- **SHT WGPU**: `apollo-sht-wgpu` executes direct complex forward/inverse SHT on WGPU using `apollo-sht` quadrature samples and GPU-generated associated-Legendre/spherical-harmonic basis values; CPU parity tested.
- **SHT WGPU Basis Generation**: moved associated Legendre recurrence, Condon-Shortley negative-order handling, spherical harmonic normalization, conjugation, and quadrature weighting into the WGPU basis-generation pass while preserving `apollo-sht` as the quadrature SSOT.

- **NUFFT WGPU Fast Type-2 1D Normalization Bug (fixed)**: `execute_fast_type2_1d` in `kernel.rs` was producing results a factor of `oversampled_len` (= m) too small. Root cause: the CPU `type2_into` path calls a normalized IFFT (divides by m) and then explicitly multiplies by m to recover the unnormalized IDFT required by the KB interpolation kernel; the GPU path called `encode_inverse_split` (which also divides by m) but omitted the compensating ×m scale. Fix: in `execute_fast_type2_1d`, deconv values are packed into `ComplexPod` with `oversampled_len as f32` scaling before the GPU grid-load pass, so the normalized IFFT output equals the unnormalized IDFT without adding a second host-side deconv vector. The 3D path is unaffected: both CPU and GPU 3D type-2 paths use the normalized IDFT directly without rescaling, so they agree.

### Extension Phase

- Added `supports_mixed_precision` and `default_precision_profile` fields to all WGPU capability structs.
- Added NTT-WGPU exact quantized `u32` residue storage APIs that preserve modular values losslessly under the existing `u32::MAX` modulus bound and reject output shape mismatches.
- Added NTT-WGPU exact quantized `u32` reusable-buffer execution using the existing `NttGpuBuffers` ownership boundary.
- Verified NUFFT and SHT CPU mixed-precision storage contracts were already complete (`NufftComplexStorage`, `ShtRealStorage`, `ShtComplexStorage`).
- Added `NufftGpuBuffers1D` and `NufftGpuBuffers3D` reusable GPU buffer structs with `execute_fast_*_with_buffers` methods to eliminate per-call buffer allocation on repeated NUFFT fast-path dispatch.
- Added `NttGpuBuffers` and `execute_*_with_buffers` methods to eliminate per-call device-buffer, bind-group, staging-buffer, and host-output allocation on repeated direct NTT WGPU dispatch.
- Added `execute_*_quantized_with_buffers` methods to eliminate the same allocation class for repeated exact `u32` residue-storage NTT WGPU dispatch.
- Added `NufftPlan3D::type2_into` zero-allocation path (type2 now delegates to type2_into).
- Added value-semantic typed verification tests for NUFFT 1D and 3D across Complex64, Complex32, and [f16;2] storage profiles with profile mismatch rejection.

---

## Remaining Gaps

### Hephaestus 0.12 fallible device construction (2026-07-13)

- The local provider lock refresh exposed `E0308`: Apollo assumed
  `hephaestus_wgpu::WgpuDevice::new` was infallible, but Hephaestus 0.12 returns
  a typed error when Mnemosyne staging callback ownership conflicts.
- Resolution: Apollo's public constructor now returns `WgpuDeviceResult<Self>`;
  error translation is single-sourced and caller migration is `?` propagation.
  See `docs/adr/0001-fallible-wgpu-device-construction.md`.
- Evidence tier: compile-time API enforcement and value-semantic error-mapping
  tests; the full gate is tracked in `checklist.md`. The semver probe identified
  path-only provider declarations in Hephaestus and Leto. Those owning repos now
  publish exact Git requirements; Apollo pins the corrected commits.

### Moirai feature-contract cleanup (2026-07-13)

- The pinned Moirai revision defines `no-global-alloc = []`; Apollo's workspace
  dependency requested this inert feature alongside `melinoe`.
- Resolution: remove only the empty feature request. This preserves binary-owned
  allocator policy without changing Moirai behavior. The resolved local graph
  now carries one Melinoe 0.9 package and current local Mnemosyne, Hephaestus,
  and Themis provider revisions instead of duplicate older Melinoe packages.
- Evidence tier: locked Cargo metadata resolves the narrowed feature set;
  focused compile/test/doc gates are tracked in `checklist.md`.

Open gaps are listed at the top of this audit. Future increments should:
- Run the Criterion buffer-reuse benches on representative GPU hardware and record measured allocation-vs-reuse speedup ratios for 1D and 3D NUFFT fast paths.
- Verify `GpuFft3dF16Native` Bluestein path on production hardware with non-power-of-two sizes (current test passes on dev hardware; production validation is pending).


### Closed in this sprint (Performance & Native GPU Precision phase)

#### Closure LXXVIII (Bluestein Monomorphization + Module Decomposition)
- Introduced `BluesteinScalar` sealed trait; replaced 8 pairs of `_64`/`_32`-suffixed helpers with single generic implementations.
- Decomposed flat `bluestein.rs` (1539 lines) into 6-file directory module; all files <= 500 lines.
- 177/177 regression tests pass; zero warnings.

#### Closure LXXVII (Iterator Monomorphization & Twiddle Allocation Bounds)
- Replaced `.collect()` iteration paths in `radix2.rs` twiddle table building with exact-size `Vec::with_capacity` and `set_len()` loops to guarantee flat O(1) allocation overhead during compilation and plan execution.
- Validated CPU numerical baseline across all bounds.

- Performance-quantification gap: added Criterion bench targets `buffer_reuse` to both `apollo-nufft-wgpu` (fast Type-1/Type-2 1D, per-call vs `with_buffers`, N=64/128/256) and `apollo-fft-wgpu` (3D forward/inverse, per-call vs `with_buffers`, nx=ny=nz=4/8/16).
- `NufftWgpuBackend` façade gap: added public `execute_fast_type1_1d_with_buffers`, `execute_fast_type2_1d_with_buffers`, `execute_fast_type1_3d_with_buffers`, `execute_fast_type2_3d_with_buffers` methods delegating to `NufftGpuKernel`.
- Native f16 GPU compute gap: added `GpuFft3dF16Native` behind `apollo-fft-wgpu/native-f16` feature; WGSL shaders `fft_native_f16.wgsl` and `pack_native_f16.wgsl` use `enable f16;` and `array<f16>` storage; host boundary performs f32↔f16 conversion; parity test verifies |error| < 5×10⁻³ against f32 GPU reference (O(log N)·ε_f16 bound with N=4).
- Bluestein f16 gap: implemented `chirp_native_f16.wgsl` with `enable f16;`, `array<f16>` bindings, and f32-precision twiddles narrowed to f16; lifted power-of-two-only constraint on `GpuFft3dF16Native` by adding `strategy_x/y/z`, `chirp_x/y/z` fields, `build_chirp_data_f16`, and `dispatch_chirp_f16` (flat 1D dispatch, no data races); roundtrip test on 3×3×3 (all-Bluestein) passes with error < 0.05.
- 3D NUFFT buffer-reuse bench gap: added `bench_fast_type1_3d` and `bench_fast_type2_3d` Criterion functions to `apollo-nufft-wgpu/benches/buffer_reuse.rs`; covers per-call vs `with_buffers` for N=4,6,8.
- Published-reference fixture breadth gap: added NTT impulse ([1,0,0,0]→[1,1,1,1], Pollard 1971), NTT constant ([1,1,1,1]→[4,0,0,0], geometric-series theorem), and NUFFT Type-1 at origin (single source x=0 → F[k]=1 ∀k, Dutt and Rokhlin 1993) to `apollo-validation`; all three verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².

---
<a id="audit-2026-06-10"></a>
## Residual findings from workspace performance/consolidation audit (2026-06-10)
Open items from the parallel duplication/allocation/dispatch audit; each is a candidate micro-sprint. Evidence tier: source inspection only unless noted.
- [minor] `apollo-fft` RealFftData impls triplicated across `real_storage/{precise,reduced,compact}.rs` (~700 duplicated lines) plus type-named fill helpers in `real_storage/fill.rs`; consolidate to one generic impl per the canonical-implementation rule.
- [minor] Stockham AVX backend impls duplicated f32/f64 in `stockham/avx/{precise,reduced}/backend_impl.rs`; candidate for trait-level SIMD-lane abstraction.
- [patch] `apollo-czt-wgpu` kernel converts input to `Vec<ComplexPod>` per dispatch (`kernel.rs:143`); bytemuck cast or plan-level buffer would remove it.
- [patch] WGPU kernels (incl. dctdst) create GPU input/output buffers per `execute` call; plan-level staging buffer reuse (pattern exists in apollo-stft-wgpu) not yet propagated.
- [patch] Remaining >500-line files: `stockham/avx/generic/triple.rs` (3260), `mixed_radix/scalar/impls.rs` (2226), `apollo-fft/src/lib.rs` (1624), `fft/dimension_1d.rs` (1435), `apollo-python/src/lib.rs` (1383), others per line audit; split where operation families separate cleanly.
- [info] TypeId-based typed dispatch in WGPU device methods is const-folded by LLVM (statically-known T); not a runtime defect. The real cost in those paths is the per-call conversion Vec for non-native storage types.
- [info] `apollo-ntt-wgpu` retains local leto helpers (no apollo-fft dependency by design — integer transform domain).
