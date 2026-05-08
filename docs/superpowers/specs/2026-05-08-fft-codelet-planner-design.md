# Apollo FFT Generated Codelet Planner Design

Status: proposed for implementation after review.
Date: 2026-05-08.
Scope: `apollo-fft` CPU power-of-two and smooth-composite execution, with a
matching WGPU strategy boundary for generated shader leaves.

## Goal Extraction

Deliverable: replace the current Stockham-only CPU performance strategy with a
research-backed generated codelet planner that can beat RustFFT on same-host
Criterion runs while preserving Apollo's public natural-order, no-bit-reversal
contract.

Success criteria:

- No standalone bit-reversal, bit-reversed transpose, or permutation pass is
  introduced in production Apollo FFT execution.
- Hot paths allocate zero heap memory and use caller- or plan-owned scratch.
- Arithmetic remains native to the selected scalar precision. Generic codelets
  do not widen `T` to another concrete scalar type for convenience.
- Codelet selection is monomorphized by scalar type, layout, ISA, radix chain,
  and leaf length through traits, associated types, const generics, and sealed
  strategy types.
- A generated plan is enabled by default only after same-run Criterion shows it
  is faster than both the current Apollo path and RustFFT for the targeted size
  family.
- `cargo asm` promotion requires no scheduler-visible bit reversal, no
  non-ABI vector spills in generated leaf hot loops, and no bounds-check or
  divide panics in the loop body.

Non-goals:

- Do not add compatibility shims, public suffix APIs, or legacy radix facades.
- Do not replace the domain `FftStorage<T>` contract.
- Do not use runtime graph traversal, dynamic trait dispatch, or heap-allocated
  kernel descriptors inside butterfly loops.
- Do not enable split-radix, VLUT, or graph-search variants unless benchmarks
  and assembly prove they improve the real hot path.

## Research Basis

FFTW's codelet generator is the closest proven architecture. Its `genfft`
pipeline emits a directed acyclic graph for a DFT, applies algebraic
simplification, schedules the result, and prints imperative code. This directly
matches Apollo's current failure mode: handwritten Stockham leaves improve local
loops, but the planner still forces full-buffer traversals where generated
fixed leaves would keep work in registers or L1. Source:
[FFTW code-generation documentation](https://www.fftw.org/doc/Generating-your-own-code.html).

Johnson and Frigo's FFT practice chapter emphasizes that high-performance FFTs
are dominated by memory hierarchy, large radices, planner choices, and
generated kernels rather than textbook operation counts. The same chapter
identifies explicit bit reversal as avoidable. Source:
[arXiv:2602.23525](https://arxiv.org/pdf/2602.23525).

SPIRAL models FFT implementation as a search over mathematically equivalent
formulas and maps the selected formula to SIMD code. This supports a planner
that searches codelet decompositions instead of encoding one fixed traversal.
Source: [SPIRAL Proceedings IEEE paper](https://www.cs.cmu.edu/~mmv/papers/05SpiralIEEE.pdf).

Recent graph-search FFT work models SIMD FFT schedule selection as an
optimization problem and reports that larger FFT leaves can lose when register
pressure spills twiddles. This validates Apollo's observed rejected radix-16
attempts and makes spill-free live-range control an acceptance gate rather than
an implementation detail. Source:
[arXiv:2604.04311](https://arxiv.org/abs/2604.04311).

GPU Stockham work reaches the same practical constraint from another backend:
radix and fusion choices are bounded by register pressure, local memory, and
scattered access. Source:
[arXiv:2603.27569](https://arxiv.org/abs/2603.27569).

Modified split-radix reduces arithmetic count, but Apollo should use it only
inside generated register leaves. As a whole-buffer traversal replacement it can
increase irregular memory pressure, which is the current measured bottleneck.
Source: [Johnson and Frigo 2007](https://math.mit.edu/~stevenj/papers/JohnsonFr07.pdf).

## Formal Specification

Let `N = A * B`. The DFT is

```text
X[k1 + A*k2] =
  sum_{n2=0}^{B-1} sum_{n1=0}^{A-1}
    x[n1 + A*n2] * W_N^((n1 + A*n2) * (k1 + A*k2)).
```

By separating powers of `W_N`, this decomposes into:

```text
inner_A[n2, k1] = sum_{n1=0}^{A-1} x[n1 + A*n2] * W_A^(n1*k1)
twiddled[n2, k1] = inner_A[n2, k1] * W_N^(n2*k1)
outer_B[k1, k2] = sum_{n2=0}^{B-1} twiddled[n2, k1] * W_B^(n2*k2)
```

Therefore, any generated leaf that computes the same `A`-point DFT and any
outer codelet chain that computes the same `B`-point DFT is observationally
equivalent to the original `N`-point DFT when twiddle exponents match the
factorization. Local register transposes are permitted because they are basis
changes internal to the leaf. A standalone memory permutation is rejected unless
it is part of a tiled matrix transpose required by the six-step algorithm and
the output order remains natural.

Inverse normalization contract: forward transforms remain unnormalized;
inverse transforms apply exactly one final `1/N` normalization. Intermediate
leaves never apply hidden normalization.

## Architecture

Domain layer:

- Keep `FftStorage<T>` as the canonical storage boundary.
- Add domain-visible shape descriptors only if they are scalar/layout neutral:
  `TransformLen<const N: usize>`, `RadixChain<const K: usize>`, and
  `NativePrecision`.
- Do not import ISA modules, WGPU, Criterion, or generated infrastructure.

Application layer:

- Replace single-path Stockham scheduling with a `PlanDag<T, L, P>` model whose
  nodes are mathematical operations: `GeneratedLeaf<N>`, `MixedRadix<R>`,
  `StockhamStage<S>`, `TwiddleApply`, and `MatrixTransposeTile`.
- Use a Viterbi-style shortest-path planner over legal decompositions. Edge
  costs come from checked-in stage-isolation benchmarks and are keyed by scalar
  type, ISA, length, layout, and cache footprint.
- Store the selected plan as a closed enum, not `dyn Trait`. Dispatch occurs at
  plan-node boundaries, never inside butterfly loops.
- Keep Stockham autosort as a legal schedule node, but stop requiring every
  power-of-two transform to be executed as full-buffer Stockham passes.

Infrastructure layer:

- Add `apollo-fft-codegen` as a development/build tool that emits checked-in
  Rust and WGSL leaf modules. Production builds consume generated source and do
  not run an optimizer at runtime.
- Generated CPU leaves implement a sealed `Codelet<T, Layout, Isa, const N>`
  trait. Concrete public names do not encode `f32`, `f64`, `avx`, or old radix
  facades.
- Generated WGPU leaves use the same mathematical schedule descriptors and a
  backend-specific emitter. Native `SHADER_F16` codelets use `enable f16;` and
  f16 storage/arithmetic, not f32 widening.
- Twiddle layouts are generated per schedule. Broadcast twiddles remain scalar
  broadcast tables. Vector-packed twiddle tables are emitted only when SIMD
  lanes carry distinct exponents.

## Codelet Strategy

Initial leaf set:

- f32 AVX2/FMA: `N = 32, 64, 128, 256, 512`.
- f64 AVX2/FMA: `N = 16, 32, 64, 128, 256, 512`.
- WGPU f16/f32: power-of-two leaves bounded by workgroup register pressure and
  `maxComputeWorkgroupStorageSize`.

Each generated leaf is constructed from a canonical DFT DAG:

- Apply algebraic CSE before schedule selection.
- Prefer decimation choices that delay twiddle loads until the consumer branch.
- Store partial outputs as soon as the dependency cut is complete.
- Reject schedules whose live vector count exceeds the target ISA budget.
- Reject schedules whose assembly shows non-ABI vector spills.

Split-radix is allowed only as a DAG candidate inside this generator. It is not
a separate public algorithm, and it is not selected by arithmetic count alone.

SVDAG lesson: use structural sharing at code-generation time to canonicalize
subgraphs and reduce repeated twiddle products. Do not add a runtime graph data
structure to the FFT hot path.

## Planner Strategy

For a runtime length `N`, the application planner computes legal decompositions
from:

- generated leaf sizes,
- mixed-radix factors `{3, 5, 7, 8, 9, 11, 12, 16}`,
- Stockham stages for residual power-of-two segments,
- six-step tiled matrix transforms for cache-bound large sizes.

The selected path minimizes:

```text
cost = cycles_leaf
     + cycles_twiddle
     + cycles_transpose
     + bytes_moved / measured_bandwidth
     + spill_penalty
     + dispatch_penalty
```

The planner must prefer no-copy plans. Copy-back is allowed only when ping-pong
parity requires it and measured A/B proves it beats a prepass or extra
traversal.

## Verification Plan

Correctness:

- Analytical DFT fixtures for every generated leaf length.
- Forward parity against the current scalar Stockham recurrence.
- Inverse roundtrip with exactly one `1/N` normalization.
- Property tests over impulse, constant, sinusoid, conjugate symmetry, and
  adversarial non-aligned buffers.
- Cross-backend parity for CPU/WGPU supported precisions and layouts.

Source and architecture guards:

- Production source rejects `bitrev`, `bit_reverse`, and bit-reversal
  permutation symbols outside tests, comments, and third-party source.
- Generated code rejects `Vec`, `Box`, `RefCell`, `dyn Trait`, and allocation
  paths in hot modules.
- Generic codelet bodies reject `as f32`, `as f64`, `.to_f32()`, and `.to_f64()`
  except in documented non-generic twiddle construction.

Performance:

- Stage-isolation Criterion benchmarks for leaf-only, twiddle-only,
  transpose-only, and full-plan execution.
- Same-run RustFFT comparison for `N = 64, 256, 512, 1024, 4096, 8192, 65536`.
- `perf stat -d` collection for cycles, instructions, L1 misses, LLC misses,
  branches, and stalled cycles.
- `cargo asm` promotion report for every selected generated leaf.

Promotion gate:

- A generated path is default-enabled only when same-run median latency is at
  least 5% faster than RustFFT for the targeted size family and all correctness
  tests pass.
- If a generated path beats current Apollo but not RustFFT, it remains behind
  an internal benchmark-only strategy.
- If a generated leaf spills non-ABI vectors, the scheduler rejects that DAG and
  chooses a smaller leaf or earlier store schedule.

## Expected Failure Modes

- Large leaves can lose to smaller leaves through register spills. The planner
  must use assembly-derived spill penalties.
- Vector twiddle LUTs can increase memory traffic when lanes share an exponent.
  The emitter must prove lane-exponent diversity before using a packed twiddle
  table.
- Split-radix can reduce arithmetic but lose to regular radix schedules through
  irregular memory access. Selection requires measured edge cost.
- AVX-512 and f16 support can fragment the strategy space. Add them as `Isa`
  and `NativePrecision` implementations, not public method families.
- Checked-in generated code can increase repository size. The generator must
  emit deterministic source and a manifest hash for traceability.

## Implementation Plan Boundary

The first implementation plan should deliver one complete vertical slice:

1. Add the `apollo-fft-codegen` crate and manifest format.
2. Generate one f32 and one f64 AVX2/FMA leaf family for `N = 64, 128, 256`.
3. Add leaf-only correctness and assembly gates.
4. Add planner integration for power-of-two `N = 512, 1024, 4096`.
5. Benchmark against current Apollo and RustFFT.
6. Default-enable only the size families that pass the promotion gate.

This design intentionally defers AVX-512, native CPU f16, and WGPU generated
leaves until the CPU AVX2 planner proves the architecture.
