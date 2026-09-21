# 0069 — Lightweight direct DCT kernel ownership

- Status: Accepted
- Date: 2026-09-21
- Item: `backlog.md#apollo-dct-core-001`

## Context

`apollo-dctdst` owns the DCT-III formula, but the package also requires Apollo's
FFT, array, parallel-execution, memory, and accelerator integration stack.
Consus reconstructs JPEG blocks with the same DCT-III basis and cannot take
that dependency closure without coupling a bounded raster codec to unrelated
execution infrastructure. Its cached eight-point basis also avoids evaluating
cosines for every decoded block, while Apollo's direct slice kernel currently
recomputes them on every call.

The current typed DCT/DST path converts `f32` and reduced-precision storage to
`f64` before execution. That storage adapter is not a valid arithmetic contract
for a shared scalar-generic kernel: `f32` computation must remain `f32`.

## Decision

Add `apollo-dctdst-core`, a lightweight crate that owns direct DCT mathematics.
It depends only on Eunomia's scalar vocabulary. The first slice provides one
generic, fixed-capacity, precomputed DCT-III plan over `T: RealField` and
`const N: usize`, with unnormalized and orthonormal normalization. Construction
evaluates the basis once; execution applies the stored matrix into caller-owned
arrays without allocation or transcendental operations.

The orthonormal contract is

```text
y[j] = X[0]/sqrt(N)
     + sqrt(2/N) * sum(k=1..N-1, X[k] cos(pi*k*(j+1/2)/N)).
```

At `N = 8`, this is the one-dimensional inverse transform in JPEG T.81
section A.3.3. Apollo's existing unnormalized convention remains
`X[0]/2 + sum(...)`.

The same private coefficient function constructs both the precomputed plan and
the dynamic slice operation. `apollo-dctdst` routes its direct DCT-III entry
point through that operation and removes its scalar and Hermes-basis copies.
Other DCT/DST kinds remain in `apollo-dctdst` until their own bounded migration;
this slice does not duplicate or redesign them.

## Alternatives

- Depending on the full `apollo-dctdst` package from a raster codec was
  rejected because it imports unrelated FFT, array, runtime, memory and GPU
  integration.
- Keeping a JPEG-specific cosine table was rejected because it leaves two
  authoritative DCT-III equations.
- Widening generic inputs to `f64` was rejected because it makes `f32` a storage
  parameter rather than the arithmetic precision selected by the caller.
- Computing the basis on every block was rejected because the block size and
  transform kind are invariant across decoding.

## Consequences

The dependency direction is `apollo-dctdst -> apollo-dctdst-core -> eunomia`.
Format owners may depend on the core without importing Apollo's execution
stack. The core admits `f32` and `f64` through `RealField`; reduced-precision
storage stays outside this kernel until it has a native arithmetic contract.

The fixed-capacity plan stores `N * N` scalar weights. This is appropriate for
small direct transforms such as JPEG's eight-point blocks. Variable-length and
FFT-derived plans remain in `apollo-dctdst`.

## Verification

- Instantiate one generic suite for `f32` and `f64`.
- Check the unnormalized and orthonormal basis against analytical impulse
  values, including the DC and first AC terms at `N = 8`.
- Check orthonormal row inner products against the identity with error bounds
  derived from `N` and the scalar epsilon.
- Check the cached plan against the dynamic direct operation.
- Run all existing `apollo-dctdst` regressions after routing DCT-III through the
  core.
