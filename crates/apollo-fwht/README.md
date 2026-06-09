# Apollo FWHT

`apollo-fwht` owns the fast Walsh-Hadamard transform implementation for Apollo.

## Architecture

```text
src/
  domain/          length validation and error contracts
  application/     reusable FWHT plan and in-place execution policy
  infrastructure/  CPU convenience wrappers
```

The plan is the single source of truth for the power-of-two length contract.
The kernel owns the butterfly schedule and selects Moirai parallel chunks for
large stages.

## Mathematical Contract

For `n = 2^m`, the transform applies the Sylvester Hadamard matrix `H_n`.
The entries are

```text
H_n[i,j] = (-1)^(popcount(i & j))
```

The identity `H_n H_n = n I` makes the unnormalized FWHT self-inverse up to
division by `n`. Apollo therefore uses the same butterfly kernel for forward
and inverse execution, with inverse scaling applied after the second transform.

## Execution Surfaces

- `forward` and `inverse` allocate returned arrays.
- `forward_into` and `inverse_into` copy into caller-owned output and then run
  in place.
- `forward_typed_into` and `inverse_typed_into` accept `f64`, `f32`, and mixed
  `f16` storage profiles while reusing the same generic butterfly schedule.
- `forward_inplace` and `inverse_inplace` execute with `O(1)` auxiliary storage.
- Complex variants use the same butterfly topology over `Complex64`.

## Precision Contract

Typed execution uses Apollo's shared `PrecisionProfile` contract:

- `HIGH_ACCURACY_F64`: `f64` storage with `f64` arithmetic.
- `LOW_PRECISION_F32`: `f32` storage with `f32` arithmetic.
- `MIXED_PRECISION_F16_F32`: `f16` storage with `f32` arithmetic and one
  quantization at the output boundary.

Profile mismatch is rejected with `FwhtError::PrecisionMismatch`; precision is a
storage/compute contract, not a duplicated algorithm family.

## Verification

The crate verifies analytical two-point output, real and complex roundtrips,
caller-owned parity against allocating paths, typed `f64`/`f32`/`f16` storage
paths, invalid length rejection, precision-profile mismatch rejection, and
property-based roundtrips over power-of-two lengths.
