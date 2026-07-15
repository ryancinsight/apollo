# Apollo FrFT

`apollo-frft` owns the fractional Fourier transform reference implementation
for Apollo.

## Architecture

```text
src/
  domain/          length/order validation and error contracts
  application/     reusable FrFT plan and direct execution policy
  infrastructure/
    transport/
      cpu/         CPU convenience boundary
      gpu/         Hephaestus device boundary and typed kernels
```

`FrftPlan` is the single source of truth for transform length, fractional order,
rotation parameters, and direct-kernel execution.

## Mathematical Contract

The FrFT of order `a` rotates a signal in the time-frequency plane by

```text
alpha = a*pi/2
```

Integer quarter rotations are exact discrete operators:

- `a = 0 mod 4`: identity
- `a = 1 mod 4`: centered unitary DFT
- `a = 2 mod 4`: reversal
- `a = 3 mod 4`: centered unitary inverse DFT

Non-integer orders use the direct cotangent/cosecant kernel over centered
discrete coordinates.

## Execution Surfaces

- `forward` and `inverse` allocate returned arrays.
- `forward_into` and `inverse_into` write into caller-owned output buffers.
- `forward_typed_into` and `inverse_typed_into` support `Complex64`,
  `Complex32`, and mixed `[f16; 2]` storage profiles.
- With the `wgpu` feature, `FrftWgpuBackend` uses Hephaestus typed bindings
  and command streams for both the direct and unitary transforms. Leto views
  are the host boundary and returned arrays use Mnemosyne storage.

## Precision Contract

Typed execution uses Apollo's shared `PrecisionProfile` contract:

- `HIGH_ACCURACY_F64`: `Complex64` storage.
- `LOW_PRECISION_F32`: `Complex32` storage.
- `MIXED_PRECISION_F16_F32`: `[f16; 2]` storage, with lane 0 as real and lane
  1 as imaginary.

Lower storage profiles reuse the authoritative `Complex64` FrFT plan and
quantize once at the storage boundary. Profile/storage mismatch is rejected
with `FrftError::PrecisionMismatch`.

The optional accelerator contract is deliberately narrower: `FrftGpuStorage`
admits native `Complex32` and explicit `[f16; 2]` promotion only. `Complex64`
is excluded at compile time, so a GPU call cannot silently narrow a
high-accuracy CPU computation.

## Unitary Accelerator Theorem

The Candan--Gr\u00fcnbaum implementation evaluates

```text
DFrFT_a(x) = V diag(exp(-i a k pi / 2)) V^T x.
```

The Leto eigensolver constructs the real orthonormal basis `V`; therefore
`V^T V = I` and the phase diagonal has unit-modulus entries. Their product is
unitary, so `||DFrFT_a(x)||_2 = ||x||_2` and
`DFrFT_-a(DFrFT_a(x)) = x` in exact arithmetic. The accelerator encodes the
projection, phase, and reconstruction as three ordered Hephaestus passes;
each stream boundary provides the dependency ordering required by the next
pass. Real-device CPU differential, norm, reversal, and roundtrip tests are
the empirical evidence tier for `Complex32`; no machine-checked proof is
performed.

## Verification

The crate verifies identity order, continuity near the centered DFT boundary,
integer-order inverse reconstruction, caller-owned inverse parity, and invalid
plan rejection. The accelerator suite verifies direct CPU parity, unitary
identity/reversal/norm/roundtrip laws, Leto host views, and the sealed
`Complex32`/`[f16; 2]` storage contract.
