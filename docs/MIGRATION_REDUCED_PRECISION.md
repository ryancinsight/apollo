# Reduced-precision storage migration

Apollo's compact reduced-precision storage now uses Eunomia's owned types. The
old `apollo_fft::f16` export and every `half::f16` production representation
are removed without compatibility aliases.

## Replace the scalar type

Update imports and type annotations:

```rust
use apollo_fft::F16;

let sample = F16::from_f32(1.5);
assert_eq!(sample.to_f32(), 1.5);
```

`F16` preserves the binary16 storage layout and exposes checked conversion
through Eunomia. Use `Bf16` from `eunomia` when bfloat16 storage is required.

## Replace complex storage

Use Eunomia's generic complex type with the owned component:

```rust
use apollo_fft::F16;
use eunomia::Complex;

let sample: Complex<F16> = Complex::new(F16::from_f32(1.0), F16::ZERO);
```

Apollo's compact complex FFT route accepts `Complex<F16>`. The representation is
two adjacent 16-bit lanes, and the cached `f32` execution plan performs the
single widening boundary internally.

## Dependency change

Remove direct `half` usage from Apollo consumers. `half` remains only as
Eunomia's development-time differential oracle for the reduced-precision
contract; WGPU may still resolve it transitively through Naga, which does not
provide Apollo's storage representation.

This is a major-version API migration for external consumers. In-repository
callers must migrate in the same change; no forwarding alias or adapter is
provided.
