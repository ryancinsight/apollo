# Apollo DHT

`apollo-dht` owns the discrete Hartley transform for real-valued Apollo
signals.

## Architecture

```text
src/
  domain/          length contracts, errors, and Hartley spectrum storage
  application/     reusable DHT plan
  infrastructure/  real cas-kernel execution
  verification/    analytical, inverse, and property tests
```

`DhtPlan` is the single source of truth for validated signal length and
execution. Leto views remain borrowed through the storage-generic 2D/3D
kernels, and Mnemosyne owns output and scratch storage. Hermes supplies direct
row reductions and Moirai supplies bounded parallel row scheduling.

Typed caller-owned paths support high-accuracy `f64`, low-precision `f32`, and
mixed `f16` storage profiles. The typed paths reuse the authoritative `f64`
Hartley kernel and quantize once at the storage boundary, so precision support
does not fork the mathematical implementation.

With the `wgpu` feature, Apollo owns the Hartley WGSL formulas while
Hephaestus owns typed buffers, authored-kernel preparation, command streams,
synchronization, and transfer. `HartleyGpuStorage` admits native `f32` and the
declared mixed `f16`/`f32` profile; high-accuracy `f64` storage is rejected at
compile time because this accelerator kernel computes in `f32`.

## Mathematical Contract

The DHT computes

```text
H[k] = sum_n x[n] cas(2*pi*k*n/N), cas(theta) = cos(theta) + sin(theta)
```

The Hartley kernel is self-inverse up to scale:

```text
DHT(DHT(x)) = N x
```

so inverse execution reuses the same transform and applies `1 / N`.

## Verification

Tests cover impulse response, constant-signal DC behavior, Parseval scaling,
double-transform scaling, inverse execution, invalid contracts, and randomized
roundtrips. Typed tests cover `f64`, `f32`, mixed `f16`, and precision/profile
mismatch rejection. GPU verification adds real-device CPU differential,
self-inverse, exact unit-impulse, Leto contiguous/strided, caller-owned output,
and typed mixed-storage checks.
