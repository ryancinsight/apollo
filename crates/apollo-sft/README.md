# Apollo SFT

`apollo-sft` owns sparse Fourier transform modeling and execution. Dense FFT
kernels remain in `apollo-fft`; sparse support selection and reconstruction
remain here.

## Architecture

```text
src/
  domain/          sparse spectrum storage and validation
  application/     sparse FFT plan and execution
  infrastructure/  dense reference kernel for verification
  verification/    sparse-support, inverse, and contract tests
```

`SparseFftPlan` is the plan-level SSOT for signal length, sparsity, bucket
count, trial count, and threshold.

Typed execution supports `Complex64`, `Complex32`, and mixed `[f16; 2]`
storage through one generic sparse API. The dense FFT, deterministic top-`k`
selector, and sparse inverse remain the `Complex64` owner path; typed storage
converts represented input into the owner path and quantizes only retained
coefficients or reconstructed samples at the API boundary.

With the `wgpu` feature, the accelerator boundary uses one typed Hephaestus
direct-DFT kernel and an ordered command stream. Hephaestus owns every device
buffer, pipeline, binding, dispatch, submission, and transfer; Apollo keeps
only sparse-domain selection and reconstruction policy. Contiguous Leto views
borrow host storage and strided views materialize exactly one logical-order
copy. Dense inverse outputs allocate directly in Mnemosyne-backed Leto storage.
The sealed `SftGpuStorage` contract admits `Complex32` and `[f16; 2]` only, so
the concrete accelerator cannot silently narrow `Complex64` input storage.
`SparseSpectrum` remains the CPU `Complex64` SSOT; inverse acceleration rejects
a coefficient that is not exactly representable in `f32` instead of changing
its value during staging. `SftWgpuBackend::quantize_spectrum` is the explicit
lossy conversion for callers who intentionally choose concrete accelerator
precision.

## Mathematical Contract

Forward execution computes dense Fourier coefficients, orders them by
magnitude, and retains the configured top-`k` sparse support. Inverse execution
evaluates the inverse dense Fourier sum from retained sparse coefficients.

For the accelerator's dense DFT convention,

```text
X[k] = sum_n x[n] exp(-2 pi i n k / N)
x[n] = (1/N) sum_k X[k] exp(2 pi i n k / N)
```

the root-of-unity identity
`sum_k exp(2 pi i k(n-m)/N) = N delta_nm` proves dense inverse recovery in
exact arithmetic. The top-`k` projection is intentionally outside that inverse
theorem: it reconstructs the retained sparse spectrum, not discarded bins.
The exact theorem is documented in the kernel module; real-device CPU
differential and inverse tests provide empirical evidence for finite precision.

## Verification

Tests cover constructor metadata, invalid contracts, sparse spectrum insertion,
dominant coefficient retention, inverse reconstruction on retained support,
direct DFT reference parity, zero-signal behavior, pure-tone support, `k = n`,
DC-only constant signals, typed `Complex64` parity, represented-input
`Complex32` and mixed `[f16; 2]` parity, typed inverse roundtrip, sparse
frequency/value shape rejection, and profile/storage mismatch rejection.

The all-feature accelerator suite additionally covers real-device CPU forward
differential, dense inverse reconstruction, Leto contiguous and strided input,
typed `f16` storage, invalid plans, and rejection of non-representable sparse
`Complex64` coefficients. The `SftGpuStorage` compile-fail doctest verifies
that `Complex64` cannot enter the concrete accelerator typed API.
