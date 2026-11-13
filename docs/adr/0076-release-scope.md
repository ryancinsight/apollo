# ADR 0076: Apollo release scope

## Status

Accepted (2026-11-12). Revised 2026-11-13: every workspace member now has a
disposition, milestones are capabilities, and planned versions follow the
highest manifest version (judge verdict on PR 705).

## Context

Apollo is the Atlas stack's FFT and integral-transform provider over Eunomia,
Leto, Hermes, Hephaestus, Mnemosyne, and Moirai. The workspace has 26 members
(`Cargo.toml` `[workspace] members`). Manifest versions differ by crate
(`apollo-fft` 0.29.5 down to `xtask` 0.1.1; `apollo-dctdst-core` and
`apollo-bench` track the workspace 0.17.2), and release tags are per crate
(`crate-<package>-v<version>`, [ADR 0002](0002-git-source-release-contract.md)).
Open work is the capability audit, memory discipline, the reference
scoreboard, and source-size ratchets (`backlog.md`). This ADR fixes what the
1.0 release contains; the milestone ladder at the head of `backlog.md` orders
the work.

## Decision

Each member's capability areas are classified below as Delivered (shipped at
the basis commit), In scope (a ladder exit or a ladder-current item serves it),
or Non-goal for 1.0. Ladder milestones: `M-numpy-fft-semantics` (0.30.0),
`M-spectral-analysis` (0.31.0), `M-imaging-transforms` (0.32.0),
`M-backend-and-python-reach` (0.33.0), `M-stable-transform-suite` (1.0.0).
Tightening, memory, ratchet, and verification items join the current
milestone and gate no exit.

| Member | Delivered | In scope for 1.0 | Non-goal for 1.0 |
|---|---|---|---|
| apollo-fft | Dense CPU FFT at ranks 1-3, real half spectrum, static plans, plan caches, composite, prime and Bluestein routes | Normalization modes, axis and rank-N transforms, frequency-axis utilities, fallible API, reference-parity performance, bounded caches, backend execution seam, WGPU real-to-complex | AVX-512 path (no host), CUDA beyond 1-D `f32` powers of two, WGSL `f64` |
| apollo-fft-macros | Generated routing kernels | Only what apollo-fft routing needs | None |
| apollo-leto-interop | Zero-copy Leto view materialization, Mnemosyne output construction | Axis views for the axis forms | None |
| apollo-alloc | Deterministic in-page measurement buffers (`publish = false`) | None | Publication |
| apollo-bench | Sequential measurement runtime and comparator (`publish = false`) | None | Publication |
| apollo-validation | Published-reference fixtures, parity and benchmark runners | Suite tour example | None |
| apollo-dctdst-core | Direct DCT-III fixed-capacity plans | Review item DCT-CORE-001 | Array, runtime or device dependencies |
| apollo-dctdst | 1-D DCT/DST, square 2-D and cube 3-D, fast kernels, accelerator dispatch | Rectangular 2-D/3-D, orthonormal mode, axis form | Types V-VIII |
| apollo-dht | Real-to-real DHT plans, self-inverse kernels | Source-size ratchet only | None |
| apollo-hilbert | Analytic signal, envelope, phase | Precision-profile truth | Python bindings |
| apollo-nufft | 1-D and 3-D Type-1/2 Kaiser-Bessel plans, direct references, WGPU | 2-D plan, tolerance-driven parameters, fallible constructors, book chapter | Type-3, GPU 2-D |
| apollo-radon | Parallel-beam projection, adjoint, ramp-filtered backprojection | Fan-beam geometry, more filters, typed and `_into` backprojection | None |
| apollo-sht | Spherical harmonic plans, Gauss-Legendre quadrature | None | None |
| apollo-sdft | Sliding DFT streaming plans | None | None |
| apollo-mellin | Mellin moments, log-resampling, log-frequency spectra | None | None |
| apollo-sft | Sparse Fourier transform plans, direct recovery | Book chapter | None |
| apollo-wavelet | Haar, Daubechies-4, Ricker, real Morlet; 1-D power-of-two periodic | More filter families, any length, symmetric and zero boundaries, 2-D DWT, complex Morlet | Wavelet packets, SWT, inverse CWT |
| apollo-stft | Hann-window plans, centered-frame reconstruction | Window family, custom-window inversion, one-sided spectra | Streaming STFT, GPU window variants |
| apollo-czt | Direct and Bluestein chirp z-transform | Zoom-FFT band constructor | Fast and non-square inverse |
| apollo-frft | Direct O(N^2) and unitary plans | O(N log N) chirp-convolution path | None |
| apollo-fwht | Walsh-Hadamard plans, in-place kernels | Scratch release | None |
| apollo-gft | Graph Laplacian spectral plans | Scratch release | None |
| apollo-qft | Dense unitary QFT plans | None | None |
| apollo-ntt | Radix-2 and circle-group NTT, validated moduli | None | Non-power-of-two lengths, Montgomery and Barrett arithmetic |
| apollo-python | FFT, NUFFT, DCT, FWHT, DHT bindings; precision and backend introspection (`publish = false`) | NumPy-compatible `rfft`/`irfft` with `norm` and `axes`, `gil_used` audit | Bindings for STFT, wavelet, Hilbert and every other sibling |
| xtask | Repository tooling (`publish = false`) | None | Publication |

## Version basis

Planned versions track `apollo-fft`, the highest-versioned member (manifest
0.29.5; newest tag by version `crate-apollo-fft-v0.29.4`). The ladder uses
successive minors above that tag, then 1.0.0. Sibling crates version
independently by the change class of their own changes and never fall below
their manifest versions. The newest tag by date (`crate-apollo-dctdst-core-v0.17.2`)
is not the basis: following it would roll `apollo-fft` back.

## Veto points

Choices no source determines, taken as the recommended option:

1. Version basis above, with five rungs ordered by dependency.
2. NUFFT Type-3 is a non-goal (the audit defers it to a later increment).
3. CUDA beyond 1-D `f32` powers of two is a non-goal (no CUDA host exercised).
4. Inverse CWT, wavelet packets, SWT and the fast or non-square CZT inverse
   are non-goals.
5. The Python surface is verified by a Rust-side binding test; the pytest
   NumPy-parity suite supplements it and is not an exit.
6. AVX-512 stays a non-goal while no host reports `avx512f`.

## Alternatives

- One release without a ladder: rejected, memory and capability slices span
  many crates and the ladder gives each a runnable check.
- Versions from the newest tag by date: rejected, it rolls `apollo-fft` back.
- Process milestones (gates, ratchets): rejected, they are quality states.

## Consequences

- Open items carry one `milestone:`; feature items name their exit.
- Release and publication need explicit authority; the 1.0.0 block closes only
  its exit checks and standing conditions.
- Overturning evidence: a consumer requirement for a non-goal, or a host that
  reports `avx512f` or a CUDA device, reopens that area by revision note.
