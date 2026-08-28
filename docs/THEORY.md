# Apollo Transform Theory

Apollo uses classical transform constructions with crate-local ownership and a
shared verification convention: every implemented transform must state the
mathematical map it computes and must validate computed values against an
analytical invariant.

## FFT Normalization

Apollo FFT paths use a shared normalization convention:

- Forward transforms are unnormalized.
- Inverse transforms apply `1 / N` over the transformed domain.

## Fourier Algorithms

- Iterative Cooley-Tukey radix-2 factorization for power-of-two CPU lengths.
- Rader prime-length DFT reduction: for prime `N`, the non-zero indices form
  a cyclic multiplicative group, so the non-DC DFT bins reduce to a length
  `N-1` cyclic convolution. For `N-1 = 2m`, Apollo can use the half-cyclic
  Liu-Tolimieri/Winograd-style CRT split:
  `x^(2m)-1 = (x^m-1)(x^m+1)`. The residues compute one cyclic and one
  negacyclic length-`m` convolution, and recombination recovers the original
  lower and upper halves exactly in exact arithmetic. Production routing is
  scalar-policy driven: current f32 and f64 routing enables the split at
  `N-1 >= 1024` because the split and recombination overhead does not amortize
  at the lower threshold in the current CPU kernel family.
  The half-cyclic spectrum cache builder constructs these residues directly,
  avoiding a temporary full length-`N-1` kernel allocation.
  Runtime Rader caches only the generator order `g^q`; the scatter order is
  derived by `g^{-q}=g^(N-1-q)` because the non-zero residues form a cyclic
  group of order `N-1`.
- CZT evaluates `X[k] = sum_n x[n] a^-n w^(nk)` directly for reference and
  uses `nk = (n^2 + k^2 - (k-n)^2) / 2` to reduce fast execution to one
  zero-padded convolution.
- FrFT order `a` rotates by `alpha = a*pi/2`; integer quarter-rotations reduce
  to identity, centered DFT, reversal, and centered inverse DFT. Apollo keeps
  these singular limits finite in plan state and verifies inverse recovery for
  the centered DFT case.
- NTT evaluates a DFT over `Z_q` using a primitive `n`th root of unity
  `omega = g^((q - 1) / n) mod q`. Orthogonality of finite root sums gives the
  inverse formula with `omega^-1` and `n^-1 mod q`.
- Real-to-complex half-spectrum packing that preserves Hermitian symmetry.

## Invariants

- A real-valued spatial signal produces Hermitian-symmetric spectral coefficients.
- The Nyquist bin is real-valued when the corresponding dimension is even.
- Parseval-style energy checks are evaluated under Apollo's FFTW-compatible normalization.
- Direct DFT kernels are retained as analytical reference surfaces for tests and
  validation, not as production dense FFT plan paths.

## Numerical Notes

- Roundtrip stability is assessed with relative and absolute tolerances.
- GPU parity compares against the CPU plan family rather than raw bitwise identity.
- Dense WGPU FFT kernels live in Hephaestus and consume Leto layout metadata;
  Apollo transform crates retain only their domain-specific GPU algorithms.
  Every accelerator path proves parity against its Apollo CPU implementation
  before advertising numerical execution support.

## Real-To-Real Transforms

- DHT computes `H[k] = sum_n x[n] (cos(2 pi kn/N) + sin(2 pi kn/N))`.
  Theorem: `DHT(DHT(x)) = N x`; Apollo verifies raw involution and normalized
  inverse reconstruction.
- FWHT applies the Sylvester Hadamard matrix `H_n` for power-of-two lengths.
  Theorem: `H_n H_n = n I`; Apollo verifies analytical two-point output,
  real/complex roundtrip reconstruction, and caller-owned parity.
- DCT-II/DCT-III and DST-II/DST-III compute their unnormalized analytical
  cosine/sine projections. Apollo verifies known two-point projections and the
  standard `2/N` inverse-pair scaling, including caller-owned inverse output.

## Time, Scale, And Phase Transforms

- SDFT tracks a fixed window by recurrence and is verified against direct DFT
  recomputation after each update.
- SFT computes dense Apollo FFT coefficients, retains the top-K magnitudes with
  deterministic frequency-index tie-breaking, and reconstructs by expanding the
  sparse support before inverse FFT evaluation.
- STFT uses centered Hann-window frames. Inverse reconstruction overlap-adds
  inverse FFT frames and divides by the accumulated squared-window weights,
  recovering every covered sample with non-zero weight in exact arithmetic.
- Wavelet DWT uses orthogonal filter banks; inverse reconstruction is verified
  for Haar and Daubechies-4. CWT uses direct real wavelet correlations with
  Ricker and DC-corrected real Morlet kernels. The Morlet correction subtracts
  `exp(-omega0^2/2)` from the carrier term so the continuous mother wavelet has
  zero integral.
- Hilbert analysis constructs the analytic signal by preserving DC/Nyquist,
  doubling positive frequencies, and removing negative frequencies. This yields
  `z[n] = x[n] + iH{x}[n]`; Apollo verifies quadrature, envelope, and
  real-part preservation.
- Mellin analysis evaluates `M(s) = int f(r) r^(s-1) dr` over positive scale
  coordinates and uses the substitution `r = exp(u)` for log-frequency spectra.
  Apollo verifies constant and power-law moments against analytical integrals.

## Spatial And Geometric Transforms

- SHT uses orthonormal complex spherical harmonics over Gauss-Legendre latitude
  nodes and uniform longitude nodes. Apollo verifies constant and single-mode
  coefficients.
- GFT diagonalizes the graph Laplacian and verifies roundtrip reconstruction
  over deterministic eigenpair ordering.
- Radon uses a discrete parallel-beam model over pixel-center point masses with
  linear detector splatting. The forward and adjoint kernels use the same
  interpolation weights, so `<R f, p> = <f, R* p>` in exact arithmetic. Apollo
  verifies axis projections, adjoint identity, mass conservation, and ramp
  filter DC rejection.

