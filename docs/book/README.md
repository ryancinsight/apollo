# apollo — Spectral Transforms for Atlas

`apollo` is the forward-focused spectral transform library of the Atlas
stack.  It provides FFT, DHT, NTT, wavelet, STFT, and a family of
specialized transforms with a shared plan cache and GPU dispatch hooks.

## Design goals

- **Plan cache** — transform plans are computed once and cached by size and
  precision; repeated calls reuse the cached plan without re-allocation.
- **Generic over precision** — the same `fft_1d_array` works for `f32` and
  `f64`; the plan selects the correct kernel for the precision.
- **GPU hooks** — CPU kernels and GPU kernels share one plan type; the
  dispatch chooses the device at call time.
- **Autodiff hooks** — the plan seam integrates with `coeus` autodiff so
  gradient computations through FFT operations stay in one codebase.

## What this book covers

1. Real and complex 1D/2D/3D FFT with `fft_1d_array` and `ifft_1d_array`.
2. Frequency grids via `fftfreq` and `rfftfreq`.
3. Parseval's theorem and how to verify spectral energy.
4. DHT, NTT, wavelet, and STFT transforms.
5. The plan cache architecture.
6. GPU transform dispatch.
