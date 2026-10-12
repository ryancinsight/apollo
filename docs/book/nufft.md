# Non-Uniform Fast Fourier Transform

The NUFFT evaluates Fourier sums at non-uniform points (type 1:
non-uniform samples to uniform modes) and its adjoint (type 2: uniform
modes to non-uniform points) via spreading, oversampled FFT, and
deconvolution. Apollo owns the spreading, interpolation, and
deconvolution logic in `apollo-nufft` — `NufftPlan1D` in one dimension —
and delegates the dense FFT execution to `apollo-fft`.
