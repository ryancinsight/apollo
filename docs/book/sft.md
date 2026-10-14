# Sparse Fourier Transform

The SFT recovers a sparse spectrum — a few dominant frequencies among many
bins — without computing the dense transform, by hashing bins into buckets
and voting for the heavy hitters. Apollo owns the support selection and
reconstruction in `apollo-sft`, built around `SparseFftPlan`, and keeps the
dense kernels in `apollo-fft`.
