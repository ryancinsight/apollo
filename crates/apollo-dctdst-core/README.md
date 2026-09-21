# Apollo DCT/DST Core

`apollo-dctdst-core` owns direct scalar discrete cosine transform mathematics
without Apollo's FFT, array, execution-runtime, memory, or accelerator
dependencies.

The first operation is a fixed-capacity DCT-III plan. It precomputes either the
unnormalized or orthonormal basis and then transforms caller-owned arrays
without allocation.

For input coefficients `X` of length `N`, the dynamic operation computes

```text
y[j] = X[0]/2 + sum(k=1..N-1, X[k] cos((pi/N) k (j+1/2))).
```

The orthonormal plan replaces the DC weight with `sqrt(1/N)` and each remaining
weight with `sqrt(2/N) cos((pi/N) k (j+1/2))`. Arithmetic executes in the
selected `RealField` precision. IEEE-754 NaNs propagate through an affected
sum, infinities follow ordinary multiplication and addition, and an exact zero
result may have a positive sign because accumulation begins at `+0`.

`DctIiiPlan<T, N>` stores exactly `N * N * size_of::<T>()` inline bytes for its
basis and allocates no heap memory. The plan therefore suits small fixed sizes;
callers choose `N` with stack capacity in mind.

```rust
use apollo_dctdst_core::{DctIiiPlan, Normalization};

let plan = DctIiiPlan::<f64, 2>::new(Normalization::Orthonormal)?;
let mut output = [0.0; 2];
plan.transform(&[1.0, 0.0], &mut output);
// Adjacent representable bounds enclose the exact value 1/sqrt(2).
assert!(output.into_iter().all(|sample| {
    (0.707_106_781_186_547_4..=0.707_106_781_186_547_7).contains(&sample)
}));
# Ok::<(), apollo_dctdst_core::PlanError>(())
```
