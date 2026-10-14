# Discrete Cosine and Sine Transforms

The DCT and DST are real-to-real transforms: the DCT-II expands a signal
in cosines, the DST-II in sines, with the inverse scaling folded into the
matching inverse plans. Apollo owns variable-length plans, fast execution,
and accelerator dispatch in `apollo-dctdst`, built around `DctDstPlan`.

```rust
{{#include ../../crates/apollo-dctdst/examples/book_dct_self_inverse.rs}}
```

## Eight kinds, one plan

`RealTransformKind` covers DCT-I through DCT-IV and DST-I through DST-IV,
each documented with its Rao and Yip definition and self-inversion scale.
`DctDstPlan::new` binds a length to a kind; `forward` and `inverse` move
plain `f64` slices both ways, with `forward_2d`/`forward_3d` (and matching
inverses) for image and volume shapes. The sample exercises the DCT-IV
identity: two forward passes scale the signal by exactly `N/2`.

## Dual-kernel dispatch

Every kind runs two kernels behind the plan: an FFT-backed `O(N log N)`
fast path at length 16 and above, and the direct analytical kernel below
it, where the setup it would need costs more than the `O(N²)` sum. The
direct kernels are also exported as free functions — `dct1`, `dct2`,
`dct4`, `dst1`, `dst2`, `dst3`, `dst4` — for one-shot transforms that
never amortize a plan.
