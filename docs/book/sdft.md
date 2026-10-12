# Sliding Discrete Fourier Transform

The SDFT maintains the DFT of a fixed sliding window over a streaming
signal, updating bins incrementally as each sample enters and leaves
instead of recomputing the window transform from scratch. Apollo owns the
sliding state in `apollo-sdft`, built around `SdftPlan`.

```rust
{{#include ../../crates/apollo-sdft/examples/book_sdft_update.rs}}
```

## Tracked bins

`SdftPlan::new(window_len, bin_count)` fixes the window length `N` and the
number of tracked bins `K`. Each `SdftState::update` call advances every
tracked bin in the modulated form, so per-sample work scales with the
tracked bin count rather than the cost of a fresh window FFT.
`state_from_window` seeds the state
from an initial window, and `direct_bins` recomputes the same bins from
scratch whenever an independent check is wanted.

## Bounded drift

A pure recurrence would accumulate rounding without bound. Apollo instead
re-sums one bin per refresh period in round-robin order, so no bin carries
more than `max(K, N)` advances of rounding. `SdftPlan::drift_bound` states
the resulting guarantee as a closed form in the unit roundoff `u`, the
signal amplitude, `N`, and `K`; the bound is independent of the update
count because every bin is refreshed on that cadence.
