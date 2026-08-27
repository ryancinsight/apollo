// Production: the register-resident 128-point base butterfly and its
// measurement probes (ATLAS-APOLLO-BASE-BUTTERFLY-128); routed from the
// power-of-two plan's log2 == 7 arm with incumbent fallback.
pub(crate) mod base128;
pub(crate) mod batched;
// Test-gated: correct on every oracle but slower than the batched route as
// built; its pinned probe is the same-process comparison instrument.
pub(crate) mod butterflies;
#[cfg(test)]
mod resident;
#[cfg(test)]
mod test_support;
// Test-gated deliberately, not provisionally: the N = 16 codelet is correct
// against a direct-DFT oracle but measured slower than the incumbent sized
// kernel pinned on both core types, so it ships as a measurement instrument
// with the probe that declined it. Promotion is gated on the in-register
// permutation primitive (`HS-INTERLEAVE-PAIRS`), which removes the
// store-forward stalls that cost it the comparison.
#[cfg(test)]
mod codelet;
pub(crate) mod four_step;
pub(crate) mod good_thomas;
mod lane_capability;
pub(crate) mod rader;
pub(crate) mod radix_composite;
pub(crate) mod stockham;
pub(crate) mod winograd;
