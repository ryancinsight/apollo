pub(crate) mod batched;
// Test-gated: correct on every oracle, blocked on a hermes vectorize
// codegen defect for large kernel bodies (the module doc carries the
// evidence); its pinned probe is the four-engine same-process instrument.
pub(crate) mod butterflies;
#[cfg(test)]
mod resident;
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
pub(crate) mod rader;
pub(crate) mod radix_composite;
pub(crate) mod stockham;
pub(crate) mod winograd;
