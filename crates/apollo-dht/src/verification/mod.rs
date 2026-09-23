//! Verification modules for the DHT.

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet APOLLO-UNWRAP-1: pre-existing debt"
    )
)]

#[cfg(test)]
mod multidimensional;
#[cfg(test)]
mod one_dimensional;
