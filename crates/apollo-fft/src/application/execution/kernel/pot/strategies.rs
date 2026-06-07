//! Zero-sized strategy types for power-of-two FFT execution.
//!
//! Each strategy is a ZST (or const-generic ZST) that parameterizes a
//! monomorphized kernel. Adding a new strategy (e.g. in-place DIT, four-step
//! tiled) requires only a new ZST and a blanket or specific impl; call sites
//! using `impl PoTStrategy` or `<S: PoTStrategy>` get the specialized code
//! with zero runtime overhead.

use core::marker::PhantomData;

/// Marker trait for all PoT execution strategies.
/// Strategies must be ZSTs or const-generic ZSTs to guarantee zero-cost.
pub trait PoTStrategy: Copy + Clone + 'static + Send + Sync {}

/// Stockham autosort (out-of-place ping-pong, natural order, no bit reversal).
/// ZST wired (plan PowerOfTwo + dispatch concrete log2 arms for 512/1024; sized execs construct exact).
/// Enables zero-cost monomorph per-log2 selection (replaces runtime match).
#[derive(Copy, Clone, Default, Debug)]
pub struct StockhamAutosort;
impl PoTStrategy for StockhamAutosort {}

/// Phantom-tagged size class for arity-specialized monomorph schedules (zero-sized, zero-cost).
/// Wired: exact SizedPoT<StockhamAutosort, LOG2> constructed in plan/dispatch for hot (9=512,10=1024).
/// Example: `SizedPoT<StockhamAutosort, 9>` for N=512 (explicit arm + test guard).
#[derive(Copy, Clone, Default, Debug)]
pub struct SizedPoT<S: PoTStrategy, const LOG2: u32> {
    pub(crate) _p: PhantomData<S>,
}

impl<S: PoTStrategy, const LOG2: u32> PoTStrategy for SizedPoT<S, LOG2> {}

impl<S: PoTStrategy, const LOG2: u32> SizedPoT<S, LOG2> {
    /// Zero-cost constructor for the ZST (used in PoT wiring in plan/dispatch
    /// to tag monomorphized Stockham/Direct paths by exact log2).
    #[inline]
    pub(crate) const fn new() -> Self {
        Self { _p: PhantomData }
    }
}
