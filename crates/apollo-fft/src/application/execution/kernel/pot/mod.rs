//! Power-of-two FFT strategies and entry points.
//!
//! Provides zero-cost (monomorphized) and ZST-driven abstractions for PoT
//! execution regimes. Strategies are encoded as ZSTs so that the compiler
//! selects schedules at compile time with no runtime dispatch cost in hot paths.
//!
//! The hierarchy isolates PoT concerns from general mixed-radix so that
//! Stockham, Winograd-codelet, and future radix-4/8 in-place variants can
//! coexist behind a single facade while sharing butterfly primitives.

// Windows-gated: the instrument pins threads through Win32, and the hybrid
// scheduling it exists to control for is a Windows behavior.
#[cfg(all(test, windows))]
mod core_matrix;
#[cfg(test)]
mod crossover;
pub(crate) mod route;
pub mod strategies;

pub(crate) use route::{one_dimensional_uses_four_step, FourStep, PotRoute};
pub use strategies::{PoTStrategy, SizedPoT, StockhamAutosort};

/// Constructs a `SizedPoT<StockhamAutosort, LOG2>` ZST, binds it to the
/// given identifier, and executes the provided block.
///
/// Centralizes the `SizedPoT::new()` + const-generic wiring shared across
/// dispatch (`try_pot_zst!`) and stockham (`zst_stockham_dispatch!`).
/// The caller names the ZST variable (convention: `_s`) and references it
/// in the body for zero-cost monomorphization.
#[macro_export]
macro_rules! with_pot_zst {
    ($log2:expr, $zst:ident, $body:block) => {{
        let $zst: $crate::application::execution::kernel::pot::SizedPoT<
            $crate::application::execution::kernel::pot::StockhamAutosort,
            $log2,
        > = $crate::application::execution::kernel::pot::SizedPoT::new();
        $body
    }};
}
