//! Analysis and synthesis windows for the STFT.
//!
//! Every window here is symmetric over the frame, `w[i] = w[N - 1 - i]`
//! exactly, with the phase `x = i / (N - 1)`; a one-sample frame is `[1]`. The plan
//! uses one window for both analysis and synthesis, so its weighted
//! overlap-add inverse divides by the per-sample sum of `w²`.

use crate::domain::contracts::error::{StftError, StftResult};
use leto::Array1;
use std::f64::consts::PI;

/// A named window family.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Window {
    /// `0.5 - 0.5 cos(2πx)`.
    Hann,
    /// `0.54 - 0.46 cos(2πx)`.
    Hamming,
    /// `0.42 - 0.5 cos(2πx) + 0.08 cos(4πx)`.
    Blackman,
    /// A flat top over `1 - α` of the frame with raised-cosine tapers over
    /// the remaining `α`, `0 ≤ α ≤ 1`: `α = 0` is rectangular and `α = 1`
    /// is Hann (Harris, "On the use of windows for harmonic analysis with
    /// the discrete Fourier transform", Proc. IEEE 66(1), 1978).
    Tukey {
        /// The tapered fraction of the frame.
        alpha: f64,
    },
}

impl Window {
    /// The window's `n` coefficients.
    ///
    /// # Errors
    ///
    /// [`StftError::InvalidWindowParameter`] for a Tukey `alpha` outside
    /// `[0, 1]` or not finite.
    pub fn coefficients(self, n: usize) -> StftResult<Array1<f64>> {
        if let Self::Tukey { alpha } = self {
            if !(0.0..=1.0).contains(&alpha) {
                return Err(StftError::InvalidWindowParameter);
            }
        }
        if n == 1 {
            return Ok(Array1::from(vec![1.0]));
        }
        let span = (n - 1) as f64;
        // Each window is evaluated at the nearer end, so `w[i]` and
        // `w[N - 1 - i]` are the same value, not two roundings of it.
        Ok(Array1::from_shape_fn([n], |[i]| {
            let x = i.min(n - 1 - i) as f64 / span;
            match self {
                Self::Hann => 0.5 - 0.5 * (2.0 * PI * x).cos(),
                Self::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
                Self::Blackman => 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos(),
                Self::Tukey { alpha } => tukey(alpha, x),
            }
        }))
    }
}

/// The Tukey window at phase `x`, symmetric about `x = 1/2`.
fn tukey(alpha: f64, x: f64) -> f64 {
    // `2 edge >= alpha` rather than `edge >= alpha / 2`: a subnormal
    // `alpha` halves to zero and would read the taper as flat.
    let edge = x.min(1.0 - x);
    if alpha == 0.0 || 2.0 * edge >= alpha {
        1.0
    } else {
        0.5 * (1.0 + (PI * (2.0 * edge / alpha - 1.0)).cos())
    }
}

/// Whether the weighted overlap-add inverse of a `signal_len`-sample signal
/// reconstructs every sample: the `Σ w²` it divides by must exceed `ε` times
/// the largest such weight.
///
/// A frame's transform carries error relative to that frame's own largest
/// magnitude, `|w|max |x|max`, not relative to the sample being recovered.
/// So a sample covered only by small window values collects absolute error
/// `γ |w|max |x|max w_small` and is divided by a weight of order `w_small²`,
/// leaving a relative error bounded by `γ √(largest / weight)` with
/// `γ = 16 ⌈log₂ N⌉ ε` (Higham Thm 24.2). The amplification is the square
/// root of the weight ratio, not its reciprocal: the error reaches the
/// sample's own magnitude only near `weight / largest ≈ γ²`, which is
/// `1.1e-28` at `N = 8` and `1.3e-27` at `N = 1024`.
///
/// The floor is `ε` rather than that crossing, deliberately: at `ε largest`
/// the bound is `γ / √ε`, about `7e-7` at `N = 8` and `2.4e-6` at
/// `N = 1024`, so an accepted configuration still reconstructs to roughly
/// six significant digits, while below it the analysis stops promising even
/// half the mantissa.
///
/// The bound is an estimate validated by measurement, not a proof: it takes a
/// frame's transform error as `γ |w|max |x|max`, which asserts
/// `‖wx‖₂ ≤ |w|max |x|max` and is false by up to `√N` (the reconstruction
/// test's sibling bound carries that `√N`). It survives because
/// `γ = 16 ⌈log₂ N⌉ ε` overestimates the real round-trip error by more than
/// `√N` does; measured margin is flat in `N`, about 48x at `N = 8` and 35x at
/// `N = 8192`, rather than shrinking as `1/√N`. The family
/// `[t, 1, 1, 1, t, 1, 1, 1]` at `hop = 4` puts the weight ratio at `t²`, so
/// `t` from 1 down to just above `√ε` sweeps fifteen decades of ratio within
/// the accepted range; `a_plan_at_the_floor_holds_the_square_root_law`
/// measures across it.
///
/// Frame `m` starts at `m hop - N / 2` for `m ≤ ⌈L / hop⌉`, the framing the
/// forward and inverse share. From `N / 2` to `L - N / 2` every covering
/// frame exists, so the weight is periodic in `hop` there; the samples within
/// `N / 2 + hop` of either end hold one full period and every partially
/// covered sample, and the interior's largest weight is the global one. The
/// check therefore reads only those samples.
#[must_use]
pub(crate) fn wola_weights_defined(window: &[f64], hop: usize, signal_len: usize) -> bool {
    let n = window.len();
    let half = n / 2;
    let last_frame = signal_len.div_ceil(hop);
    let weight = |i: usize| -> f64 {
        // Frames covering `i`: `m hop - half ≤ i < m hop - half + n`.
        let highest = ((i + half) / hop).min(last_frame);
        let lowest = (i + half + 1).saturating_sub(n).div_ceil(hop);
        (lowest..=highest)
            .map(|m| window[i + half - m * hop])
            .map(|w| w * w)
            .sum()
    };
    let reach = (half + hop + 1).min(signal_len);
    let edges: Vec<f64> = (0..reach)
        .chain(signal_len - reach..signal_len)
        .map(weight)
        .collect();
    let largest = edges.iter().fold(0.0f64, |m, &w| m.max(w));
    largest > 0.0 && edges.iter().all(|&w| w > f64::EPSILON * largest)
}

#[cfg(test)]
mod tests {
    use super::{wola_weights_defined, Window};
    use crate::domain::contracts::error::StftError;

    /// The closed forms at the ends and the middle of an odd frame, where
    /// `x = 0, 1/2, 1` and the cosines are exactly `1, -1, 1`.
    #[test]
    fn coefficients_match_the_closed_forms() {
        let at = |window: Window| {
            let w = window.coefficients(9).expect("valid window");
            (w[0], w[4], w[8])
        };
        assert_eq!(at(Window::Hann), (0.0, 1.0, 0.0));
        assert_eq!(
            at(Window::Hamming),
            (0.08000000000000002, 1.0, 0.08000000000000002)
        );
        let (b0, b4, b8) = at(Window::Blackman);
        assert!(b0.abs() < 1e-16 && (b4 - 1.0).abs() < 1e-15 && b8.abs() < 1e-16);
        assert_eq!(at(Window::Tukey { alpha: 0.0 }), (1.0, 1.0, 1.0));
        assert_eq!(at(Window::Tukey { alpha: 1.0 }), at(Window::Hann));
        let (t0, t4, t8) = at(Window::Tukey { alpha: 0.5 });
        assert_eq!((t0, t4, t8), (0.0, 1.0, 0.0));
    }

    #[test]
    fn every_window_is_symmetric() {
        for window in [
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::Tukey { alpha: 0.3 },
        ] {
            for n in [2usize, 7, 16] {
                let w = window.coefficients(n).expect("valid window");
                for i in 0..n {
                    assert_eq!(w[i], w[n - 1 - i], "{window:?} n={n} i={i}");
                }
            }
        }
    }

    #[test]
    fn tukey_rejects_alpha_outside_the_unit_interval() {
        for alpha in [-0.1, 1.5, f64::NAN] {
            assert_eq!(
                Window::Tukey { alpha }.coefficients(8),
                Err(StftError::InvalidWindowParameter)
            );
        }
    }

    /// A symmetric Hann is zero at both ends, so a hop of the whole frame
    /// leaves some samples with no energy; half and quarter hops cover them.
    #[test]
    fn weights_follow_the_hop() {
        let hann = Window::Hann.coefficients(16).expect("valid");
        let hann = hann.as_slice().expect("contiguous");
        assert!(!wola_weights_defined(hann, 16, 64));
        assert!(wola_weights_defined(hann, 8, 64));
        assert!(wola_weights_defined(hann, 4, 64));
    }

    /// Windows whose residues all carry energy in the interior yet leave the
    /// signal's first or last samples uncovered (the review's reproductions),
    /// and a residue with energy far below `ε` of the rest.
    #[test]
    fn edges_and_negligible_energy_are_refused() {
        let late = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let early = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!wola_weights_defined(&late, 2, 16));
        assert!(!wola_weights_defined(&early, 2, 16));
        let mut faint = [1.0; 8];
        faint[3] = 1e-160;
        faint[7] = 1e-160;
        assert!(!wola_weights_defined(&faint, 4, 32));
        assert!(wola_weights_defined(&[1.0; 8], 4, 32));
    }

    /// The Tukey taper at a subnormal `alpha` still starts at zero.
    #[test]
    fn tukey_at_subnormal_alpha_tapers() {
        let w = Window::Tukey { alpha: 5e-324 }
            .coefficients(9)
            .expect("valid");
        assert_eq!((w[0], w[4], w[8]), (0.0, 1.0, 0.0));
    }
}
