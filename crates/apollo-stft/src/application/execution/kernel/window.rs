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
    let edge = x.min(1.0 - x);
    if alpha == 0.0 || edge >= alpha / 2.0 {
        1.0
    } else {
        0.5 * (1.0 + (PI * (2.0 * edge / alpha - 1.0)).cos())
    }
}

/// Whether `window` overlap-adds at `hop`: every residue of the sample
/// index modulo `hop` receives a non-zero `Σ w²` from the frames that cover
/// it, which is what the weighted overlap-add inverse divides by in the
/// interior of the signal (the nonzero overlap-add condition).
#[must_use]
pub(crate) fn overlap_adds(window: &[f64], hop: usize) -> bool {
    (0..hop).all(|residue| {
        window
            .iter()
            .skip(residue)
            .step_by(hop)
            .map(|w| w * w)
            .sum::<f64>()
            > 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::{overlap_adds, Window};
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
    /// leaves residue 0 with no energy; half and quarter hops cover it.
    #[test]
    fn overlap_add_condition_follows_the_hop() {
        let hann = Window::Hann.coefficients(16).expect("valid");
        let hann = hann.as_slice().expect("contiguous");
        assert!(!overlap_adds(hann, 16));
        assert!(overlap_adds(hann, 8));
        assert!(overlap_adds(hann, 4));
    }
}
