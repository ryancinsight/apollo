//! Independent estimator and perturbation bounds for the three-tone scene.

use core::num::NonZeroUsize;
use std::f64::consts::{PI, TAU};

use eunomia::Complex64;

use super::{phase_delta, LEN, PEEL_ROUNDS};
use crate::estimate_peaks;

use super::super::{Estimate, Tone};

/// The geometric series `sum_{t<N} exp(2 pi i u t / N)`. `1 - exp(i x)`
/// uses half-angle identities so the quotient retains relative accuracy when
/// its denominator is small.
fn geometric_sum(u: f64) -> Complex64 {
    let n = LEN as f64;
    let reduced = u - n * (u / n).round();
    if reduced == 0.0 {
        return Complex64::new(n, 0.0);
    }
    let fraction = reduced - reduced.round();
    let one_minus_phasor = |angle: f64| {
        let half_sine = (angle / 2.0).sin();
        Complex64::new(2.0 * half_sine * half_sine, -angle.sin())
    };
    one_minus_phasor(TAU * fraction) / one_minus_phasor(TAU * reduced / n)
}

fn reference_contribution(estimate: Estimate, position: f64) -> Complex64 {
    estimate.a * geometric_sum(estimate.position - position)
        + estimate.a.conj() * geometric_sum(-estimate.position - position)
}

/// Runs the independent estimator, including estimate-and-subtract.
pub(super) fn reference_rounds(
    spectrum: &[Complex64],
    bins: &[usize],
    rounds: usize,
) -> Vec<Vec<Option<Estimate>>> {
    let mut order: Vec<usize> = (0..bins.len()).collect();
    order.sort_by(|&a, &b| {
        spectrum[bins[b]]
            .norm()
            .total_cmp(&spectrum[bins[a]].norm())
    });
    let mut estimates = vec![None; bins.len()];
    let mut by_round = vec![estimates.clone()];
    let step = PI / LEN as f64;
    for _ in 0..rounds {
        for &i in &order {
            let read = |bin: usize| {
                estimates
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .filter_map(|(_, estimate)| *estimate)
                    .fold(spectrum[bin], |residual, estimate| {
                        residual - reference_contribution(estimate, bin as f64)
                    })
            };
            let bin = bins[i];
            let before = read(bin - 1);
            let peak = read(bin);
            let after = read(bin + 1);
            let ratio = ((before - after) / (peak * 2.0 - before - after)).re;
            let delta = (ratio * step.tan()).atan() / step;
            if !delta.is_finite() || delta.abs() > 0.5 {
                estimates[i] = None;
                continue;
            }
            let direct = geometric_sum(delta);
            let image = geometric_sum(-(2.0 * bin as f64 + delta));
            let determinant = direct.norm_sqr() - image.norm_sqr();
            if determinant <= 0.0 {
                estimates[i] = None;
                continue;
            }
            estimates[i] = Some(Estimate {
                position: bin as f64 + delta,
                a: (peak * direct.conj() - peak.conj() * image) / determinant,
            });
        }
        by_round.push(estimates.clone());
    }
    by_round
}

pub(super) fn production_rounds(
    spectrum: &[Complex64],
    bins: &[usize],
) -> Vec<Vec<Option<Estimate>>> {
    let mut by_round = vec![vec![None; bins.len()]];
    by_round.extend((1..=PEEL_ROUNDS).map(|round| {
        estimate_peaks(
            spectrum,
            bins,
            NonZeroUsize::new(round).expect("invariant: rounds start at one"),
        )
        .expect("invariant: scene bins are valid")
        .into_iter()
        .map(|estimate| estimate.map(Estimate::from))
        .collect()
    }));
    by_round
}

pub(super) fn assert_within_truth_bounds(
    lhs: Estimate,
    lhs_bounds: (f64, f64),
    rhs: Estimate,
    rhs_bounds: (f64, f64),
    tone: &Tone,
    label: &str,
) {
    assert!(
        (lhs.position - rhs.position).abs() <= lhs_bounds.0 + rhs_bounds.0,
        "{label}: position difference exceeds the summed truth bounds"
    );
    assert!(
        (2.0 * lhs.a.norm() - 2.0 * rhs.a.norm()).abs() <= 2.0 * (lhs_bounds.1 + rhs_bounds.1),
        "{label}: amplitude difference exceeds the summed truth bounds"
    );
    let truth_radius = tone.a().norm();
    let lhs_phase = (lhs_bounds.1 / truth_radius).asin();
    let rhs_phase = (rhs_bounds.1 / truth_radius).asin();
    assert!(
        lhs_phase.is_finite()
            && rhs_phase.is_finite()
            && phase_delta(lhs.a.arg(), rhs.a.arg()).abs() <= lhs_phase + rhs_phase,
        "{label}: phase difference exceeds the angular truth bounds"
    );
}

fn contribution(position: f64, a: Complex64, bin: f64) -> Complex64 {
    a * geometric_sum(position - bin) + a.conj() * geometric_sum(-position - bin)
}

fn residual(tone: &Tone, estimate: Option<Estimate>, bin: f64) -> f64 {
    let truth = contribution(tone.position, tone.a(), bin);
    estimate.map_or_else(
        || truth.norm(),
        |estimate| {
            let subtracted = contribution(estimate.position, estimate.a, bin);
            let size = estimate.a.norm()
                * (geometric_sum(estimate.position - bin).norm()
                    + geometric_sum(-estimate.position - bin).norm());
            let argument_rounding = 2.0
                * PI
                * (LEN - 1) as f64
                * f64::EPSILON
                * (estimate.position.abs() + bin.abs())
                * estimate.a.norm();
            (subtracted - truth).norm() + argument_rounding + 16.0 * f64::EPSILON * size
        },
    )
}

fn offset_bound(tone: &Tone, bin: usize, extra: [f64; 3]) -> Option<f64> {
    let term = |m: usize| contribution(tone.position, tone.a(), bin as f64 + m as f64 - 1.0);
    let denominator = term(1) * 2.0 - term(0) - term(2);
    let exact_ratio = (term(0) - term(2)) / denominator;
    let denominator_spill = 2.0 * extra[1] + extra[0] + extra[2];
    let swamp = denominator_spill / denominator.norm();
    if swamp >= 1.0 {
        return None;
    }
    let ratio = (extra[0] + extra[2] + exact_ratio.norm() * denominator_spill)
        / (denominator.norm() * (1.0 - swamp));
    let step = PI / LEN as f64;
    let closure =
        ((exact_ratio.re * step.tan()).atan() / step - (tone.position - bin as f64)).abs();
    Some(step.tan() / step * ratio + closure + (LEN as f64 + 8.0) * f64::EPSILON)
}

fn solve_bound(tone: &Tone, bin: usize, estimate: Estimate, extra: f64) -> Option<f64> {
    let delta = tone.position - bin as f64;
    let estimated_delta = estimate.position - bin as f64;
    let image = |offset: f64| geometric_sum(-(2.0 * bin as f64 + offset));
    let direct_estimate = geometric_sum(estimated_delta);
    let image_estimate = image(estimated_delta);
    let gap = direct_estimate.norm() - image_estimate.norm();
    if gap <= 0.0 {
        return None;
    }
    let moved =
        (geometric_sum(delta) - direct_estimate).norm() + (image(delta) - image_estimate).norm();
    let position_rounding = f64::EPSILON * (estimate.position.abs() + bin as f64);
    let moved = moved + 2.0 * PI * (LEN - 1) as f64 * position_rounding;
    Some((tone.a().norm() * moved + extra) / gap + 16.0 * f64::EPSILON * tone.a().norm())
}

/// Checks every round against truth using only the independently evaluated
/// geometric series. Returns the final position and complex-amplitude bounds.
pub(super) fn check_geometric_rounds(
    spectrum: &[Complex64],
    tones: &[Tone],
    bins: &[usize],
    by_round: &[Vec<Option<Estimate>>],
    bin_error: &impl Fn(f64) -> f64,
    label: &str,
) -> Vec<(f64, f64)> {
    let rounds = by_round.len() - 1;
    let mut order: Vec<usize> = (0..bins.len()).collect();
    order.sort_by(|&a, &b| {
        spectrum[bins[b]]
            .norm()
            .total_cmp(&spectrum[bins[a]].norm())
    });
    let mut final_bounds = vec![(0.0, 0.0); bins.len()];
    for round in 1..=rounds {
        for (place, &i) in order.iter().enumerate() {
            let seen = |j: usize| {
                let earlier = order.iter().position(|&other| other == j) < Some(place);
                by_round[if earlier { round } else { round - 1 }][j]
            };
            let extra: [f64; 3] = core::array::from_fn(|m| {
                let position = bins[i] as f64 + m as f64 - 1.0;
                bin_error(position)
                    + (0..tones.len())
                        .filter(|&j| j != i)
                        .map(|j| residual(&tones[j], seen(j), position))
                        .sum::<f64>()
            });
            let offset = offset_bound(&tones[i], bins[i], extra);
            let estimate = by_round[round][i];
            let must_resolve = offset
                .is_some_and(|bound| (tones[i].position - bins[i] as f64).abs() + bound <= 0.5);
            if round == rounds || must_resolve {
                let offset = offset
                    .unwrap_or_else(|| panic!("{label}, round {round}, tone {i}: no offset bound"));
                let estimate = estimate.unwrap_or_else(|| {
                    panic!("{label}, round {round}, tone {i}: estimate rejected")
                });
                assert!(
                    (estimate.position - tones[i].position).abs() <= offset,
                    "{label}, round {round}, tone {i}: position exceeds {offset:e}"
                );
                let solve =
                    solve_bound(&tones[i], bins[i], estimate, extra[1]).unwrap_or_else(|| {
                        panic!("{label}, round {round}, tone {i}: no image-solve bound")
                    });
                assert!(
                    (estimate.a - tones[i].a()).norm() <= solve,
                    "{label}, round {round}, tone {i}: complex amplitude exceeds {solve:e}"
                );
                if round == rounds {
                    final_bounds[i] = (offset, solve);
                }
            }
        }
    }
    final_bounds
}
