//! Every routable length agrees with a naive DFT.
//!
//! The plan builder picks a strategy from the factorization of `n` (radix,
//! coprime split, Rader, ...). Existing tests exercise a handful of friendly
//! lengths, so a strategy that is selected for a length it cannot serve
//! returns a wrong spectrum without any test noticing. This sweep closes that
//! gap the only way it can be closed: by checking every length against an
//! oracle that shares no code with the planner.
//!
//! The oracle is the defining sum, evaluated directly. It is O(n^2) and makes
//! no claim to speed; its value is that it cannot be wrong in the same way the
//! planner is wrong.
//!
//! Tolerance is derived, not tuned. The oracle accumulates `n` products
//! sequentially, so its own error is bounded by about `n * eps * sum|x|`, and
//! with `|x| <= 1` that is at worst `n^2 * eps`. At the top of the swept range
//! (`n = 2048`, `eps = 2.2e-16`) that bound is 9.2e-10. `TOL` sits an order
//! above it, and two orders is still nine orders below the ~1e1 errors that
//! misrouting produces, so the check separates the two cleanly rather than
//! straddling them.

use apollo_fft::fft_1d_slice;

/// Upper bound on the oracle's own accumulation error across the swept range.
/// See the module note: `n^2 * eps` at `n = 2048` is 9.2e-10.
const TOL: f64 = 1e-8;

/// Naive DFT: `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)`.
///
/// Shares no factorization, twiddle table, or kernel with the planner, which
/// is the whole point of using it as the oracle.
fn naive_dft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    let scale = -2.0 * std::f64::consts::PI / n as f64;
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (j, &xj) in x.iter().enumerate() {
                // `j * k` overflows f64's exact-integer range only far beyond
                // this sweep; reducing mod n first also keeps the angle small.
                let angle = scale * ((j * k) % n) as f64;
                re += xj * angle.cos();
                im += xj * angle.sin();
            }
            (re, im)
        })
        .collect()
}

/// Deterministic, bounded, and not symmetric — a symmetric signal can hide a
/// misrouting whose error cancels in the spectrum.
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let t = j as f64 / n as f64;
            (7.0 * t).sin() + 0.5 * (23.0 * t).cos() - 0.25 * t
        })
        .collect()
}

/// Outcome for one length. A panic is a *correct* outcome for a length the
/// planner cannot serve: it refuses instead of returning corrupt output.
enum Outcome {
    Agrees,
    Refused,
    Disagrees(f64),
}

fn check(n: usize) -> Outcome {
    let x = signal(n);
    let Ok(spectrum) = std::panic::catch_unwind(|| fft_1d_slice::<f64>(&x)) else {
        return Outcome::Refused;
    };
    let expect = naive_dft(&x);
    let worst = spectrum
        .iter()
        .zip(&expect)
        .map(|(got, &(re, im))| (got.re - re).abs().max((got.im - im).abs()))
        .fold(0.0f64, f64::max);
    if worst <= TOL {
        Outcome::Agrees
    } else {
        Outcome::Disagrees(worst)
    }
}

#[test]
fn every_length_either_agrees_with_the_oracle_or_refuses() {
    // Panics are an expected outcome here; the default hook would print a
    // backtrace for each one and bury the report.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut wrong: Vec<(usize, f64)> = Vec::new();
    let mut refused: Vec<usize> = Vec::new();
    for n in 2..=2048 {
        match check(n) {
            Outcome::Agrees => {}
            Outcome::Refused => refused.push(n),
            Outcome::Disagrees(err) => wrong.push((n, err)),
        }
    }

    std::panic::set_hook(hook);

    assert!(
        wrong.is_empty(),
        "{} length(s) returned a spectrum that disagrees with the naive DFT \
         beyond {TOL:e}. A wrong answer from a published transform is worse \
         than a refusal: {:?}{}",
        wrong.len(),
        &wrong[..wrong.len().min(12)],
        if wrong.len() > 12 { " ..." } else { "" }
    );

    // Refusals are recorded, not asserted away: they are the honest failure
    // mode for a length with no correct route, and their count is the size of
    // the coverage gap that a general Bluestein strategy would close.
    if !refused.is_empty() {
        eprintln!(
            "{} length(s) have no correct route and refuse at plan \
             construction: {:?}{}",
            refused.len(),
            &refused[..refused.len().min(24)],
            if refused.len() > 24 { " ..." } else { "" }
        );
    }
}

/// The same check above the swept range, at the lengths the tracked defect
/// named specifically.
///
/// The oracle is quadratic, so the range sweep stops at 2048. These are the
/// squares of the primes just above the supported radix set and the three
/// composites the tracking item called out by number — the largest lengths
/// where the routing was known to be wrong.
#[test]
fn known_hard_lengths_above_the_swept_range_agree() {
    let lengths = [
        2209, // 47^2
        2809, // 53^2
        3481, // 59^2
        3721, // 61^2
        4489, // 67^2
        5041, // 71^2
        6241, // 79^2
        6726, 6727, 6728, // called out by the tracking item
        6889, // 83^2
        7921, // 89^2
    ];

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcomes: Vec<(usize, Outcome)> = lengths.into_iter().map(|n| (n, check(n))).collect();
    std::panic::set_hook(hook);

    let bad: Vec<String> = outcomes
        .iter()
        .filter_map(|(n, outcome)| match outcome {
            Outcome::Agrees => None,
            Outcome::Refused => Some(format!("{n}: refused")),
            Outcome::Disagrees(err) => Some(format!("{n}: off by {err:e}")),
        })
        .collect();
    assert!(bad.is_empty(), "lengths without a correct route: {bad:?}");
}
