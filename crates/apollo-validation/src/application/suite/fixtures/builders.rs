//! Helper builders for [`PublishedFixtureReport`] assembly.

use super::super::metrics::max_complex_abs_delta;
use crate::domain::report::PublishedFixtureReport;
use eunomia::Complex64;

pub(crate) const PUBLISHED_FIXTURE_LIMIT: f64 = 1.0e-12;

pub(crate) fn published_complex_fixture<'a, I, J>(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: I,
    expected: J,
) -> PublishedFixtureReport
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    let max_abs_error = max_complex_abs_delta(actual, expected);
    PublishedFixtureReport {
        transform: transform.to_string(),
        fixture: fixture.to_string(),
        reference: reference.to_string(),
        max_abs_error,
        threshold: PUBLISHED_FIXTURE_LIMIT,
        passed: max_abs_error <= PUBLISHED_FIXTURE_LIMIT,
    }
}

pub(crate) fn published_real_fixture(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: &[f64],
    expected: &[f64],
) -> PublishedFixtureReport {
    published_real_fixture_with_threshold(
        transform,
        fixture,
        reference,
        actual,
        expected,
        PUBLISHED_FIXTURE_LIMIT,
    )
}

pub(crate) fn published_real_fixture_with_threshold(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: &[f64],
    expected: &[f64],
    threshold: f64,
) -> PublishedFixtureReport {
    let max_abs_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max);
    PublishedFixtureReport {
        transform: transform.to_string(),
        fixture: fixture.to_string(),
        reference: reference.to_string(),
        max_abs_error,
        threshold,
        passed: max_abs_error <= threshold,
    }
}
