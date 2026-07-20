use super::error::RecordInvariant;
use super::ComparisonError;
use csv::StringRecord;
use std::collections::BTreeMap;
use std::path::Path;

const HEADER: [&str; 8] = [
    "case",
    "min_ns",
    "median_ns",
    "median_lower_ns",
    "median_upper_ns",
    "median_confidence_ppm",
    "samples",
    "iterations_per_sample",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ReportRecord {
    pub(super) lower_nanoseconds: u128,
    pub(super) upper_nanoseconds: u128,
    pub(super) confidence_parts_per_million: u32,
}

pub(super) fn read_report(path: &Path) -> Result<BTreeMap<String, ReportRecord>, ComparisonError> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|source| ComparisonError::read_report(path, source))?;
    let headers = reader
        .headers()
        .map_err(|source| ComparisonError::read_report(path, source))?;
    if !headers.iter().eq(HEADER) {
        return Err(ComparisonError::unexpected_header(
            path,
            headers.iter().collect::<Vec<_>>().join(","),
        ));
    }

    let mut records = BTreeMap::new();
    for (index, result) in reader.records().enumerate() {
        let row = u64::try_from(index + 2).expect("invariant: report row index fits in u64");
        let record = result.map_err(|source| ComparisonError::read_report(path, source))?;
        let case = field(path, row, &record, 0, "case")?.to_owned();
        let minimum_nanoseconds = parse_u128(path, row, &record, 1, "min_ns")?;
        let median_nanoseconds = parse_u128(path, row, &record, 2, "median_ns")?;
        let median_lower_nanoseconds = parse_u128(path, row, &record, 3, "median_lower_ns")?;
        let median_upper_nanoseconds = parse_u128(path, row, &record, 4, "median_upper_ns")?;
        let median_confidence_parts_per_million =
            parse_u32(path, row, &record, 5, "median_confidence_ppm")?;
        let samples = parse_u128(path, row, &record, 6, "samples")?;
        let iterations = parse_u128(path, row, &record, 7, "iterations_per_sample")?;

        validate_record(
            path,
            row,
            &case,
            minimum_nanoseconds,
            median_nanoseconds,
            median_lower_nanoseconds,
            median_upper_nanoseconds,
            median_confidence_parts_per_million,
            samples,
            iterations,
        )?;
        let prior = records.insert(
            case.clone(),
            ReportRecord {
                lower_nanoseconds: median_lower_nanoseconds,
                upper_nanoseconds: median_upper_nanoseconds,
                confidence_parts_per_million: median_confidence_parts_per_million,
            },
        );
        if prior.is_some() {
            return Err(ComparisonError::duplicate_case(path, row, &case));
        }
    }

    if records.is_empty() {
        return Err(ComparisonError::empty_report(path));
    }
    Ok(records)
}

#[allow(clippy::too_many_arguments)]
fn validate_record(
    path: &Path,
    row: u64,
    case: &str,
    minimum_nanoseconds: u128,
    median_nanoseconds: u128,
    median_lower_nanoseconds: u128,
    median_upper_nanoseconds: u128,
    median_confidence_parts_per_million: u32,
    samples: u128,
    iterations: u128,
) -> Result<(), ComparisonError> {
    let invariant = if case.is_empty() {
        Some(RecordInvariant::EmptyCase)
    } else if minimum_nanoseconds > median_lower_nanoseconds {
        Some(RecordInvariant::MinimumExceedsLowerBound)
    } else if !(median_lower_nanoseconds..=median_upper_nanoseconds).contains(&median_nanoseconds) {
        Some(RecordInvariant::MedianOutsideBounds)
    } else if median_confidence_parts_per_million > 1_000_000 {
        Some(RecordInvariant::ConfidenceExceedsOne)
    } else if samples == 0 {
        Some(RecordInvariant::ZeroSamples)
    } else if iterations == 0 {
        Some(RecordInvariant::ZeroIterations)
    } else {
        None
    };

    if let Some(invariant) = invariant {
        return Err(ComparisonError::invalid_record(path, row, case, invariant));
    }
    Ok(())
}

fn field<'record>(
    path: &Path,
    row: u64,
    record: &'record StringRecord,
    index: usize,
    column: &'static str,
) -> Result<&'record str, ComparisonError> {
    record
        .get(index)
        .ok_or_else(|| ComparisonError::missing_field(path, row, column))
}

fn parse_u128(
    path: &Path,
    row: u64,
    record: &StringRecord,
    index: usize,
    column: &'static str,
) -> Result<u128, ComparisonError> {
    let value = field(path, row, record, index, column)?;
    value
        .parse()
        .map_err(|source| ComparisonError::invalid_integer(path, row, column, value, source))
}

fn parse_u32(
    path: &Path,
    row: u64,
    record: &StringRecord,
    index: usize,
    column: &'static str,
) -> Result<u32, ComparisonError> {
    let value = field(path, row, record, index, column)?;
    value
        .parse()
        .map_err(|source| ComparisonError::invalid_integer(path, row, column, value, source))
}
