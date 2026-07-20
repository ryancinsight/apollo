use super::error::RecordInvariant;
use super::ComparisonError;
use crate::statistics::median::MedianInterval;
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
    "ordered_samples_ns",
    "iterations_per_sample",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ReportRecord {
    ordered_samples_nanoseconds: Box<[u128]>,
}

impl ReportRecord {
    pub(super) fn interval(&self, compared_cases: usize) -> Option<MedianInterval> {
        let simultaneous_intervals = compared_cases.checked_mul(2)?;
        MedianInterval::from_ordered_samples(
            &self.ordered_samples_nanoseconds,
            simultaneous_intervals,
        )
    }

    pub(super) fn sample_count(&self) -> usize {
        self.ordered_samples_nanoseconds.len()
    }
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
        let parsed = ParsedRecord {
            case: field(path, row, &record, 0, "case")?.to_owned(),
            minimum_nanoseconds: parse_u128(path, row, &record, 1, "min_ns")?,
            median_nanoseconds: parse_u128(path, row, &record, 2, "median_ns")?,
            median_lower_nanoseconds: parse_u128(path, row, &record, 3, "median_lower_ns")?,
            median_upper_nanoseconds: parse_u128(path, row, &record, 4, "median_upper_ns")?,
            median_confidence_parts_per_million: parse_u32(
                path,
                row,
                &record,
                5,
                "median_confidence_ppm",
            )?,
            ordered_samples_nanoseconds: parse_samples(
                path,
                row,
                &record,
                6,
                "ordered_samples_ns",
            )?,
            iterations: parse_u128(path, row, &record, 7, "iterations_per_sample")?,
        };
        parsed.validate(path, row)?;
        let prior = records.insert(
            parsed.case.clone(),
            ReportRecord {
                ordered_samples_nanoseconds: parsed.ordered_samples_nanoseconds,
            },
        );
        if prior.is_some() {
            return Err(ComparisonError::duplicate_case(path, row, &parsed.case));
        }
    }

    if records.is_empty() {
        return Err(ComparisonError::empty_report(path));
    }
    Ok(records)
}

struct ParsedRecord {
    case: String,
    minimum_nanoseconds: u128,
    median_nanoseconds: u128,
    median_lower_nanoseconds: u128,
    median_upper_nanoseconds: u128,
    median_confidence_parts_per_million: u32,
    ordered_samples_nanoseconds: Box<[u128]>,
    iterations: u128,
}

impl ParsedRecord {
    fn validate(&self, path: &Path, row: u64) -> Result<(), ComparisonError> {
        let samples = &self.ordered_samples_nanoseconds;
        let interval = MedianInterval::from_ordered_samples(samples, 1);
        let central = samples
            .get((samples.len().saturating_sub(1)) / 2)
            .zip(samples.get(samples.len() / 2))
            .map(|(lower, upper)| lower + (upper - lower) / 2);

        let invariant = if self.case.is_empty() {
            Some(RecordInvariant::EmptyCase)
        } else if samples.is_empty() {
            Some(RecordInvariant::ZeroSamples)
        } else if !samples.windows(2).all(|pair| pair[0] <= pair[1]) {
            Some(RecordInvariant::SamplesNotOrdered)
        } else if samples.first().copied() != Some(self.minimum_nanoseconds) {
            Some(RecordInvariant::MinimumMismatch)
        } else if central != Some(self.median_nanoseconds) {
            Some(RecordInvariant::MedianMismatch)
        } else if interval.is_none() {
            Some(RecordInvariant::UnsupportedSampleCount)
        } else if interval.is_some_and(|interval| {
            interval.lower_nanoseconds != self.median_lower_nanoseconds
                || interval.upper_nanoseconds != self.median_upper_nanoseconds
                || interval.confidence_parts_per_million != self.median_confidence_parts_per_million
        }) {
            Some(RecordInvariant::MedianIntervalMismatch)
        } else if self.iterations == 0 {
            Some(RecordInvariant::ZeroIterations)
        } else {
            None
        };

        if let Some(invariant) = invariant {
            return Err(ComparisonError::invalid_record(
                path, row, &self.case, invariant,
            ));
        }
        Ok(())
    }
}

fn parse_samples(
    path: &Path,
    row: u64,
    record: &StringRecord,
    index: usize,
    column: &'static str,
) -> Result<Box<[u128]>, ComparisonError> {
    let value = field(path, row, record, index, column)?;
    if value.is_empty() {
        return Ok(Box::new([]));
    }
    value
        .split(';')
        .map(|sample| {
            sample.parse().map_err(|source| {
                ComparisonError::invalid_integer(path, row, column, sample, source)
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
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
