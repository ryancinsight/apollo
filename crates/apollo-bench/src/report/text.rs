use super::BenchmarkRecord;
use core::fmt::Write;

pub(super) fn render(records: &[BenchmarkRecord]) -> String {
    let mut output = String::from("case,min_ns,median_ns,samples,iterations_per_sample\n");
    for record in records {
        writeln!(
            output,
            "{},{},{},{},{}",
            record.case,
            record.minimum_nanoseconds,
            record.median_nanoseconds,
            record.sample_count,
            record.iterations_per_sample
        )
        .expect("invariant: formatting a String cannot fail");
    }
    output
}
