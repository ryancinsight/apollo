use super::BenchmarkRecord;
use core::fmt::Write;

pub(super) fn render(records: &[BenchmarkRecord]) -> String {
    let mut output = String::from(
        "case,min_ns,median_ns,median_lower_ns,median_upper_ns,median_confidence_ppm,samples,iterations_per_sample\n",
    );
    for record in records {
        write_csv_field(&record.case.to_string(), &mut output);
        writeln!(
            output,
            ",{},{},{},{},{},{},{}",
            record.minimum_nanoseconds,
            record.median_nanoseconds,
            record.median_lower_nanoseconds,
            record.median_upper_nanoseconds,
            record.median_confidence_parts_per_million,
            record.sample_count,
            record.iterations_per_sample
        )
        .expect("invariant: formatting a String cannot fail");
    }
    output
}

fn write_csv_field(field: &str, output: &mut String) {
    if !field.contains([',', '"', '\n', '\r']) {
        output.push_str(field);
        return;
    }

    output.push('"');
    for character in field.chars() {
        if character == '"' {
            output.push('"');
        }
        output.push(character);
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::case::BenchmarkCase;
    use crate::measurement::SampleSummary;
    use crate::report::BenchmarkRecord;

    #[test]
    fn csv_quotes_separator_quote_and_newline_labels() {
        let summary = SampleSummary::from_samples(vec![7], 1)
            .expect("invariant: literal sample set is non-empty");
        let record = BenchmarkRecord::new(
            BenchmarkCase::new("group,with", "quoted\"label", "line\nbreak"),
            summary,
        );

        assert_eq!(
            render(&[record]),
            "case,min_ns,median_ns,median_lower_ns,median_upper_ns,median_confidence_ppm,samples,iterations_per_sample\n\"group,with/quoted\"\"label/line\nbreak\",7,7,7,7,0,1,1\n"
        );
    }
}
