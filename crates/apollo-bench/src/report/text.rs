use super::BenchmarkRecord;
use core::fmt::Write;

pub(super) fn render(records: &[BenchmarkRecord]) -> String {
    let mut output = String::from(
        "case,min_ps,median_ps,median_lower_ps,median_upper_ps,median_confidence_ppm,ordered_samples_ps,iterations_per_sample\n",
    );
    for record in records {
        write_csv_field(&record.case.to_string(), &mut output);
        write!(
            output,
            ",{},{},{},{},{},",
            record.minimum_picoseconds,
            record.median_picoseconds,
            record.median_lower_picoseconds,
            record.median_upper_picoseconds,
            record.median_confidence_parts_per_million,
        )
        .expect("invariant: formatting a String cannot fail");
        write_ordered_samples(&record.ordered_samples_picoseconds, &mut output);
        writeln!(output, ",{}", record.iterations_per_sample)
            .expect("invariant: formatting a String cannot fail");
    }
    output
}

fn write_ordered_samples(samples: &[u128], output: &mut String) {
    let mut samples = samples.iter();
    if let Some(first) = samples.next() {
        write!(output, "{first}").expect("invariant: formatting a String cannot fail");
    }
    for sample in samples {
        write!(output, ";{sample}").expect("invariant: formatting a String cannot fail");
    }
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
        let summary = SampleSummary::from_samples(vec![7; 6], 1)
            .expect("invariant: six samples support the one-case interval");
        let record = BenchmarkRecord::new(
            BenchmarkCase::new("group,with", "quoted\"label", "line\nbreak"),
            summary,
        );

        assert_eq!(
            render(&[record]),
            "case,min_ps,median_ps,median_lower_ps,median_upper_ps,median_confidence_ppm,ordered_samples_ps,iterations_per_sample\n\"group,with/quoted\"\"label/line\nbreak\",7,7,7,7,968750,7;7;7;7;7;7,1\n"
        );
    }
}
