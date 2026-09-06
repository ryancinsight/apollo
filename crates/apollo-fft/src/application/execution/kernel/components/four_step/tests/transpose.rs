//! A transpose permutes complete complex representations without arithmetic.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::{layout::cast_slice, Complex};
use leto::LetoError;
use leto_ops::transpose_complex_matrices;

trait Payloads: MixedRadixScalar<Complex = Complex<Self>> + From<u16> {
    const VALUES: [Self; 12];
}

impl Payloads for f32 {
    const VALUES: [Self; 12] = [
        0.0,
        -0.0,
        Self::INFINITY,
        Self::NEG_INFINITY,
        Self::from_bits(0x7fc0_1234),
        Self::from_bits(0xffc0_5678),
        Self::from_bits(0x7f80_1234),
        Self::from_bits(0xff80_5678),
        Self::from_bits(1),
        Self::from_bits(0x8000_0001),
        1.25,
        -3.5,
    ];
}

impl Payloads for f64 {
    const VALUES: [Self; 12] = [
        0.0,
        -0.0,
        Self::INFINITY,
        Self::NEG_INFINITY,
        Self::from_bits(0x7ff8_0000_0000_1234),
        Self::from_bits(0xfff8_0000_0000_5678),
        Self::from_bits(0x7ff0_0000_0000_1234),
        Self::from_bits(0xfff0_0000_0000_5678),
        Self::from_bits(1),
        Self::from_bits(0x8000_0000_0000_0001),
        1.25,
        -3.5,
    ];
}

fn signal<F: Payloads>(len: usize) -> Vec<Complex<F>> {
    (0..len)
        .map(|index| {
            let payload = F::VALUES[index % F::VALUES.len()];
            // Every element has a unique component, so repeated special-value
            // payloads cannot hide a permutation of matrix positions.
            let position = F::from(u16::try_from(index).expect("test index fits u16"));
            if index % 2 == 0 {
                Complex::new(payload, position)
            } else {
                Complex::new(position, payload)
            }
        })
        .collect()
}

fn check_permutation<F: Payloads>() {
    // Register widths 2/4 and tile sides 16/32 are crossed on either axis. The
    // Cartesian product includes square, asymmetric and both tail directions.
    const EXTENTS: [usize; 12] = [0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33];
    for rows in EXTENTS {
        for columns in EXTENTS {
            let len = rows * columns;
            for source_offset in 0..4 {
                for destination_offset in 0..4 {
                    // Four consecutive complex offsets include AVX-unaligned
                    // addresses for either scalar width, regardless of Vec alignment.
                    let source = signal::<F>(source_offset + len + 3);
                    let before = signal::<F>(destination_offset + len + 5);
                    let mut expected = before.clone();
                    for (index, value) in
                        expected[destination_offset..][..len].iter_mut().enumerate()
                    {
                        // Output is columns x rows; quotient and remainder
                        // recover its coordinates independently of tiled traversal.
                        *value = source[source_offset + (index % rows) * columns + index / rows];
                    }
                    let mut actual = before;
                    transpose_complex_matrices(
                        &source[source_offset..source_offset + len],
                        &mut actual[destination_offset..destination_offset + len],
                        1,
                        rows,
                        columns,
                    )
                    .expect("exact matrix slices satisfy the provider contract");
                    assert_eq!(
                        cast_slice::<_, u8>(&actual),
                        cast_slice::<_, u8>(&expected),
                        "{rows} x {columns}, source offset {source_offset}, destination offset {destination_offset}"
                    );
                }
            }
        }
    }
}

fn check_rejection<F: Payloads>() {
    for (rows, columns, source_len, destination_len) in [
        (4, 4, 15, 16),
        (4, 4, 16, 15),
        (17, 3, 50, 51),
        (3, 17, 51, 50),
        (usize::MAX, 2, 4, 4),
        (2, usize::MAX, 4, 4),
    ] {
        let source = signal::<F>(source_len);
        let before = signal::<F>(destination_len);
        let mut destination = before.clone();
        let expected_error = match rows.checked_mul(columns) {
            None => LetoError::Overflow {
                reason: "complex matrix element count",
            },
            Some(expected) => {
                let (role, actual) = if source_len == expected {
                    ("destination", destination_len)
                } else {
                    ("source", source_len)
                };
                LetoError::StorageError {
                    reason: format!(
                        "complex matrix transpose {role} length {actual} does not match expected {expected}"
                    ),
                }
            }
        };
        let outcome = transpose_complex_matrices(&source, &mut destination, 1, rows, columns);
        assert_eq!(
            outcome,
            Err(expected_error),
            "invalid extent {rows} x {columns}"
        );
        assert_eq!(
            cast_slice::<_, u8>(&destination),
            cast_slice::<_, u8>(&before),
            "rejection must precede every write for {rows} x {columns}"
        );
    }
    for (rows, columns) in [(0, usize::MAX), (usize::MAX, 0)] {
        let before = signal::<F>(5);
        let mut destination = before.clone();
        transpose_complex_matrices::<F>(&[], &mut destination[..0], 1, rows, columns)
            .expect("zero extent accepts empty matrix slices");
        assert_eq!(
            cast_slice::<_, u8>(&destination),
            cast_slice::<_, u8>(&before),
            "zero extent is a no-op for {rows} x {columns}"
        );
        transpose_complex_matrices::<F>(&[], &mut [], 1, rows, columns)
            .expect("zero extent accepts empty matrix slices");
    }
}

fn check_scalar<F: Payloads>() {
    check_permutation::<F>();
    check_rejection::<F>();
}

#[test]
fn transpose_preserves_bits_and_validates_before_writes() {
    check_scalar::<f32>();
    check_scalar::<f64>();
}
