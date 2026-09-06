//! Invariant failures shared by transform kernels and axis-layout execution.

use leto::LetoError;
use leto_ops::SquareTransposeError;

// Keep formatting and owned-error cleanup in one concrete failure path rather
// than instantiating them alongside every scalar, shape, and transform direction.
#[cold]
#[inline(never)]
#[track_caller]
pub(super) fn layout_failure(error: LetoError, message: &str) -> ! {
    panic!("{message}: {error:?}");
}

#[cold]
#[inline(never)]
#[track_caller]
pub(super) fn square_failure(error: SquareTransposeError, message: &str) -> ! {
    panic!("{message}: {error:?}");
}

#[cfg(test)]
mod tests {
    use super::{layout_failure, square_failure};
    use leto::LetoError;
    use leto_ops::SquareTransposeError;
    use std::panic::{catch_unwind, UnwindSafe};

    fn panic_message(action: impl FnOnce() + UnwindSafe) -> String {
        *catch_unwind(action)
            .expect_err("an invariant failure must panic")
            .downcast::<String>()
            .expect("an invariant failure preserves the formatted string payload")
    }

    #[test]
    fn layout_failure_preserves_invariant_context_and_error() {
        let cases = [
            (
                "invariant: FourStep factors exactly cover source and transpose workspace",
                LetoError::Overflow {
                    reason: "complex matrix element count",
                },
                "invariant: FourStep factors exactly cover source and transpose workspace: Overflow { reason: \"complex matrix element count\" }",
            ),
            (
                "invariant: FFT matrix batch satisfies Leto's transpose contract",
                LetoError::StorageError {
                    reason: "complex matrix transpose destination length 5 does not match expected 6".into(),
                },
                "invariant: FFT matrix batch satisfies Leto's transpose contract: StorageError { reason: \"complex matrix transpose destination length 5 does not match expected 6\" }",
            ),
            (
                "invariant: validated FFT view shape has a C-order layout",
                LetoError::Overflow {
                    reason: "stride product",
                },
                "invariant: validated FFT view shape has a C-order layout: Overflow { reason: \"stride product\" }",
            ),
            (
                "invariant: dense FFT view slice matches its logical shape",
                LetoError::StorageError {
                    reason: "view storage is too short".into(),
                },
                "invariant: dense FFT view slice matches its logical shape: StorageError { reason: \"view storage is too short\" }",
            ),
            (
                "invariant: FFT view staging matches its logical shape",
                LetoError::ShapeMismatch {
                    lhs: vec![2, 3],
                    rhs: vec![3, 2],
                },
                "invariant: FFT view staging matches its logical shape: ShapeMismatch { lhs: [2, 3], rhs: [3, 2] }",
            ),
        ];
        for (message, error, expected) in cases {
            assert_eq!(panic_message(|| layout_failure(error, message)), expected);
        }
    }

    #[test]
    fn square_failure_preserves_invariant_context_and_dimensions() {
        let message = "invariant: equal FourStep factors exactly cover the transform storage";
        let cases = [
            (
                SquareTransposeError::Overflow { side: usize::MAX },
                format!(
                    "invariant: equal FourStep factors exactly cover the transform storage: Overflow {{ side: {} }}",
                    usize::MAX
                ),
            ),
            (
                SquareTransposeError::Length {
                    side: 3,
                    expected: 9,
                    actual: 8,
                },
                "invariant: equal FourStep factors exactly cover the transform storage: Length { side: 3, expected: 9, actual: 8 }".into(),
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(panic_message(|| square_failure(error, message)), expected);
        }
    }
}
