//! Fresh-process execution for tests that mutate or measure process-wide state.
//!
//! A mutex cannot undo permanent idle-hook registration or exclude ordinary
//! transforms from global allocation windows. Run the original test body alone
//! in the same test executable, under both libtest and nextest. The parent never
//! changes its environment or the state measured by the child.

use std::process::Command;

const CHILD_TEST: &str = "APOLLO_FFT_ISOLATED_TEST";

/// Returns `true` after the parent runs this test in a fresh process. The child
/// returns `false` so the caller continues into its original assertions.
/// `test` is the crate-qualified name built with `module_path!()`.
pub(crate) fn isolate(test: &str) -> bool {
    if std::env::var(CHILD_TEST).as_deref() == Ok(test) {
        return false;
    }

    // libtest's names omit the crate prefix present in module_path!().
    let (_, name) = test.split_once("::").expect("crate-qualified test name");
    let output = Command::new(std::env::current_exe().expect("library test executable"))
        .args([
            "--exact",
            name,
            "--nocapture",
            "--test-threads=1",
            "--include-ignored",
        ])
        .env(CHILD_TEST, test)
        .output()
        .expect("start isolated library test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "isolated test {name} failed ({}):\n{stdout}\n{stderr}",
        output.status
    );
    // An incorrect --exact name otherwise succeeds with zero tests executed.
    assert!(
        stdout.contains("test result: ok. 1 passed;"),
        "isolated test {name} did not execute exactly once:\n{stdout}\n{stderr}"
    );
    true
}

#[test]
#[should_panic(expected = "did not execute exactly once")]
fn an_unmatched_test_filter_is_a_failure() {
    isolate(concat!(module_path!(), "::unregistered_test"));
}
