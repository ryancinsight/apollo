//! Preserve native interpreter tests and Maturin's extension link mode.
//!
//! Maturin >= 1.9.4 selects PYO3_BUILD_EXTENSION_MODULE for wheels;
//! ordinary Cargo tests use PyO3's interpreter-library discovery. The
//! provider helper emits target-specific cdylib linker arguments.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
