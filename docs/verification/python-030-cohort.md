# Apollo Python 0.30 cohort renewal

## Decision and preservation

[ADR 0076](../adr/0076-python-030-ffi-cohort.md) records the approved shared
Python FFI boundary. The private binding advances to 0.3.0, PyO3/build-config
0.30 and Python/ABI3 3.9. Native package versions retain the published PR #690
cohort. The only changed Rust line relative to that main explicitly requires
the GIL; transform code, tests and compiler profiles remain byte-identical.
PR #674 remains the preserved historical source qualification.

## Local verification

The renewed canonical lock keeps Eunomia 0.10 and Aequitas 0.3.1. Rust 1.97
passes formatting, strict workspace all-feature/all-target Clippy,
warning-denied docs, 1,643 native cases with 15 existing skips, and 17 doctests
with one existing ignore. Rust 1.95 checks the complete all-feature workspace.
An ordinary macOS ARM64 development wheel has `cp39-abi3` metadata, requires
Python 3.9, has no libpython dependency, and passes all 35 existing Python cases
under CPython 3.13 / NumPy 2.5.4. The installed distribution identity is checked.

A disposable shared Rust graph qualifies reviewed Eunomia 0.11 with NumPy 0.30,
PyO3 0.30.1 and the renewed Leto 0.47 source, while retaining Aequitas 0.3.1.
The old Apollo 0.29 FFI graph fails Cargo's native-link identity constraint;
the renewed graph has exactly one identity for each provider/FFI crate. Both
actual binding unit targets compile and execute in that graph. Two external
caller cases check analytic FFT/NumPy values at both native widths and typed
matrix/FFT values returned by the two installed qualified wheels. Strict caller
Clippy passes. The qualified Apollo wheel passes the same 35 Python cases.

The first embedded-Python run cannot locate NumPy; exposing the selected
interpreter's existing site-packages makes the unchanged test pass. An initial
probe names the virtual Eunomia workspace instead of its package and is
invalid. A later disposable caller assumes the Python wrapper exposes the
extension's version; its corrected lookup and distribution metadata checks
preserve every numerical assertion. The failed setup evidence remains in the
external review packet; no source/test/gate suppression addresses those errors.

## Remaining acceptance

These are local source-qualified results. No Aequitas 0.4/final canonical
backend/Python cohort, full wheel matrix, free-threaded safety or physical
NVIDIA/AMD acceptance follows. Miri remains deferred and explicitly unverified.
Independent exact-head review and unchanged hosted gates remain required.
There is no push, merge, publication, dispatch or credential/settings change.
