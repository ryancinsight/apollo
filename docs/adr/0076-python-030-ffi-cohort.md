# ADR 0076: Python binding in the PyO3 0.30 cohort

- Status: Accepted
- Date: 2026-10-20
- Item: APOLLO-PYTHON-030-COHORT [major]

## Context and contracts

The approved coordinated provider migration selects PyO3/NumPy 0.30 and
Eunomia 0.11. Apollo's private binding still selects PyO3 0.29. A Rust process
cannot link both PyO3 FFI versions because they claim the same `python` links
identity. A lock on the older graph does not admit the future cohort.

PyO3 0.30 removes `abi3-py38`; its minimum stable ABI is Python 3.9. Keeping
the declared Python 3.8 floor would promise an unsupported build. This is a
breaking binding change, so the pre-1.0 binding advances from 0.2.3 to 0.3.0.
Native libraries retain their already published versions and contracts.

## Decision

Select one PyO3 0.30 identity for both the binding and its build helper, use
`abi3-py39`, and require Python 3.9 or newer in distribution metadata and the
existing wheel recipe. Maturin enables `pyo3/extension-module` for wheels;
native Rust tests retain interpreter linking. Remove the obsolete ABI3
interpreter-version suppression. The module explicitly requires
the GIL because free-threaded operation has not been certified. Apollo uses
NumPy through Python's buffer boundary and does not add a Rust NumPy adapter.

Retain the canonical floating Git dependencies, current published provider
minimums, and source admissions for Eunomia 0.10/0.11 and Leto 0.46/0.47.
Do not add revision pins, forwarding interfaces, numerical conversions or
alternative transform implementations. Preserve the original PR #674 source
qualification as historical evidence; current native compatibility is shipped
by PR #690.

## Migration and alternatives

Python 3.8 users must retain the older binding; 0.3.0 requires Python 3.9.
Rust embedding consumers select PyO3/NumPy 0.30 and a consistent provider
cohort together. Python transform names, precision profiles, normalization,
layout rejection, and computed values retain their existing contracts.

Retaining PyO3 0.29 fails the shared Rust FFI graph. Advertising Python 3.8
with PyO3 0.30 has no corresponding ABI feature. A duplicate adapter or a
second Python FFI identity does not resolve nominal type/link ownership.

## Verification and failure modes

Require locked resolution, native Python-boundary values, strict lint/docs,
the unchanged Python smoke suite against an installed ordinary wheel, and
wheel metadata/link inspection. A disposable common Rust caller must compile
with the actual binding in the same resolution graph and inspect computed
Eunomia NumPy, Leto and Apollo values. Qualified future-provider overrides
belong only to that disposable probe, never production manifests or locks.

Local CPython/macOS results do not establish the complete wheel matrix,
free-threaded safety, physical NVIDIA/AMD execution, or the final canonical
backend/Python cohort. Required hosted/release gates and independent review
remain separate. No publication, dispatch or credential change follows from
this local decision. Miri remains deferred and explicitly unverified.
