# ADR 0028: FFT provider acquisition boundary

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking provider-boundary cleanup

## Context

`apollo-fft::WgpuBackend::try_default` acquires a Hephaestus device and wraps
it in the FFT composition backend. The method adds no transform resource
requirement or provider capability; it duplicates Hephaestus acquisition and
converts a typed provider error into an Apollo string.

## Decision

Delete `WgpuBackend::try_default` without an alias. Callers acquire
`hephaestus_wgpu::WgpuDevice` directly and pass it to `WgpuBackend::new`.
Benchmarks and device-present regressions skip only `AdapterUnavailable`; any
other provider or fixed-plan failure is surfaced.

## Theorem and evidence boundary

The change is an API-boundary deletion and does not alter the FFT operator.
Existing inverse-pair, Parseval, and CPU/GPU differential contracts remain the
numerical evidence. Source-residue scans, compilation, value-semantic tests,
and SemVer classification establish only release-boundary evidence.

## Consequences

- Hephaestus solely owns FFT adapter acquisition and typed errors.
- Apollo retains FFT plan composition through `WgpuBackend::new`.
