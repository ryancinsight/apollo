# ADR 0029: Native-f16 FFT provider acquisition boundary

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking provider-boundary cleanup

## Context

`GpuFft3dF16Native::try_new` acquires a `ShaderF16`-qualified Hephaestus
device and converts the typed provider error into `String`. It duplicates the
provider-acquisition role that Hephaestus owns. Its only in-repository callers
are device-present tests, where `let Ok` currently suppresses all failures.

## Decision

Delete `GpuFft3dF16Native::try_new` without an alias. Retain
`try_from_device(device, nx, ny, nz)` as the native-half plan-construction
boundary. The caller acquires `WgpuDevice` directly from Hephaestus with
`DeviceFeature::ShaderF16`; tests skip only
`HephaestusError::AdapterUnavailable` and panic for every other provider
fault.

## Theorem and evidence boundary

The f16 DFT convention, inverse-pair law, and the existing
`gamma_k * ||x||_1` rounding bounds are unchanged. This decision changes only
ownership of provider acquisition and error visibility. SemVer classification,
source scans, compilation, and the existing value-semantic tests establish
release-boundary and empirical evidence, not a machine-checked theorem or a
claim that an arbitrary host exposes `ShaderF16`.

## Consequences

- Hephaestus solely owns feature-qualified native-f16 device acquisition and
  its typed errors.
- Apollo retains native-half plan validation and execution through
  `try_from_device`.
- Existing callers must acquire the provider device before constructing the
  plan; no compatibility wrapper remains.
