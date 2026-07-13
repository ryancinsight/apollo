# ADR 0001: Fallible WGPU device construction

- Status: Accepted
- Date: 2026-07-13
- Change class: major

## Context

Hephaestus 0.12 registers Mnemosyne WGPU staging callbacks when wrapping a
caller-owned device and queue. Registration can fail when a different callback
pair already owns the process staging backend. Apollo's wrapper exposed an
infallible `WgpuDevice::new`, which could no longer represent this provider
contract without a panic or hidden fallback.

## Decision

`apollo_wgpu_helpers::WgpuDevice::new` returns `WgpuDeviceResult<Self>` and
maps the Hephaestus error into Apollo's existing typed device error. The same
private translation function serves caller-owned and default acquisition paths.
Callers propagate failure with `?`. The helper crate version advances from
0.1.0 to 0.2.0.

## Rejected alternatives

- Unwrap or expect: rejects an input-dependent provider failure by panicking.
- Ignore registration failure: violates Mnemosyne's process ownership invariant.
- Retain Hephaestus 0.11: leaves Apollo on a stale provider graph and defers the
  contract mismatch.
- Add a second compatibility constructor: creates dual API paths for one role.

## Verification

Rust's return type enforces failure handling at every call site. Unit tests
assert exact adapter/device error variants and messages. Workspace format,
clippy, nextest, doctest, and documentation gates validate integration.
