# ADR 0026: Provider acquisition forwarders

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking provider-boundary cleanup

## Context

Fifteen transform GPU backends publish a `try_default` method that only calls
`hephaestus_wgpu::WgpuDevice::try_default` with a label and wraps the result in
the transform backend. The factories duplicate provider acquisition ownership
and allow their test callers to erase any acquisition failure with `.ok()`.

## Decision

Delete the fifteen forwarding factories without an alias. Verification and
benchmark callers acquire `WgpuDevice` from Hephaestus and construct their
transform backend through its existing `new(device)` algorithm boundary.
Device-present verification skips only `HephaestusError::AdapterUnavailable`;
all other acquisition failures fail the test.

The NUFFT and STFT acquisition paths are excluded: each requests a
transform-specific storage-buffer limit and is not a zero-behavior forwarder.
The FFT adapter is also excluded because it is the shared transform-composition
boundary rather than a per-transform factory.

## Theorem and evidence boundary

This change modifies neither transform formulas nor numerical tolerances. The
existing transform theorems and CPU differentials remain their respective
analytical and empirical evidence. API-surface scans, value-semantic tests, and
SemVer classification establish release-boundary evidence only; they do not
prove accelerator execution.

## Consequences

- Hephaestus is the sole owner of default device acquisition for the selected
  transforms.
- Apollo backends retain transform algorithms but no longer wrap the provider
  factory.
- A faulty present provider can no longer be misclassified as an absent adapter
  by the affected verification callers.
