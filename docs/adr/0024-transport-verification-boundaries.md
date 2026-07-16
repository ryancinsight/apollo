# ADR 0024: Transport verification boundaries

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking verification-boundary cleanup

## Context

Thirteen transform transports publish `verification` modules whose contents
compile only under `cfg(test)`. The paths expose internal CPU oracles,
finite-precision contracts, and provider-acquisition fixtures as release API,
even though production callers cannot use them.

## Decision

Gate every transport verification module with `cfg(test)` and reduce its
visibility to `pub(crate)`. The existing concern-named verification trees and
the four cohesive sub-500-line verification leaves remain in their owning
transforms. No cross-transform test wrapper, provider adapter, or forwarding
module is introduced.

The transform theorems, CPU oracles, fixtures, and derived tolerances are
unchanged. The release-boundary change is verified by compilation and
value-semantic test suites; it is not a machine-checked proof of accelerator
behavior.

## Consequences

- `infrastructure::transport::gpu::verification` is no longer a public path
  in the affected pre-1.0 crates.
- Test code remains colocated with its transform-specific contract, while
  Hephaestus retains device mechanics and Apollo retains mathematical evidence.
