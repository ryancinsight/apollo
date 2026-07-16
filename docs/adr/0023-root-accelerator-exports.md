# ADR 0023: Root accelerator exports

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking API cleanup

## Context

Thirteen transform crates expose `wgpu_backend`, a public module that only
re-exports the same typed Hephaestus accelerator API already exported from the
crate root. The module owns no device acquisition, buffer, queue, pipeline, or
kernel logic. Keeping both paths creates a redundant public hierarchy and
preserves an obsolete consumer-local name for an Atlas provider boundary.

## Decision

Delete every root `wgpu_backend` forwarding module in the audited transform
set. Retain each existing feature-gated root re-export as the single public
accelerator path. Consumers import typed accelerator items directly from their
transform crate; they never import a raw WGPU surface from Apollo.

No transform theorem or numerical contract changes: the same Leto host
boundary and Hephaestus-typed execution remain in force. The claim is verified
by source inspection and value-semantic transform tests; it is not a
machine-checked proof of GPU behavior.

## Consequences

- `apollo_<transform>::wgpu_backend::*` is removed for the affected pre-1.0
  crates, requiring a pre-1.0 breaking version increment.
- The root exports are the sole migration target; no compatibility alias or
  forwarding adapter remains.
- Hephaestus retains provider ownership, while Apollo keeps transform-specific
  contracts and CPU-oracle verification.
