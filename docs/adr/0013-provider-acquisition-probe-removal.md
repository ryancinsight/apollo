# ADR 0013: Remove consumer availability probes

## Status

Accepted for the next D8 provider-boundary slice.

## Context

Sixteen Apollo transform crates expose `wgpu_available() -> bool`. Each is an
unused public forwarding function that evaluates its transform backend's typed
Hephaestus acquisition result with `is_ok()`. The boolean erases the acquisition
failure and duplicates a provider concern across consumers.

Transform-specific backend constructors remain necessary where a transform must
request a non-default provider limit. They do not reimplement WGPU mechanics;
they pass a typed limit requirement to Hephaestus.

## Options

1. Retain the boolean probes as convenience APIs.
2. Add a shared Apollo availability helper.
3. Remove the probes and require consumers to observe typed acquisition.

## Decision

Choose option 3. Delete every unused `wgpu_available` definition and root
re-export in one pre-1.0 breaking change, update every affected crate version,
and keep no compatibility alias. Consumers use the existing transform backend
constructor and handle the provider error. A shared Apollo helper would repeat
the discarded failure-erasure at a new location; generic device acquisition
belongs to Hephaestus.

## Consequences

The public surface becomes smaller and failure modes remain typed. The change
does not alter transform mathematics, GPU dispatch, Leto host boundaries, or
Hephaestus ownership. SemVer classification and value-semantic transform tests
provide API and execution evidence; neither is a machine-checked proof of GPU
correctness.
