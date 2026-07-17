# ADR 0033: Remove unused CPU marker aliases

## Status

Accepted on 2026-07-16.

## Context

Fourteen Apollo GPU transport manifests export a public
`CpuTransformMarker` type alias to the owning transform plan. A workspace-wide
reference scan finds no caller, re-export, test, or documentation consumer.
The aliases add a second public name for an existing plan without carrying a
trait bound, capability, or typestate invariant.

## Decision

Delete the aliases from every affected GPU transport manifest. The owning
transform plan and the existing crate dependency remain the sole dependency
direction contract; no compatibility alias is retained.

## Dependency-direction theorem

Let `G` be a GPU transport crate and `P` its owning transform plan. Because the
alias has no references and expands only to `P`, removing it changes neither
the type graph nor any reachable implementation. Cargo's existing dependency
edge `G -> P` remains the authoritative direction. This is a structural proof
from the reference scan and type definition, not a machine-checked theorem;
per-crate compilation and source-residue scans provide executable evidence.

## Consequences

The GPU manifests expose only provider-boundary types with active consumers.
The removal is pre-1.0 breaking for the fourteen package surfaces and does not
alter transform mathematics, Hephaestus acquisition, or Leto storage.
