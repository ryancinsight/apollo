# ADR 0002: Git-source release contract

- Status: Accepted
- Date: 2026-07-13
- Change class: arch

## Context

Apollo consumes unpublished Atlas crates through exact Git revisions and
uses sibling path patches for coordinated local development. Cargo packaging
cannot publish Apollo crates independently because crates.io cannot resolve
those first-party dependencies. Treating `cargo package` as the release
boundary would describe a distribution channel the dependency graph does not
support.

## Decision

Apollo releases as a Git tag with a committed lockfile, pinned Rust toolchain,
and exact first-party revisions. Direct Git dependency revisions live in the
root manifest. The checkout composite records exact revisions for locally
patched and transitive providers, including Hephaestus and Themis, whose path
identities intentionally replace Git sources in the resolved local lock graph.
CI reconstructs that sibling layout and runs formatting, clippy, nextest,
doctests, documentation, and Python binding tests. Crates.io publication is out
of contract until every public first-party dependency is published under the
required package name and version.

The workspace follows Hephaestus's public WGPU ABI. Apollo 0.15.0 consumes
Hephaestus 0.13.0 and WGPU 30 as one provider-owned contract; no second WGPU
type family or downstream adapter exists.

## Rejected alternatives

- Publish path-only manifests: Cargo replaces paths with registry resolution,
  where the first-party packages do not exist.
- Remove provider dependencies for packaging: produces a different artifact
  than the tested workspace.
- Upgrade Apollo independently of Hephaestus: forks the GPU device contract and
  duplicates backend types.
- Leave provider revisions floating: prevents a tag from reproducing the
  verified dependency graph.

## Verification

Release eligibility requires locked metadata, the local pre-merge gate, Python
binding tests, supply-chain checks, and an API comparison against the previous
Git revision. The GitHub checkout composite is the revision SSOT for path-only
and transitive providers; all other checkout revisions match the root manifest.
