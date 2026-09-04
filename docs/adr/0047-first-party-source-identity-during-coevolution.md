# ADR 0047: First-Party Source Identity During Co-evolution

- **Status:** Accepted (2026-09-03)
- **Class:** [patch] [arch]
- **Item:** `ATLAS-APOLLO-FIRST-PARTY-LAYOUT-2026-09-03`
- **Driving evidence:** [`backlog.md#atlas-apollo-first-party-layout`](../../backlog.md#atlas-apollo-first-party-layout)

## Context

Apollo consumes first-party Eunomia, Mnemosyne, Hermes, Leto, Hephaestus, and
Moirai packages. During coordinated changes, a transitive provider can follow
its default branch and cause Cargo to resolve the same package name from a
second commit. Rust treats those crate instances as different types, so
layout, buffer, and planner values cannot cross the resulting boundary.

The consumer graph must remain reproducible while the upstream pull requests
are open. A direct consumer pin cannot repair a provider edge that still
follows an older source identity.

## Decision

Apollo's workspace dependency table is the source of truth for temporary
co-evolution pins. The current graph follows these reviewed revisions:

| Provider | Revision | Upstream change |
| --- | --- | --- |
| Eunomia | `fdbf122` | PR #87 |
| Mnemosyne | `da5c6be` | PR #123 |
| Hermes | `5a399ee` | PR #155 |
| Leto | `1caa846` | PR #164 |
| Hephaestus | `7ca992d` | PR #270 |
| Moirai | `773c117` | PR #256 |

Each pin is documented at the manifest edge and is removed after its
upstream change merges, followed by standalone lockfile regeneration. Provider
repositories own their transitive source edges; Apollo consumes their reviewed
revision and does not add a downstream `[patch]`, path override, or conversion
adapter.

## Alternatives Rejected

- A consumer-level `[patch]` would hide an incorrect provider edge and make
  standalone consumers resolve a different graph.
- Converting values between duplicate crate instances would preserve the
  duplicate source graph and add compatibility code at every consumer seam.
- Leaving Leto or Hephaestus on earlier revisions would reintroduce the old
  Moirai and Mnemosyne source identities after the direct pins were corrected.

## Verification

The standalone lockfile check passes. The lockfile and neutral-directory
`cargo tree` source scans contain one revision for each provider in the table
above; the previous Leto, Moirai, and Mnemosyne source entries are absent. The
exact graph passes workspace check, warning-denied Clippy, nextest (1,187/1,187
with 29 skipped), doctests, warning-denied rustdoc, and formatting.

The decision changes dependency resolution only; it does not change Apollo's
public API or numerical kernels.
