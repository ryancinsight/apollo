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

The standalone lockfile check passes, and the exact graph passes workspace
check, warning-denied Clippy, nextest, doctests, warning-denied rustdoc, and
formatting.

Source identity is **not** one revision per provider, contrary to what this
section previously recorded. The committed lock resolves Eunomia through two
sources, Mnemosyne through three, and Moirai through two — four sources in
excess of one per repository. The count is now measured rather than asserted:
atlas's canonical lock guard reports it on every run and fails when it exceeds
`.provider-identity-baseline`, which this repository commits at 4 and lowers as
the dependency-ordered unpin sweep closes each fork
([`backlog.md#apollo-provider-identity`](../../backlog.md#apollo-provider-identity)).

The decision changes dependency resolution only; it does not change Apollo's
public API or numerical kernels.

## Revisions

- **2026-09-04.** The verification section asserted one revision per provider
  without measuring it; a lock scan showed four sources in excess of one per
  repository. Corrected, and the claim replaced by a committed bound the lock
  guard enforces. A merged co-evolution pin is additionally established as not
  independently removable: dropping four merged pins at once doubled the
  excess, because transitive first-party consumers still pinned the older
  revisions.
