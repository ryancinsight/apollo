# Leto 0.47 compatibility for borrowed CSR consumers

Base: `f266bb0b319cda8dccde6090936967130f5af7da`. Target: apollo-fft 0.29.2; affected family patches in CHANGELOG.md. Classification: [patch].
This isolated export leaves every existing checkout and its owner untouched.

## Contract

Leto draft #362 implements accepted ADR 0036. Its 0.47 addition preserves the
existing default-index/ordering and scalar/SIMD APIs; its minor API comparison
against 0.46 reports no removal. This provider uses the existing layout, array
and operation vocabulary. No new CSR API appears in its public interface.

The old `^0.46.0` requirement rejects the source graph when Coeus selects 0.47.
Accept `>=0.46.0, <0.48.0` for both Leto layers. The interval retains the
current 0.46 cohort and admits the verified additive 0.47 line; later pre-1.0
minor lines require a new compatibility audit. Every first-party source stays
canonical, with one resolved provider identity. No conversion adapter, path
patch, manifest revision pin, backend feature or Rust source changes.

Published packages whose dependency closure changes receive patch versions.
The committed lock retains merged provider defaults (currently Leto 0.46).
A separate review consumer must exercise Leto 0.47 and both matching provider
bindings before Coeus claims its complete graph. This source compatibility
change does not authorize provider merge, registry publication or Pages.

## Verification state

The original Rust implementation/test tree is unchanged. Native, strict
Clippy, exact-source API, fresh-consumer and hosted checks remain required on
the committed candidate. The 0.46 committed-graph result and the 0.47 review
consumer result are separate gates; neither substitutes for actual GPU
hardware execution or publication receipts.

Main merges FFT-layout recovery #671 and publishes FFT 0.29.1 during review.
This candidate retains that complete source as its base and targets 0.29.2.
