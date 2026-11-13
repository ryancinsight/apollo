# Milestone ladder ([ADR 0076](../docs/adr/0076-release-scope-1-0.md))
<a id="M-fft-core"></a> **M-fft-core** (0.18.0) needs: (none) — native gates, memory discipline, ratchets.
- exit: pass fft core tests (`cargo test -p apollo-fft --locked --all-features` passes).
- exit: pass dht tests (`cargo test -p apollo-dht --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-transforms-capability"></a> **M-transforms-capability** (0.19.0) needs: M-fft-core — capability and perf parity.
- exit: pass sibling transform tests (`cargo test -p apollo-nufft --locked --all-features` and `cargo test -p apollo-dctdst --locked --all-features` pass).
- exit: pass wavelet and radon tests (`cargo test -p apollo-wavelet --locked --all-features` and `cargo test -p apollo-radon --locked --all-features` pass).
- standing: no open correctness item.
<a id="M-1.0"></a> **M-1.0** (1.0.0) needs: M-transforms-capability — stable transform release (authority-gated).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
