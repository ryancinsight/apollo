
use anyhow::{bail, Result};
use std::env;

mod benchmark;
mod provider_audit;

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("provider-audit") => provider_audit::run(args),
        Some("benchmark") => benchmark::run(args),
        Some("-h" | "--help" | "help") => {
            print_help();
            Ok(())
        }
        Some(command) => bail!("unknown xtask command `{command}`"),
        None => {
            print_help();
            Ok(())
        }
    }
}

fn print_help() {
    println!(
        "Usage:\n  cargo run -p xtask -- provider-audit [--root <path>]\n  cargo run -p xtask -- benchmark [--full] [--skip-run --csv <path>] [--root <path>]\n\nProvider audit options:\n  --root <path>       Workspace root to inspect. Defaults to the current directory.\n\nBenchmark options:\n  --full              Dense 1..=500 sweep instead of the bounded default set.\n  --skip-run          Re-render from a saved CSV instead of measuring.\n  --csv <path>        CSV to read with --skip-run.\n  --root <path>       Workspace root. Defaults to the current directory."
    );
}
