#![allow(missing_docs)]

use anyhow::{bail, Result};
use std::env;

mod provider_audit;

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("provider-audit") => provider_audit::run(args),
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
        "Usage:\n  cargo run -p xtask -- provider-audit [--root <path>]\n\nProvider audit options:\n  --root <path>       Workspace root to inspect. Defaults to the current directory."
    );
}
