use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const PROVIDER_REQUIREMENTS: &[(&str, &str)] = &[
    (
        "moirai",
        "monomorphized CPU scheduling with bounded work queues, scoped non-'static closures, by-reference and by-value non-Clone iteration, caller-owned output collection, deterministic chunking, and no Box<dyn> or Arc<Vec<T>> hot-path storage",
    ),
    (
        "mnemosyne",
        "optional scratch and plan-cache allocation for aligned transform workspaces, reusable thread-local regions, Cow-backed borrowed views, zero-sized allocation policies, and no default global allocator requirement",
    ),
    (
        "melinoe",
        "branded zero-copy slice and Cow boundaries for scratch, staging, and validation views with ZST policy markers and no shared mutable state in mathematical kernels",
    ),
    (
        "hermes",
        "monomorphized SIMD vector kernels, preferred-architecture ZST routing, copy-on-write SIMD views, and no runtime-erased dispatch in transform hot paths",
    ),
    (
        "wgpu",
        "GPU kernels keep device buffers behind infrastructure crates while CPU planning, host staging, and verification stay provider-agnostic",
    ),
];

const SOURCE_PATTERNS: &[(&str, &str)] = &[
    ("moirai", "moirai"),
    ("mnemosyne", "mnemosyne"),
    ("melinoe", "melinoe"),
    ("hermes", "hermes"),
    ("hermes_simd", "hermes_simd"),
    ("rayon", "rayon"),
    ("arc", "Arc<"),
    ("mutex", "Mutex<"),
    ("box_dyn", "Box<dyn"),
    ("dyn_trait", "dyn "),
    ("to_vec", ".to_vec("),
    ("collect_vec", "collect::<Vec"),
    ("cow", "Cow<"),
    ("wgpu", "wgpu"),
];

pub(crate) fn run(args: impl Iterator<Item = String>) -> Result<()> {
    let root = parse_args(args)?;
    let audit = ProviderAudit::collect(&root)?;
    println!("{}", audit.render());
    Ok(())
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<PathBuf> {
    let mut root = PathBuf::from(".");
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                root = PathBuf::from(args.next().context("--root requires a path")?);
            }
            "-h" | "--help" => {
                println!(
                    "Usage:\n  cargo run -p xtask -- provider-audit [--root <path>]\n\nOptions:\n  --root <path>       Workspace root to inspect. Defaults to the current directory."
                );
                return Ok(root);
            }
            other => bail!("unknown provider-audit option `{other}`"),
        }
    }
    Ok(root)
}

#[derive(Debug)]
struct ProviderAudit {
    root: PathBuf,
    crates: Vec<CrateAudit>,
    workspace: WorkspaceAudit,
}

#[derive(Debug, Default)]
struct WorkspaceAudit {
    moirai_workspace_dep: bool,
    mnemosyne_workspace_dep: bool,
    melinoe_workspace_dep: bool,
    hermes_workspace_dep: bool,
    ndarray_rayon_feature: bool,
}

#[derive(Debug)]
struct CrateAudit {
    name: String,
    manifest: PathBuf,
    manifest_usage: ManifestUsage,
    source_usage: BTreeMap<&'static str, usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ManifestUsage {
    moirai: bool,
    mnemosyne: bool,
    melinoe: bool,
    hermes: bool,
    rayon: bool,
    ndarray_rayon_feature: bool,
}

impl ProviderAudit {
    fn collect(root: &Path) -> Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("failed to canonicalize {}", root.display()))?;
        let manifests = collect_manifests(&root)?;
        let workspace = collect_workspace_usage(&root)?;
        let crates = manifests
            .into_iter()
            .map(|manifest| collect_crate_audit(&root, &manifest))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            root,
            crates,
            workspace,
        })
    }

    fn render(&self) -> String {
        let mut output = String::new();
        output.push_str("# Apollo Provider Audit\n\n");
        writeln!(&mut output, "Root: `{}`\n", self.root.display())
            .expect("writing to String cannot fail");
        output.push_str("## Workspace Dependency Surface\n");
        push_bool_line(
            &mut output,
            "Moirai git workspace dependency",
            self.workspace.moirai_workspace_dep,
        );
        push_bool_line(
            &mut output,
            "Mnemosyne workspace dependency",
            self.workspace.mnemosyne_workspace_dep,
        );
        push_bool_line(
            &mut output,
            "Melinoe workspace dependency",
            self.workspace.melinoe_workspace_dep,
        );
        push_bool_line(
            &mut output,
            "Hermes workspace dependency",
            self.workspace.hermes_workspace_dep,
        );
        push_bool_line(
            &mut output,
            "ndarray rayon/matrixmultiply-threading feature",
            self.workspace.ndarray_rayon_feature,
        );
        output.push('\n');

        output.push_str("## Crate Usage\n");
        output.push_str(
            "| Crate | Manifest | Moirai | Mnemosyne | Melinoe | Hermes | Rayon | ndarray rayon | Arc | Mutex | dyn | Vec clones | Cow | WGPU |\n",
        );
        output.push_str(
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        );
        for crate_audit in &self.crates {
            let dyn_count = count(&crate_audit.source_usage, "box_dyn")
                + count(&crate_audit.source_usage, "dyn_trait");
            let vec_clone_count = count(&crate_audit.source_usage, "to_vec")
                + count(&crate_audit.source_usage, "collect_vec");
            writeln!(
                &mut output,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                crate_audit.name,
                crate_audit.manifest.display(),
                mark(
                    crate_audit.manifest_usage.moirai
                        || count(&crate_audit.source_usage, "moirai") > 0
                ),
                mark(
                    crate_audit.manifest_usage.mnemosyne
                        || count(&crate_audit.source_usage, "mnemosyne") > 0
                ),
                mark(
                    crate_audit.manifest_usage.melinoe
                        || count(&crate_audit.source_usage, "melinoe") > 0
                ),
                mark(
                    crate_audit.manifest_usage.hermes
                        || count(&crate_audit.source_usage, "hermes") > 0
                        || count(&crate_audit.source_usage, "hermes_simd") > 0
                ),
                mark(
                    crate_audit.manifest_usage.rayon
                        || count(&crate_audit.source_usage, "rayon") > 0
                ),
                mark(crate_audit.manifest_usage.ndarray_rayon_feature),
                count(&crate_audit.source_usage, "arc"),
                count(&crate_audit.source_usage, "mutex"),
                dyn_count,
                vec_clone_count,
                count(&crate_audit.source_usage, "cow"),
                count(&crate_audit.source_usage, "wgpu"),
            )
            .expect("writing to String cannot fail");
        }
        output.push('\n');

        output.push_str("## Provider Requirements\n");
        for (provider, requirement) in PROVIDER_REQUIREMENTS {
            writeln!(&mut output, "- `{provider}`: {requirement}.")
                .expect("writing to String cannot fail");
        }
        output.push_str("\n## Dependency Order\n");
        output.push_str(
            "- Moirai, Mnemosyne, Melinoe, and Hermes are consumed from Git dependencies; provider changes must be committed and pushed before Apollo can update dependency revisions.\n",
        );
        output.push_str(
            "- Apollo must not add a local path override for provider work in committed manifests.\n",
        );
        output
    }
}

fn collect_workspace_usage(root: &Path) -> Result<WorkspaceAudit> {
    let manifest = root.join("Cargo.toml");
    let text = fs::read_to_string(&manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?;
    Ok(WorkspaceAudit {
        moirai_workspace_dep: text.contains("moirai") && text.contains("github.com"),
        mnemosyne_workspace_dep: text.contains("mnemosyne"),
        melinoe_workspace_dep: text.contains("melinoe"),
        hermes_workspace_dep: text.contains("hermes-simd") && text.contains("github.com"),
        ndarray_rayon_feature: text.contains("features = [\"rayon\"")
            || text.contains("features = [\"rayon\",")
            || text.contains("features = [\"rayon\","),
    })
}

fn collect_manifests(root: &Path) -> Result<Vec<PathBuf>> {
    let mut manifests = Vec::new();
    collect_manifests_inner(root, &mut manifests)?;
    manifests.sort();
    Ok(manifests)
}

fn collect_manifests_inner(path: &Path, manifests: &mut Vec<PathBuf>) -> Result<()> {
    if should_skip_path(path) {
        return Ok(());
    }
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_manifests_inner(&path, manifests)?;
        } else if entry.file_name() == "Cargo.toml" {
            manifests.push(path);
        }
    }
    Ok(())
}

fn collect_crate_audit(root: &Path, manifest: &Path) -> Result<CrateAudit> {
    let manifest_text = fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?;
    let package_name = package_name(&manifest_text);
    let has_package = package_name.is_some();
    let name = package_name.unwrap_or_else(|| "workspace".to_string());
    let manifest_usage = manifest_usage(&manifest_text);
    let source_usage = if has_package {
        manifest
            .parent()
            .filter(|crate_root| !crate_root.ends_with("xtask"))
            .map_or_else(BTreeMap::new, source_usage)
    } else {
        BTreeMap::new()
    };
    Ok(CrateAudit {
        name,
        manifest: manifest
            .strip_prefix(root)
            .unwrap_or(manifest)
            .to_path_buf(),
        manifest_usage,
        source_usage,
    })
}

fn package_name(text: &str) -> Option<String> {
    let mut in_package = false;
    for line in text.lines().map(str::trim) {
        if line == "[package]" {
            in_package = true;
            continue;
        }
        if in_package && line.starts_with('[') {
            return None;
        }
        if in_package && line.starts_with("name") {
            return line
                .split_once('=')
                .map(|(_, value)| value.trim().trim_matches('"').to_string());
        }
    }
    None
}

fn manifest_usage(text: &str) -> ManifestUsage {
    ManifestUsage {
        moirai: text.contains("moirai"),
        mnemosyne: text.contains("mnemosyne"),
        melinoe: text.contains("melinoe"),
        hermes: text.contains("hermes-simd") || text.contains("hermes_simd"),
        rayon: text.contains("rayon"),
        ndarray_rayon_feature: text.contains("ndarray") && text.contains("rayon"),
    }
}

fn source_usage(crate_root: &Path) -> BTreeMap<&'static str, usize> {
    let mut usage = BTreeMap::new();
    collect_source_usage(crate_root, &mut usage);
    usage
}

fn collect_source_usage(path: &Path, usage: &mut BTreeMap<&'static str, usize>) {
    if should_skip_path(path) {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_source_usage(&path, usage);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            if let Ok(text) = fs::read_to_string(&path) {
                for (key, pattern) in SOURCE_PATTERNS {
                    let count = text.match_indices(pattern).count();
                    if count > 0 {
                        *usage.entry(key).or_default() += count;
                    }
                }
            }
        }
    }
}

fn should_skip_path(path: &Path) -> bool {
    path.components().any(|component| {
        let value = component.as_os_str().to_string_lossy();
        matches!(
            value.as_ref(),
            ".git" | "target" | "docs" | "benchmark_results"
        )
    })
}

fn push_bool_line(output: &mut String, label: &str, value: bool) {
    writeln!(output, "- {label}: {}", mark(value)).expect("writing to String cannot fail");
}

fn mark(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn count(usage: &BTreeMap<&'static str, usize>, key: &'static str) -> usize {
    usage.get(key).copied().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn provider_audit_reports_git_providers_and_crate_usage() -> Result<()> {
        let root = temp_workspace("provider-audit-usage")?;
        fs::create_dir_all(root.join("crates/apollo-demo/src"))?;
        fs::write(
            root.join("Cargo.toml"),
            r#"[workspace]
members = ["crates/apollo-demo"]

[workspace.dependencies]
moirai = { git = "https://github.com/ryancinsight/Moirai.git", default-features = false, features = ["parallel"] }
melinoe = { git = "https://github.com/ryancinsight/melinoe.git", default-features = false, features = ["alloc"] }
hermes-simd = { git = "https://github.com/ryancinsight/hermes.git", default-features = false, features = ["std"] }
ndarray = { version = "0.16", features = ["rayon","matrixmultiply-threading"] }
"#,
        )?;
        fs::write(
            root.join("crates/apollo-demo/Cargo.toml"),
            r#"[package]
name = "apollo-demo"
version = "0.1.0"
edition = "2021"

[dependencies]
moirai = { workspace = true }
melinoe = { workspace = true }
hermes-simd = { workspace = true }
"#,
        )?;
        fs::write(
            root.join("crates/apollo-demo/src/lib.rs"),
            "use std::{borrow::Cow, sync::Arc}; use hermes_simd as hermes; fn f(v: &[u8]) { let _: Cow<'_, [u8]> = Cow::Borrowed(v); let _ = Arc::new(v.to_vec()); let _ = core::any::type_name::<hermes::Scalar>(); }",
        )?;

        let audit = ProviderAudit::collect(&root)?;
        let rendered = audit.render();

        assert!(rendered.contains("Moirai git workspace dependency: yes"));
        assert!(rendered.contains("Mnemosyne workspace dependency: no"));
        assert!(rendered.contains("Melinoe workspace dependency: yes"));
        assert!(rendered.contains("Hermes workspace dependency: yes"));
        assert!(
            rendered
                .contains("| apollo-demo | crates\\apollo-demo\\Cargo.toml | yes | no | yes | yes |")
        );
        assert!(
            rendered
                .contains("Moirai, Mnemosyne, Melinoe, and Hermes are consumed from Git dependencies")
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn provider_audit_rejects_unknown_options() {
        let result = parse_args(["--bad".to_string()].into_iter());
        assert!(result.is_err());
    }

    fn temp_workspace(label: &str) -> Result<PathBuf> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system time before unix epoch")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("{label}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}
