use super::ComparisonError;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) fn discover_reports(root: &Path) -> Result<BTreeMap<PathBuf, PathBuf>, ComparisonError> {
    let mut reports = BTreeMap::new();
    visit(root, root, &mut reports)?;
    if reports.is_empty() {
        return Err(ComparisonError::no_reports(root));
    }
    Ok(reports)
}

fn visit(
    root: &Path,
    directory: &Path,
    reports: &mut BTreeMap<PathBuf, PathBuf>,
) -> Result<(), ComparisonError> {
    let entries = fs::read_dir(directory)
        .map_err(|source| ComparisonError::read_directory(directory, source))?;
    let mut entries = entries
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| ComparisonError::read_entry(directory, source))?;
    entries.sort_by_cached_key(std::fs::DirEntry::path);

    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|source| ComparisonError::read_entry(&path, source))?;
        if file_type.is_dir() {
            visit(root, &path, reports)?;
        } else if file_type.is_file()
            && path.extension().is_some_and(|extension| extension == "csv")
        {
            let relative = path
                .strip_prefix(root)
                .expect("invariant: discovered report remains below its root")
                .to_path_buf();
            reports.insert(relative, path);
        }
    }

    Ok(())
}
