#!/usr/bin/env python3
"""Transfer benchmark sources without changing either revision's dependency graph.

Manifest compatibility is a conservative precondition, not an API proof. The
subsequent locked builds establish whether the candidate sources compile with
each revision's providers. In particular, transitive Hermes versions may differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tomllib
from pathlib import Path


SOURCE_ROOTS = (Path("crates/apollo-bench/src"), Path("crates/apollo-fft/benches"))
BENCHMARKS = ("kernel_strategy", "prime_compose", "half_cyclic_rader")


def checked_path(root: Path, relative: Path) -> Path:
    """Reject redirects before reading or changing a checkout-relative path."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"path must stay inside {root}: {relative}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"linked path is not an instrument input: {path}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"path escapes checkout: {path}")
    return path


def source_files(root: Path) -> dict[Path, bytes]:
    """Read the complete source set, rejecting links and non-regular files."""
    files = {}
    for relative in SOURCE_ROOTS:
        directory = checked_path(root, relative)
        if not directory.is_dir():
            raise ValueError(f"missing instrument directory: {directory}")
        pending = [directory]
        while pending:
            for path in sorted(pending.pop().iterdir()):
                relative_path = path.relative_to(root)
                checked_path(root, relative_path)
                mode = path.stat().st_mode
                if stat.S_ISDIR(mode):
                    pending.append(path)
                elif stat.S_ISREG(mode):
                    if path.name in {"Cargo.toml", "Cargo.lock"}:
                        raise ValueError(f"manifest inside source transfer: {path}")
                    files[relative_path] = path.read_bytes()
                else:
                    raise ValueError(f"non-regular instrument input: {path}")
    for benchmark in BENCHMARKS:
        relative = SOURCE_ROOTS[1] / f"{benchmark}.rs"
        if relative not in files:
            raise ValueError(f"missing benchmark source: {root / relative}")
    if SOURCE_ROOTS[0] / "lib.rs" not in files:
        raise ValueError(f"missing instrument library: {root / SOURCE_ROOTS[0]}")
    return files


def dependencies(table: dict, workspace: dict) -> dict:
    """Resolve inherited dependency declarations without resolving versions."""
    result = {}
    for name, declaration in table.items():
        declaration = {"version": declaration} if isinstance(declaration, str) else dict(declaration)
        if declaration.pop("workspace", False):
            inherited = workspace["dependencies"][name]
            inherited = {"version": inherited} if isinstance(inherited, str) else dict(inherited)
            features = set(inherited.get("features", [])) | set(declaration.get("features", []))
            inherited.update(declaration)
            if features:
                inherited["features"] = sorted(features)
            declaration = inherited
        if "features" in declaration:
            declaration["features"] = sorted(declaration["features"])
        result[name] = declaration
    return result


def manifest_contract(root: Path) -> dict:
    """Select requirements of the copied sources, leaving production deps alone."""
    def read(relative: str) -> dict:
        return tomllib.loads(checked_path(root, Path(relative)).read_text(encoding="utf-8"))

    workspace = read("Cargo.toml")["workspace"]
    instrument = read("crates/apollo-bench/Cargo.toml")
    transform = read("crates/apollo-fft/Cargo.toml")
    contract = {}
    for name, manifest, kinds in (
        ("instrument", instrument, ("dependencies", "dev-dependencies", "build-dependencies")),
        ("benchmarks", transform, ("dev-dependencies",)),
    ):
        for kind in kinds:
            contract[f"{name}.{kind}"] = dependencies(manifest.get(kind, {}), workspace)
        for target, table in manifest.get("target", {}).items():
            for kind in kinds:
                contract[f"{name}.{target}.{kind}"] = dependencies(table.get(kind, {}), workspace)
        for field in ("edition", "rust-version"):
            value = manifest["package"].get(field)
            if isinstance(value, dict) and value.get("workspace"):
                value = workspace["package"][field]
            contract[f"{name}.{field}"] = value
    # Target declarations and feature wiring determine how the transferred
    # sources are compiled. Refuse unsupported migrations instead of editing
    # historical manifests to admit them.
    for field in ("lib", "bin", "features"):
        contract[f"instrument.{field}"] = instrument.get(field)
    contract["benchmarks.targets"] = transform.get("bench", [])
    for benchmark in BENCHMARKS:
        target = [item for item in transform.get("bench", []) if item["name"] == benchmark]
        if len(target) != 1 or target[0].get("harness", True):
            raise ValueError(f"{root}: {benchmark} requires one harness-free bench target")
        if target[0].get("path", f"benches/{benchmark}.rs") != f"benches/{benchmark}.rs":
            raise ValueError(f"{root}: {benchmark} lies outside its source transfer")
    features = transform.get("features", {})
    pending = ["kernel-strategy-bench"]
    visited = {}
    while pending:
        feature = pending.pop()
        if feature in visited:
            continue
        if feature not in features:
            raise ValueError(f"{root}: missing benchmark feature {feature}")
        visited[feature] = sorted(features[feature])
        pending.extend(value for value in features[feature] if value in features)
    contract["benchmarks.features"] = visited
    return contract


def protected_files(root: Path) -> dict[Path, bytes]:
    """Snapshot manifests and the lock so preparation can assert preservation."""
    paths = {Path("Cargo.lock"), Path("Cargo.toml")}
    paths.update(path.relative_to(root) for path in root.glob("crates/**/Cargo.toml"))
    return {path: checked_path(root, path).read_bytes() for path in sorted(paths)}


def hashes(files: dict[Path, bytes]) -> dict[str, str]:
    return {path.as_posix(): hashlib.sha256(content).hexdigest() for path, content in sorted(files.items())}


def prepare(candidate: Path, baseline: Path) -> dict:
    """Preflight both trees, then replace only the baseline instrument sources."""
    candidate, baseline = candidate.resolve(strict=True), baseline.resolve(strict=True)
    if candidate.is_relative_to(baseline) or baseline.is_relative_to(candidate):
        raise ValueError("candidate and baseline must be disjoint checkout roots")
    contracts = manifest_contract(candidate), manifest_contract(baseline)
    differences = sorted(key for key in contracts[0].keys() | contracts[1].keys()
                         if contracts[0].get(key) != contracts[1].get(key))
    if differences:
        raise ValueError("incompatible instrument manifest requirements: " + ", ".join(differences))
    protected = protected_files(candidate), protected_files(baseline)
    sources, previous = source_files(candidate), source_files(baseline)
    # Read and validate the entire input and destination set before the first
    # deletion. All mutation destinations descend from the two fixed roots.
    for path in sources.keys() | previous.keys():
        checked_path(baseline, path)
    removed = sorted(previous.keys() - sources.keys())
    written = sorted(path for path, content in sources.items() if previous.get(path) != content)
    for path in removed:
        checked_path(baseline, path).unlink()
    for relative in SOURCE_ROOTS:
        directories = [path for path in (baseline / relative).rglob("*") if path.is_dir()]
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            if not any(directory.iterdir()):
                checked_path(baseline, directory.relative_to(baseline)).rmdir()
    for path in written:
        destination = checked_path(baseline, path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sources[path])
    if source_files(baseline) != sources:
        raise RuntimeError("transferred instrument source set differs from candidate")
    if (protected_files(candidate), protected_files(baseline)) != protected:
        raise RuntimeError("preparation changed a manifest or lockfile")
    return {"candidate": str(candidate), "baseline": str(baseline),
            "source_sha256": hashes(sources),
            "candidate_manifest_lock_sha256": hashes(protected[0]),
            "baseline_manifest_lock_sha256": hashes(protected[1]),
            "written": [path.as_posix() for path in written],
            "removed": [path.as_posix() for path in removed]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.candidate, args.baseline)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"benchmark preparation failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
