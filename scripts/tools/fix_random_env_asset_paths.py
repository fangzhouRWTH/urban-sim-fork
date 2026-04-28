#!/usr/bin/env python3
"""Scan and repair USD asset paths used by random_env.

This tool targets the assets loaded by ``random_env.py`` and fixes texture/material
references that still point to machine-specific absolute paths such as:

* ``/home/hollis/projects/IsaacUrban/...``
* the current checkout's absolute repo path

The rewrite path prefers repository-relative references so the assets remain portable
if the checkout is moved to a different directory.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from pxr import Sdf, Usd
except ImportError:  # pragma: no cover - exercised only in environments without USD.
    Sdf = None
    Usd = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_ROOTS = (
    REPO_ROOT / "assets/robots/coco_one",
    REPO_ROOT / "assets/usds",
    REPO_ROOT / "assets/peds",
)
LEGACY_ROOT_HINTS = (
    Path("/home/hollis/projects/IsaacUrban"),
    REPO_ROOT,
)
TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr", ".mdl"}
USD_EXTENSIONS = {".usd", ".usda", ".usdc"}
PATH_PATTERN = re.compile(r"/home/[^\s@\"']+")


@dataclass
class PathIssue:
    usd_file: Path
    prim_path: str
    attribute_name: str
    current_path: str
    mapped_path: str | None
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("scan", "rewrite"),
        help="Use 'scan' to report problematic asset references or 'rewrite' to update them in place.",
    )
    parser.add_argument(
        "--asset-root",
        action="append",
        default=[],
        help="Optional extra asset root to scan. Defaults to random_env asset roots.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summaries unless an unresolved reference remains.",
    )
    return parser.parse_args()


def iter_asset_roots(extra_roots: Iterable[str]) -> list[Path]:
    roots = list(DEFAULT_TARGET_ROOTS)
    roots.extend(REPO_ROOT / root for root in extra_roots)
    deduped = []
    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def iter_usd_files(asset_roots: Iterable[Path]) -> list[Path]:
    usd_files: list[Path] = []
    for root in asset_roots:
        usd_files.extend(path for path in root.rglob("*") if path.suffix.lower() in USD_EXTENSIONS)
    return sorted(set(usd_files))


def build_candidate_index(asset_roots: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for root in asset_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXTURE_EXTENSIONS and path.suffix.lower() not in USD_EXTENSIONS:
                continue
            index[path.name].append(path.resolve())
    return index


def looks_like_asset_path(raw_path: str) -> bool:
    cleaned = raw_path.strip().strip("@")
    suffix = Path(cleaned).suffix.lower()
    return cleaned.startswith("/home/") or suffix in TEXTURE_EXTENSIONS or suffix in USD_EXTENSIONS


def to_relative_path(target: Path, base_dir: Path) -> str:
    rel_path = os.path.relpath(target, base_dir).replace(os.sep, "/")
    if not rel_path.startswith("."):
        rel_path = f"./{rel_path}"
    return rel_path


def existing_relative_candidate(raw_path: str, usd_file: Path) -> Path | None:
    relative_candidate = (usd_file.parent / raw_path).resolve()
    if relative_candidate.exists():
        return relative_candidate
    return None


def search_candidate_by_basename(raw_path: str, usd_file: Path, candidate_index: dict[str, list[Path]]) -> Path | None:
    basename = Path(raw_path).name
    if not basename:
        return None

    local_search_roots = [
        usd_file.parent,
        *list(usd_file.parents[:3]),
    ]
    for root in local_search_roots:
        for subdir in ("materials", "textures"):
            candidate = root / subdir / basename
            if candidate.exists():
                return candidate.resolve()
        candidate = root / basename
        if candidate.exists():
            return candidate.resolve()

    if basename in candidate_index:
        suffix_parts = Path(raw_path).parts[-2:]
        suffix_hint = "/".join(suffix_parts)
        matches = candidate_index[basename]
        for candidate in matches:
            if suffix_hint and candidate.as_posix().endswith(suffix_hint):
                return candidate
        if len(matches) == 1:
            return matches[0]

    return None


def remap_path(raw_path: str, usd_file: Path, candidate_index: dict[str, list[Path]]) -> Path | None:
    raw_path = raw_path.strip().strip("@")
    if not raw_path:
        return None

    path_obj = Path(raw_path)
    if not path_obj.is_absolute():
        relative_candidate = existing_relative_candidate(raw_path, usd_file)
        if relative_candidate is not None:
            return relative_candidate
        return search_candidate_by_basename(raw_path, usd_file, candidate_index)

    if path_obj.exists():
        try:
            path_obj.relative_to(REPO_ROOT)
        except ValueError:
            pass
        else:
            return path_obj

    for legacy_root in LEGACY_ROOT_HINTS:
        try:
            suffix = path_obj.relative_to(legacy_root)
        except ValueError:
            continue
        direct_candidate = (REPO_ROOT / suffix).resolve()
        if direct_candidate.exists():
            return direct_candidate

    return search_candidate_by_basename(raw_path, usd_file, candidate_index)


def extract_attr_paths(value) -> tuple[str, object] | None:
    if Sdf is not None and isinstance(value, Sdf.AssetPath):
        return "asset", value.path
    if isinstance(value, str) and looks_like_asset_path(value):
        return "string", value
    return None


def make_issue(
    usd_file: Path,
    prim_path: str,
    attribute_name: str,
    current_path: str,
    mapped_candidate: Path | None,
    reason: str,
) -> PathIssue:
    return PathIssue(
        usd_file=usd_file,
        prim_path=prim_path,
        attribute_name=attribute_name,
        current_path=current_path,
        mapped_path=to_relative_path(mapped_candidate, usd_file.parent) if mapped_candidate else None,
        reason=reason,
    )


def should_ignore_missing_path(current_path: str, attribute_name: str) -> bool:
    # Imported glTF materials often preserve the source MDL identifier as metadata even though
    # the runtime texture bindings live on separate shader inputs. Treat these as informational
    # rather than blockers for the random_env texture cleanup.
    return attribute_name == "info:mdl:sourceAsset" and (current_path.startswith("gltf/") or "/" not in current_path)


def scan_path_value(
    usd_file: Path,
    prim_path: str,
    attribute_name: str,
    current_path: str,
    candidate_index: dict[str, list[Path]],
) -> PathIssue | None:
    mapped_candidate = remap_path(current_path, usd_file, candidate_index)
    if Path(current_path).is_absolute():
        return make_issue(usd_file, prim_path, attribute_name, current_path, mapped_candidate, "absolute path")
    if mapped_candidate is None:
        if should_ignore_missing_path(current_path, attribute_name):
            return None
        return make_issue(usd_file, prim_path, attribute_name, current_path, None, "missing target")
    return None


def iter_layer_paths(layer: Sdf.Layer) -> Iterable[str]:
    spec_paths: list[str] = []
    layer.Traverse("/", lambda path: spec_paths.append(path))
    return spec_paths


def iter_reference_like_entries(spec, list_name: str):
    list_op = getattr(spec, list_name)
    for bucket_name in ("explicitItems", "prependedItems", "appendedItems"):
        items = list(getattr(list_op, bucket_name, []))
        if items:
            yield bucket_name, items


def rebuild_reference(item, new_path: str):
    if isinstance(item, Sdf.Reference):
        return Sdf.Reference(assetPath=new_path, primPath=item.primPath, layerOffset=item.layerOffset, customData=item.customData)
    return Sdf.Payload(assetPath=new_path, primPath=item.primPath, layerOffset=item.layerOffset)


def scan_usd_file(usd_file: Path, candidate_index: dict[str, list[Path]]) -> list[PathIssue]:
    if Sdf is None:
        raise RuntimeError("USD Python bindings are required for structured scanning.")

    issues: list[PathIssue] = []
    layer = Sdf.Layer.FindOrOpen(str(usd_file))
    if layer is None:
        return [make_issue(usd_file, "", "", "", None, "unable to open layer")]

    for index, sublayer_path in enumerate(layer.subLayerPaths):
        issue = scan_path_value(usd_file, "/", f"subLayerPaths[{index}]", sublayer_path, candidate_index)
        if issue is not None:
            issues.append(issue)

    for spec_path in iter_layer_paths(layer):
        spec = layer.GetObjectAtPath(spec_path)
        spec_type = spec.__class__.__name__
        if spec_type == "AttributeSpec":
            extracted = extract_attr_paths(spec.default)
            if extracted is None:
                continue
            _, current_path = extracted
            issue = scan_path_value(usd_file, str(spec.path.GetPrimPath()), spec.name, current_path, candidate_index)
            if issue is not None:
                issues.append(issue)
        elif spec_type == "PrimSpec":
            for list_name in ("referenceList", "payloadList"):
                for bucket_name, items in iter_reference_like_entries(spec, list_name):
                    for item_index, item in enumerate(items):
                        if not item.assetPath:
                            continue
                        issue = scan_path_value(
                            usd_file,
                            str(spec.path),
                            f"{list_name}.{bucket_name}[{item_index}]",
                            item.assetPath,
                            candidate_index,
                        )
                        if issue is not None:
                            issues.append(issue)

    return issues


def rewrite_usd_file(usd_file: Path, candidate_index: dict[str, list[Path]]) -> tuple[int, list[PathIssue]]:
    if Sdf is None:
        raise RuntimeError("USD Python bindings are required for rewrite mode.")

    layer = Sdf.Layer.FindOrOpen(str(usd_file))
    if layer is None:
        return 0, [make_issue(usd_file, "", "", "", None, "unable to open layer")]

    rewrite_count = 0
    unresolved: list[PathIssue] = []

    new_sublayers = list(layer.subLayerPaths)
    for index, sublayer_path in enumerate(layer.subLayerPaths):
        mapped_candidate = remap_path(sublayer_path, usd_file, candidate_index)
        if mapped_candidate is None:
            unresolved.append(make_issue(usd_file, "/", f"subLayerPaths[{index}]", sublayer_path, None, "missing target"))
            continue
        new_path = to_relative_path(mapped_candidate, usd_file.parent)
        if sublayer_path != new_path:
            new_sublayers[index] = new_path
            rewrite_count += 1
    if new_sublayers != layer.subLayerPaths:
        layer.subLayerPaths = new_sublayers

    for spec_path in iter_layer_paths(layer):
        spec = layer.GetObjectAtPath(spec_path)
        spec_type = spec.__class__.__name__
        if spec_type == "AttributeSpec":
            extracted = extract_attr_paths(spec.default)
            if extracted is None:
                continue
            kind, current_path = extracted
            mapped_candidate = remap_path(current_path, usd_file, candidate_index)
            if mapped_candidate is None:
                if should_ignore_missing_path(current_path, spec.name):
                    continue
                unresolved.append(
                    make_issue(usd_file, str(spec.path.GetPrimPath()), spec.name, current_path, None, "missing target")
                )
                continue
            new_path = to_relative_path(mapped_candidate, usd_file.parent)
            if current_path == new_path:
                continue
            if kind == "asset":
                spec.default = Sdf.AssetPath(new_path)
            else:
                spec.default = new_path
            rewrite_count += 1
        elif spec_type == "PrimSpec":
            for list_name in ("referenceList", "payloadList"):
                list_op = getattr(spec, list_name)
                for bucket_name, items in iter_reference_like_entries(spec, list_name):
                    updated_items = list(items)
                    bucket_changed = False
                    for item_index, item in enumerate(items):
                        if not item.assetPath:
                            continue
                        mapped_candidate = remap_path(item.assetPath, usd_file, candidate_index)
                        if mapped_candidate is None:
                            if should_ignore_missing_path(item.assetPath, f"{list_name}.{bucket_name}[{item_index}]"):
                                continue
                            unresolved.append(
                                make_issue(
                                    usd_file,
                                    str(spec.path),
                                    f"{list_name}.{bucket_name}[{item_index}]",
                                    item.assetPath,
                                    None,
                                    "missing target",
                                )
                            )
                            continue
                        new_path = to_relative_path(mapped_candidate, usd_file.parent)
                        if item.assetPath == new_path:
                            continue
                        updated_items[item_index] = rebuild_reference(item, new_path)
                        rewrite_count += 1
                        bucket_changed = True
                    if bucket_changed:
                        setattr(list_op, bucket_name, updated_items)

    if rewrite_count > 0:
        layer.Save()
    return rewrite_count, unresolved


def fallback_scan(usd_files: Iterable[Path]) -> list[tuple[Path, str]]:
    findings: list[tuple[Path, str]] = []
    for usd_file in usd_files:
        text = usd_file.read_bytes().decode("utf-8", errors="ignore")
        for match in PATH_PATTERN.findall(text):
            findings.append((usd_file, match))
    return findings


def print_issues(issues: list[PathIssue], quiet: bool) -> None:
    if not issues:
        print("No problematic asset references found.")
        return

    for issue in issues:
        if quiet and issue.reason != "missing target":
            continue
        print(f"{issue.usd_file}")
        if issue.prim_path:
            print(f"  prim: {issue.prim_path}")
        if issue.attribute_name:
            print(f"  attr: {issue.attribute_name}")
        if issue.current_path:
            print(f"  current: {issue.current_path}")
        if issue.mapped_path:
            print(f"  mapped:  {issue.mapped_path}")
        print(f"  reason:  {issue.reason}")


def main() -> int:
    args = parse_args()
    asset_roots = iter_asset_roots(args.asset_root)
    usd_files = iter_usd_files(asset_roots)

    if Usd is None:
        if args.mode == "rewrite":
            print(
                "rewrite mode requires USD Python bindings (pxr). "
                "Run this script with Isaac Sim python or a Python environment that has usd-core installed.",
                file=sys.stderr,
            )
            return 2
        findings = fallback_scan(usd_files)
        if findings:
            for usd_file, raw_path in findings:
                print(f"{usd_file}\n  current: {raw_path}\n  reason:  absolute path (fallback scan)")
            return 1
        print("No obvious absolute paths found in fallback scan.")
        return 0

    candidate_index = build_candidate_index(asset_roots)

    if args.mode == "scan":
        issues: list[PathIssue] = []
        for usd_file in usd_files:
            issues.extend(scan_usd_file(usd_file, candidate_index))
        print_issues(issues, quiet=args.quiet)
        print(f"Scanned {len(usd_files)} USD file(s); found {len(issues)} problematic reference(s).")
        return 1 if issues else 0

    total_rewrites = 0
    unresolved: list[PathIssue] = []
    changed_files = 0
    for usd_file in usd_files:
        rewrite_count, missing_issues = rewrite_usd_file(usd_file, candidate_index)
        total_rewrites += rewrite_count
        if rewrite_count > 0:
            changed_files += 1
        unresolved.extend(missing_issues)

    print(f"Rewrote {total_rewrites} asset reference(s) across {changed_files} USD file(s).")
    if unresolved:
        print_issues(unresolved, quiet=args.quiet)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
