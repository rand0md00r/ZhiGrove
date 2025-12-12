import argparse
import os
from pathlib import Path
from typing import Set
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove unreferenced files under docs/assets based on links in markdown."
    )
    parser.add_argument(
        "--docs-root", default="docs", help="Docs root directory (default: docs)"
    )
    parser.add_argument(
        "--assets-root",
        default="docs/assets",
        help="Assets root directory (default: docs/assets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list deletions without removing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print matched asset references",
    )
    return parser.parse_args()


IMG_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def collect_asset_refs(docs_root: Path, assets_root: Path, verbose: bool) -> Set[Path]:
    refs: Set[Path] = set()
    for md_file in docs_root.rglob("*.md"):
        # skip assets directory itself
        if assets_root in md_file.parents:
            continue
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        for m in IMG_PATTERN.finditer(text):
            raw = m.group(1).strip()
            if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("#"):
                continue
            if raw.startswith("/"):
                # already from docs root
                candidate = (docs_root / raw.lstrip("/")).resolve()
            else:
                candidate = (md_file.parent / raw).resolve()
            try:
                candidate.relative_to(assets_root)
            except ValueError:
                continue
            refs.add(candidate)
            if verbose:
                print(f"ref: {md_file} -> {candidate}")
    return refs


def collect_existing_assets(assets_root: Path) -> Set[Path]:
    return {p.resolve() for p in assets_root.rglob("*") if p.is_file()}


def main():
    args = parse_args()
    docs_root = Path(args.docs_root).resolve()
    assets_root = Path(args.assets_root).resolve()

    if not docs_root.exists():
        raise SystemExit(f"Docs root not found: {docs_root}")
    if not assets_root.exists():
        print(f"Assets root not found: {assets_root} (nothing to clean)")
        return

    refs = collect_asset_refs(docs_root, assets_root, args.verbose)
    existing = collect_existing_assets(assets_root)
    stale = sorted(existing - refs)

    if not stale:
        print("No stale assets found.")
        return

    print(f"Found {len(stale)} stale assets:")
    for p in stale:
        print(f"  {p}")

    if args.dry_run:
        print("Dry-run: no files removed.")
        return

    for p in stale:
        try:
            p.unlink()
        except OSError as e:
            print(f"Failed to remove {p}: {e}")

    # remove empty directories
    for dirpath, dirnames, filenames in os.walk(assets_root, topdown=False):
        path_dir = Path(dirpath)
        if not any(path_dir.iterdir()):
            try:
                path_dir.rmdir()
            except OSError:
                pass

    print("Cleanup completed.")


if __name__ == "__main__":
    main()
