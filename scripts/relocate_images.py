import argparse
import datetime
import os
import re
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Move local images referenced in a markdown file into "
            "docs/assets/<section>/<doc-stem>/ and rewrite the markdown links."
        )
    )
    parser.add_argument("markdown", help="Path to the markdown file")
    parser.add_argument(
        "--docs-root",
        default="docs",
        help="Docs root directory (default: docs)",
    )
    parser.add_argument(
        "--assets-root",
        default="docs/assets",
        help="Assets root directory (default: docs/assets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without moving files",
    )
    return parser.parse_args()


def is_remote(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def relocate_images(md_path: Path, docs_root: Path, assets_root: Path, dry_run: bool = False):
    # 跳过 assets 目录下的 md
    if md_path.is_dir():
        print(f"跳过目录: {md_path}")
        return
    if assets_root in md_path.parents:
        print(f"跳过 assets 内文件: {md_path}")
        return

    content = md_path.read_text(encoding="utf-8")
    stem = md_path.stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 相对 docs 的路径，用于确定 section（00-inbox/…、10-knowledge/… 等）
    try:
        rel_from_docs = md_path.relative_to(docs_root)
    except ValueError:
        raise SystemExit(f"{md_path} 不在 docs 根目录 {docs_root} 下")

    section = rel_from_docs.parts[0] if rel_from_docs.parts else stem
    # Markdown image pattern: ![alt](path)
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    counter = 1
    replacements = []
    assets_dir = None

    def repl(match: re.Match):
        nonlocal counter, assets_dir
        alt_text = match.group(1)
        src = match.group(2).strip()

        # Skip remote, anchors, or already-absolute site paths
        if is_remote(src) or src.startswith("#") or src.startswith("/"):
            return match.group(0)

        # Resolve relative to the markdown file
        src_path = (md_path.parent / src).resolve()
        if not src_path.exists():
            print(f"⚠️  找不到图片文件: {src}")
            return match.group(0)

        suffix = src_path.suffix or ".png"
        if assets_dir is None:
            assets_dir = assets_root / section / stem
            assets_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{stem}-{timestamp}-{counter}{suffix}"
        dest_path = assets_dir / dest_name
        counter += 1

        # /assets/section/stem/file
        rel_link = dest_path.relative_to(docs_root)
        new_link = "/" + rel_link.as_posix()

        replacements.append((src_path, dest_path, new_link))
        return f"![{alt_text}]({new_link})"

    new_content = pattern.sub(repl, content)

    if dry_run:
        for src, dest, link in replacements:
            print(f"[dry-run] {src} -> {dest} (markdown: {link})")
        return

    for src, dest, link in replacements:
        shutil.move(src, dest)
        print(f"移动: {src} -> {dest} (markdown 使用 {link})")

    if replacements:
        md_path.write_text(new_content, encoding="utf-8")
        print(f"已更新 markdown: {md_path}")
    else:
        print("未找到需要移动的本地图片。")


def main():
    args = parse_args()
    md_path = Path(args.markdown).expanduser().resolve()
    docs_root = Path(args.docs_root).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve()

    if not md_path.exists():
        raise SystemExit(f"找不到文件: {md_path}")
    if md_path.is_dir():
        print(f"跳过目录: {md_path}")
        return
    if md_path.suffix.lower() != ".md":
        print(f"跳过非 markdown: {md_path}")
        return
    if not docs_root.exists():
        raise SystemExit(f"找不到 docs 根目录: {docs_root}")

    relocate_images(md_path, docs_root, assets_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
