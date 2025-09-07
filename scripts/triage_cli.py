#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式收纳脚本：把 00-inbox/week-xx 中的 .md 逐一移动到 10/20/30/40/50 仓位，
并在文件顶部注入相应的 card 头（YAML front matter + 简明骨架）。
- 默认支持三类模板：
  * knowledge：TL;DR / What / Why / How / Gotchas
  * idea     ：问题 / 原因 / 下一步行动
  * experiment：包含 EXP CONFIG META 代码块 + 目标/设置差异/指标/结论/下一步
- 原始内容会被保存在文末 "## Raw Notes" 段落，确保信息不丢失。
- 若处于 git 仓库，使用 `git mv`；否则使用 shutil.move。
- 在 00-inbox/week-xx/TRIAGE.md 记录移动映射。

用法：
    python scripts/triage_cli.py 00-inbox/week-36
"""

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

CHOICES = {
    "k": ("10-knowledge", "knowledge"),
    "i": ("30-ideas", "idea"),
    "e": ("40-experiments", "experiment"),
    "p": ("20-papers", "paper"),
    "r": ("50-reports", "report"),
    "s": ("SKIP", "skip"),
}

# ====== Templates ======

EXP_CONFIG_META = """\
# ======================================================================
# EXP CONFIG META (paste at top of your .yaml / .toml / .ini config file)
# ----------------------------------------------------------------------
# EXP_ID      : <exp_yyyymmdd_x>          # e.g., exp_20250903_A
# TITLE       : <short one-liner>         # e.g., 5L transencoder, metaquery=256(no-decouple), LR×5
# PURPOSE     : <what this config is testing/ablation>
# KEY_DIFFS   : <the 2–4 knobs that distinguish this config from baseline>
# DATASET     : <e.g., 24M t2i>
# HARDWARE    : <e.g., A100×64>
# BRANCH/COMMIT: <wyq_dev / abc1234>      # optional
# CORE PARAMS : lr=<...>, batch=<...>, res=<...>, seed=<...>
# TAGS        : [<ablation>, <lrx5>, <5L>, <no-decouple>]   # optional
# NOTES       : <any caveat, e.g., warmup↑, grad_clip needed, etc.>
# ======================================================================
"""

KNOWLEDGE_SCAFFOLD = """\
## TL;DR（≤3点）
- 
- 
- 

## What（是什么）
- 

## Why（为什么这么做/何时使用）
- 

## How（最小复现配方，≤5步）
- 

## Gotchas（坑点与边界）
- 

"""

IDEA_SCAFFOLD = """\
## 问题
- 

## 原因/洞察
- 

## 下一步行动
- [ ] 
- [ ] 

"""

EXPERIMENT_SCAFFOLD = f"""\
## EXP CONFIG META（把下块注释贴进你的配置文件顶部）

{EXP_CONFIG_META}

## 目标
- 

## 设置（与基线的差异）
- 数据/分辨率/学习率/权重 等关键改动（≤4 条）

## 指标与记录（训练/推理）
- <iters> / <metric: value>
- <样例路径或截图>

## 结论
- <保留/否决/待复验>

## 下一步
- 
"""

RAW_NOTES_HEADER = "\n## Raw Notes\n\n"

# ====== Helpers ======

def find_repo_root(start: Path) -> Optional[Path]:
    p = start
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return None

def has_yaml_front_matter(text: str) -> bool:
    return text.lstrip().startswith("---")

def parse_title_from_filename(fname: str) -> str:
    # 从文件名推测 title：去掉扩展名；若有 [YYMMDD-HHMM] 前缀，取后半部分
    base = fname.rsplit(".", 1)[0]
    m = re.match(r"\[\d{6}-\d{4}\](.*)", base)
    title = m.group(1).strip(" -_") if m else base
    return title or base

def week_from_inbox_path(week_dir: Path) -> str:
    # 返回 'week-xx'（默认用目录名）
    return week_dir.name

def build_yaml(title: str, dtype: str, origin: str, tags: str) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    tags_clean = ",".join([t.strip() for t in tags.split(",") if t.strip()]) if tags else ""
    return (
        "---\n"
        f"title: {title}\n"
        f"created: {now}\n"
        f"updated: {now.split(' ')[0]}\n"
        f"origin: {origin}\n"
        f"type: {dtype}\n"
        "status: draft\n"
        f"tags: [{tags_clean}]\n"
        "links: []\n"
        "---\n\n"
    )

def scaffold_for(dtype: str) -> str:
    if dtype == "knowledge":
        return KNOWLEDGE_SCAFFOLD
    if dtype == "idea":
        return IDEA_SCAFFOLD
    if dtype == "experiment":
        return EXPERIMENT_SCAFFOLD
    # paper/report：只给个 Raw Notes，避免误导
    return ""

def inject_head_and_scaffold(path: Path, dtype: str, origin: str, tags: str):
    text = path.read_text(encoding="utf-8")
    title = parse_title_from_filename(path.name)
    if not has_yaml_front_matter(text):
        head = build_yaml(title, dtype, origin, tags)
        body = scaffold_for(dtype)
        new_text = head + body + RAW_NOTES_HEADER + text
        path.write_text(new_text, encoding="utf-8")
    else:
        # 已有 YAML：仅确保有 Raw Notes 标题用于承载原文（若没有则追加）
        if "## Raw Notes" not in text:
            text = text.rstrip() + RAW_NOTES_HEADER
            path.write_text(text, encoding="utf-8")

def safe_move(src: Path, dst: Path, repo_root: Optional[Path]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if repo_root is not None:
        # 使用 git mv，保持历史
        subprocess.run(["git", "-C", str(repo_root), "mv", "-f", str(src), str(dst)], check=True)
    else:
        shutil.move(str(src), str(dst))

def append_triage_log(week_dir: Path, src_rel: Path, dst_rel: Path):
    triage = week_dir / "TRIAGE.md"
    line = f"- moved: `{src_rel.as_posix()}` → `{dst_rel.as_posix()}`\n"
    with open(triage, "a", encoding="utf-8") as f:
        if triage.stat().st_size == 0:
            f.write(f"# Triage Log for {week_dir.name}\n\n")
        f.write(line)

def prompt_choice(prompt: str, valid: set, default: Optional[str] = None) -> str:
    while True:
        ans = input(f"{prompt} ").strip().lower()
        if not ans and default:
            return default
        if ans in valid:
            return ans
        print(f"无效输入，可选：{', '.join(sorted(valid))}")

# ====== Main ======

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("week_dir", type=str, help="路径：00-inbox/week-xx")
    ap.add_argument("--ext", type=str, default=".md", help="只处理此扩展名（默认 .md）")
    args = ap.parse_args()

    week_dir = Path(args.week_dir).resolve()
    assert week_dir.exists() and week_dir.is_dir(), f"目录不存在：{week_dir}"
    origin = week_from_inbox_path(week_dir)

    repo_root = find_repo_root(week_dir)
    if repo_root:
        print(f"[i] 检测到 Git 仓库：{repo_root}")
    else:
        print("[i] 未检测到 Git 仓库，将使用普通移动。")

    files = sorted([p for p in week_dir.glob(f"*{args.ext}") if p.is_file() and p.name.lower() != "triage.md"])
    if not files:
        print("[i] 无待处理文件。")
        return

    print(f"[i] 待处理 {len(files)} 个文件：\n" + "\n".join(f"- {p.name}" for p in files))
    print("\n选择目标：k=knowledge / i=idea / e=experiment / p=paper / r=report / s=skip\n")

    # 预估默认类型的小启发式（可按需加）
    def guess_choice(p: Path) -> str:
        name = p.name.lower()
        if any(k in name for k in ["exp", "实验", "测评", "ablation"]):
            return "e"
        if any(k in name for k in ["idea", "想法", "晨思", "讨论"]):
            return "i"
        if any(k in name for k in ["论文", "paper", "reading"]):
            return "p"
        if any(k in name for k in ["方案", "报告", "申请"]):
            return "r"
        return "k"

    for src in files:
        print("\n" + "=" * 72)
        print(f"[{src.name}]")
        default_key = guess_choice(src)
        default_dir, default_type = CHOICES[default_key]
        ans = prompt_choice(f"放到哪个目录？(k/i/e/p/r/s) [默认 {default_key}→{default_dir}]", set(CHOICES.keys()), default_key)
        target_dir, dtype = CHOICES[ans]
        if dtype == "skip":
            print("→ 跳过")
            continue

        tags = input("可选：输入逗号分隔的 tags（回车跳过）: ").strip()

        # 先在原地注入 head & scaffold，再移动
        try:
            inject_head_and_scaffold(src, dtype, origin, tags)
        except Exception as e:
            print(f"[!] 写入头信息失败：{e}")
            continue

        # 构造目标路径（保持原文件名）
        dst = (week_dir.parents[1] / target_dir / src.name).resolve()

        try:
            safe_move(src, dst, repo_root)
        except subprocess.CalledProcessError as e:
            print(f"[!] git mv 失败：{e}")
            continue
        except Exception as e:
            print(f"[!] 移动失败：{e}")
            continue

        # TRIAGE 记录
        try:
            src_rel = src.relative_to(week_dir.parents[1]) if week_dir.parents[1] in src.parents else Path(src.name)
            dst_rel = dst.relative_to(week_dir.parents[1])
            append_triage_log(week_dir, src_rel, dst_rel)
        except Exception as e:
            print(f"[!] 写 TRIAGE 失败：{e}")

        print(f"✓ 已移动：{src.name} → {target_dir}/")

    print("\n全部完成 ✅")

if __name__ == "__main__":
    main()
