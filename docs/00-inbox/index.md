# 📥 Inbox 工作流（快速上手）

> **唯一目标：快速落地 → 周日一次性归档 → Inbox 清零**

---

## 1) 先记录（随手、低摩擦）
- **推荐**：在当周文件夹 `00-inbox/week-XX/` 直接新建 `.md`
- **文件名**：`[YYMMDD-HHMM]-简述.md`（不纠结，能区分就行）
- **内容最少三行**：
  - 今天学到：…
  - 遇到问题：…
  - 下一步：…

``` bash
python scripts/triage_cli.py 00-inbox/week-XX


# 交互选择：k=knowledge / i=idea / e=experiment / p=paper / r=report / s=skip
# 脚本会：
# 1) 在文件顶部加入对应卡片头（YAML + 骨架）
# 2) 原文自动放入文末 “## Raw Notes”
# 3) 自动移动到 10/20/30/40/50 对应目录
# 4) 在 00-inbox/week-XX/TRIAGE.md 记录迁移日志
```
