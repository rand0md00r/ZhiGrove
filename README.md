# ZhiGrove · 知林

> 个人研究型知识库：多模态大模型 / Flow Matching / MoE / 机器人学 / 函数调用优化等。

## 🚀 快速开始

### 一键检查状态
```bash
# 检查知识库健康状态
./scripts/quick_start.sh status

# 或者简写
./scripts/quick_start.sh s
```

### 快速记录
```bash
# 记录想法
./scripts/quick_start.sh idea "我的新想法"

# 创建论文笔记
./scripts/quick_start.sh paper "论文标题"

# 记录实验
./scripts/quick_start.sh experiment "实验名称"

# 创建报告
./scripts/quick_start.sh report "报告标题"

# 记录到收件箱
./scripts/quick_start.sh inbox "临时内容"
```

### 查看帮助
```bash
./scripts/quick_start.sh help
```

## 📚 使用方式

- **目录约定、模板与脚本**：见 `templates/` 与 `scripts/`
- **收件箱 → 轻整理 → 沉淀**：见 `/00-inbox → /30-ideas → /10-knowledge`
- **完整工作流**：见 `WORKFLOW.md`

### 近期更新

**论文**  
- 2025-08-27 · [2025-08-20-](20-papers/2025/2025-08-20-.md)
- 2025-08-27 · [2025-08-20-vtla-preference-learning](20-papers/2025/2025-08-20-vtla-preference-learning.md)

**闪念/会议**  
- 2025-08-27 · [20250819-ideation-cfg-schedule](30-ideas/2025/20250819-ideation-cfg-schedule.md)

**实验**

## 📁 目录导航

- [00-inbox/](00-inbox/) – 临时收件箱（快速记录）
- [10-knowledge/](10-knowledge/) – 领域知识手册（沉淀区）
- [20-papers/](20-papers/) – 论文学习笔记
- [30-ideas/](30-ideas/) – 闪念&会议小结
- [40-experiments/](40-experiments/) – 实验日志
- [50-reports/](50-reports/) – 周报/月报/汇报材料

## 🏷️ 标签体系（示例）

`llm`, `moe`, `flow-matching`, `multimodal`, `robotics`, `rl`, `function-call`, `diffusion`, `clip`, `cfg`, `evaluation`

## 📋 约定

- **Frontmatter 统一字段**：`title/date/tags/status/links/related/summary`
- **图片就近存放**：当前笔记同级 `assets/`；通用图放 `assets/`
- **提交规范**：Conventional Commits

## 🛠️ 工具脚本

- `scripts/quick_start.sh` - **统一入口脚本（推荐使用）**
- `scripts/quick_note.py` - 快速记录核心功能
- `scripts/status_check.py` - 状态检查脚本
- `scripts/build_index.py` - 索引更新脚本
- `scripts/README.md` - 脚本功能说明

> **注意**：`new.sh` 已被整合到 `quick_start.sh` 中，避免功能重复

## 📖 使用指南

- [工作流指南](WORKFLOW.md) - 每日/每周/每月的工作流程
- [收件箱指南](00-inbox/README.md) - 如何快速记录和整理
- [知识沉淀指南](10-knowledge/README.md) - 如何整理和沉淀知识
- [脚本说明](scripts/README.md) - 脚本功能和使用方法

## 🔄 授权

- 私有（2025/08/20）