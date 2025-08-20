# 🛠️ 脚本功能说明

## 📋 脚本概览

| 脚本名称 | 功能 | 状态 | 说明 |
|----------|------|------|------|
| `quick_start.sh` | 统一入口脚本 | ✅ 推荐使用 | 整合了所有功能，消除重复 |
| `quick_note.py` | 快速记录核心 | ✅ 核心功能 | 支持多种笔记类型 |
| `status_check.py` | 状态检查 | ✅ 功能完整 | 检查知识库健康状态 |
| `build_index.py` | 索引更新 | ✅ 功能完整 | 自动更新README |
| `new.sh.backup` | 传统创建脚本 | ⚠️ 已备份 | 功能已被整合，建议不再使用 |

## 🚀 推荐使用方式

### 主要入口：`quick_start.sh`
```bash
# 检查状态
./scripts/quick_start.sh status

# 创建内容
./scripts/quick_start.sh idea "想法内容"
./scripts/quick_start.sh paper "论文标题"
./scripts/quick_start.sh experiment "实验名称"
./scripts/quick_start.sh report "报告标题"
./scripts/quick_start.sh inbox "临时内容"

# 更新索引
./scripts/quick_start.sh update

# 查看帮助
./scripts/quick_start.sh help
```

## 🔄 脚本重构说明

### 为什么重构？
1. **功能重复**：`new.sh` 和 `quick_start.sh` 都有创建笔记的功能
2. **维护困难**：两个脚本维护相同的功能
3. **用户体验**：用户不知道用哪个脚本

### 重构后的优势
1. **统一入口**：一个脚本处理所有操作
2. **功能完整**：支持所有笔记类型
3. **易于维护**：减少重复代码
4. **用户体验**：清晰的命令结构

## 📝 脚本功能对比

### 创建笔记功能
| 功能 | 原 new.sh | 新 quick_start.sh | 改进 |
|------|------------|-------------------|------|
| 想法记录 | ✅ | ✅ | 模板更丰富 |
| 论文笔记 | ✅ | ✅ | 支持更多字段 |
| 实验记录 | ✅ | ✅ | 自动创建子目录 |
| 报告创建 | ❌ | ✅ | 新增功能 |
| 收件箱记录 | ❌ | ✅ | 新增功能 |

### 其他功能
| 功能 | 原 new.sh | 新 quick_start.sh | 说明 |
|------|------------|-------------------|------|
| 状态检查 | ❌ | ✅ | 新增功能 |
| 索引更新 | ❌ | ✅ | 新增功能 |
| 帮助信息 | ❌ | ✅ | 新增功能 |

## 💡 使用建议

### 1. **日常使用**
- 使用 `./scripts/quick_start.sh` 作为主要工具
- 根据内容类型选择合适的命令

### 2. **快速记录**
```bash
# 有想法时
./scripts/quick_start.sh idea "我的新想法"

# 学习论文时
./scripts/quick_start.sh paper "论文标题"

# 做实验时
./scripts/quick_start.sh experiment "实验名称"
```

### 3. **定期维护**
```bash
# 检查状态
./scripts/quick_start.sh status

# 更新索引
./scripts/quick_start.sh update
```

## 🔧 技术细节

### 脚本架构
```
quick_start.sh (统一入口)
    ↓
quick_note.py (核心功能)
    ↓
创建不同类型的笔记
```

### 支持的类型
- `idea`: 想法记录
- `paper`: 论文笔记
- `experiment`: 实验记录
- `report`: 报告创建
- `inbox`: 收件箱记录

### 自动功能
- 自动创建目录结构
- 自动生成文件名
- 自动填充模板
- 自动创建子目录（实验类型）

## 📚 相关文档

- [工作流指南](../../WORKFLOW.md)
- [收件箱指南](../../00-inbox/README.md)
- [知识沉淀指南](../../10-knowledge/README.md)
- [可视化工作流](../../WORKFLOW_VISUAL.md)

---

**记住：现在只需要使用 `quick_start.sh` 一个脚本就能完成所有操作！**
