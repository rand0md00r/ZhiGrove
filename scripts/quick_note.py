#!/usr/bin/env python3
"""
快速记录脚本 - 支持多种记录方式
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

def create_quick_note(content, note_type="idea", title=None):
    """创建快速笔记"""
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M")
    
    if not title:
        title = f"快速记录-{time}"
    
    # 确定目录和文件路径
    year = datetime.now().strftime("%Y")
    
    if note_type == "idea":
        dir_path = f"30-ideas/{year}"
        filename = f"{date}-{title}.md"
    elif note_type == "paper":
        dir_path = f"20-papers/{year}"
        filename = f"{date}-{title}.md"
    elif note_type == "experiment":
        dir_path = f"40-experiments/exp-{date}-{title}"
        filename = "log.md"
    elif note_type == "report":
        dir_path = f"50-reports/weekly/{year}"
        filename = f"week-{datetime.now().strftime('%V')}.md"
    elif note_type == "knowledge":
        dir_path = f"10-knowledge"
        filename = f"{date}-{title}.md"
    else:
        dir_path = f"00-inbox"
        filename = f"{date}-{title}.md"
    
    # 创建目录
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 如果是实验，创建子目录
    if note_type == "experiment":
        Path(f"{dir_path}/results").mkdir(exist_ok=True)
        Path(f"{dir_path}/configs").mkdir(exist_ok=True)
    
    file_path = os.path.join(dir_path, filename)
    
    # 生成内容
    if note_type == "idea":
        template = f"""---
title: {title}
date: {date}
tags: []
status: draft
links: []
summary: {content[:100] if len(content) > 100 else content}
---

## 💡 想法概述
{content}

## 🔍 相关资源
- 

## 📝 下一步行动
- [ ] 

## 🏷️ 标签
- 
"""
    elif note_type == "paper":
        template = f"""---
title: {title}
date: {date}
tags: []
status: draft
links:
  paper:
  code:
  project:
summary: {content[:100] if len(content) > 100 else content}
---

## 1. 任务与动机
- 试图解决什么问题？哪里比前人更进一步？

## 2. 方法概述（一句话 + 框图/要点）
- 要点 1
- 要点 2

## 3. 核心技术细节（我能复现吗）
- 模块/损失/训练设置简要

## 4. 实验与结论
- 关键指标表/图 + 我在乎的 ablation

## 5. 启发与局限
- 我可以把它用在（OpenUni / Flow Matching / MoE / Function-Call）哪
- 潜在问题或改进点

## 6. TODO
- [ ] 跑个最小复现实验 / 对比到现有流水线
"""
    elif note_type == "experiment":
        template = f"""---
title: {title}
date: {date}
tags: []
status: running
summary: {content[:100] if len(content) > 100 else content}
---

## 🎯 实验目标
{content}

## 📊 实验设置
- 数据集：
- 模型：
- 超参数：

## 🔬 实验过程
- 

## 📈 实验结果
- 

## 💭 分析与结论
- 

## 📝 下一步
- [ ] 
"""
    elif note_type == "report":
        template = f"""---
title: {title}
date: {date}
tags: []
status: draft
summary: {content[:100] if len(content) > 100 else content}
---

## 📊 本周总结
{content}

## 🎯 主要成果
- 

## 📝 遇到的问题
- 

## 🔍 解决方案
- 

## 📋 下周计划
- [ ] 

## 💡 思考与反思
- 
"""
    elif note_type == "knowledge":
        template = f"""---
title: {title}
date: {date}
tags: []
status: draft
category: knowledge
difficulty: intermediate
prerequisites: []
related: []
links:
  official: 
  paper: 
  code: 
  tutorial: 
  project: 
summary: {content[:100] if len(content) > 100 else content}
---

# {title}

## 📚 概述

### 核心概念
{content}

### 关键特点
- 

### 应用场景
- 

## 🔍 详细内容

### 基本原理
- 

### 核心算法/方法
- 

### 技术架构
- 

## 💡 最佳实践

### 使用建议
- 

### 常见陷阱
- 

### 性能优化
- 

## 🧪 实践案例

### 示例代码
```python
# 代码示例
```

### 实际应用
- 

### 效果评估
- 

## 🔗 相关知识

### 前置知识
- 

### 相关技术
- 

### 扩展阅读
- 

## 📊 总结与反思

### 核心收获
- 

### 适用条件
- 

### 局限性
- 

### 改进方向
- 

## 📝 更新记录

| 日期 | 更新内容 | 更新人 |
|------|----------|--------|
| {date} | 初始创建 | |

## 🏷️ 标签

- 技术领域：
- 难度等级：
- 应用领域：
- 相关项目：

---

> **注意**：这是一个知识沉淀文档，内容应该经过验证和测试，确保准确性和实用性。
"""
    else:
        template = f"""# {title}

> 记录时间：{date} {time}

{content}

---
标签：
相关：
"""
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"✅ 已创建: {file_path}")
    return file_path

def main():
    parser = argparse.ArgumentParser(description="快速创建笔记")
    parser.add_argument("content", help="笔记内容")
    parser.add_argument("-t", "--type", choices=["idea", "paper", "experiment", "report", "knowledge", "inbox"], 
                       default="idea", help="笔记类型")
    parser.add_argument("--title", help="笔记标题")
    
    args = parser.parse_args()
    
    try:
        file_path = create_quick_note(args.content, args.type, args.title)
        print(f"📝 笔记已保存到: {file_path}")
        print(f"💡 提示：使用 'code {file_path}' 在编辑器中打开")
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 