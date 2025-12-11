---
title: exp_card_0903_transencodder
created: 2025-09-07 16:57
updated: 2025-09-07
origin: week-36
type: experiment
status: draft
tags: [transencoder]
links: []
---

## EXP CONFIG META（把下块注释贴进你的配置文件顶部）

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

## Raw Notes

**配置卡片**
```
# ===================================================================================
# 实验配置：internvl_query_5TransEncoder_sdvae_23m_256pix_bs4096_0903.py
# -----------------------------------------------------------------------------------
# InternVL3 Query 5TransEncoder SDVAE 23M 配置
# 
# 基础配置：
# 本配置用于训练 InternVL3-Query-CrossFlow 架构，采用 5 层 TransEncoder 作为文本特征压缩器，
# 搭配 SDVAE 23M 图像自编码器，输入分辨率为 256 像素，批量大小 4096。
# - LLM Backbone: InternVL3-2B
# - 图像自编码器: dc-ae-f32c32-sana-1.1-diffusers
# - 图像分辨率: 256x256
# - 批量大小: 4096
# - 支持多种损失与训练策略（包括Flow Matching、KL 正则）
# - 适配 deepspeed 分布式训练
# 
# 实验参数：
# - 文本编码器: 5 层 TransEncoder，隐藏维度1536，注意力头8
# - LR : 5e-5
# - max_iterations: 120000(20 epochs)
#  
# ===================================================================================
```
# 实验记录

**配置文件**：`configs/pretrain/internvl_query_5TransEncoder_sdvae_23m_256pix_bs4096_0903.py `  
**日期**：`2025年9月3日`  

## 1. 实验目的（一句话）
- `使用轻量化的TransEncoder，将MetaQuery投影压缩到高斯空间`
- `提高了学习率`

## 2. 配置差异（相对基线）
- 关键差异：`transencoder=5层；metaquery=256(不解耦)；LR×5`

## 3. 训练与推理记录（简要）
**训练快照（挑代表迭代点）**

**打印日志 1k**
``` 

```

**打印日志 5k**
``` 

```

**打印日志 10k**
``` 

```

**推理样例（可选）**
**检查点 10k iters**
``` 

```

**检查点 20k iters**
``` 

```

## 4. 结果分析（要点式）
- ① `<要点1：例如收敛更稳/更快或出现nan>`  
- ② `<要点2：平台loss对比基线的差异>`  
- ③ `<要点3：资源/吞吐/稳定性观察>`

## 5. 实验结论（一句话 + 下一步）
- **结论**：`<成功 / 部分成功 / 未达到预期>`（理由：`<最关键依据>`）  
- **下一步**：`<例如：回退LR×3；延长warmup；改T/V比例；共享/独立头对照；增加门控等>`
