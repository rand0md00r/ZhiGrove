---
title: OpenUni训练方法
created: 2025-09-07 17:01
updated: 2025-09-07
origin: week-35
type: knowledge
status: draft
tags: [OpenUni,train]
links: []
---

## TL;DR（≤3点）
- 总结了OpenUni的训练方法；
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


## Raw Notes

# 训练是怎么做的

- 整体思路：用最小连接模块把一个冻结的多模态 LLM与一个**扩散/Flow-Matching DiT（SANA）**连接起来。连接模块只含 N=256 个可学习 query 和 6 层 Transformer 的轻量 connector，把 LLM 的条件特征送入扩散模型的 cross-attention 中。LLM 始终冻结，视觉生成能力主要靠 connector+扩散模型学习。
arXiv

- 两阶段训练（Two-stage）：

  - Stage I（预训练，23M对）：只训练 learnable queries + connector，扩散模型冻结（LLM 也冻结）。目标是先把 LLM 条件正确“翻译”成扩散模型可用的条件信号。超参：AdamW，LR=1e-4，Batch=512，Cosine 调度，Warm-up 1k 步，训练 100k 步，GradClip=1.0，WD=0.05，betas=(0.9,0.95)。
arXiv
+1

  - Stage II（高质微调，60K对）：解冻扩散模型并与 connector 一起微调（LLM 仍冻结）。超参：AdamW，LR=1e-5，Batch=256，Cosine 调度，Warm-up 100 步，训练 10k 步，其他同上。
arXiv

- 提示模板与 CFG：文本到图像训练时的提示统一成
User: Generate an image <caption>\n Assistant:，并对 10% 样本把 <caption> 置空以支持推理时的 Classifier-Free Guidance (CFG)。
arXiv

- 模型变体：提供 B-512、L-512、L-1024 三个版本（分别对应 InternVL3-1B/2B 与 SANA-0.6B/1.6B，不同分辨率）。训练配方对三者一致。


# 数据集是怎么构造的

- 预训练语料（23M 图文对）：来自多个公开集合的大合并集，并统一用 LLM 重新标注（re-caption）。论文点名包含：text-to-image-2M、LAION-Aesthetic-6M、Megalith-10M、RedCaps-5M 等，合计约 2300 万 图文对；这些图像的文本描述全部由 LLM 生成/改写，并计划公开。
arXiv

- 高质量微调集（60K 图文对）：采用 BLIP3-o 发布的 6 万高质指令型图文数据。其构造方法是：用 GPT-4o 生成多样 caption，再使用 DALL·E-3、Midjourney 等模型合成图像，从而得到“指令→图像”的高一致性样本，用于 Stage II 的对齐与质量提升。
arXiv

- 数据使用方式：Stage I 用 23M 合并集先学“桥接”；Stage II 用 60K 高质指令数据做小步微调，增强指令跟随与画质鲁棒性。
