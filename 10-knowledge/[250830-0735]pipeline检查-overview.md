---
title: pipeline检查
created: 2025-09-07 17:02
updated: 2025-09-07
origin: week-35
type: knowledge
status: draft
tags: [flow-matching, VAE, latent-scaling, KL, training-checklist]
links: []
---

## TL;DR（≤3点）
- **先看尺度**：确保 `z₀`（文本头部采样）与 `z₁`（VAE latent）在**同一缩放**与**同一统计量级**（典型 `∥z∥₂` 批均值同量级，差不超 2–3×）。
- **再看目标**：噪声自由 FM 下，路径 `x_t=(1-t)x₀+t x₁`，目标速度 **`v* = x₁ - x₀`（常数）**；别把 `ε`/score 目标混进去，别反号。
- **防“卡死”**：给 KL **warmup** + 对 `logvar` 合理 **clamp**；`v*` 计算时对 `x₀,x₁` **detach**，只让预测 `u_θ` 回传。

## What（是什么）
- 这是一个从「**尺度**、**目标**、**梯度**、**结构**」四个维度快速排障的 **Flow Matching（FM） 文本→图像潜空间**训练检查单。
- 你的 pipeline：文本 → VLM → `q_token` → 线性头（3层）→ `(μ, logvar)` → **采样** `z₀` →（noise-free）FM → 目标图像 latent `z₁`。

## Why（为什么这么做/何时使用）
- FM 在噪声自由设定下对**尺度极敏感**；任何一端的单位或缩放错位都会让 `v*=x₁-x₀` 被**一侧主导**，训练发散或头部/KL “卡死”。
- 目标写错（把 DDPM 的 `ε` 当目标；或把速度反号）会导致模型学到错误物理量，损失不降。
- KL 与 `logvar` 容易在早期**过强/过小**，需要暖启动与数值保护。

## How（最小复现配方，≤5步）
1. **对齐 VAE 缩放**  
   核对 autoencoder 的 `scaling_factor`（SD 家族常见 **0.18215**；也有其他值）。确保：
   - `encode()` / `encode_moments()` 返回的量**是否已乘**缩放；
   - 你的 `z₁` 与解码 `decode()` 的接口**互逆**（编码乘、解码除，或反之，切勿重复/漏乘）。
2. **量纲自检**  
   同一批次打印
   - `E[∥z₀∥₂]`, `E[∥z₁∥₂]`, `E[∥x_t∥₂]`（t~U[0,1]）、`E[∥v*∥₂]`；  
   - 期望它们同量级；若 `z₁` 比 `z₀` 大 **>3×**，优先修正缩放而非引入额外比值校正。
3. **目标定义核对（noise-free FM）**  
   - 路径：`x_t=(1-t)x₀+t x₁`；目标：`v*(x_t,t)=x₁-x₀`（与 t 无关）。  
   - 训练：最小化 `E_{t,x₀,x₁} [‖u_θ(x_t, t, cond) - (x₁ - x₀)‖²]`。  
   - **注意**：`x₀, x₁` 在构造 `v*` 时 **detach**，防止“移动靶”。
4. **KL 稳定化**  
   - 令 `q_φ(z₀|text)=N(μ, diag(σ²))`，对 `KL(q_φ‖N(0,I))` 做 **warmup**（如前 10% 训练从 0 线性升至目标权重）。  
   - 对 `logvar=log σ²` **clamp 到 [-6,6]**（或依据数据调），避免早期 σ→0 导致 KL 爆大。
5. **小集合过拟合**  
   - 固定一小撮（如 128 条）样本，观察**数小时内**：FM MSE 下降、`cos(u_θ, v*)↑`、KL 平稳；若不能过拟合，回到 1–4 排查。

## Gotchas（坑点与边界）
- **尺度优先**：别用临时比值 `r=∥z₁∥/∥z₀∥` 去“救火”，先把 VAE 的缩放口径搞对。
- **目标不混淆**：FM 的速度目标 `x₁-x₀` ≠ DDPM 的噪声 `ε`，也 ≠ score；签名错/混合会让损失表面看降实则学偏。
- **时间编码**：若给 `t` 做正弦位置编码/MLP，**注意幅值**（过大易主导网络），可做 LayerNorm 或小幅度初始化。
- **损失归一化**：MSE/维度平均要与 KL 权重在**同一数量级**；记录两者量纲，避免某项“淹没”另一项。
- **梯度路径**：`z₀` 由 reparam 采样得到，**允许**梯度回到 `(μ, logvar)`（经 reparam trick），但**不允许**通过 `v*` 反向“改目标”（对 `x₀,x₁` detach）。
- **数值精度**：半精度下启用 GradScaler / 动态 loss scaling；`x_t` 的构造尽量 FP32 计算后再 cast。
- **数据口径**：文本与图像对齐、增广一致性、VAE 的 train/eval 模式（某些 VAE 在 eval 关闭噪声/BN）。
- **评估口径**：仅在**相同** VAE、分辨率、latent 维度、缩放下比较指标；跨口径比较没有意义。


## Raw Notes

从「尺度、目标、梯度、结构」四类问题排

现在的pipeline是：
- 文本 → VLM → q_token → 三层 Linear → (μ, logvar) → 采样 z₀ →（noise-free）Flow Matching → 目标图像潜变量 z₁

# To check list

- 1. VAE 潜空间尺度对齐（最容易踩坑）

SD/VAEs 往往用 scale=0.18215。确保 z₁ 和你头部输出的 z₀ 在同一量纲。

  - 若 encode() 已返回规范化 latent，就不要再乘/除；

  - 打印 ||z₁||₂ 与 ||z₀||₂ 的 batch 均值：两者差不要超过 2–3 倍。若差距大，先把头部权重缩小或把 z₁ 做相同缩放。

- 问题1：x0和x1的尺度差异过大，∥x0∥≈151~155， ∥x1∥≈896~1224，在 FM 里目标速度 v = x1 - x0 会被 x1 的尺度完全主导，头部和 KL 会“卡死”。
  - （不需要）加个 r = (z1.pow(2).mean().sqrt() / (z0.pow(2).mean().sqrt() + 1e-8)).clamp(1e-3, 1e3) 
  - 原因：没有采样：
  ``` python
  mu, logvar = autoencoder.encode_moments(x)         # 只算参数，不采样
  # = posterior = autoencoder.encode(x)              # 某些实现返回分布对象
  #   posterior.mean(), posterior.sample()
  
  z1 = autoencoder.sample(x)  # 内部：encode_moments -> reparam -> (可能)乘 scaling_factor
  ```
  - mu, logvar：像素 x 的后验分布参数，单张图是对角高斯N(μ(x),σ2(x)I)
  - sample(x)：返回一次采样的潜变量（训练期常用）。
  - scaling_factor：很多 LDM/SD 家族把潜空间统一缩放（编码时乘、解码时除）。别重复乘或漏乘，否则会出现你刚看到的“z1 量级巨大”。
  

- 2. KL权重与logvar范围

  - 平台常见因：logvar 一开始太小（σ→0），KL 过强把头部压死。

  - 暂时 clamp logvar ∈ [-6, 6]，并 KL warmup（前 10% step 从 0 线性升到目标值）。

- 3. FLow目标是否正确？

