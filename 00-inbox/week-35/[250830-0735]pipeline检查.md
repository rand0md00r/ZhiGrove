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

