
## TL;DR（≤3点）

* **把 LLaMA 的 MLP（SwiGLU）换成 MoE**：注意力仍是**密集共享**，只在若干层把 FFN 替换为“多专家+路由器”。
* **路由 = 轻量线性 + Top-K + 归一权重**；每个 token 只激活 **K 个专家**（常见 K=1 或 2），计算稀疏，参数巨量可见。
* 训练稳定三件套：**容量上限（capacity）**、**负载均衡损失（aux loss）**、**All-to-All 专家并行**。

---

## What（是什么）

以 LLaMA 基础块（RMSNorm → Self-Attn → RMSNorm → MLP/SwiGLU）为底座，把 **MLP** 替换成 **MoE-FFN**。对每个 token 隐状态 $h\in\mathbb{R}^d$：

1. **路由打分（gating）**

$$
s = W_r h \in \mathbb{R}^{E},\quad
\text{选 } \mathcal{T}=\text{TopK}(s)
$$

可选温度/噪声：$s \leftarrow (s+\text{noise})/\tau$。

2. **权重归一**（只在选中的 $\mathcal{T}$ 上 softmax）

$$
g_i(h)=\frac{\exp(s_i)}{\sum_{j\in \mathcal{T}}\exp(s_j)},\quad i\in\mathcal{T}
$$

3. **专家前馈**（每个专家都是一套独立的 SwiGLU-FFN）

$$
E_i(h)=W^{(i)}_{\text{down}}\!\left(\mathrm{SiLU}(W^{(i)}_{\text{up}}h)\odot(W^{(i)}_{\text{gate}}h)\right)
$$

4. **聚合输出**

$$
\mathrm{MoE}(h)=\sum_{i\in\mathcal{T}} g_i(h)\,E_i(h)
$$

（可选：再加一个**共享专家** $E_{\text{shared}}$ 恒被激活，改善长尾稳健性。）

注意：**注意力层不变**；MoE 只替换 FFN，并可“隔层”使用（如每 2\~4 层一个 MoE-FFN）。

---

## Why（为什么这么做/何时使用）

* **更高“表观参数量”**：E 个专家 × FFN 参数，但**每步只算 K 个**，在相近 FLOPs 下提升模型表达力/迁移性。
* **吞吐与成本**：训练/推理仍近似稀疏计算成本；适合大规模预训练与指令跟随扩展。
* **可控专家化**：不同专家可隐式学到不同模式/域；在多域数据上更有益。

---

## How（最小复现配方，≤5步）

1. **在若干 LLaMA 层把 FFN 替换为 MoE-FFN**（保持维度一致；专家数 $E\in\{8,16,32\}$，Top-K 常用 1 或 2，K=2 质量更好、通信更重）。
2. **实现路由与容量**：为每层设容量

$$
C=\left\lceil \text{capacity\_factor}\cdot \frac{B\cdot L\cdot K}{E}\right\rceil
$$

每个专家最多接收 $C$ 个 token；溢出按策略 **drop** 或 **随机回退**。
3\) **All-to-All 专家并行**：把不同专家分布到不同 GPU（expert parallel），用 A2A 把被分配的 token 打包发送/回收。
4\) **加负载均衡损失**（Switch/GShard 风格）：

* 令 $f_i$=分到专家 $i$ 的 token 份额，$p_i$=路由概率质量在专家 $i$ 的份额；
* 辅助损失 $L_{\text{aux}} = E \cdot \sum_i f_i \, p_i$（或等价的均衡正则/熵正则），系数一般 $10^{-2}\!\sim\!10^{-1}$。

5. **训练细节**：路由温度/噪声、容量系数（1.0–1.5）、路由 logits 的 z-loss（抑制过饱和）、混合并行（DP+TP+EP），以及 **drop-tokens vs dropless** 路由策略按资源取舍。

---

## Gotchas（坑点与边界）

* **通信成为瓶颈**：Top-2 + 大 E 会放大 All-to-All；用专家分组、流水或“分块 GEMM（megablocks）”缓解。
* **负载不均/溢出**：没有 aux loss 或温度太低时，少数专家被“吸满”；监控各专家的 **利用率直方图**。
* **数值稳定**：路由 logits 可加微噪声/温度；必要时 clip；K=1（Switch）更稳但上限略低。
* **放置策略**：不是所有层都用 MoE。实践里常“**间隔放置**”或“**后段更密**”（高层语义更受益）。
* **推理并行**：生产推理多用 **expert sharding**；批很小时通信开销相对更高，需权衡 E 与并行度。
* **度量**：除了 PPL/任务指标，**看专家负载熵、容量利用率、溢出率** 才能定位路由是否健康。

---

### 常见超参参考（经验值）

* $E=8$ 或 $16$，**Top-K=2**，capacity\_factor $=1.25$；
* 路由温度 $\tau\in[0.5,1.0]$，或加小噪声（Noisy Top-K）；
* $L_{\text{aux}}$ 系数 $=0.01$ 左右；
* 仅 **部分层** 使用 MoE（例如每 2 层一个 MoE-FFN）。

---

**一句话**：MoE-LLaMA = **“注意力密集 + MoE-FFN 稀疏”**。用一个轻量路由器把 token 分发给少数专家（Top-K），再按门控权重加权求和；想把它跑稳，核心在 **容量与负载均衡**，想把它跑快，核心在 **专家并行的 All-to-All**。
