---
title: token熵
created: 2025-09-07 17:02
updated: 2025-09-07
origin: week-35
type: knowledge
status: draft
tags: [entropy, token, perplexity, temperature, training-signal]
links: []
---

## TL;DR（≤3点）
- **定义**：位置 \(t\) 的 token 熵 \(H_t=-\sum_v p_t(v)\log p_t(v)\)，度量**下一个词分布的不确定性**；与样本惊讶度 \(-\log p_t(y_t)\) 不同（它是期望惊讶度）。
- **等价式**：若 \(p_t(v)=\exp(\beta z_{t,v})/Z_t\)，则 \(H_t=\log Z_t-\beta\,\mathbb{E}_{p_t}[z]\)；**温度升高**（\(T=1/\beta\)）→ 熵**单调增**。
- **用途**：定位**高不确定位置**（筛选/加权）、温度/采样策略调参、与困惑度 \(\mathrm{PPL}=\exp(H)\) 对应做监控与报警。


## What（是什么）
- **Token 熵**是**Shannon 熵**在“下一词分布”上的实例：衡量模型对下一个 token 的不确定性。
- 取自然对数单位为 **nat**，上界为 \(\log|V_t^{\text{mask}}|\)（在有效词表上的均匀分布），下界为 0（one-hot）。


## Why（为什么这么做/何时使用）
- **训练观测**：监控 \(H_t\) 的统计可发现**过度自信/塌缩**或**过平**的问题（校准）。
- **数据与目标选择**：用高熵位置做**掩码/加权**，集中算力在“真正不确定”的 token 上（如 Top-K/Top-ρ）。
- **采样与温度**：指导推理期的 **T / top-p / top-k** 调参；温度上升应伴随熵上升，否则提示 logits 量纲或掩码异常。


## How（最小复现配方，≤5步）
1. 从模型获得 logits \(z_t\)，按温度 \(T\) 计算 \(p_t(v)=\mathrm{softmax}(z_t/T)\)（训练监控通常取 \(T{=}1\)）。
2. 计算 \(H_t=-\sum_v p_t(v)\log p_t(v)\)；**仅在有效词表上**求和并重归一（mask）。
3. 统计与聚合：对 batch/序列取均值、分位数（如 P50/P90），输出曲线与报警阈值。
4. 选点：设分位阈值 \(\tau_\rho\)（如 80/90 分位）或序列 Top-K，把 \(\{t:H_t\ge\tau\}\) 作为**高熵位置**。
5. 融合训练：在策略/监督目标中对这些位置**加权或掩码**；若仅做“路标”，**对 \(H_t\) detach**，避免将“更不确定”作为优化方向。


## Gotchas（坑点与边界）
- **与 NLL 的区别**：\(-\log p_t(y_t)\) 是样本惊讶度；\(H_t\) 是其**期望**。混用会误读。
- **温度与单位**：训练监控建议固定 \(T{=}1\)；推理温度改变采样熵但**不改变**训练时度量口径。注意 \(\log\) 基底（nat vs bit）。
- **数值稳定**：用 \(\log\text{-}\mathrm{sum}\exp\) 计算 \(\log Z\)；掩码后要**重归一**。
- **标签平滑/正则**：label smoothing 会抬高目标熵；对照时需注明是否启用。
- **回传策略**：把熵作为**正则项**回传与把熵作为**选择/加权信号**（detach）在优化语义上不同，勿混搭。



## Raw Notes

很好，这里把“token 熵”从**严格公式**出发讲清楚，并点明与常见量（交叉熵、负对数似然、困惑度）以及温度/采样的关系。整段**不含伪代码**，只给出可直接落地的数学表达。

---

# 1) 场景与符号

* 设第 $t$ 个生成位置的**logits 向量**为 $z_t\in\mathbb{R}^V$（$V$=词表大小），温度 $T>0$，$\beta=1/T$。
* **策略分布**（对下一个词元的条件分布）：

  $$
  p_t(v)\;=\;\pi_\theta(v\mid \text{context}_t)\;=\;\frac{\exp(\beta\, z_{t,v})}{Z_t},\quad
  Z_t=\sum_{j=1}^V \exp(\beta\, z_{t,j}).
  $$

> 这里的“context$_t$”就是输入 + 先前已生成的 token 前缀 $y_{&lt;t}$。

---

# 2) Token 熵的**定义**与等价式

## 2.1 标准定义（Shannon 熵，位置 $t$ 的分布不确定性）

$$
\boxed{\,H_t \;=\; -\sum_{v=1}^{V} p_t(v)\,\log p_t(v)\,}
$$

* 若用自然对数 $\log$，单位为 **nat**；若用 $\log_2$，单位为 **bit**。
* **语义**：这是**分布**的属性，不是被采样出的具体 token 的属性。

## 2.2 与 softmax“能量”形式的恒等式

把 $p_t$ 写成 $p_t(v)=e^{\beta z_{t,v}}/Z_t$，可得到：

$$
\boxed{\,H_t \;=\; \log Z_t \;-\; \beta\;\mathbb{E}_{v\sim p_t}[\,z_{t,v}\,]\,}
$$

* 这条式子常用于分析温度/梯度（见 §4）。

---

# 3) 与**负对数似然/交叉熵/困惑度**的区别与联系

1. **样本的自信息**（被抽到的那个 token $y_t$ 的“惊讶度”）：

$$
I_t(y_t)\;=\;-\log p_t(y_t).
$$

* 这是**标量**，依赖于**采样出的具体 token**。

2. **位置分布的熵**（我们这里的 $H_t$）：

$$
H_t \;=\; \mathbb{E}_{v\sim p_t}[\,I_t(v)\,]\;=\;-\sum_v p_t(v)\log p_t(v).
$$

* 这是**该位置分布的期望惊讶度**，不依赖某个采样结果。

3. **交叉熵**（若有“真分布” $q_t$）：

$$
\mathrm{CE}(q_t,p_t)\;=\;-\sum_v q_t(v)\log p_t(v)\;=\;H(q_t)\;+\;\mathrm{KL}(q_t\Vert p_t).
$$

* 当“真分布”是**一元分布**（one-hot）时，$\mathrm{CE}=-\log p_t(y_t)$ 就是**NLL**。

4. **困惑度**（perplexity）：

$$
\mathrm{PPL}_t \;=\; \exp\!\big(H_t\big).
$$

* 这是一种把熵**指数化**后的直观度量；平均到语料上得到数据级 PPL。

---

# 4) **温度**与熵的单调性、可导性

用 §2.2 的恒等式，写成 $H_t(\beta)=\log Z_t-\beta\,\mathbb{E}_{p_t}[z]$。可得两条关键性质：

1. **对温度的单调性**
   先对 $\beta$ 求导，再链式法则到 $T$：

$$
\frac{\partial H_t}{\partial \beta}
\;=\; -\,\beta\;\mathrm{Var}_{p_t}(z_{t,\cdot})\;\le 0
\quad\Longrightarrow\quad
\boxed{\;\frac{\partial H_t}{\partial T}
\;=\;\frac{\mathrm{Var}_{p_t}(z_{t,\cdot})}{T^{3}}\;\ge 0\;}
$$

* 结论：**温度越高（分布越平），熵越大**；温度 $\to 0$ 时 $p_t$ 退化为 one-hot，熵 $\to 0$。

2. **对 logits 的梯度**（若你在目标里“真地”对熵求导）

$$
\boxed{\;\frac{\partial H_t}{\partial z_{t,k}}
\;=\; -\,\beta^2\;p_t(k)\,\Big(z_{t,k}-\mathbb{E}_{p_t}[z]\Big)
\;=\; -\,\beta\,p_t(k)\,\Big(\log p_t(k)+H_t\Big)\;}
$$

* 一些方法（如“**优势熵塑形**”）选择 **detach** 熵，不让这一梯度回传，从而把熵仅当作“加权信号”使用（见应用文献中的做法）。

---

# 5) **上下界**与**特殊情况**

* 设有效词表为 $V_t^{\text{mask}}$（把 PAD/非法/被屏蔽 token 去掉并重归一），则

  $$
  \boxed{\,0\;\le\;H_t\;\le\;\log |V_t^{\text{mask}}|\,}.
  $$

  * **下界取等**（完全确定）：$p_t$ 为 one-hot。
  * **上界取等**（完全无偏）：$p_t$ 为在 $V_t^{\text{mask}}$ 上的**均匀分布**。

* 与**均匀分布**的 KL 关系：

  $$
  \mathrm{KL}\big(p_t\;\Vert\;\mathcal{U}\big)\;=\;\log|V_t^{\text{mask}}|\;-\;H_t.
  $$

  —— 熵越大，离均匀越近；熵越小，分布越“尖”。

---

# 6) **批量/序列**上的聚合与“高熵位置”的选择

* **批内 Top-$\rho$ 分位阈值**
  把一个（微）批全部 token 的 $H_t$ 聚在一起，取分位数阈值 $\tau_\rho$：

  $$
  \text{mask}_t \;=\;\mathbf{1}\big[\,H_t \ge \tau_\rho\,\big].
  $$

  —— 在很多工作中用它\*\*筛选“高熵少数 token”\*\*进入策略梯度（$\rho\approx 10\!\sim\!30\%$ 常见）。

* **序列 Top-$K$**
  对**单条**序列的 $\{H_t\}_{t=1}^L$ 取 Top-$K$ 位置作为“**分岔锚点**”，再围绕这些锚点做结构化的探索/局部展开。

---

# 7) 与训练目标的**正确拼接方式**（只述公式，不给代码）

* **掩码式（只训高熵）**
  在 token-level 的策略目标（如 PPO/GRPO/DAPO 的 clipped surrogate）里，将**优势/损失**乘上 $\text{mask}_t$：

  $$
  \max_\theta\;\mathbb{E}\big[\;\text{mask}_t\cdot
  \min\big(r_t(\theta)A_t,\;\mathrm{clip}(r_t(\theta),1-\varepsilon_{\!l},1+\varepsilon_{\!h})A_t\big)\;\big].
  $$

* **优势“熵塑形”（不回传熵梯度）**

  $$
  \tilde A_t \;=\; A_t \;+\; \min\!\Big(\alpha\,H_t^{\text{detach}},\;\frac{|A_t|}{\kappa}\Big),
  \qquad
  \max_\theta\;\mathbb{E}\big[\min(r_t\tilde A_t,\;\mathrm{clip}(\cdot)\tilde A_t)\big].
  $$

  —— 只改变**步长幅度/优先级**，不改变原有优化方向。

* **结构化探索（熵作锚点评分）**
  用 $\{H_t\}$ 选定锚点，再从这些前缀做**局部 rollouts**估计中间价值 $V(\text{prefix})$，把它变成**更密集的优势**（可加自适应缩放），再进常规策略更新。

---

# 8) 常见容易混淆的点（一口气厘清）

1. **“token 熵” vs “负对数似然”**
   $-\log p_t(y_t)$ 是**样本**的惊讶度；$H_t$ 是**分布**在该位置的期望惊讶度。两者不是同一量，但 $\mathbb{E}_{y_t\sim p_t}[-\log p_t(y_t)]=H_t$。

2. **“加熵正则” vs “用熵做路标”**
   在损失里加 $+\beta\sum H_t$ 会**反向传导“变得更不确定”**；而“优势熵塑形/高熵掩码”只是**用 $H_t$** 来**决定更新力度或范围**，可以选择 **detach** 避免改变优化目标。

3. **温度/采样与熵**
   训练时 $H_t$ 通常由 $T=1$ 的分布计算；推理期若调采样温度/Top-p/Top-k，会**改变采样分布的熵**，但**不改变训练时度量的 $H_t$**（除非你也显式改训练温度）。

---

> 结论（一句话）：
> **token 熵**就是**该位置 softmax 分布**的不确定性 $H_t=-\sum p\log p$；用 softmax 形式可写成 $H_t=\log Z_t-\beta\,\mathbb{E}_p[z]$。它**不等同**于某个采样 token 的 $-\log p$，而是**其期望**。在训练中，熵既可**作为选择/加权信号**（不回传），也可（少见地）**作为正则项**回传——两者数学与优化含义不同，务必区分。
