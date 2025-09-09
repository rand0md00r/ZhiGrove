# 训练 Loss 调试思路

> 适用于同时包含 **Flow Matching**、**DINO 语义监督**、**KL 正则** 的训练。目标：**验证每个分支是否起作用**，以及**快速定位异常**。

---

## 0) 日志与可观测性基线

**命名建议**

- 损失：`loss_flow` / `loss_dino` / `loss_kl`；原始 KL：`kl_raw`
- 语义对齐：`dino_global`（越小越好）
- 重参数化统计：`z_init/mean`、`z_init/var`（= 采样后的 `z_text`）
- 梯度强度：`dbg/grad_flow`、`dbg/grad_dino_t_head`、`dbg/grad_mu_head`、`dbg/grad_logvar_head`…
- 梯度覆盖：`dbg/<tag>_has_grad`、`dbg/<tag>_trainable`（其比值≈**累积步数 GA**）
- 参数更新幅度：`dbg/update_ratio/<tag>`

**强制打印技巧**

- 对任一监控键 `k` 附加 **影子项**：`loss_<k_sanitized> = 0 * <k>`，不影响优化，但能被 logger 打印。

**MMEngine 建议**

```python
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)
# LoggerHook.interval 设为你的打印节奏（如每 50/100 step）
```

---

## 1) 关键指标怎么看

### 1.1 总览（建议每 50–200 step 打印）

| 指标名 | 代表含义 | 正常趋势 | 红旗信号 | 处理动作 |
|---|---|---|---|---|
| `loss` | 总损失 | 中/长窗缓慢下降 | 长期平台期 | 调整 LR/裁剪/权重平衡；下钻子项 |
| `loss_flow` | FM 主目标 | 持续下降 | 长期不降 | 提 LR/放宽裁剪/减正则；查梯度覆盖 |
| `loss_dino` | 语义监督项 | warmup 期占比小，逐步起量 | 早期占比过高或抖动大 | 降 `dino_alpha`、拉长 warmup |
| `loss_kl` / `kl_raw` | 先验约束 | `kl_raw` 缓慢收敛 | 持续增大 | 增 KL 权重、约束 `logvar` |
| `dino_global` | DINO 对齐度 | **下降**（越小越好） | 不降反升 | 降 `alpha`/延长 warmup/关局部 |
| `z_init/mean` | 采样均值 | 近 0 小幅波动 | \|mean\| > 0.1 | mu/logvar 后加 LN/居中化 |
| `z_init/var` | 采样方差 | 收敛到 ≈0.8–1.0 | <0.3 或 >1.5 | 调 KL 权重、clamp `logvar` |
| `dbg/grad_flow` | Flow 梯度范数 | 随 `loss_flow` 下降而变小 | 长期≈0 或爆高 | 调 LR/裁剪/数值稳定性 |
| `dbg/flow_has_grad` / `dbg/flow_trainable` | 梯度覆盖 | 比值≈**GA** | 偏离 >±10% 或 =0 | 校准 GA/no_sync/flush 时机 |
| `dbg/grad_dino_t_head` | DINO 梯度 | 随 warmup 逐步变大 | 早期过大 | 降 `alpha`、拉长 warmup |
| `dbg/update_ratio/<tag>` | 参数步长占比 | 稳定非零 | 长期≈0 | 提 LR/减正则/查冻结 |

> **推进性的判定**：当 `loss_flow ↓`、`dbg/grad_flow > 0`、`has_grad/trainable ≈ GA` 同时成立，说明 Flow 分支确实在学习。

---

## 2) 三类 Loss 的“有效性验证”

### 2.1 Flow Matching
- **主观察**：`loss_flow` 持续下降；`dbg/grad_flow` 非零且平滑收敛；`has_grad/trainable ≈ GA`。
- **辅助**：对比 `‖z_text‖` 与目标 `‖x1‖` 的尺度，避免失配导致梯度极小/极大。
- **异常修复**  
  - 不降且梯度很小：提 LR、放宽裁剪、减弱其它正则。  
  - 不降且梯度很大：降 LR、加裁剪、检查归一化/数值稳定。

### 2.2 DINO 语义监督
- **主观察**：warmup 期间 `loss_dino` 占比小；`dbg/grad_dino_t_head` 随步增大；`dino_global` 缓慢下降。
- **对照**：临时关闭 DINO 或只保留 global 分支，确认不会干扰 FM。
- **异常修复**：降低 `dino_alpha`、延长 warmup、关闭局部 patch 对齐。

### 2.3 KL 正则
- **主观察**：`kl_raw` 缓慢收敛；`z_init/mean ≈ 0`，`z_init/var → 0.8–1.0`。
- **异常修复**  
  - `kl_raw ↑` 且 `z_init/var ↑`：增 KL 权重、clamp `logvar`、mu/logvar 头后加 LN。  
  - 方差坍缩：降低 KL 权重或做 β-anneal。

---

## 3) 梯度监控（DDP/ZeRO 友好，无额外 backward）

**做法**：给关键模块参数注册 `p.register_hook`，在**真实 backward**期间累加 `g.pow(2).sum()` 与计数；**下一步**把快照写入日志（一步延迟，零冲突）。

- 记录：`dbg/grad_<tag>`、`dbg/<tag>_has_grad`、`dbg/<tag>_trainable`  
- 保障打印：为每个 `dbg/*` 加 `loss_dbg_* = 0 * dbg/*`

**为何不用 `autograd.grad`**：与 reentrant checkpoint 不兼容；ZeRO/DDP 不支持同参多次规约，易触发 *already been reduced*。

---

## 4) 参数“更新幅度”（Step Ratio）

- 定义：`update_ratio = ||Δθ|| / (||θ|| + ε)`（模块内 L2 汇总）
- 作用：验证“梯度×LR”是否让参数移动；长期≈0 代表学不动（LR 太小/正则过大/被裁剪/被冻结）
- 记录：`dbg/update_ratio/<tag>`（每步对比上一快照）

---

## 5) 与分布式 & Checkpoint 的注意

- 尽量将所有 `checkpoint(...)` 设为 `use_reentrant=False`。  
- 如果做不到，**只用 hook 监控**，不要做额外 backward。  
- 出现 *already been reduced*：说明同一迭代做了多次 backward → 去掉监控用 backward。

---

## 6) 常见问题速查

| 症状 | 可能原因 | 快速修复 |
|---|---|---|
| `loss_flow` 平台且 `grad_flow` 很小 | LR 小 / 裁剪或正则过强 | 提 LR / 放宽裁剪 / 减正则 |
| `loss_flow` 抖动大且 `grad_flow` 爆高 | LR 高 / 数值不稳 | 降 LR / 加裁剪 / 加 LN |
| `dino_global` 不降 | 语义分支过强或 warmup 失效 | 降 `alpha` / 延长 warmup / 关局部 |
| `z_init/var ↑` + `kl_raw ↑` | KL 太弱 | 增 β、clamp `logvar`、加 LN |
| `has_grad/trainable` ≠ GA | 累积/flush/no_sync 不一致 | 校准 GA 与 flush 时机 |
| 日志缺少 `dbg/*` | 无影子 `loss_*` | 为每个监控键加 0 权重 `loss_*` |

---

## 7) 建议阈值（可做告警）

- `has_grad/trainable` 偏离 GA **> ±10%** 连续 100 step  
- `dbg/grad_flow < 1e-2` 且 `loss_flow` 连续 100 step 不降  
- `dino_global` 在 warmup 中段以后 **200 step 不降**  
- `|z_init/mean| > 0.1` 或 `z_init/var < 0.3 / > 1.5` 连续 200 step  
- `kl_raw` 连续 200 step 上升

---

## 8) 最小监控清单（Quick Start）

1. 损失：`loss_flow`、`loss_dino`、`loss_kl`、`kl_raw`、`dino_global`  
2. z 分布：`z_init/mean`、`z_init/var`  
3. 梯度（hook）：`dbg/grad_flow`、`dbg/flow_has_grad`、`dbg/flow_trainable`、`dbg/grad_dino_t_head`  
4. 更新幅度：`dbg/update_ratio/flow`（及其它关键模块）  
5. 打印保障：为每个键加 0 权重 `loss_*` 影子项

---

## 9) 参考代码骨架（监控思路）

> 仅示意：**hook 累积 → 下一步 flush**；每个写入附带 0 权重 `loss_*` 影子项。

```python
# 1) 安装参数 hook（在 __init__）
def _install_grad_taps(self):
    def add_hooks(mod, tag):
        if mod is None: return
        self._trainable_counts[tag] = 0
        for p in mod.parameters():
            if not p.requires_grad: continue
            self._trainable_counts[tag] += 1
            def _hook(g, _tag=tag):
                st = self._grad_acc.setdefault(_tag, {'sq': 0.0, 'cnt': 0})
                st['sq'] += float(g.detach().float().pow(2).sum().item())
                st['cnt'] += 1
            p.register_hook(_hook)
    add_hooks(self.fm_transformers, 'flow')
    add_hooks(self.dino_t_head, 'dino_t_head')
    add_hooks(self.mu_head, 'mu_head')
    add_hooks(self.logvar_head, 'logvar_head')

# 2) 训练步尾：把“上一轮”统计写入 ret；本轮统计滚动成“上一轮”
def _flush_grad_stats(self, ret: dict):
    import math, torch
    for tag, st in self._last_grad_stats.items():
        gn = math.sqrt(st.get('sq', 0.0))
        got = st.get('cnt', 0)
        tot = self._trainable_counts.get(tag, 0)
        ret[f'dbg/grad_{tag}'] = torch.tensor(gn, device=self.device)
        ret[f'dbg/{tag}_has_grad'] = torch.tensor(got, device=self.device)
        ret[f'dbg/{tag}_trainable'] = torch.tensor(tot, device=self.device)
        ret[f'loss_dbg_grad_{tag}'] = ret[f'dbg/grad_{tag}'] * 0.0  # 影子项
    self._last_grad_stats = self._grad_acc
    self._grad_acc = {}
```

---

## 10) 小结

- **趋势优先**：`loss_flow↓`、`dino_global↓`、`kl_raw` 稳定；`z_init` 均值≈0、方差≈1。  
- **梯度证据**：强度非零、覆盖≈GA、`update_ratio` 非零。  
- **定位方法**：对照实验 + 阈值告警 + hook 梯度，快速判断“哪个分支不工作 / 过强 / 过弱”，精准调参。
