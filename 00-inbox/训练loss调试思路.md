
# 训练 Loss 调试思路（DINO 语义监督 / KL / Flow Matching）

> 目标：给出**可复制**的诊断套路与**最小化代码插桩**，确保三类损失都在“真起作用”：能产生有效梯度、得到正确统计趋势、与推理公式一致。

---

## 目录
- [总览](#总览)
- [统一日志规范（ret 字典）](#统一日志规范ret-字典)
- [A. DINO 语义监督](#a-dino-语义监督)
- [B. KL Loss（VAE 后验约束）](#b-kl-lossvae-后验约束)
- [C. Flow Matching Loss](#c-flow-matching-loss)
- [多任务权衡与常见修复](#多任务权衡与常见修复)
- [训练内/外快速消融](#训练内外快速消融)
- [最小插桩代码模板](#最小插桩代码模板)
- [常见陷阱清单](#常见陷阱清单)

---

## 总览

**核心判断准则：**
1) **看梯度，不看标量**：分别对每个 loss 单独反传一次，记录梯度范数 → 判断监督强弱；  
2) **看统计趋势**：DINO 的输出熵、KL 的 μ/σ、Flow 的 cos/NMSE/端点误差；  
3) **看推理一致性**：训练中的目标向量场与推理时积分公式、尺度是否一致；  
4) **看条件利用度**：shuffle 条件后性能下降幅度应显著。

---

## 统一日志规范（ret 字典）

所有指标写入 `ret[...]`，便于 TensorBoard / JSON 可视化：
- **梯度**：`grad/dino, grad/kl, grad/flow, grad/total`
- **参数步长**：`step/z_head_update_ratio, step/flow_head_update_ratio`
- **DINO 熵**：`ent/teacher, ent/student`
- **KL 统计**：`mu/abs_mean, mu/rms, logvar/mean, std/mean|min|max, z_init/mean, z_init/var, z_init/std_per_dim_mean`
- **Flow 对齐**：`flow/cos, flow/nmse, flow/speed_ratio, flow/endpt_rmse@K, flow/cond_gain_cos, flow/nmse_t{0..9}, flow/cos_t{0..9}`

---

## A. DINO 语义监督

**目的**：student 拟合 teacher 的软目标分布；确保不会“过尖/过钝”导致梯度饱和或无效。

### 诊断步骤
1. **梯度强度**  
   - 单独对 `L_dino` backward：`ret["grad/dino"]`  
   - 经验阈：`grad/dino` < `grad/total` 的 **5%~10%** → 监督弱或温度不当。
2. **输出熵（温度/中心化）**  
   - 记录 `ent/teacher, ent/student`  
   - 教师熵过低（≈0）→ 目标太尖，学生早饱和，梯度趋零。  
   - 学生熵极低+梯度小 → 增大 `T_s` 或 `w_dino`。
3. **参数步长（是否动到关键头）**  
   - 监控 `step/z_head_update_ratio`（q→z 对齐头）。长期 < 1e-5 说明几乎没更新。

### 调参建议
- 调权重：增大 `w_dino`（×2/×4）；  
- 调温度：**提高 `T_t`（教师）或 `T_s`（学生）**，避免目标/输出过尖；  
- **分组学习率**：对投影/对齐头设更高 LR（×2~×5）；  
- 必要时用 **GradNorm/PCGrad** 平衡多目标。

---

## B. KL Loss（VAE 后验约束）

**目的**：使 `q(z|x)=N(μ, diag(σ²))` 接近先验 `N(0, I)`，避免潜变量漂移/崩坏。

### 诊断步骤
1. **梯度强度**  
   - 单独对 `L_kl` backward：`ret["grad/kl"]`；过小 → 增大 `w_kl` 或检查实现。
2. **统计趋势（最关键）**  
   - `mu/abs_mean` **下降**、`std/mean` **→ 1**、`logvar/mean` **→ 0**；  
   - 采样后 `z_init/mean ~ 0, z_init/var ~ 1`；  
   - 对比消融：**`w_kl=0`** 时上述趋势应消失/反向；有差异说明 KL **在起作用**。
3. **尺度一致性**  
   - 若图像 VAE 使用缩放常数（如 Stable Diffusion 的 **0.18215**），**确保 KL 在未缩放空间上计算**，或相应调整先验方差（否则目标不是 `I`）。

### 常见问题与修复
- KL 用错输入：**必须基于 μ、logσ²** 而非采样后的 `z`；  
- KL 的权重过小、被其它分量淹没 → 调大 `w_kl`；  
- `std` 崩到 0（posterior collapse）→ 适当**降低 `w_kl`** 或用 KL warmup/β-VAE 日程。

---

## C. Flow Matching Loss

**目的**：学习从 `z0` 到 `z1` 的目标向量场，使得**自由积分**能到达终点（或近似）。

### 诊断步骤
1. **梯度强度**  
   - 单独对 `L_flow` backward：`ret["grad/flow"]`；长期低于总梯度 10% → 调 `w_flow` 或检查目标公式。
2. **向量场对齐（训练内指标）**  
   - **NMSE**：`flow/nmse = ||v_pred - v_tgt||² / ||v_tgt||²`（越低越好）  
   - **Cos 相似度**：`flow/cos`（→1 越好）  
   - **速度比**：`flow/speed_ratio = ||v_pred|| / ||v_tgt||`（≈1 最好，偏离大说明尺度不匹配）
3. **自由积分端点误差（训练外硬指标）**  
   - Euler/Heun 积分 K 步得到 `z1_hat`，记录 `flow/endpt_rmse@K = ||z1_hat - z1||`；  
   - 曲线随 K 增大应**单调下降**或趋稳；若不降，多半是**训练目标与推理积分公式不一致**。
4. **t-切片诊断**  
   - 记录 `flow/nmse_t{0..9}, flow/cos_t{0..9}`；边界段（t≈0/1）常见劣化 → **重采样/重权**或修正 α′(t)/β′(t)。
5. **条件利用度**  
   - 条件乱序对比：`flow/cond_gain_cos = cos(cond) - cos(shuffle)`；>0 且越大越好，表示确实在用条件信息而非无条件平均场。

### 常见问题与修复
- **训练/推理不一致**：训练目标 `v*` 带 α′(t)、β′(t)，推理却直接 `z += v* dt`（漏乘/错乘）→ **统一公式**；  
- **尺度不一致**：`z`/`v` 在 SD-VAE 0.18215 缩放问题 → 训练与推理**同一尺度**；  
- **边界学不好**：对 t 边缘重采样/加权；  
- **被其它损失淹没**：增大 `w_flow` 或用 GradNorm/PCGrad。

---

## 多任务权衡与常见修复

- **按梯度而非按损失调权**：让 `grad/dino : grad/flow : grad/kl` 维持在 **可控比值**（如 0.2~0.5 : 1 : 0.1~0.3）。  
- **参数分组 LR**：给对齐头（z_head）与速度头（flow_head）更高 LR。  
- **优化器技巧**：对“头部”关闭/降低权重衰减；对 backbone 保守。  
- **自动方法**：GradNorm、PCGrad、/或简单余弦重加权日程。

---

## 训练内/外快速消融

- **仅开某一分量**：如只开 Flow（`w_dino=w_kl=0`）跑 200 步，`cos↑ nmse↓ endpt_rmse↓` 应显著；  
- **关闭某一分量**：如关 KL（`w_kl=0`），μ 不再向 0 收缩、σ 不再向 1 收敛；  
- **条件乱序**：Flow/DINO 的 cond shuffle 指标应显著下降。

---

## 最小插桩代码模板

> 将下述片段放入 `train_step`（注意：示例变量名需替换为你的实际张量）

```python
# ===== 通用：梯度范数 =====
def total_grad_norm(params):
    sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            sq += (g * g).sum().item()
    return sq ** 0.5

# ======= 假设你已得到这些变量 =======
# L_flow, L_dino, L_kl : 标量 loss
# mu, logvar, z0, z1, t : KL/Flow 所需张量
# teacher_logits, student_logits, T_t, T_s
# model.velocity(x_t, t, cond) : 速度头
# model.z_head / model.flow_head : 关键头模块
# optimizer, ret = {}, {}
# ====================================

# --- A. 单分量梯度强度 ---
for name, L in {"flow": L_flow, "dino": L_dino, "kl": L_kl}.items():
    optimizer.zero_grad(set_to_none=True)
    L.backward(retain_graph=True)
    ret[f"grad/{name}"] = total_grad_norm(model.parameters())

# --- 总梯度 ---
optimizer.zero_grad(set_to_none=True)
(L_flow*w_flow + L_dino*w_dino + L_kl*w_kl).backward()
ret["grad/total"] = total_grad_norm(model.parameters())

# --- 关键头参数步长（示例：z_head/flow_head）---
with torch.no_grad():
    if hasattr(model, "z_head"):
        snap = {p: p.detach().clone() for p in model.z_head.parameters()}
    if hasattr(model, "flow_head"):
        snap_flow = {p: p.detach().clone() for p in model.flow_head.parameters()}

optimizer.step()

with torch.no_grad():
    if hasattr(model, "z_head"):
        ratios = []
        for p in model.z_head.parameters():
            base = snap[p]
            ratios.append(((p - base).norm() / (base.norm() + 1e-12)).detach())
        ret["step/z_head_update_ratio"] = torch.stack(ratios).mean().item()
    if hasattr(model, "flow_head"):
        ratios = []
        for p in model.flow_head.parameters():
            base = snap_flow[p]
            ratios.append(((p - base).norm() / (base.norm() + 1e-12)).detach())
        ret["step/flow_head_update_ratio"] = torch.stack(ratios).mean().item()

# --- DINO：输出熵 ---
def entropy_from_logits(logits, temp=1.0):
    p = torch.softmax(logits / temp, dim=-1)
    return (-p * (p.clamp_min(1e-12).log())).sum(dim=-1).mean()

ret["ent/teacher"] = entropy_from_logits(teacher_logits, T_t).item()
ret["ent/student"] = entropy_from_logits(student_logits, T_s).item()

# --- KL：统计趋势 ---
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z_init = mu + eps * std

with torch.no_grad():
    ret["z_init/mean"] = z_init.mean().item()
    ret["z_init/var"]  = z_init.var(unbiased=False).item()
    ret["mu/abs_mean"] = mu.abs().mean().item()
    ret["mu/rms"]      = (mu.pow(2).mean().sqrt().item())
    ret["logvar/mean"] = logvar.mean().item()
    ret["std/mean"]    = std.mean().item()
    ret["std/min"]     = std.min().item()
    ret["std/max"]     = std.max().item()

    z_flat = z_init.view(z_init.size(0), -1)
    ret["z_init/std_per_dim_mean"] = z_flat.std(dim=0, unbiased=False).mean().item()

# --- Flow：目标、对齐、切片 ---
import torch.nn.functional as F

def fm_targets_linear(z0, z1, t):
    x_t = z0 + t * (z1 - z0)          # 若用 α(t),β(t)，替换此式与 v_tgt 公式
    v_tgt = (z1 - z0).detach()
    return x_t, v_tgt

with torch.no_grad():
    x_t, v_tgt = fm_targets_linear(z0, z1, t)

v_pred = model.velocity(x_t, t, cond)    # 你的速度预测
res = v_pred - v_tgt

nmse = (res.pow(2).mean()) / (v_tgt.pow(2).mean().clamp_min(1e-12))
cos  = F.cosine_similarity(v_pred.flatten(1), v_tgt.flatten(1), dim=1).mean()
spd_ratio = (v_pred.flatten(1).norm(dim=1) / (v_tgt.flatten(1).norm(dim=1) + 1e-12)).mean()

ret["flow/nmse"] = nmse.item()
ret["flow/cos"]  = cos.item()
ret["flow/speed_ratio"] = spd_ratio.item()

with torch.no_grad():
    bins = torch.tensor([0., .1,.2,.3,.4,.5,.6,.7,.8,.9,1.], device=t.device)
    idx  = torch.bucketize(t.flatten(), bins) - 1
    for b in range(10):
        m = (idx == b)
        if m.any():
            nmse_b = ((res.view(res.size(0), -1)[m]**2).mean() /
                      (v_tgt.view(v_tgt.size(0), -1)[m]**2).mean().clamp_min(1e-12))
            cos_b  = F.cosine_similarity(
                        v_pred.view(v_pred.size(0), -1)[m],
                        v_tgt.view(v_tgt.size(0), -1)[m],
                        dim=1).mean()
            ret[f"flow/nmse_t{b}"] = nmse_b.item()
            ret[f"flow/cos_t{b}"]  = cos_b.item()

# --- Flow：条件利用度（shuffle）---
with torch.no_grad():
    order = torch.randperm(cond.size(0), device=cond.device)
    v_cond = v_pred
    v_shuf = model.velocity(x_t, t, cond[order])
    cos_cond = F.cosine_similarity(v_cond.flatten(1), v_tgt.flatten(1), dim=1).mean()
    cos_shuf = F.cosine_similarity(v_shuf.flatten(1), v_tgt.flatten(1), dim=1).mean()
    ret["flow/cond_gain_cos"] = (cos_cond - cos_shuf).item()

# --- 训练外（或 eval 钩子）自由积分端点误差 ---
@torch.no_grad()
def rollout_euler(model, z0, cond, steps=32):
    z = z0.clone().detach()
    dt = 1.0 / steps
    for s in range(steps):
        t_mid = (s + 0.5) / steps
        t_in  = torch.full_like(z[..., :1], t_mid)
        v = model.velocity(z, t_in, cond)
        z = z + v * dt
    return z

@torch.no_grad()
def record_endpt_rmse(ret, model, z0, z1, cond):
    for K in (8, 16, 32, 64):
        z1_hat = rollout_euler(model, z0, cond, steps=K)
        ret[f"flow/endpt_rmse@{K}"] = (z1_hat - z1).pow(2).mean().sqrt().item()

# 在 eval 或定期 step 调用：
# record_endpt_rmse(ret, model, z0_eval, z1_eval, cond_eval)
