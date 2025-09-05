太好了，我直接把“CLIP → VICReg”的落地改造写成**最小打补丁方案**。你把下面几段代码按模块补进去就能跑（保持你现有 FM 与 KL 流程不变）。

---

# 1) 新增超参 & 投影头（projectors）

在 `__init__` 里增加 VICReg 的开关、权重和头（分支法，不影响 FM 使用的 `z_text`）：

```python
# ---- VICReg config (add in __init__) ----
self.use_vicreg = True
self.vic_alpha = 0.3          # 主权重，建议训练初期warmup
self.vic_lambda_self = 0.1    # 单分支稳定项
self.vic_inv_w = 25.0
self.vic_var_w = 25.0
self.vic_cov_w = 1.0
self.vic_gamma = 1.0
self.vic_stopgrad_image = False  # 若FM被拖慢可置True
self.vic_warmup_steps = 5000     # 线性warmup到 vic_alpha/vic_lambda_self

d = self.d_align if hasattr(self, 'd_align') else 1024  # 你已有 d_align=1024

# 文本侧 VICReg 头：z_text (B, 8192) → (B, d)
self.vic_t_head = nn.Sequential(
    nn.Linear(8192, 2048), nn.GELU(),
    nn.Linear(2048, d)
)

# 图像侧 VICReg 头：VAE latent (B,4,H,W) → (B, d)
# 轻量：1x1 conv提升特征维度→GAP→Linear
self.vic_i_head = nn.Sequential(
    nn.Conv2d(4, 256, kernel_size=1),
    nn.GELU(),
    nn.AdaptiveAvgPool2d(1),  # (B,256,1,1)
    nn.Flatten(),             # (B,256)
    nn.Linear(256, d)
)
```

> 说明
>
> * VICReg 只在 `vic_t_head(z_text)` 与 `vic_i_head(vae_latent)` 上做；FM 仍吃 `z_text`，二者解耦。
> * 如果后续发现 FM 收敛变慢，把 `self.vic_stopgrad_image=True` 或把 `self.vic_alpha` 降到 0.1\~0.2。

---

# 2) 全局 gather（带梯度）与 VICReg 损失

把这些方法加到类里（与 `compute_clip_loss` 平级）：

```python
def _gather_with_grad(self, t: torch.Tensor) -> torch.Tensor:
    """DDP下可回传的all_gather；单卡时原样返回。"""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.nn.functional import all_gather as ag
        parts = list(ag(t))
        return torch.cat(parts, dim=0)
    return t

def _vicreg_terms(self, y: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4):
    """
    返回 (variance_term, covariance_term)。内部用fp32做统计，数值更稳。
    y: (N, d)
    """
    y32 = y.float()
    std = y32.var(dim=0, unbiased=False).add(eps).sqrt()      # (d,)
    var_term = torch.relu(gamma - std).mean()

    yc = y32 - y32.mean(dim=0, keepdim=True)
    N, d = yc.shape
    c = (yc.t() @ yc) / max(N - 1, 1)                         # (d, d)
    off = c - torch.diag(torch.diag(c))
    cov_term = (off.pow(2).sum()) / d
    return var_term, cov_term

def compute_vicreg_loss(self, y_t: torch.Tensor, y_i: torch.Tensor):
    """
    y_t, y_i: (B, d) —— projector之后的向量。
    做全局all_gather后计算三项：invariance + variance + covariance。
    """
    # gather成全局batch
    yt = self._gather_with_grad(y_t)
    yi = self._gather_with_grad(y_i)

    # 1) invariance（在fp32上计）
    inv = (yt.float() - yi.float()).pow(2).mean()

    # 2) variance / covariance（分别对两分支）
    v_t, c_t = self._vicreg_terms(yt, gamma=self.vic_gamma)
    v_i, c_i = self._vicreg_terms(yi, gamma=self.vic_gamma)

    loss = (self.vic_inv_w * inv
            + self.vic_var_w * (v_t + v_i)
            + self.vic_cov_w * (c_t + c_i))
    terms = {
        'vic_inv': inv.detach(),
        'vic_var_t': v_t.detach(), 'vic_var_i': v_i.detach(),
        'vic_cov_t': c_t.detach(), 'vic_cov_i': c_i.detach(),
    }
    return loss, terms

def compute_vicreg_self(self, y: torch.Tensor):
    """
    单分支VICReg（无对齐项），增强稳定性，防塌缩：var+cov。
    """
    v, c = self._vicreg_terms(self._gather_with_grad(y), gamma=self.vic_gamma)
    return (self.vic_var_w * v + self.vic_cov_w * c), {'vic_self_var': v.detach(), 'vic_self_cov': c.detach()}
```

---

# 3) 拿到图像侧向量（来自 VAE latent）

新增一个小工具函数（类内）：

```python
@torch.no_grad()
def _encode_image_latent(self, x_img: torch.Tensor):
    """
    与 flow_matching_loss 里的编码一致：x_img -> 256 -> moments -> sample -> z_img (B,4,H,W)
    """
    x_img_256 = F.interpolate(x_img, size=(256, 256), mode='bilinear', align_corners=False)
    moments = self.autoencoder(x_img_256.to(dtype=self._dtype, device=self.device), fn='encode_moments').squeeze(0)
    z_img = self.autoencoder.sample(moments).to(dtype=self._dtype, device=self.device)
    return z_img  # (B,4,H,W)
```

> 你也可以做一点“视角扰动”提升鲁棒性：例如对 `x_img` 做随机resize/crop后再encode；或者对 `moments` 走一次 reparameterize（上面 sample 已经做了）。

---

# 4) 在 `text2image_loss` 中接入 VICReg（替换 CLIP）

把 `text2image_loss` 的 C 段改成如下（保留你已有的 FM & KL）：

```python
# C. 损失计算
loss_flow = self.flow_matching_loss(z_text, x_img)
loss_kl = self.compute_kl_loss(mu, log_var).mean()

# ---- VICReg: 替代 CLIP ----
vic_logs = {}
if self.use_vicreg:
    with torch.set_grad_enabled(True):
        # 文本侧向量
        y_t = self.vic_t_head(z_text)  # (B,d)

        # 图像侧向量（来自VAE latent）
        z_img_latent = self._encode_image_latent(x_img)            # (B,4,H,W)
        y_i = self.vic_i_head(z_img_latent)                        # (B,d)
        if self.vic_stopgrad_image:
            y_i = y_i.detach()

        # warmup（用内部步数计数）
        self._train_steps = getattr(self, '_train_steps', 0) + 1
        alpha_scale = min(1.0, self._train_steps / float(self.vic_warmup_steps))
        alpha_eff = self.vic_alpha * alpha_scale
        lambda_eff = self.vic_lambda_self * alpha_scale

        loss_vic, vic_terms = self.compute_vicreg_loss(y_t, y_i)

        # 可选：单分支稳定项（对text和image各算一次）
        loss_self_t, self_terms_t = self.compute_vicreg_self(y_t)
        loss_self_i, self_terms_i = self.compute_vicreg_self(y_i)
        loss_vic_total = alpha_eff * loss_vic + lambda_eff * (loss_self_t + loss_self_i)

        vic_logs.update(vic_terms)
        vic_logs.update({f'{k}_t': v for k, v in self_terms_t.items()})
        vic_logs.update({f'{k}_i': v for k, v in self_terms_i.items()})
else:
    loss_vic_total = torch.tensor(0.0, device=self.device, dtype=self._dtype)

# ---- 权重合成 ----
kl_weight = 1e-3
loss_total = loss_flow + loss_vic_total + kl_weight * loss_kl

# 调试/日志
if not hasattr(self, '_debug_step'):
    self._debug_step = 0
self._debug_step += 1
if self._debug_step % 100 == 0:
    msg = f"[DEBUG] Step {self._debug_step}: FM={loss_flow.item():.4f}, VIC={loss_vic_total.item():.4f}, KL={loss_kl.item():.4f}"
    if self.use_vicreg:
        msg += f", alpha_eff={alpha_eff:.3f}, lambda_eff={lambda_eff:.3f}"
    print(msg)

ret = {
    'loss': loss_total,          # mmengine 习惯：提供总loss
    'loss_flow': loss_flow,
    'loss_vic': loss_vic_total.detach(),
    'loss_kl': kl_weight * loss_kl,
    'kl_raw': loss_kl.detach(),
}
# 额外记录VIC内部项
ret.update(vic_logs)
return ret
```

> 关键点
>
> * **完全不再使用 CLIP**，你已有 `compute_clip_loss` 可以留作历史。
> * `self.vic_alpha` / `self.vic_lambda_self` 做**线性warmup**，更稳。
> * `self.vic_stopgrad_image=True` 可只拉文本侧到图像分布，避免图像侧被扰动。
> * `ret['loss']` 给个总和，便于 mmengine 直接 backward；你也可以只返回分项，由上层加和。

---

# 5) 修一个你代码里的小坑（`null_indicator` 未定义）

`flow_matching_loss` 里使用了 `null_indicator`，但函数签名没有这个参数。简单修法：

```python
def flow_matching_loss(self, z_text: torch.Tensor, x_img: torch.Tensor, null_indicator: torch.Tensor = None) -> torch.Tensor:
    ...
    if null_indicator is None:
        null_indicator = (torch.rand(batch_size, device=x1.device) < 0.1)
    ...
    preds = self.fm_transformers(x_t, log_snr=log_snr, null_indicator=null_indicator)
    ...
```

并在 `text2image_loss` 调用时保持不传即可（或传入你自己的 CFG 掩码）。

---

# 6) 训练建议（无 CLIP）

* **先跑默认**：`vic_alpha=0.3`、`vic_lambda_self=0.1`、`d_align=1024`、`warmup=5k steps`。
* 如果 **FM 收敛变慢**：把 `vic_alpha` 降到 0.1\~0.2，或 `vic_stopgrad_image=True`。
* 如果出现 **塌缩**（std<1、特征趋同）：增大全局 batch（或梯度累计）、确认 `_gather_with_grad` 生效、适当提高 `vic_var_w`，并确保 projector 里不要加最后激活。
* **监控**：记录 `vic_inv/var/cov`、以及 `y_t/y_i` 的 per-dim std 均值（>1 为佳）和 off-diagonal 能量（越低越好）。

---

# 7) 可选的小优化

* projector 里把 `nn.Linear → nn.SyncBatchNorm`（需要把线性改成 `Linear → LayerNorm/BN → GELU` 的结构；多卡建议 `SyncBatchNorm`）。
* 图像侧 `vic_i_head` 也可以换成：`Conv1x1(4→d)` + `AdaptiveAvgPool2d(1)` 直接得到 `(B,d)`，更简洁，但初期 256 通道更稳。
* 若你想保持一点“方向一致性”但仍不引入负样本，可额外加一个**极轻量余弦回归项**：`MSE( y_t/||·||, y_i/||·|| )`，权重 0.02 左右。

---

把上面 1\~5 的代码块贴进去，就完成了从 CLIP 到 VICReg 的替换与接入。需要我把这些改动整理成一个独立的 `VICRegHead` / `loss hook` 文件再对接到 `mmengine` 配置里吗？我可以顺手给一个最小 YAML 配置片段。
