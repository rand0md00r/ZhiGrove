下面给你一套**可直接粘贴**的 DINOv2 语义监督方案，包括：

1. 官方优先的 DINOv2 加载与封装（支持 `torch.hub`，并内置 `timm` 兜底）；
2. 在你的主类里接入 DINOv2 的 projector 与损失；
3. 三种损失任选：**Cosine**（默认）、**MSE(带归一化)**、**VICReg（学生侧 var/cov）**。

> 设计要点：**分支法**。只在 `z_text` 的**投影**上做 DINO 语义监督；FM 主干仍吃 `z_text` 本体，互不打架。图像侧（DINO）**冻结**、**stop-grad**。

---

# A) 新增文件：DINOv2 封装（官方优先）

保存为：`src/models/dino/dinov2_backbone.py`

```python
# src/models/dino/dinov2_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2Backbone(nn.Module):
    """
    官方优先的 DINOv2 加载封装：
      1) 先尝试 torch.hub: facebookresearch/dinov2
      2) 失败则兜底 timm: 'vit_large_patch14_dinov2.lvd142m' 等
    forward(x) 返回:
      {'global': (B,D), 'patch': (B,N,D)} —— 若无法拿到 patch，就返回 None
    """
    def __init__(self, variant: str = "vitl14", img_size: int = 224, use_timm_fallback: bool = True):
        super().__init__()
        self.variant = variant.lower()
        self.img_size = img_size
        self.model = None
        self.embed_dim = None
        self._load_official_or_timm(use_timm_fallback)

        # 推断输出维度
        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            # timm 的 ViT 一般用 embed_dim / num_features
            self.embed_dim = getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("Cannot infer DINOv2 embed dim; please set manually.")

        # 统一 eval + 冻结
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def _load_official_or_timm(self, use_timm_fallback: bool):
        # 1) 官方 torch.hub
        hub_ok = False
        try:
            import torch.hub
            name_map = {
                "vitl14": "dinov2_vitl14",
                "vitb14": "dinov2_vitb14",
                "vitg14": "dinov2_vitg14",
                "vits14": "dinov2_vits14",
            }
            hub_name = name_map.get(self.variant, "dinov2_vitl14")
            self.model = torch.hub.load("facebookresearch/dinov2", hub_name)
            hub_ok = True
        except Exception:
            hub_ok = False

        # 2) 兜底 timm
        if (not hub_ok) and use_timm_fallback:
            try:
                import timm
                timm_name_map = {
                    "vitl14": "vit_large_patch14_dinov2.lvd142m",
                    "vitb14": "vit_base_patch14_dinov2.lvd142m",
                    "vitg14": "vit_gigantic_patch14_224",  # 可能无权重，仅示例
                    "vits14": "vit_small_patch14_dinov2.lvd142m",
                }
                tm = timm_name_map.get(self.variant, "vit_large_patch14_dinov2.lvd142m")
                self.model = timm.create_model(tm, pretrained=True)
            except Exception as e:
                raise RuntimeError(f"Load DINOv2 failed (hub & timm). Error: {e}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: (B,3,H,W) 已标准化到 ImageNet 分布并 resize 到 self.img_size
        返回:
          {'global': (B,D), 'patch': (B,N,D or None)}
        """
        m = self.model
        out = {"global": None, "patch": None}

        # 官方 DINOv2: 大多有 forward_features，返回 dict，含 x_norm_clstoken / x_norm_patchtokens
        if hasattr(m, "forward_features"):
            feats = m.forward_features(x)
            # 官方 repo 常见键
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    out["global"] = feats["x_norm_clstoken"]  # (B,D)
                elif "cls_token" in feats:
                    out["global"] = feats["cls_token"]
                # 尝试拿 patch
                if "x_norm_patchtokens" in feats:
                    out["patch"] = feats["x_norm_patchtokens"]  # (B,N,D)
                elif "patch_tokens" in feats:
                    out["patch"] = feats["patch_tokens"]
            else:
                # timm 大多返回序列 (B,N+1,D)
                if feats.ndim == 3:
                    out["global"] = feats[:, 0]        # cls
                    out["patch"] = feats[:, 1:]        # patches
                else:
                    # 某些 timm 模型 forward_features 返回的已是 pooled
                    out["global"] = feats
        else:
            # 兜底：直接前向
            feats = m(x)
            if feats.ndim == 3:
                out["global"] = feats[:, 0]
                out["patch"] = feats[:, 1:]
            else:
                out["global"] = feats

        return out
```

---

# B) 在你的主类里接入 DINOv2

**1. 在 imports 顶部**加一行：

```python
from src.models.dino.dinov2_backbone import DINOv2Backbone
```

**2. 在 `__init__` 末尾附近**加入 DINO 配置、加载与投影头（紧跟 dtype/to 之后即可）：

```python
# ---- DINOv2 semantic supervision ----
self.use_dino = True                 # 语义监督开关
self.dino_variant = 'vitl14'         # 'vitl14' / 'vitb14' / ...
self.dino_img_size = 224             # 224起步更稳，518可选
self.dino_loss_type = 'cosine'       # 'cosine' | 'mse' | 'vicreg'
self.dino_alpha = 0.2                # 总权重（建议 warmup）
self.dino_use_patch = False          # 先全局，稳了再开 patch
self.dino_patch_grid = (2, 2)        # 开patch时用 g×g 网格
self.dino_w_global = 1.0
self.dino_w_patch = 0.2
self.dino_stopgrad_image = True      # 冻结教师，图像侧不回传

# 加载 DINOv2（官方优先）
self.dino_model = DINOv2Backbone(variant=self.dino_variant, img_size=self.dino_img_size)
self.dino_dim = int(self.dino_model.embed_dim)

# 文本侧投影：8192 -> 4096 -> dino_dim
self.dino_t_head = nn.Sequential(
    nn.Linear(8192, 4096), nn.GELU(),
    nn.LayerNorm(4096),
    nn.Linear(4096, self.dino_dim)
).to(self._dtype)

# 可选：局部（patch）投影头，首次用到再 lazy build（见 compute_dino_loss）
self.dino_local_head = None
```

**3. 在类内新增预处理与特征函数**

```python
@torch.no_grad()
def _dino_preprocess(self, x_img: torch.Tensor) -> torch.Tensor:
    """
    输入 [-1,1] → [0,1] → ImageNet mean/std → resize 到 DINO 输入尺寸
    """
    x = (x_img + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1,3,1,1)
    x = (x - mean) / std
    x = F.interpolate(x, size=(self.dino_img_size, self.dino_img_size), mode='bilinear', align_corners=False)
    return x

@torch.no_grad()
def _dino_features(self, x_img: torch.Tensor, return_patches: bool = False):
    x = self._dino_preprocess(x_img).to(self.device, dtype=self._dtype)
    feats = self.dino_model(x)  # {'global': (B,D), 'patch': (B,N,D)}
    g = feats['global']
    p = feats['patch'] if return_patches else None
    if self.dino_stopgrad_image:
        g = g.detach()
        if p is not None:
            p = p.detach()
    return g.to(self._dtype), (p.to(self._dtype) if p is not None else None)
```

**4. 在类内新增 DINO 语义损失**（三选一：`cosine`/`mse`/`vicreg`）

```python
def _dino_cos_loss(self, y_t, g):
    y_t = F.normalize(y_t, dim=-1)
    g   = F.normalize(g,   dim=-1)
    return (1.0 - (y_t * g).sum(dim=-1)).mean()

def _dino_mse_loss(self, y_t, g, normalize=True, whiten=False, eps=1e-5):
    if normalize:
        y_t = F.normalize(y_t, dim=-1)
        g   = F.normalize(g,   dim=-1)
    if whiten:
        y_t = (y_t - y_t.mean(0)) / (y_t.std(0) + eps)
        g   = (g   - g.mean(0))   / (g.std(0)   + eps)
    return F.mse_loss(y_t, g)

def _gather_with_grad(self, t: torch.Tensor) -> torch.Tensor:
    # 你已有 _all_gather_cat；这里给个可回传版（若你愿意可直接用已有）
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.nn.functional import all_gather as ag
        parts = list(ag(t))
        return torch.cat(parts, dim=0)
    return t

def _vicreg_student_on_teacher(self, y_t, g, inv_w=25., var_w=10., cov_w=0.5, gamma=1.0, eps=1e-4):
    """
    把 DINO (g) 当冻结教师：只在 y_t 上做 var/cov；invariance 用 y_t 与 g 的 MSE。
    """
    yt = self._gather_with_grad(y_t)
    gg = self._gather_with_grad(g.detach())

    inv = (yt.float() - gg.float()).pow(2).mean()

    y = yt.float()
    std = y.var(dim=0, unbiased=False).add(eps).sqrt()
    var = torch.relu(gamma - std).mean()

    yc = y - y.mean(dim=0, keepdim=True)
    N, d = yc.shape
    c = (yc.t() @ yc) / max(N - 1, 1)
    off = c - torch.diag(torch.diag(c))
    cov = off.pow(2).mean()  # 用 mean，稳定

    loss = inv_w*inv + var_w*var + cov_w*cov
    logs = {'dino_vic_inv': inv.detach(), 'dino_vic_var': var.detach(), 'dino_vic_cov': cov.detach()}
    return loss, logs

def compute_dino_loss(self, z_text: torch.Tensor, x_img: torch.Tensor):
    """
    全局 + 可选 patch 语义监督
    返回: loss, logs
    """
    # 1) DINO 全局特征
    g_global, p_tokens = self._dino_features(x_img, return_patches=self.dino_use_patch)

    # 2) 文本侧投影到 DINO 维度
    y_t_global = self.dino_t_head(z_text)

    # 3) 选择全局损失
    if self.dino_loss_type == 'cosine':
        loss_global = self._dino_cos_loss(y_t_global, g_global)
    elif self.dino_loss_type == 'mse':
        loss_global = self._dino_mse_loss(y_t_global, g_global, normalize=True, whiten=False)
    elif self.dino_loss_type == 'vicreg':
        loss_global, vic_logs = self._vicreg_student_on_teacher(y_t_global, g_global)
    else:
        raise ValueError(f"Unknown dino_loss_type: {self.dino_loss_type}")

    loss = self.dino_w_global * loss_global
    logs = {'dino_global': loss_global.detach()}

    if self.dino_loss_type == 'vicreg':
        logs.update(vic_logs)

    # 4) 可选：粗粒度 patch 对齐（g×g 自适应池化）
    if self.dino_use_patch:
        # 将 z_text 复原为 latent map，再做 g×g 池化
        # 注意：与你 FM 对齐的 latent 分辨率要一致（训练时 4×32×32 常见）
        B = z_text.size(0)
        C, Hh, Ww = 4, 32, 32   # 如你的训练是 256 latent → 4×32×32；按需改
        z_map = einops.rearrange(z_text, 'b (c h w) -> b c h w', c=C, h=Hh, w=Ww)

        gH, gW = self.dino_patch_grid
        pooled = F.adaptive_avg_pool2d(z_map, output_size=(gH, gW))   # (B,4,gH,gW)
        pooled = pooled.view(B, C, gH*gW).transpose(1, 2)             # (B,K,4), K=gH*gW

        # lazy build 局部投影
        if self.dino_local_head is None:
            self.dino_local_head = nn.Sequential(
                nn.Linear(C, 256), nn.GELU(), nn.LayerNorm(256),
                nn.Linear(256, self.dino_dim)
            ).to(self._dtype).to(self.device)

        y_loc = self.dino_local_head(pooled)  # (B,K,D)

        # 教师侧 patch 聚合到 K 个区域（简单起步：平均）
        if p_tokens is not None:
            # 简单做 mean；你也可以用注意力池化或最近邻匹配
            p_agg = p_tokens.mean(dim=1, keepdim=True).expand(B, gH*gW, self.dino_dim)
        else:
            # 若拿不到 patch，就退化为 global
            p_agg = g_global.unsqueeze(1).expand(B, gH*gW, self.dino_dim)

        # 局部损失：默认 cosine
        loc_cos = F.cosine_similarity(F.normalize(y_loc, dim=-1), F.normalize(p_agg, dim=-1), dim=-1)  # (B,K)
        loss_patch = 1.0 - loc_cos.mean()

        loss = loss + self.dino_w_patch * loss_patch
        logs['dino_patch'] = loss_patch.detach()

    return loss, logs
```

**5. 在 `text2image_loss()` 里接入 DINO 语义监督**

在你计算完 `loss_flow` 和 `loss_kl` 之后，加上：

```python
# ---- DINOv2 semantic supervision ----
if getattr(self, 'use_dino', False):
    # warmup（可选）
    self._dino_steps = getattr(self, '_dino_steps', 0) + 1
    warm = min(1.0, self._dino_steps / 5000.0)  # 5k步线性到位
    alpha_eff = self.dino_alpha * warm

    loss_dino, dino_logs = self.compute_dino_loss(z_text, x_img)
    ret = {
        'loss_flow': loss_flow,
        'loss_dino': alpha_eff * loss_dino,   # mmengine 会自动求和所有以 loss_ 开头
        'loss_kl': 1e-3 * loss_kl,
        'kl_raw': loss_kl.detach(),
    }
    ret.update(dino_logs)
else:
    ret = {
        'loss_flow': loss_flow,
        'loss_kl': 1e-3 * loss_kl,
        'kl_raw': loss_kl.detach(),
    }
```

> 说明
>
> * 不需要显式返回 `loss`；mmengine 会把 `loss_*` 相加作为总损。
> * 初期建议 `dino_loss_type='cosine'`，`dino_alpha=0.2`，**slow warmup**。稳定后再尝试 `mse`（带 normalize）或 `vicreg`（只在学生侧做 var/cov）。
> * 若你后续也接入了 “VAE latent 上的 VICReg”，建议 **DINO 用 Cosine**，以免两条 VICReg 互相掰手腕。

---

## 训练指引（起步配方）

* `dino_loss_type='cosine'`
* `dino_alpha=0.2`，5k step 线性 warmup
* `dino_use_patch=False`（先只全局） → 稳了再开 `True`，`dino_patch_grid=(2,2)`，`dino_w_patch=0.2`
* `dino_stopgrad_image=True`（强烈建议）
* 观察日志：`dino_global` 逐步下降、`loss_flow` 不被明显拖慢，生成不再是“彩色马赛克”，应出现主体语义。

---

## 常见问题

* **载不动官方权重**：自动走 `timm` 兜底；你也可把 `use_timm_fallback=False` 改成 `True/False` 控。
* **patch 特征拿不到**：封装里已处理，会退化到只用 global。
* **尺度不匹配**：损失里统一做了 `F.normalize`（Cosine/MSE），避免被尺度主导。
* **FM 收敛变慢**：降 `dino_alpha` 到 0.1～0.15 或加长 warmup；先不开 patch。

---

需要我把 `DINOv2Backbone` 注册到你们的 `BUILDER` 里，以及给一段最小 YAML 配置吗？
