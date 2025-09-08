下面给你一套**基于 DINOv2 官方代码库定义**（`facebookresearch/dinov2`）的、可直接接入你当前 `InternVL3QueryFMTransformer` 的实现。包含：

* 官方模型加载（支持 **torch.hub** 入口名，如 `dinov2_vitl14`；也支持本地 hub / 纯源码导入）
* 预处理与特征抽取（拿到 `x_norm_clstoken` 作为 global、`x_norm_patchtokens` 作为 patch）
* 三种语义监督损失：**Cosine**（默认）、**L2/MSE（已规范化）**、**VICReg（仅学生侧 var/cov）**

> 注：官方 repo 提供了 `torch.hub` 入口（如 `dinov2_vitl14`）供加载，并在 `forward_features` 中返回 `x_norm_clstoken` 和 `x_norm_patchtokens` 这类键值；下面的实现正是按此接口写的。([Hugging Face][1], [GitHub][2])
> 若你完全离线，`torch.hub.load(repo_or_dir=本地目录, source='local')` 也能用（目录里需有官方的 `hubconf.py`）。([PyTorch 文档][3])

---

## 1) 新增文件：官方 DINOv2 封装

**路径**：`src/models/dino/dinov2_official.py`

```python
# -*- coding: utf-8 -*-
# src/models/dino/dinov2_official.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class OfficialDINOv2(nn.Module):
    """
    官方优先的 DINOv2 封装（facebookresearch/dinov2）：
    - 优先用 torch.hub 的入口名（如 'dinov2_vitl14'）
    - 也支持传入本地 repo 路径 + source='local'
    - 或者 fallback 为从源码 import + 手工 load_state_dict（可选）

    forward(x) 返回:
      {'global': (B,D), 'patch': (B,N,D or None)}
    其中 global 来自 x_norm_clstoken，patch 来自 x_norm_patchtokens（若可用）
    """
    def __init__(
        self,
        variant: str = "vitl14",            # 'vits14'|'vitb14'|'vitl14'|'vitg14'
        hub_repo_or_dir: str = "facebookresearch/dinov2",
        hub_source: str = "github",         # 'github' or 'local'
        checkpoints_dir: str = None,        # 若要离线 hub，放置 ckpt 的目录（可选）
        fallback_from_source: bool = False, # 若 hub 失败，是否尝试源码 import + 手动权重
        fallback_repo_dir: str = None,      # 源码路径（根目录包含 dinov2/）
        fallback_ckpt_path: str = None,     # 对应 *.pth
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.hub_repo_or_dir = hub_repo_or_dir
        self.hub_source = hub_source
        self.embed_dim = None
        self.device_ = device
        self.dtype_ = dtype

        self.model = None
        self._load_from_hub_or_fallback(
            fallback_from_source=fallback_from_source,
            fallback_repo_dir=fallback_repo_dir,
            fallback_ckpt_path=fallback_ckpt_path,
            checkpoints_dir=checkpoints_dir
        )

        if self.device_ is not None:
            self.to(self.device_)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def _entry_name(self):
        return {
            "vits14": "dinov2_vits14",
            "vitb14": "dinov2_vitb14",
            "vitl14": "dinov2_vitl14",
            "vitg14": "dinov2_vitg14",
        }.get(self.variant, "dinov2_vitl14")

    def _load_from_hub_or_fallback(
        self,
        fallback_from_source: bool,
        fallback_repo_dir: str,
        fallback_ckpt_path: str,
        checkpoints_dir: str = None
    ):
        # 可选：指定 TORCH_HOME 以便 hub 从本地 checkpoints 取权重
        if checkpoints_dir:
            os.makedirs(checkpoints_dir, exist_ok=True)
            os.environ.setdefault("TORCH_HOME", os.path.dirname(os.path.dirname(checkpoints_dir)))

        entry = self._entry_name()
        # 1) torch.hub（支持本地 source='local'）
        try:
            import torch.hub
            self.model = torch.hub.load(self.hub_repo_or_dir, entry, source=('local' if self.hub_source == 'local' else 'github'))
        except Exception as e:
            if not fallback_from_source:
                raise RuntimeError(f"[DINOv2] torch.hub 加载失败：{e}")
            # 2) fallback: 从源码构建 + load_state_dict
            if not (fallback_repo_dir and fallback_ckpt_path):
                raise RuntimeError("[DINOv2] 需要提供 fallback_repo_dir 与 fallback_ckpt_path 才能从源码加载。")
            sys.path.insert(0, fallback_repo_dir)
            from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
            if self.variant == "vits14":
                self.model = vit_small()
            elif self.variant == "vitb14":
                self.model = vit_base()
            else:
                self.model = vit_large()
            state = torch.load(fallback_ckpt_path, map_location="cpu")
            key = "model" if isinstance(state, dict) and "model" in state else None
            self.model.load_state_dict(state[key] if key else state, strict=True)

        # 推断维度
        self.embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("[DINOv2] 无法推断特征维度 embed_dim。")

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: 已经 resize & 标准化到 ImageNet 分布的 (B,3,H,W)
        返回: {'global': (B,D), 'patch': (B,N,D or None)}
        """
        m = self.model
        out = {"global": None, "patch": None}
        # 官方/部分 timm 都提供 forward_features
        if hasattr(m, "forward_features"):
            feats = m.forward_features(x)
            if isinstance(feats, dict):
                g = feats.get("x_norm_clstoken", feats.get("cls_token", None))
                p = feats.get("x_norm_patchtokens", feats.get("patch_tokens", None))
                out["global"] = g if g is not None else feats
                out["patch"]  = p
            else:
                # 部分实现直接返回 (B,N+1,D)
                if feats.ndim == 3:
                    out["global"] = feats[:, 0]
                    out["patch"]  = feats[:, 1:]
                else:
                    out["global"] = feats
        else:
            feats = m(x)
            if feats.ndim == 3:
                out["global"] = feats[:, 0]
                out["patch"]  = feats[:, 1:]
            else:
                out["global"] = feats
        # dtype 对齐
        for k in out:
            if out[k] is not None:
                out[k] = out[k].to(dtype=self.dtype_, device=x.device)
        return out
```

---

## 2) 在你的主类中：接入 DINOv2 + 语义损失

把下面各段**直接粘**进你的类定义里（按注释位置放）。

### 2.1 imports（在文件头部）

```python
from src.models.dino.dinov2_official import OfficialDINOv2
```

### 2.2 在 `__init__` 末尾附近（`self.to(self._dtype)` 之后）增加 DINO 配置与加载

```python
# ---- DINOv2 semantic supervision (official) ----
self.use_dino = True                   # 语义监督开关
self.dino_variant = 'vitl14'           # 'vits14'|'vitb14'|'vitl14'|'vitg14'
self.dino_img_size = 224               # 官方常用 224（或 518）；起步用 224
self.dino_loss_type = 'cosine'         # 'cosine' | 'mse' | 'vicreg'
self.dino_alpha = 0.2                  # DINO 语义损失总权重（建议 warmup）
self.dino_use_patch = False            # 先不开 patch，对齐 global
self.dino_patch_grid = (2, 2)          # 开 patch 时的 g×g 网格
self.dino_w_global = 1.0
self.dino_w_patch = 0.2
self.dino_stopgrad_image = True        # 冻结教师，不回传图像侧

# hub 模式（联网或本地 hub）。完全离线时，将 hub_source 设为 'local' 并把官方 repo 拷到本地
self.dino_model = OfficialDINOv2(
    variant=self.dino_variant,
    hub_repo_or_dir="facebookresearch/dinov2",  # 或本地路径，如 "/opt/models/dinov2"
    hub_source="github",                        # 离线请改 "local"
    checkpoints_dir=None,                       # 若需离线 hub，可设置 TORCH_HOME/hub/checkpoints
    fallback_from_source=False,                 # 若你想走源码 fallback，再改 True 并传 fallback_* 参数
    device=self.device,
    dtype=self._dtype
)
self.dino_dim = int(self.dino_model.embed_dim)

# 文本侧投影到 DINO 维度（仅用于语义监督分支）
self.dino_t_head = nn.Sequential(
    nn.Linear(8192, 4096), nn.GELU(),
    nn.LayerNorm(4096),
    nn.Linear(4096, self.dino_dim)
).to(self._dtype).to(self.device)

# 局部投影（patch）lazy build：首次用到再创建
self.dino_local_head = None
```

### 2.3 增加图片预处理 & 特征抽取函数（类内）

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
    return x.to(dtype=self._dtype, device=self.device)

@torch.no_grad()
def _dino_features(self, x_img: torch.Tensor, return_patches: bool = False):
    """
    返回：global (B, D) 与可选 patch (B, N, D)
    """
    x = self._dino_preprocess(x_img)
    feats = self.dino_model(x)  # {'global': (B,D), 'patch': (B,N,D or None)}
    g = feats['global']
    p = feats['patch'] if return_patches else None
    if self.dino_stopgrad_image:
        g = g.detach()
        if p is not None:
            p = p.detach()
    return g, p
```

### 2.4 增加三种损失（类内）

```python
def _dino_cos_loss(self, y_t, g):
    # 方向一致性：最稳
    y_t = F.normalize(y_t, dim=-1)
    g   = F.normalize(g,   dim=-1)
    return (1.0 - (y_t * g).sum(dim=-1)).mean()

def _dino_mse_loss(self, y_t, g, normalize=True, whiten=False, eps=1e-5):
    # L2 回归：建议先做归一化（或白化），避免尺度主导
    if normalize:
        y_t = F.normalize(y_t, dim=-1)
        g   = F.normalize(g,   dim=-1)
    if whiten:
        y_t = (y_t - y_t.mean(0)) / (y_t.std(0) + eps)
        g   = (g   - g.mean(0))   / (g.std(0)   + eps)
    return F.mse_loss(y_t, g)

def _gather_with_grad(self, t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.nn.functional import all_gather as ag
        parts = list(ag(t))
        return torch.cat(parts, dim=0)
    return t

def _dino_vicreg_student(self, y_t, g, inv_w=25., var_w=10., cov_w=0.5, gamma=1.0, eps=1e-4):
    """
    VICReg：把 DINO (g) 当冻结教师，仅在学生 y_t 上加 var/cov。
    invariance 用 y_t 与 g 的 MSE。
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
    cov = off.pow(2).mean()   # 用 mean，更稳

    loss = inv_w*inv + var_w*var + cov_w*cov
    logs = {'dino_vic_inv': inv.detach(), 'dino_vic_var': var.detach(), 'dino_vic_cov': cov.detach()}
    return loss, logs
```

### 2.5 组合 DINO 语义损失（类内）

```python
def compute_dino_loss(self, z_text: torch.Tensor, x_img: torch.Tensor):
    """
    计算 DINO 语义损失：global + 可选 patch
    返回: loss, logs
    """
    # 1) 教师特征
    g_global, p_tokens = self._dino_features(x_img, return_patches=self.dino_use_patch)

    # 2) 学生（文本侧）投影到 DINO 空间
    y_t_global = self.dino_t_head(z_text)  # (B, dino_dim)

    # 3) 全局损失
    if self.dino_loss_type == 'cosine':
        loss_global = self._dino_cos_loss(y_t_global, g_global)
        logs = {}
    elif self.dino_loss_type == 'mse':
        loss_global = self._dino_mse_loss(y_t_global, g_global, normalize=True, whiten=False)
        logs = {}
    elif self.dino_loss_type == 'vicreg':
        loss_global, logs = self._dino_vicreg_student(y_t_global, g_global)
    else:
        raise ValueError(f"Unknown dino_loss_type: {self.dino_loss_type}")

    loss = self.dino_w_global * loss_global
    logs['dino_global'] = loss_global.detach()

    # 4) 可选：粗粒度 patch 对齐（把 z_text 映射回 4×H×W，再做 g×g 池化）
    if self.dino_use_patch:
        B = z_text.size(0)
        # 按你 FM 训练用的 latent 尺寸设置。你在 flow_matching_loss 里从 VAE sample 得到 (B, C, Hh, Ww)
        C, Hh, Ww = 4, 32, 32
        z_map = einops.rearrange(z_text, 'b (c h w) -> b c h w', c=C, h=Hh, w=Ww)

        gH, gW = self.dino_patch_grid
        pooled = F.adaptive_avg_pool2d(z_map, output_size=(gH, gW))  # (B,4,gH,gW)
        pooled = pooled.view(B, C, gH*gW).transpose(1, 2)            # (B,K,4)

        if self.dino_local_head is None:
            self.dino_local_head = nn.Sequential(
                nn.Linear(C, 256), nn.GELU(), nn.LayerNorm(256),
                nn.Linear(256, self.dino_dim)
            ).to(self._dtype).to(self.device)

        y_loc = self.dino_local_head(pooled)    # (B,K,D)

        if p_tokens is not None:
            # 简化：把所有 patch 取均值作为目标（你也可以做最近邻匹配/注意力聚合）
            p_agg = p_tokens.mean(dim=1, keepdim=True).expand(B, gH*gW, self.dino_dim)
        else:
            # 若拿不到 patch，就退化为复用 global
            p_agg = g_global.unsqueeze(1).expand(B, gH*gW, self.dino_dim)

        # 局部损失统一用 cosine（稳定）
        loc_cos = F.cosine_similarity(F.normalize(y_loc, dim=-1), F.normalize(p_agg, dim=-1), dim=-1)  # (B,K)
        loss_patch = 1.0 - loc_cos.mean()
        loss = loss + self.dino_w_patch * loss_patch
        logs['dino_patch'] = loss_patch.detach()

    return loss, logs
```

### 2.6 在 `text2image_loss()` 中合入（C 段后面追加）

在你已经得到 `z_text, mu, log_var`、并算完 `loss_flow` 与 `loss_kl` 之后，**追加**：

```python
# ---- DINOv2 semantic supervision ----
if getattr(self, 'use_dino', False):
    # 线性 warmup 到位（例如 5k 步）
    self._dino_steps = getattr(self, '_dino_steps', 0) + 1
    warm = min(1.0, self._dino_steps / 5000.0)
    alpha_eff = self.dino_alpha * warm

    loss_dino, dino_logs = self.compute_dino_loss(z_text, x_img)

    ret = {
        'loss_flow': loss_flow,
        'loss_dino': alpha_eff * loss_dino,    # mmengine 会自动把所有 loss_* 相加
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

---

## 3) 训练建议（Quick Start）

* `self.dino_loss_type = 'cosine'`（默认最稳）
* `self.dino_alpha = 0.2`，**5k** 步线性 warmup
* `self.dino_use_patch = False`（先全局，稳了再开 patch：`(2,2)`，`dino_w_patch=0.2`）
* 教师侧 **冻结 + stop-grad**（`self.dino_stopgrad_image=True`），只“拉”文本侧。
* 观察 `dino_global` 随训练下降、生成不再是“彩色马赛克”，出现主体语义；确保 `loss_flow` 没被明显拖慢。
* 如果你后续在 VAE latent 上还会用 VICReg，建议 DINO 这条保持 **cosine**，避免两个 VICReg 互相“掰”方向。

---

### 关键官方依据（接口/入口名）

* `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')` 等入口是官方 README 所示用法。([Hugging Face][1])
* 官方 `forward_features` 中包含 `x_norm_clstoken`（全局 CLS）与 `x_norm_patchtokens`（patch token），本文实现按此读取。([GitHub][2])
* `torch.hub.load` 支持 **本地目录 + source='local'**，便于离线部署。([PyTorch 文档][3])

---

如果你要我把 **完全离线** 的两段（“本地 hub + checkpoints” 或 “源码 import + load\_state\_dict”）写成脚本放到 `tools/` 下，也可以直接给你。

[1]: https://huggingface.co/Jacobmadwed/ocmhelp/blame/b5d16b2e410ccb34ab6e0f7c66c7878ad2429c9e/torchhub/facebookresearch_dinov2_main/README.md?utm_source=chatgpt.com "torchhub/facebookresearch_dinov2_main/README.md"
[2]: https://github.com/facebookresearch/dinov2/issues/398?utm_source=chatgpt.com "Issue #398 · facebookresearch/dinov2 - intermediate tokens"
[3]: https://docs.pytorch.org/docs/stable/hub.html?utm_source=chatgpt.com "torch.hub — PyTorch 2.8 documentation"
