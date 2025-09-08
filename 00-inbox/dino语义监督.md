# -*- coding: utf-8 -*-
# src/models/dino/dinov2_official.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2Backbone(nn.Module):
    """
    官方优先的 DINOv2 封装（facebookresearch/dinov2）：
    - 优先用 torch.hub 的入口名（如 'dinov2_vitl14'）
    - 也支持传入本地 repo 路径 + source='local'
    - 或者 fallback 为从源码 import + 手工 load_state_dict（可选）

    forward(x) 返回:
      {'global': (B,D), 'patch': (B,N,D or None)}
    其中 global 来自 x_norm_clstoken，patch 来自 x_norm_patchtokens（若可用）

    Args:
        variant: 模型变体，可选 'vits14'|'vitb14'|'vitl14'|'vitg14'
        hub_repo_or_dir: torch.hub 的 repo 或本地路径
        hub_source: torch.hub 的源，可选 'github' 或 'local'
        checkpoints_dir: 若要离线 hub，放置 ckpt 的目录（可选）
        fallback_from_source: 若 hub 失败，是否尝试源码 import + 手动权重
        fallback_repo_dir: 源码路径（根目录包含 dinov2/）
        fallback_ckpt_path: 对应 *.pth
        device: 设备
        dtype: 数据类型
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
        device: torch.device = None,        # 设备
        dtype: torch.dtype = torch.float32, # 数据类型
    ):
        super().__init__()
        self.variant = variant.lower()
        self.hub_repo_or_dir = hub_repo_or_dir
        self.hub_source = hub_source
        self.embed_dim = None               # 特征维度
        self.device_ = device
        self.dtype_ = dtype                  # 数据类型

        self.model = None
        self._load_from_hub_or_fallback(
            fallback_from_source=fallback_from_source,
            fallback_repo_dir=fallback_repo_dir,
            fallback_ckpt_path=fallback_ckpt_path,
            checkpoints_dir=checkpoints_dir
        )

        if self.device_ is not None:
            self.to(self.device_)             # 设备
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

    def _load_state_dict_any(self, ckpt_path: str):

        assert os.path.exists(ckpt_path), f"CKPT not found: {ckpt_path}"
        if ckpt_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise RuntimeError("Please `pip install safetensors` or convert safetensors to .pth/.bin offline.") from e
            raw = load_file(ckpt_path)  # dict[str,Tensor]
        else:
            raw = torch.load(ckpt_path, map_location="cpu")  # 支持 .pth/.bin

        # 兼容常见封装格式
        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            sd = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]
        elif isinstance(raw, dict):
            sd = raw
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(raw)}")

        # 去常见前缀
        def strip_prefix(k: str):
            for p in ("model.", "module.", "backbone.", "encoder."):
                if k.startswith(p): return k[len(p):]
            return k
        sd = {strip_prefix(k): v for k, v in sd.items()}

        # 去掉分类头（若有）
        sd = {k: v for k, v in sd.items() if not k.startswith("head.") and not k.startswith("fc.")}

        return sd

    def _load_from_hub_or_fallback(
        self,
        fallback_from_source: bool,
        fallback_repo_dir: str,
        fallback_ckpt_path: str,
        checkpoints_dir: str = None
    ):

        entry = self._entry_name()
        # 1) torch.hub（支持本地 source='local'）

        # === fallback: 官方源码 + 本地权重 ===
        import sys, os, torch
        sys.path.insert(0, self.fallback_repo_dir)
        from dinov2.models.vision_transformer import vit_small, vit_base, vit_large

        if self.variant == "vits14":
            self.model = vit_small()
        elif self.variant == "vitb14":
            self.model = vit_base()
        else:
            self.model = vit_large()

        sd = self._load_state_dict_any(self.fallback_ckpt_path)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[DINOv2] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

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
