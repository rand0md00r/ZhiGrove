# -*- coding: utf-8 -*-
# src/models/dino/dinov2_backbone.py
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2Backbone(nn.Module):
    """
    离线版 DINOv2 封装（不使用 torch.hub）：
      - 从官方源码构建骨干（需要提供 dinov2 源码根目录）
      - 从本地权重文件夹自动加载权重（支持 model.safetensors / pytorch_model.bin / *.pth）
      - forward(x) 返回 {'global': (B,D), 'patch': (B,N,D 或 None)}

    参数：
      variant: 'vits14' | 'vitb14' | 'vitl14'（默认） | 'vitg14'(部分环境可能无对齐权重)
      repo_dir:  官方 dinov2 源码根目录（包含 dinov2/ 子目录）
      weights_dir: 本地权重文件夹（内含 model.safetensors 或 pytorch_model.bin 等）
      weights_path: 可选，若你想指定具体权重文件路径，优先级高于 weights_dir
      device:  放到哪个设备
      dtype:   模型参数 dtype
    """
    def __init__(
        self,
        variant: str = "vitl14",
        repo_dir: str = "/vepfs/group03/wyq/ug_uni/dinov2-main",
        weights_dir: str = "/vepfs/public/model-public/dinov2-large",
        weights_path: str = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.repo_dir = repo_dir
        self.weights_dir = weights_dir
        self.weights_path = weights_path
        self.device_ = device
        self.dtype_ = dtype

        # 1) 导入官方源码
        self._import_official_repo(self.repo_dir)

        # 2) 构建骨干
        self.model = self._build_backbone(self.variant)

        # 3) 加载本地权重
        ckpt = self._resolve_ckpt_path(self.weights_path, self.weights_dir)
        state = self._load_state_dict_any(ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[DINOv2] load_state_dict from '{ckpt}': "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

        # 4) 推断输出维度
        self.embed_dim = getattr(self.model, "embed_dim", None) or \
                         getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            raise RuntimeError("[DINOv2] 无法推断特征维度 embed_dim。")

        # 5) 冻结 & 设备/精度
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        if self.device_ is not None:
            self.to(self.device_)

    # ---------- 构建 / 加载 ----------
    def _import_official_repo(self, repo_dir: str):
        if not (repo_dir and os.path.isdir(repo_dir)):
            raise FileNotFoundError(
                f"[DINOv2] 官方源码目录不存在：{repo_dir}\n"
                f"请将 facebookresearch/dinov2 仓库克隆到本地，并把该路径传入 repo_dir。"
            )
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

    def _build_backbone(self, variant: str):
        # 官方实现位于 dinov2.models.vision_transformer
        from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
        if variant == "vits14":
            model = vit_small()
        elif variant == "vitb14":
            model = vit_base()
        elif variant == "vitl14":
            model = vit_large()
        else:
            # 大多数环境只提供到了 vit_large；vit_giant 可能没有现成权重
            print(f"[DINOv2] 未知/不推荐的变体 '{variant}'，回退到 vit_large。")
            model = vit_large()
        return model.to(dtype=self.dtype_)

    def _resolve_ckpt_path(self, weights_path: str, weights_dir: str) -> str:
        if weights_path:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"[DINOv2] 指定权重不存在：{weights_path}")
            return weights_path

        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(f"[DINOv2] 权重文件夹不存在：{weights_dir}")

        # 按优先级寻找：safetensors > bin > pth
        patterns = [
            "**/model.safetensors", "**/*.safetensors",
            "**/pytorch_model.bin", "**/*.bin",
            "**/*.pth", "**/*.ckpt",
        ]
        candidates = []
        for pat in patterns:
            candidates += glob.glob(os.path.join(weights_dir, pat), recursive=True)
        if not candidates:
            raise FileNotFoundError(
                f"[DINOv2] 在目录 '{weights_dir}' 下未找到可用权重文件 "
                f"(model.safetensors / pytorch_model.bin / *.pth)。"
            )
        # 选择最靠前的匹配
        ckpt = sorted(candidates)[0]
        print(f"[DINOv2] 自动选择权重：{ckpt}")
        return ckpt

    def _load_state_dict_any(self, ckpt_path: str):
        assert os.path.exists(ckpt_path), f"CKPT not found: {ckpt_path}"
        if ckpt_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise RuntimeError(
                    "请安装 safetensors：`pip install safetensors`，"
                    "或在有网机器上将 .safetensors 转为 .pth 后再拷贝。"
                ) from e
            raw = load_file(ckpt_path)  # dict[str, Tensor]
        else:
            raw = torch.load(ckpt_path, map_location="cpu")  # 支持 .bin/.pth/.ckpt

        # 兼容常见包装
        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            sd = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]
        elif isinstance(raw, dict):
            sd = raw
        else:
            raise RuntimeError(f"[DINOv2] 不支持的权重格式：{type(raw)}")

        # 去除常见前缀
        def strip_prefix(k: str):
            for p in ("model.", "module.", "backbone.", "encoder."):
                if k.startswith(p):
                    return k[len(p):]
            return k
        sd = {strip_prefix(k): v for k, v in sd.items()}
        # 删掉分类头（若存在）
        sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "fc."))}
        return sd

    # ---------- 前向 ----------
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: 已标准化到 ImageNet 分布、且尺寸已被外部处理成 (B,3,H,W)
        返回：
          {'global': (B,D), 'patch': (B,N,D 或 None)}
        """
        m = self.model
        out = {"global": None, "patch": None}

        if hasattr(m, "forward_features"):
            feats = m.forward_features(x)
            if isinstance(feats, dict):
                g = feats.get("x_norm_clstoken", feats.get("cls_token", None))
                p = feats.get("x_norm_patchtokens", feats.get("patch_tokens", None))
                out["global"] = g if g is not None else feats
                out["patch"] = p
            else:
                # 某些实现直接返回 (B,N+1,D)
                if feats.ndim == 3:
                    out["global"] = feats[:, 0]
                    out["patch"] = feats[:, 1:]
                else:
                    out["global"] = feats
        else:
            feats = m(x)
            if isinstance(feats, torch.Tensor) and feats.ndim == 3:
                out["global"] = feats[:, 0]
                out["patch"] = feats[:, 1:]
            else:
                out["global"] = feats

        # dtype 对齐
        for k, v in out.items():
            if v is not None:
                out[k] = v.to(dtype=self.dtype_, device=x.device)
        return out




# === Dino v2（离线） ===
dino_v2 = dict(
    type=DINOv2Backbone,
    variant='vitl14',
    repo_dir="/vepfs/group03/wyq/ug_uni/dinov2-main",           # 官方源码根目录（含 dinov2/）
    weights_dir="/vepfs/public/model-public/dinov2-large",      # 你的本地模型文件夹
    # 如果你想指定具体文件，也可加：
    # weights_path="/vepfs/public/model-public/dinov2-large/model.safetensors",
)
