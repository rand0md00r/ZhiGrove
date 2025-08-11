# stunning-palm-tree
---
2025年8月11日10:25:24

``` python
# src/models/openuni/fm_utils.py
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class TimeStepSampler:
    """Base: sample timesteps t ∈ [0,1] for flow matching."""
    def sample_time(self, x_start: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LogitNormalSampler(TimeStepSampler):
    """t = sigmoid(N(mean, std^2)) per https://arxiv.org/pdf/2403.03206.pdf ."""
    def __init__(self, normal_mean: float = 0.0, normal_std: float = 1.0):
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)

    @torch.no_grad()
    def sample_time(self, x_start: torch.Tensor) -> torch.Tensor:
        x_normal = torch.normal(
            mean=self.normal_mean,
            std=self.normal_std,
            size=(x_start.shape[0],),
            device=x_start.device,
        )
        return torch.sigmoid(x_normal)


def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand 1D t to the shape of x (broadcastable)."""
    t_expanded = t
    while t_expanded.ndim < x.ndim:
        t_expanded = t_expanded.unsqueeze(-1)
    return t_expanded.expand_as(x)


def psi(t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor,
        sigma_min: float, sigma_max: float) -> torch.Tensor:
    # Linear path used in your FM impl: ((t*(sigma_min/sigma_max - 1) + 1) * x + t * x1)
    t = expand_t(t, x)
    return (t * (sigma_min / sigma_max - 1) + 1) * x + t * x1


def Dt_psi(t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor,
           sigma_min: float, sigma_max: float) -> torch.Tensor:
    # ∂ψ/∂t for the path above
    assert x.shape[0] == x1.shape[0]
    return (sigma_min / sigma_max - 1) * x + x1


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    x: [B, L, D] or [B, D]; mask: [B, L] (True=valid).
    Return: [B, D]
    """
    if x.ndim == 2:
        return x
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.dtype)
    m = m / (m.sum(dim=1, keepdim=True) + 1e-6)
    return (x * m.unsqueeze(-1)).sum(dim=1)
```

---
``` python

# src/models/openuni/loss_fns.py
import torch

def kl_loss(mu: torch.Tensor, log_var: torch.Tensor,
            coef_mu: float = 0.3, power: int = 6) -> torch.Tensor:
    """
    Slightly modified KL: mu -> 0 via (coef_mu*mu)^power, var -> 1.
    Return per-sample loss [B].
    """
    return -0.5 * torch.sum(1 + log_var - (coef_mu * mu) ** power - log_var.exp(), dim=1)


def mse_mean_over_spatial(err: torch.Tensor, start_dim: int = 1,
                          mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Mean-of-square over spatial/feature dims; optional mask over tokens.
    Return per-sample loss [B].
    """
    if mask is not None:
        return (err.pow(2).mean(dim=-1) * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)
```

---


``` python
# src/models/openuni/internvl3_sana_hf.py
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.nn.utils.rnn import pad_sequence

from mmengine.model import BaseModel
from mmengine.logging import print_log
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from src.models.connector import ConnectorConfig, ConnectorEncoder
from src.models.encoder import TransEncoder, FrozenCLIPEmbedder
from src.models.diffusion import FMTransFormers
from src.models.encoder.autoencoder import FrozenAutoencoderKL
from src.models.cliploss import ClipLoss
from ml_collections import ConfigDict

from src.models.openuni.fm_utils import (
    IMAGENET_MEAN, IMAGENET_STD,
    LogitNormalSampler, masked_mean_pool,
    psi, Dt_psi
)
from src.models.openuni.loss_fns import kl_loss, mse_mean_over_spatial


class OpenUniInternVL3SANAHF(BaseModel):
    """
    Frozen VLM (self.lmm) 提供视觉/文本特征，
    文本侧经 TransEncoder 变换为 latent（均值/方差 + reparam），
    与图像侧目标 latent 做 Flow Matching；同时做 CLIP-style 对齐与 KL 正则。
    """
    def __init__(self,
                 lmm,
                 vae,
                 tokenizer,
                 prompt_template,
                 connector,
                 # optional blocks
                 transformer=None,                   # 已不用于训练，仅保留构造兼容
                 train_scheduler=None,               # 未使用（训练侧FM自研）
                 test_scheduler=None,                # 未使用（若后续做采样/inference再接回）
                 # arch hyper-params
                 num_queries=256,
                 vit_input_size=448,
                 max_length=2048,
                 proj_type='enc_proj',
                 transencoder=None,
                 # training switches
                 pretrained_pth=None,
                 freeze_lmm=True,
                 freeze_transformer=True,
                 use_activation_checkpointing=True,
                 lora_modules=None,
                 lora_rank=8,
                 lora_alpha=8,
                 # IO & precision
                 dtype='bf16',
                 clip_target_source='pixels',  # 'pixels' | 'latents_decode'
                 ):
        super().__init__()

        # ---- precision ----
        self._dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32

        # ---- backbone / tokenizer ----
        self.lmm = BUILDER.build(lmm)
        if freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = freeze_lmm

        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.vit_input_size = vit_input_size
        self.max_length = max_length

        # token id for image placeholders
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            prompt_template['IMG_CONTEXT_TOKEN']
        )
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std',  torch.tensor(IMAGENET_STD),  persistent=False)

        # ---- connector & projector ----
        self.num_queries = num_queries
        self.connector = ConnectorEncoder(ConnectorConfig(**connector))
        self.proj_type = proj_type

        # projector depends on proj_type
        if self.proj_type == 'proj_enc':
            assert self.connector.config.hidden_size == self.transformer.config.caption_channels
            self.projector = nn.Linear(self.llm.config.hidden_size, self.connector.config.hidden_size)
        elif self.proj_type == 'enc_proj':
            assert self.connector.config.hidden_size == self.llm.config.hidden_size
            # 训练中未使用 transformer captioner，这里 projector 仅保留占位
            self.projector = nn.Linear(self.connector.config.hidden_size,  self.llm.config.hidden_size)
        elif self.proj_type == 'proj_enc_proj':
            self.projector = nn.ModuleList([
                nn.Linear(self.llm.config.hidden_size, self.connector.config.hidden_size),
                nn.Linear(self.connector.config.hidden_size, self.llm.config.hidden_size),
            ])
        else:
            raise ValueError(f'Unknown proj type: {self.proj_type}')

        # meta queries
        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size)
        )
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))

        # activation ckpt
        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        # load ckpt
        if pretrained_pth is not None:
            state = guess_load_checkpoint(pretrained_pth)
            _ = self.load_state_dict(state, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        # ---- text & image heads ----
        self.frozen_clip = FrozenCLIPEmbedder().eval()  # 文本侧 encoder（保持冻结）
        img_feat_dim = getattr(self.llm.config, 'hidden_size', 1536)
        # 将 VLM 视觉 token 池化后的向量映射到对齐空间（与 z_text 的维度一致）
        self.img_clip_head = nn.Linear(img_feat_dim, 8192, bias=False)

        # CLIP 对齐：可学习温度
        self.clip_logit_scale = nn.Parameter(torch.tensor(1/0.07).log())
        self.clip_loss = ClipLoss()

        # ---- TransEncoder (text VE) ----
        assert transencoder is not None, "transencoder config is required"
        self.context_encoder = TransEncoder(
            d_model=transencoder.d_model,
            N=transencoder.num_blocks,
            num_token=transencoder.num_token,
            head_num=transencoder.num_attention_heads,
            d_ff=transencoder.hidden_dim,
            latten_size=transencoder.latten_size,
            down_sample_block=transencoder.down_sample_block,
            dropout=transencoder.dropout_prob,
            last_norm=transencoder.last_norm
        )

        # ---- Flow backbone ----
        self.latent_channels = 8  # 你的实现中固定为 8
        stage_configs = [
            ConfigDict({
                "block_type": "TransformerBlock",
                "dim": 1024, "hidden_dim": 2048, "num_attention_heads": 16,
                "num_blocks": 65, "max_height": 16, "max_width": 16,
                "image_input_ratio": 1, "input_feature_ratio": 2,
                "final_kernel_size": 3, "dropout_prob": 0.0,
                "pe_type": "sinusoidal", "norm_type": "TDRMSN",
                "gradient_checking": True
            }),
            ConfigDict({
                "block_type": "ConvNeXtBlock",
                "dim": 512, "hidden_dim": 1024, "kernel_size": 7,
                "num_blocks": 33, "max_height": 32, "max_width": 32,
                "image_input_ratio": 1, "input_feature_ratio": 1,
                "final_kernel_size": 3, "dropout_prob": 0.0,
                "pe_type": "sinusoidal", "norm_type": "TDRMSN",
                "gradient_checking": True
            }),
        ]
        self.fm_transformers = FMTransFormers(latent_channels=self.latent_channels, stage_configs=stage_configs)
        self.fm_transformers.set_cfgs(stage_configs)
        self.fm_transformers.to(self.device, dtype=self._dtype)

        # ---- VAE (FrozenAutoencoderKL，供 encode_moments) ----
        config_autoencoder = {
            'pretrained_path': '/vepfs/DI/yaqi/understand_gen/models/stable-diffusion/autoencoder_kl.pth',
            'scale_factor': 0.23010
        }
        ddconfig = dict(
            double_z=True, z_channels=4, resolution=256,
            in_channels=3, out_ch=3, ch=128, ch_mult=[1,2,4,4],
            num_res_blocks=2, attn_resolutions=[], dropout=0.0
        )
        self.autoencoder = FrozenAutoencoderKL(
            ddconfig, 4, config_autoencoder['pretrained_path'], config_autoencoder['scale_factor']
        ).to(self.device)

        # ---- FM path params ----
        self.time_step_sampler = LogitNormalSampler()
        self.sigma_min: float = 1e-5
        self.sigma_max: float = 1.0
        self.timescale: float = 1.0

        # ---- misc ----
        self.clip_target_source = clip_target_source  # 'pixels' or 'latents_decode'
        self.to(self._dtype)

    # ---------- helpers ----------
    @property
    def llm(self):
        return self.lmm.language_model

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self._dtype

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.connector.gradient_checkpointing = False

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()
        return self

    @torch.no_grad()
    def pixels_to_latents(self, x):
        scaling = self.vae.config.scaling_factor
        z = self.vae.encode(x)[0] * scaling
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        scaling = self.vae.config.scaling_factor
        x_rec = self.vae.decode(z / scaling)[0]
        return x_rec

    def prepare_forward_input(self, x, inputs_embeds=None, input_ids=None,
                              attention_mask=None, past_key_values=None):
        """Append x (queries) to tokens & build mask/positions."""
        b, l,

```
