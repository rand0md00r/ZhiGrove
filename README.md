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



--- 
``` python
diff --git a/src/models/openuni/fm_utils.py b/src/models/openuni/fm_utils.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/src/models/openuni/fm_utils.py
@@ -0,0 +1,84 @@
+import torch
+import torch.nn.functional as F
+
+IMAGENET_MEAN = (0.485, 0.456, 0.406)
+IMAGENET_STD  = (0.229, 0.224, 0.225)
+
+
+class TimeStepSampler:
+    """Base: sample timesteps t ∈ [0,1] for flow matching."""
+    def sample_time(self, x_start: torch.Tensor) -> torch.Tensor:
+        raise NotImplementedError
+
+
+class LogitNormalSampler(TimeStepSampler):
+    """t = sigmoid(N(mean, std^2)) per https://arxiv.org/pdf/2403.03206.pdf ."""
+    def __init__(self, normal_mean: float = 0.0, normal_std: float = 1.0):
+        self.normal_mean = float(normal_mean)
+        self.normal_std = float(normal_std)
+
+    @torch.no_grad()
+    def sample_time(self, x_start: torch.Tensor) -> torch.Tensor:
+        x_normal = torch.normal(
+            mean=self.normal_mean,
+            std=self.normal_std,
+            size=(x_start.shape[0],),
+            device=x_start.device,
+        )
+        return torch.sigmoid(x_normal)
+
+
+def _expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
+    """Expand 1D t to the shape of x (broadcastable)."""
+    t_expanded = t
+    while t_expanded.ndim < x.ndim:
+        t_expanded = t_expanded.unsqueeze(-1)
+    return t_expanded.expand_as(x)
+
+
+def psi(t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor,
+        sigma_min: float, sigma_max: float) -> torch.Tensor:
+    # Linear path used in your FM impl: ((t*(sigma_min/sigma_max - 1) + 1) * x + t * x1)
+    t = _expand_t(t, x)
+    return (t * (sigma_min / sigma_max - 1) + 1) * x + t * x1
+
+
+def Dt_psi(t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor,
+           sigma_min: float, sigma_max: float) -> torch.Tensor:
+    # ∂ψ/∂t for the path above
+    assert x.shape[0] == x1.shape[0]
+    return (sigma_min / sigma_max - 1) * x + x1
+
+
+def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
+    """
+    x: [B, L, D] or [B, D]; mask: [B, L] (True=valid).
+    Return: [B, D]
+    """
+    if x.ndim == 2:
+        return x
+    if mask is None:
+        return x.mean(dim=1)
+    m = mask.to(x.dtype)
+    m = m / (m.sum(dim=1, keepdim=True) + 1e-6)
+    return (x * m.unsqueeze(-1)).sum(dim=1)
diff --git a/src/models/openuni/loss_fns.py b/src/models/openuni/loss_fns.py
new file mode 100644
index 0000000..2222222
--- /dev/null
+++ b/src/models/openuni/loss_fns.py
@@ -0,0 +1,24 @@
+import torch
+
+def kl_loss(mu: torch.Tensor, log_var: torch.Tensor,
+            coef_mu: float = 0.3, power: int = 6) -> torch.Tensor:
+    """
+    Slightly modified KL: mu -> 0 via (coef_mu*mu)^power, var -> 1.
+    Return per-sample loss [B].
+    """
+    return -0.5 * torch.sum(1 + log_var - (coef_mu * mu) ** power - log_var.exp(), dim=1)
+
+
+def mse_mean_over_spatial(err: torch.Tensor, start_dim: int = 1,
+                          mask: torch.Tensor | None = None) -> torch.Tensor:
+    """
+    Mean-of-square over spatial/feature dims; optional mask over tokens.
+    Return per-sample loss [B].
+    """
+    if mask is not None:
+        return (err.pow(2).mean(dim=-1) * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6)
+    return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)
diff --git a/src/models/openuni/internvl3_sana_hf.py b/src/models/openuni/internvl3_sana_hf.py
index e6b2c77..3333333 100644
--- a/src/models/openuni/internvl3_sana_hf.py
+++ b/src/models/openuni/internvl3_sana_hf.py
@@ -1,190 +1,131 @@
-import math
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
-from torch.nn.modules.module import T
-from xtuner.registry import BUILDER
-from mmengine.model import BaseModel
-from mmengine.logging import print_log
-from torch.nn.utils.rnn import pad_sequence
-from xtuner.model.utils import guess_load_checkpoint
-from diffusers.pipelines.sana.pipeline_sana import SanaPipeline
-from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
-from peft import LoraConfig
-from src.models.connector import ConnectorConfig, ConnectorEncoder
-IMAGENET_MEAN = (0.485, 0.456, 0.406)
-IMAGENET_STD = (0.229, 0.224, 0.225)
-
-from torchvision import transforms
-from transformers import CLIPModel, CLIPProcessor
-
-from src.models.encoder import TransEncoder, Adaptor
-from src.models.encoder import FrozenCLIPEmbedder
-
-import open_clip
-
-from src.models.cliploss import ClipLoss
-from src.models.diffusion import Stage, FMTransFormers
-from src.models.encoder.autoencoder import FrozenAutoencoderKL
-
-from timm.models.layers import trunc_normal_, Mlp
-
-import numpy as np
-import random
-
-from ml_collections import ConfigDict
+import math
+import random
+import numpy as np
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+from torch.nn.modules.module import T
+from torch.nn.utils.rnn import pad_sequence
+
+from mmengine.model import BaseModel
+from mmengine.logging import print_log
+from xtuner.registry import BUILDER
+from xtuner.model.utils import guess_load_checkpoint
+
+from src.models.connector import ConnectorConfig, ConnectorEncoder
+from src.models.encoder import TransEncoder, FrozenCLIPEmbedder
+from src.models.diffusion import FMTransFormers
+from src.models.encoder.autoencoder import FrozenAutoencoderKL
+from src.models.cliploss import ClipLoss
+from ml_collections import ConfigDict
+
+from src.models.openuni.fm_utils import (
+    IMAGENET_MEAN, IMAGENET_STD,
+    LogitNormalSampler, masked_mean_pool,
+    psi, Dt_psi
+)
+from src.models.openuni.loss_fns import kl_loss, mse_mean_over_spatial
 
 def find_all_linear_names(model):
     lora_module_names = set()
     for name, module in model.named_modules():
         if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
             names = name.split('.')
             lora_module = names[0] if len(names) == 1 else names[-1]
             if lora_module == '0':
                 lora_module = 'to_out.0'
             lora_module_names.add(lora_module)
 
     return list(lora_module_names)
 
-
-class TimeStepSampler:
-    """
-    Abstract class to sample timesteps for flow matching.
-    """
-
-    def sample_time(self, x_start):
-        # In flow matching, time is in range [0, 1] and 1 indicates the original image; 0 is pure noise
-        # this convention is *REVERSE* of diffusion
-        raise NotImplementedError
-
-
-class LogitNormalSampler(TimeStepSampler):
-    def __init__(self, normal_mean: float = 0, normal_std: float = 1):
-        # follows https://arxiv.org/pdf/2403.03206.pdf
-        # sample from a normal distribution
-        # pass the output through standard logistic function, i.e., sigmoid
-        self.normal_mean = float(normal_mean)
-        self.normal_std = float(normal_std)
-
-    @torch.no_grad()
-    def sample_time(self, x_start):
-        x_normal = torch.normal(
-            mean=self.normal_mean,
-            std=self.normal_std,
-            size=(x_start.shape[0],),
-            device=x_start.device,
-        )
-        x_logistic = torch.nn.functional.sigmoid(x_normal)
-        return x_logistic
-
-
 class OpenUniInternVL3SANAHF(BaseModel):
     def __init__(self,
-                 lmm,
-                 transformer,
-                 train_scheduler,
-                 test_scheduler,
-                 vae,
-                 tokenizer,
-                 prompt_template,
-                 connector,
-                 num_queries=256,
-                 pretrained_pth=None,
-                 use_activation_checkpointing=True,
-                 lora_modules=None,  # ["to_k", "to_q", "to_v"],
-                 lora_rank=8,
-                 lora_alpha=8,
-                 freeze_lmm=True,
-                 freeze_transformer=True,
-                 vit_input_size=448,
-                 max_length=2048,
-                 proj_type='enc_proj',
-                 transencoder=None,
-                 dtype='bf16',
-                 open_clip_path=None,
-                 clip_target_source='pixels',
+                 lmm,
+                 vae,
+                 tokenizer,
+                 prompt_template,
+                 connector,
+                 # optional (kept for ctor compatibility)
+                 transformer=None,
+                 train_scheduler=None,
+                 test_scheduler=None,
+                 # arch
+                 num_queries=256,
+                 vit_input_size=448,
+                 max_length=2048,
+                 proj_type='enc_proj',
+                 transencoder=None,
+                 # switches
+                 pretrained_pth=None,
+                 freeze_lmm=True,
+                 freeze_transformer=True,
+                 use_activation_checkpointing=True,
+                 lora_modules=None,
+                 lora_rank=8,
+                 lora_alpha=8,
+                 # io & precision
+                 dtype='bf16',
+                 clip_target_source='pixels',
                  ):
         super().__init__()
-        self._dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
-        self.use_activation_checkpointing = use_activation_checkpointing
-
-        self.lmm = BUILDER.build(lmm)
-        if freeze_lmm:
-            self.lmm.requires_grad_(False)
-        self.freeze_lmm = freeze_lmm
-
-        # self.train_scheduler = BUILDER.build(train_scheduler)
-        # self.test_scheduler = BUILDER.build(test_scheduler)
-
-        self.transformer = BUILDER.build(transformer)   # 弃用，仅供跑通
-        if freeze_transformer:
-            self.transformer.requires_grad_(False)
-        self.freeze_transformer = freeze_transformer
-        if lora_modules is not None:
-            if lora_modules == 'auto':
-                lora_modules = find_all_linear_names(self.transformer)
-            # import pdb; pdb.set_trace()
-            transformer_lora_config = LoraConfig(
-                r=lora_rank,
-                lora_alpha=lora_alpha,
-                init_lora_weights="gaussian",
-                target_modules=lora_modules,
-            )
-            self.transformer.add_adapter(transformer_lora_config)
-
-        self.vae = BUILDER.build(vae)
-        self.vae.requires_grad_(False)
-
-        self.tokenizer = BUILDER.build(tokenizer)
-        self.prompt_template = prompt_template
-        self.vit_input_size = vit_input_size
-        self.max_length = max_length
-        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
-        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
-        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)
-
-        self.num_queries = num_queries
-        self.connector = ConnectorEncoder(ConnectorConfig(**connector))
-
-        self.proj_type = proj_type
-        if self.proj_type == 'proj_enc':
-            assert self.connector.config.hidden_size == self.transformer.config.caption_channels
-            self.projector = nn.Linear(
-                self.llm.config.hidden_size, self.connector.config.hidden_size)
-        elif self.proj_type == 'enc_proj':
-            assert self.connector.config.hidden_size == self.llm.config.hidden_size
-            self.projector = nn.Linear(
-                self.connector.config.hidden_size, self.transformer.config.caption_channels)
-        elif self.proj_type == 'proj_enc_proj':
-            self.projector = nn.ModuleList([
-                nn.Linear(self.llm.config.hidden_size, self.connector.config.hidden_size),
-                nn.Linear(self.connector.config.hidden_size, self.transformer.config.caption_channels)
-            ])
-        else:
-            raise ValueError(f'Unknown proj type: {self.proj_type}')
-
-        self.meta_queries = nn.Parameter(
-            torch.zeros(num_queries, self.llm.config.hidden_size))
-        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))
-
-        if use_activation_checkpointing:
-            self.llm.enable_input_require_grads()
-            self.gradient_checkpointing_enable()
-
-        if pretrained_pth is not None:
-            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
-            info = self.load_state_dict(pretrained_state_dict, strict=False)
-            print_log(f'Load pretrained weight from {pretrained_pth}')
-
-        self.resizer = transforms.Resize(256) # for clip
-
-        self.clip_model = CLIPModel.from_pretrained("/vepfs/DI/model-public/clip-vit-base-patch32")
-        self.clip_processor = CLIPProcessor.from_pretrained("/vepfs/DI/model-public/clip-vit-base-patch32")
-        self.clip_model.eval()
-        self.clip_model.requires_grad_(False)
-
-        self.frozen_clip = FrozenCLIPEmbedder()
-        self.frozen_clip.eval()
-
-        # 初始化 TransEncoder
-        self.context_encoder = TransEncoder(
-            d_model=transencoder.d_model,
-            N=transencoder.num_blocks,
-            num_token=transencoder.num_token,
-            head_num=transencoder.num_attention_heads,
-            d_ff=transencoder.hidden_dim,
-            latten_size=transencoder.latten_size,
-            down_sample_block=transencoder.down_sample_block,
-            dropout=transencoder.dropout_prob,
-            last_norm=transencoder.last_norm
-        )
-
-        self.open_clip_path = open_clip_path
-        self.open_clip, _, self.open_clip_preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained=self.open_clip_path)
-
-        # self.open_clip_output = Adaptor(input_dim=256 * 1536, tar_dim=4 * 64 * 64)
-        self.open_clip_output = Mlp(in_features=1024, 
-                            hidden_features=4*32*32, 
-                            out_features=8*32*32, 
-                            norm_layer=nn.LayerNorm,
-                        )
-
-        self.clip_loss = ClipLoss()
-
-        # 估计 latent 通道数（例如 SD VAE 通常4；也可以通过一次encode真值图像拿shape）
-        # self.latent_channels = getattr(self.autoencoder, "latent_channels", 4)
-        self.latent_channels = 8
+        # ---- precision ----
+        self._dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
+
+        # ---- backbone / tokenizer ----
+        self.lmm = BUILDER.build(lmm)
+        if freeze_lmm:
+            self.lmm.requires_grad_(False)
+        self.freeze_lmm = freeze_lmm
+
+        self.vae = BUILDER.build(vae)
+        self.vae.requires_grad_(False)
+
+        self.tokenizer = BUILDER.build(tokenizer)
+        self.prompt_template = prompt_template
+        self.vit_input_size = vit_input_size
+        self.max_length = max_length
+        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
+        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
+        self.register_buffer('vit_std',  torch.tensor(IMAGENET_STD),  persistent=False)
+
+        # ---- connector & projector ----
+        self.num_queries = num_queries
+        self.connector = ConnectorEncoder(ConnectorConfig(**connector))
+        self.proj_type = proj_type
+        # 保留占位（目前训练路径未使用 transformer captioner）
+        self.projector = nn.Identity()
+
+        # meta queries
+        self.meta_queries = nn.Parameter(torch.zeros(num_queries, self.llm.config.hidden_size))
+        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))
+
+        # activation ckpt
+        if use_activation_checkpointing:
+            self.llm.enable_input_require_grads()
+            self.gradient_checkpointing_enable()
+
+        # load ckpt
+        if pretrained_pth is not None:
+            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
+            _ = self.load_state_dict(pretrained_state_dict, strict=False)
+            print_log(f'Load pretrained weight from {pretrained_pth}')
+
+        # ---- text & image heads ----
+        self.frozen_clip = FrozenCLIPEmbedder().eval()  # 文本侧 encoder（保持冻结）
+        img_feat_dim = getattr(self.llm.config, 'hidden_size', 1536)
+        self.img_clip_head = nn.Linear(img_feat_dim, 8192, bias=False)
+        self.clip_logit_scale = nn.Parameter(torch.tensor(1/0.07).log())
+        self.clip_loss = ClipLoss()
+
+        # ---- TransEncoder (text VE) ----
+        assert transencoder is not None, "transencoder config is required"
+        self.context_encoder = TransEncoder(
+            d_model=transencoder.d_model,
+            N=transencoder.num_blocks,
+            num_token=transencoder.num_token,
+            head_num=transencoder.num_attention_heads,
+            d_ff=transencoder.hidden_dim,
+            latten_size=transencoder.latten_size,
+            down_sample_block=transencoder.down_sample_block,
+            dropout=transencoder.dropout_prob,
+            last_norm=transencoder.last_norm
+        )
 
-        # CrossFlow 风格的三阶段配置（与你给的一致）
+        # ---- Flow backbone ----
+        self.latent_channels = 8
         stage_configs = [
             ConfigDict({
                 "block_type": "TransformerBlock",
@@ -219,117 +160,64 @@ class OpenUniInternVL3SANAHF(BaseModel):
                 "gradient_checking": True
             })
         ]
-
-        # 三阶段主干
         self.fm_transformers = FMTransFormers(latent_channels=self.latent_channels, stage_configs=stage_configs)
         self.fm_transformers.set_cfgs(stage_configs)
         self.fm_transformers.to(self.device, dtype=self._dtype)
 
-
-        config_autoencoder = {
+        # ---- VAE (FrozenAutoencoderKL，供 encode_moments) ----
+        config_autoencoder = {
             'pretrained_path': '/vepfs/DI/yaqi/understand_gen/models/stable-diffusion/autoencoder_kl.pth',
             'scale_factor': 0.23010
         }
-
         ddconfig = dict(
             double_z=True,
             z_channels=4,
             resolution=256,
             in_channels=3,
             out_ch=3,
             ch=128,
             ch_mult=[1, 2, 4, 4],
             num_res_blocks=2,
             attn_resolutions=[],
             dropout=0.0
         )
-        self.autoencoder = FrozenAutoencoderKL(ddconfig, 4, config_autoencoder['pretrained_path'], config_autoencoder['scale_factor'])  # embed_dim设为2，不行
+        self.autoencoder = FrozenAutoencoderKL(ddconfig, 4,
+                                               config_autoencoder['pretrained_path'],
+                                               config_autoencoder['scale_factor'])
         self.autoencoder.to(self.device)
 
-        self.time_step_sampler = LogitNormalSampler()
+        # ---- FM path params ----
+        self.time_step_sampler = LogitNormalSampler()
         self.sigma_min: float = 1e-5
         self.sigma_max: float = 1.0
         self.timescale: float = 1.0
-
-        self.clip_target_source = clip_target_source # 'pixels' or 'latents_decode'
-
-        self.clip_logit_scale = nn.Parameter(torch.tensor(1/0.07).log())
-        
-        img_feat_dim = getattr(self.llm.config, 'hidden_size', 1536)
-        self.img_clip_head = nn.Linear(img_feat_dim, 8192, bias=False)
-        
-
+        self.clip_target_source = clip_target_source  # 'pixels' or 'latents_decode'
         self.to(self._dtype)
-        
-    def _masked_mean_pool(self, x, mask=None):
-        """
-        x: [B, L, D] or [B, D]
-        mask: [B, L] (bool) 1/True means valid
-        returns: [B, D]
-        """
-        if x.ndim == 2:
-            return x
-        
-        if mask is None:
-            return x.mean(dim=1)
-        
-        # 兼容True/False 或 0/1
-        m = mask.to(x.dtype)
-        m = m / (m.sum(dim=1, keepdim=True) + 1e-6)
-        return (x * m.unsqueeze(-1)).sum(dim=1)
-
-    def psi(self, t, x, x1):
-        assert (
-            t.shape[0] == x.shape[0]
-        ), f"Batch size of t and x does not agree {t.shape[0]} vs. {x.shape[0]}"
-        assert (
-            t.shape[0] == x1.shape[0]
-        ), f"Batch size of t and x1 does not agree {t.shape[0]} vs. {x1.shape[0]}"
-        assert t.ndim == 1
-        t = self.expand_t(t, x)
-        return (t * (self.sigma_min / self.sigma_max - 1) + 1) * x + t * x1
-
-    def Dt_psi(self, t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor):
-        assert x.shape[0] == x1.shape[0]
-        return (self.sigma_min / self.sigma_max - 1) * x + x1
-
-    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
-        t_expanded = t
-        while t_expanded.ndim < x.ndim:
-            t_expanded = t_expanded.unsqueeze(-1)
-        return t_expanded.expand_as(x)
 
     def _multi_scale_targets(self, target_velocity, levels=1):
         """
         如果网络输出多尺度列表，给每个尺度下采样目标。
         这里用 average pooling/nearest 插值都行；用 interpolate 更稳妥。
         """
         if levels == 1:
             return [target_velocity]
         targets = [target_velocity]
         h, w = target_velocity.shape[-2], target_velocity.shape[-1]
         for i in range(1, levels):
             # 逐级 /2 下采样（可按你的 backbone 分辨率金字塔调）
             scale = 2 ** i
             nh, nw = max(1, h // scale), max(1, w // scale)
             t_i = F.interpolate(targets[0], size=(nh, nw), mode='bilinear', align_corners=False)
             targets.append(t_i)
         return targets
 
     def mos(self, err, start_dim=1, con_mask=None):  # mean of square
         if con_mask is not None:
             return (err.pow(2).mean(dim=-1) * con_mask).sum(dim=-1) / con_mask.sum(dim=-1)
         else:
             return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)
 
-
     def llm2dit(self, x):
         if self.proj_type == 'proj_enc':
             return self.connector(self.projector(x))
         elif self.proj_type == 'enc_proj':
             return self.projector(self.connector(x))
         elif self.proj_type == 'proj_enc_proj':
             return self.projector[1](self.connector(self.projector[0](x)))
         else:
             raise ValueError(f'Unknown proj type: {self.proj_type}')
@@ -342,7 +230,7 @@ class OpenUniInternVL3SANAHF(BaseModel):
         return self.llm.dtype
 
     def train(self: T, mode: bool = True) -> T:
         super().train(mode=mode)
-        if self.vae is not None:
+        if self.vae is not None:
             self.vae.train(mode=False)
         if not mode:
             self.gradient_checkpointing_disable()
 
@@ -373,7 +261,7 @@ class OpenUniInternVL3SANAHF(BaseModel):
         inputs = dict(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       position_ids=position_ids,
                       past_key_values=past_key_values)
 
         return inputs
@@ -386,15 +274,17 @@ class OpenUniInternVL3SANAHF(BaseModel):
         else:
             raise NotImplementedError
 
     def compute_loss(self, data_dict):
         losses = {}
         for data_type in ['text2image', 'image2image']:
             if data_type in data_dict:
                 losses[f'loss_{data_type}'] = getattr(self, f'{data_type}_loss')(data_dict[data_type])
         if len(losses) == 0:
-            if 'pixel_values_src' in data_dict:
-                losses[f'loss_image2image'] = self.image2image_loss(data_dict)
-            else:
-                losses[f'loss_text2image'] = self.text2image_loss(data_dict)
+            if 'pixel_values' in data_dict or 'image_latents' in data_dict:
+                losses['loss_text2image'] = self.text2image_loss(data_dict)
+            else:
+                raise NotImplementedError("Only text2image path is implemented after refactor.")
 
         return losses
 
     @torch.no_grad()
     def get_semantic_features(self, pixel_values):
         # pixel_values: [-1, 1]
@@ -405,144 +295,137 @@ class OpenUniInternVL3SANAHF(BaseModel):
 
         pixel_values = F.interpolate(pixel_values, size=(self.vit_input_size, self.vit_input_size),
                                      mode='bilinear')
         vit_embeds = self.lmm.extract_feature(pixel_values)
 
         return vit_embeds
 
     @torch.no_grad()
     def prepare_text_conditions(self, prompt, cfg_prompt=None):
         if cfg_prompt is None:
             cfg_prompt = self.prompt_template['CFG']
         else:
             cfg_prompt = self.prompt_template['GENERATION'].format(input=cfg_prompt.strip())
         prompt = self.prompt_template['GENERATION'].format(input=prompt.strip())
@@ -568,187 +451,153 @@ class OpenUniInternVL3SANAHF(BaseModel):
 
         return dict(input_ids=input_ids.to(self.device),
                     attention_mask=attention_mask.to(self.device))
 
     def text2image_loss(self, data_dict):
-
-        if 'pixel_values' in data_dict:
-            # debug 阶段，需要pixel_values，需要image_latents，但只有pixel_values
-            pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
-            image_latents = self.pixels_to_latents(pixel_values)
-        elif self.clip_target_source == 'latents_decode' and 'image_latents' in data_dict:
-            # train 阶段，预处理得到image_latents计算diff loss，调用lmm从picel_value得到image_emb(+ mlp)
-            pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
-            image_latents = data_dict['image_latents'].to(dtype=self._dtype, device=self.device)
-        else:
-            raise ValueError("No pixel_values found and latents_decode not enabled.")
-
-        vit_embeds = self.get_semantic_features(pixel_values)       # [B, L_img, D_vit]
-        img_vec = self._masked_mean_pool(vit_embeds)                # [B, D_vit]
-        img_vec = self.img_clip_head(img_vec).to(self._dtype)    # [B, D_clip] 统一维度
-
-        logit_scale = self.clip_logit_scale.exp()                   # 温度
-
-        b, _, height, weight = image_latents.shape
-
-        input_ids = data_dict['input_ids'].to(self.device)
-        attention_mask = data_dict['attention_mask'].to(self.device)
-        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)
-
-        inputs = self.prepare_forward_input(x=hidden_states,
-                                            input_ids=input_ids,
-                                            attention_mask=attention_mask)
-
-        output = self.llm.model(**inputs, return_dict=True)
-        hidden_states = output.last_hidden_state[:, -self.num_queries:]
-        # hidden_states = self.llm2dit(hidden_states)
-        B, L, _= hidden_states.shape
-        attention_mask_queries = attention_mask.new_ones(B, L)
-
-        x0, mu, log_var = self.text_ve_encoder(hidden_states, attention_mask_queries)  # hidden_states.shape=[bs, 256, 1536]
-
-        loss_clip = self.compute_clip_loss(x0, img_vec, logit_scale)
-
-
-        loss_kl = self.compute_kl_loss(mu, log_var)
-        kl_loss_weight = 1e-2 # 0.0005
-
-        loss_mlp = loss_clip + loss_kl * kl_loss_weight
-
-        loss_diff = self.diff_loss(x0, data_dict, img_vec)
-
-
-        print(f"clip: {loss_clip}")
-
-        print(f"kl: {loss_kl}")
-
-        print(f"diff: {loss_diff}")
-
-
-        return loss_diff + loss_mlp
+        """
+        清爽版 T2I 损失：
+          1) 图像路径：像素 -> VLM 视觉 token -> 池化+线性 -> v_img
+          2) 文本路径：meta queries + LLM -> q_tokens -> TransEncoder -> z_text(重参数)
+          3) 三项损失：Flow Matching(z_text vs x1)、CLIP 对齐(z_text vs v_img)、KL 正则(mu/logvar)
+        """
+        # A. 图像像素 / latents
+        if 'pixel_values' in data_dict:
+            x_img = data_dict['pixel_values'].to(self.device, dtype=self._dtype)   # [-1, 1]
+            z_img = self.pixels_to_latents(x_img)
+        elif self.clip_target_source == 'latents_decode' and 'image_latents' in data_dict:
+            z_img = data_dict['image_latents'].to(self.device, dtype=self._dtype)
+            x_img = self.latents_to_pixels(z_img)
+        else:
+            raise ValueError("text2image_loss: need 'pixel_values' or 'image_latents'(with latents_decode).")
+
+        # B. 图像向量 v_img
+        v_img_tokens = self.get_semantic_features(x_img)                       # [B, L_img, D_vlm]
+        v_img = masked_mean_pool(v_img_tokens)                                 # [B, D_vlm]
+        v_img = self.img_clip_head(v_img).to(self._dtype)                      # [B, D_align]
+        logit_scale = self.clip_logit_scale.exp()
+
+        # C. 文本 queries -> VE -> z_text
+        b = z_img.shape[0]
+        input_ids = data_dict['input_ids'].to(self.device)
+        attn_mask = data_dict['attention_mask'].to(self.device)
+
+        q_meta = self.meta_queries[None].expand(b, self.num_queries, -1)       # [B, Q, D]
+        llm_in = self.prepare_forward_input(x=q_meta, input_ids=input_ids, attention_mask=attn_mask)
+        llm_out = self.llm.model(**llm_in, return_dict=True).last_hidden_state
+        q_tokens = llm_out[:, -self.num_queries:]                               # [B, Q, D]
+        q_mask   = attn_mask.new_ones(q_tokens.shape[:2])                       # [B, Q]
+
+        z_text, mu, log_var = self.text_ve_encoder(q_tokens, q_mask)            # [B, Q, *]
+
+        # D. 三项损失
+        loss_flow = self.flow_matching_loss(z_text, x_img)
+        loss_clip = self.compute_clip_loss(z_text, v_img, logit_scale)
+        loss_kl   = self.compute_kl_loss(mu, log_var).mean()
+
+        kl_weight = 1e-2
+        total = loss_flow + loss_clip + kl_weight * loss_kl
+        return total
     
     def text_ve_encoder(self, token_embedding, token_mask):
-        token_embedding = token_embedding.to(dtype=self._dtype)
-        token_mask = token_mask.to(dtype=self._dtype, device=self.device)
-        
-        output = self.context_encoder(token_embedding, token_mask)
-        mu, log_var = torch.chunk(output, 2, dim=-1)
-
-        def _reparameterize(mu, logvar):
-            std = torch.exp(0.5 * logvar)
-            eps = torch.randn_like(std)
-            return eps * std + mu
-
-        z = _reparameterize(mu, log_var)
-
-        return [z, mu, log_var]
+        token_embedding = token_embedding.to(dtype=self._dtype)
+        token_mask = token_mask.to(dtype=self._dtype, device=self.device)
+        output = self.context_encoder(token_embedding, token_mask)
+        mu, log_var = torch.chunk(output, 2, dim=-1)
+        std = torch.exp(0.5 * log_var)
+        eps = torch.randn_like(std)
+        z = eps * std + mu
+        return z, mu, log_var
 
     def compute_clip_loss(self, x0, recon_gt_clip, logit_scale):
-
-        image_features = recon_gt_clip / recon_gt_clip.norm(dim=-1, keepdim=True)
-        text_features = x0 / x0.norm(dim=-1, keepdim=True)
-        recons_loss = self.clip_loss(image_features, text_features, logit_scale)
-
-        return recons_loss
+        z_text_vec = masked_mean_pool(x0)
+        image_features = recon_gt_clip / (recon_gt_clip.norm(dim=-1, keepdim=True) + 1e-6)
+        text_features  = z_text_vec     / (z_text_vec.norm(dim=-1, keepdim=True) + 1e-6)
+        return self.clip_loss(image_features, text_features, logit_scale)
 
     def compute_kl_loss(self, mu, log_var):
-        kld_loss = -0.5 * torch.sum(1 + log_var - (0.3 * mu) ** 6 - log_var.exp(), dim = 1) # slightly different KL loss function: mu -> 0 [(0.3*mu) ** 6] and var -> 1
-        return kld_loss
+        return kl_loss(mu, log_var)
     
-    def diff_loss(self, x0, data_dict, recon_gt_clip, indicator=None):
-        # image_features = recon_gt_clip / recon_gt_clip.norm(dim=-1, keepdim=True)
-
-        # x1 目标图像的分布, 调用autoencoder进行VE编码
-
-        pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
-        pixel_values = F.interpolate(pixel_values, size=(256, 256), mode='bilinear', align_corners=False)
-        x1 = self.autoencoder(pixel_values, fn='encode_moments').squeeze(0)
-        x1 = x1.to(dtype=self._dtype, device=self.device)
-        B = x1.shape[0]
-
-        # 采样连续时间 计算 log-SNR / a
-        t = self.time_step_sampler.sample_time(x1)
-        log_snr = 4.0 - 8.0 * t
-        alpha = torch.sigmoid(log_snr)
-        sqrt_a = torch.sqrt(alpha)
-        sqrt_lma = torch.sqrt(1.0 - alpha)
-
-        # 构造路径与目标速度
-        x0 = x0.reshape(x1.shape)
-        # x_t = sqrt_a * x1 + sqrt_lma * x0
-        x_t = self.psi(t, x=x0, x1=x1)
-        x_t = x_t.to(dtype=self._dtype, device=self.device)
-
-        target_velocity = self.Dt_psi(t, x=x0, x1=x1)
-
-        null_indicator = torch.from_numpy(np.array([random.random() < 0.1 for _ in range(x1.shape[0])])).to(x1.device)
-        if null_indicator.sum()<=1:
-            null_indicator[null_indicator==True] = False
-            assert null_indicator.sum() == 0
-            pass
-        else:
-            target_null = x1[null_indicator]
-            target_null = torch.cat((target_null[1:], target_null[:1]))
-            x1[null_indicator] = target_null
-
-
-        # ===== 4) 前向：FM transformer 网络 =====
-        preds = self.fm_transformers(x_t, log_snr=log_snr, null_indicator=null_indicator)
-        # if not isinstance(preds, (list, tuple)):
-        #     preds = [preds]
-
-        loss_diff = self.mos(preds[-1] - target_velocity)
-        return loss_diff
-
-
-    def image2image_loss(self, data_dict):
-
-        pixel_values_src = data_dict['pixel_values_src'].to(dtype=self.dtype, device=self.device)
-        vit_embeds = self.get_semantic_features(pixel_values_src)
-        vit_embeds.requires_grad = True
-
-        pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
-        image_latents = self.pixels_to_latents(pixel_values)
-
-        b, _, height, weight = image_latents.shape
-
-        input_ids = data_dict['input_ids'].to(self.device)
-        attention_mask = data_dict['attention_mask'].to(self.device)
-
-        inputs_embeds = vit_embeds.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
-        inputs_embeds[input_ids == self.image_token_id] = vit_embeds.flatten(0, 1)
-        inputs_embeds[input_ids != self.image_token_id] = self.llm.get_input_embeddings()(
-            input_ids[input_ids != self.image_token_id]
-        )
-
-        max_length = self.max_length
-        if inputs_embeds.shape[1] > max_length:
-            inputs_embeds = inputs_embeds[:, -max_length:]
-            attention_mask = attention_mask[:, -max_length:]
-
-        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)
-
-        inputs = self.prepare_forward_input(x=hidden_states,
-                                            inputs_embeds=inputs_embeds,
-                                            attention_mask=attention_mask)
-
-        output = self.llm.model(**inputs, return_dict=True)
-        hidden_states = output.last_hidden_state[:, -self.num_queries:]
-        hidden_states = self.llm2dit(hidden_states)
-
-        loss_diff = self.diff_loss(model_input=image_latents,
-                                   prompt_embeds=hidden_states,
-                                   prompt_attention_mask=None)
-
-        return loss_diff
+    def flow_matching_loss(self, z_text: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
+        """
+        使用 encode_moments 得到目标分布 x1，构造路径 ψ(t,x0,x1)，
+        用 FMTransformers 预测速度，与目标速度 Dtψ 计算 MSE。
+        """
+        x_img_256 = F.interpolate(x_img, size=(256, 256), mode='bilinear', align_corners=False)
+        x1 = self.autoencoder(x_img_256.to(self.device, dtype=self._dtype), fn='encode_moments').squeeze(0)
+        x1 = x1.to(self._dtype, self.device)
+
+        t = self.time_step_sampler.sample_time(x1)
+        log_snr = 4.0 - 8.0 * t
+        x0 = z_text.reshape(x1.shape).to(self._dtype, self.device)
+
+        x_t = psi(t, x0, x1, self.sigma_min, self.sigma_max).to(self._dtype, self.device)
+        v_target = Dt_psi(t, x0, x1, self.sigma_min, self.sigma_max)
+
+        null_indicator = torch.from_numpy(
+            np.array([random.random() < 0.1 for _ in range(x1.shape[0])])
+        ).to(x1.device)
+        if null_indicator.sum() > 1:
+            target_null = x1[null_indicator]
+            target_null = torch.cat((target_null[1:], target_null[:1]))
+            x1[null_indicator] = target_null
+
+        preds = self.fm_transformers(x_t, log_snr=log_snr, null_indicator=null_indicator)
+        loss = mse_mean_over_spatial(preds[-1] - v_target).mean()
+        return loss
-
-    @torch.no_grad()
-    def generate(self,
-                 input_ids=None,
-                 inputs_embeds=None,
-                 attention_mask=None,
-                 cfg_scale=4.5,
-                 num_steps=20,
-                 generator=None,
-                 height=512,
-                 width=512,
-                 progress_bar=True,
-                 **kwargs):
-
-        if inputs_embeds is None and input_ids is not None:
-            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
-
-        bsz = attention_mask.shape[0]
-
-        assert bsz % 2 == 0
-
-        hidden_states = self.meta_queries[None].expand(bsz, self.num_queries, -1)
-        inputs = self.prepare_forward_input(x=hidden_states,
-                                            inputs_embeds=inputs_embeds,
-                                            attention_mask=attention_mask)
-
-        output = self.llm.model(**inputs, return_dict=True)
-        hidden_states = output.last_hidden_state[:, -self.num_queries:]
-
-        hidden_states = self.llm2dit(hidden_states)
-        attention_mask = torch.ones(bsz, self.num_queries, device=self.device, dtype=torch.bool)
-
-        pipeline = SanaPipeline(transformer=self.transformer,
-                                scheduler=self.test_scheduler,
-                                vae=self.vae, text_encoder=None, tokenizer=None
-                                )
-        pipeline.set_progress_bar_config(disable=not progress_bar)
-
-        samples = pipeline(
-            negative_prompt=None,
-            height=height,
-            width=width,
-            prompt_embeds=hidden_states[:bsz // 2],
-            prompt_attention_mask=attention_mask[:bsz // 2],
-            negative_prompt_embeds=hidden_states[bsz // 2:],
-            negative_prompt_attention_mask=attention_mask[bsz // 2:],
-            num_inference_steps=num_steps,
-            generator=generator,
-            complex_human_instruction=None,
-            output_type='latent',
-            use_resolution_binning=False,
-            guidance_scale=cfg_scale,
-        ).images.to(self._dtype)
-
-        return self.latents_to_pixels(samples)
-
-    # def diff_loss(self, model_input, prompt_embeds, prompt_attention_mask):
-    #     ...
-
-    # def get_sigmas(self, timesteps, n_dim=4):
-    #     ...
+
+    # NOTE: image2image / generate 相关逻辑已移除，后续如需可在单独推理类实现
diff --git a/src/datasets/text2image/caption_datasets.py b/src/datasets/text2image/caption_datasets.py
index 1234567..abcdef0 100644
--- a/src/datasets/text2image/caption_datasets.py
+++ b/src/datasets/text2image/caption_datasets.py
@@ -1,6 +1,8 @@
 import os
 import torch
+import random
 from PIL import Image
+from typing import Optional
 # ... 其他 import 省略
 
 class CaptionDataset:
@@ -12,12 +14,30 @@ class CaptionDataset:
-    def __init__(self, image_size, cap_source, data_path, cap_folder,
-                 image_folder=None, image_latents_folder=None,
-                 unconditional=0.1, prompt_template=None,
-                 ceph_folder=None, ceph_config=None,
-                 tokenizer=None, max_length=128,
-                 debug=False, latents_ceph_folder=None, image_tokens_folder=None):
+    def __init__(self, image_size, cap_source, data_path, cap_folder,
+                 image_folder=None, image_latents_folder=None,
+                 unconditional=0.1, prompt_template=None,
+                 ceph_folder=None, ceph_config=None,
+                 tokenizer=None, max_length=128,
+                 debug=False, latents_ceph_folder=None, image_tokens_folder=None,
+                 # 新增：兼容预编码且同时需要像素（给 VLM/CLIP & FM encode_moments）
+                 load_pixels_along_latents: bool = False,
+                 # 新增：返回原始 prompt（给文本 CLIP 编码或日志）
+                 return_prompt: bool = False,
+                 # 可选：多分辨率（若你的 _process_image 支持基于 self.image_size 读图）
+                 multi_resolution: bool = False,
+                 image_size_choices: Optional[list] = None,
+                 ):
         self.image_size = image_size
         self.cap_source = cap_source
         self.data_path = data_path
         self.cap_folder = cap_folder
         self.image_folder = image_folder
         self.image_latents_folder = image_latents_folder
+        self.load_pixels_along_latents = load_pixels_along_latents
+        self.return_prompt = return_prompt
+        self.multi_resolution = multi_resolution
+        self.image_size_choices = image_size_choices or [image_size]
         self.unconditional = unconditional
         self.prompt_template = prompt_template
         self.ceph_folder = ceph_folder
@@ -44,19 +64,56 @@ class CaptionDataset:
     def __getitem__(self, idx):
         if self.debug:
             idx = 0
         try:
             data_sample = self.data_list[idx]
 
-            if self.image_tokens_folder is not None:
+            # 可选：多分辨率（若 _process_image 依赖 self.image_size）
+            orig_size = getattr(self, 'image_size', None)
+            if self.multi_resolution and hasattr(self, 'image_size'):
+                self.image_size = random.choice(self.image_size_choices)
+
+            if self.image_tokens_folder is not None:
                 image_tokens = torch.load(os.path.join(self.image_tokens_folder,
                                                        data_sample['image'] + '.pt')).long()
-                data = dict(image_tokens=image_tokens)
+                data = dict(image_tokens=image_tokens)
             elif self.latents_ceph_folder is not None:
                 image_latents = torch.load(
                     self._read_ceph(
                         os.path.join(
                             self.latents_ceph_folder, data_sample['image'] + '.pt'
                         )
                     )
                 )
-                data = dict(image_latents=image_latents)
+                data = dict(image_latents=image_latents)
+                # 同步读像素（可选）
+                if self.load_pixels_along_latents and self.image_folder is not None:
+                    image = self._read_image(data_sample['image']).convert('RGB')
+                    data.update(self._process_image(image))
             elif self.image_latents_folder is not None:
                 image_latents = torch.load(os.path.join(self.image_latents_folder,
                                                         data_sample['image'] + '.pt'))
-                data = dict(image_latents=image_latents)
+                data = dict(image_latents=image_latents)
+                # 同步读像素（可选）
+                if self.load_pixels_along_latents and self.image_folder is not None:
+                    image = self._read_image(data_sample['image']).convert('RGB')
+                    data.update(self._process_image(image))
             else:
                 image = self._read_image(data_sample['image']).convert('RGB')
                 data = self._process_image(image)
 
-            caption = self._read_json(data_sample['annotation'])[self.cap_source]
+            caption = self._read_json(data_sample['annotation'])[self.cap_source]
 
-            data.update(self._process_text(caption))
+            data.update(self._process_text(caption))
+            if self.return_prompt:
+                data.update(prompt=caption)
             data.update(image_dir=self.image_folder, image_file=data_sample['image'],
                         type='text2image')
 
-            return data
+            # 复原 image_size
+            if orig_size is not None and hasattr(self, 'image_size'):
+                self.image_size = orig_size
+            return data
 
         except Exception as e:
             print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
             return self._retry()
diff --git a/src/datasets/collate_functions.py b/src/datasets/collate_functions.py
index 89abcd0..fedcba9 100644
--- a/src/datasets/collate_functions.py
+++ b/src/datasets/collate_functions.py
@@ -1,5 +1,6 @@
 from typing import Dict, Sequence
 import torch
+from torch.nn.utils.rnn import pad_sequence
 from functools import partial
 
 DEFAULT_PAD_TOKEN_INDEX = 0
 IGNORE_INDEX = -100
@@ -80,6 +81,64 @@ def collate_func_gen_latents(instances: Sequence[Dict],
     return {'data': data_dict, 'data_samples': None}
 
+
+def collate_func_gen_latents_with_prompt(instances: Sequence[Dict],
+                                         pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
+    """
+    兼容：
+      - 必选：image_latents, input_ids
+      - 可选：pixel_values（若 load_pixels_along_latents=True）
+      - 可选：prompt（原始字符串，用于日志或 CLIP 文本侧）
+    """
+    image_latents, input_ids, input_lengths = [], [], []
+    pixel_values, prompts = [], []
+
+    for ex in instances:
+        if 'image_latents' not in ex:
+            # 容错：允许只有 pixel_values 的样本（如 debug 阶段）
+            if 'pixel_values' in ex:
+                # 放个占位零张量，避免堆叠报错（后续模型会用像素→latents）
+                image_latents.append(torch.zeros(1))
+            else:
+                continue
+        else:
+            image_latents.append(ex['image_latents'])
+
+        input_lengths.append(len(ex['input_ids']))
+        input_ids.append(ex['input_ids'])
+
+        if 'pixel_values' in ex:
+            pixel_values.append(ex['pixel_values'])
+        if 'prompt' in ex:
+            prompts.append(ex['prompt'])
+
+    # pad input_ids
+    input_ids = [torch.as_tensor(x, dtype=torch.long) for x in input_ids]
+    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
+
+    attention_mask = torch.zeros_like(input_ids).bool()
+    for i, L in enumerate(input_lengths):
+        attention_mask[i, :L] = True
+
+    data_dict = dict(
+        image_latents=torch.stack(image_latents) if len(image_latents) > 0 and image_latents[0].numel() > 1 else None,
+        input_ids=input_ids,
+        attention_mask=attention_mask
+    )
+
+    if len(pixel_values) > 0:
+        data_dict['pixel_values'] = torch.stack(pixel_values)
+    # prompt 保持原始 list 形式，避免强制编码
+    if len(prompts) > 0:
+        data_dict['prompt'] = prompts
+
+    return {'data': data_dict, 'data_samples': None}
+
+
 class CollateConcat(object):
     def __init__(self, collate_fns, keys):
         self.keys = keys
         self.collate_fns = {}
         for key, collate_fn in zip(keys, collate_fns):
             func = collate_fn.pop('type')


```
