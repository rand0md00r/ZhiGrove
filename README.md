# stunning-palm-tree
---
2025年8月11日10:25:24

``` python
# src/models/openuni/internvl3_sana_hf.py
# Refactored on 2025-08-11
#
# 变更要点：
# - 使用冻结的 VLM(self.lmm) 进行图像嵌入（不再依赖 open_clip / CLIPModel）。
# - 抽离通用工具到 fm_utils.py / loss_fns.py（需确保这两个文件已存在）。
# - 精简并重写 text2image_loss：清晰三段式（图像向量、文本 VE、三项损失）。
# - 统一 dtype：通过 self._dtype 控制（bf16 / fp32）。
# - 移除未使用的开发残留（generate / image2image 等）。

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T

from mmengine.model import BaseModel
from mmengine.logging import print_log
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from ml_collections import ConfigDict

from src.models.connector import ConnectorConfig, ConnectorEncoder
from src.models.encoder import TransEncoder
from src.models.diffusion import FMTransFormers
from src.models.encoder.autoencoder import FrozenAutoencoderKL
from src.models.cliploss import ClipLoss

from src.models.openuni.fm_utils import (
    IMAGENET_MEAN, IMAGENET_STD,
    LogitNormalSampler, masked_mean_pool,
    psi, Dt_psi
)
from src.models.openuni.loss_fns import kl_loss, mse_mean_over_spatial


class OpenUniInternVL3SANAHF(BaseModel):
    """
    冻结的 VLM(self.lmm) 提供视觉 token；文本侧用 TransEncoder 估计均值/方差并重参数化得到 z_text，
    通过 Flow Matching 与 autoencoder 的目标分布对齐，同时做 CLIP 风格的对比损失与 KL 正则。
    """

    def __init__(self,
                 # 必需
                 lmm,
                 vae,
                 tokenizer,
                 prompt_template,
                 connector,
                 # 兼容（不在训练路径中使用，仅保留构造兼容）
                 transformer=None,
                 train_scheduler=None,
                 test_scheduler=None,
                 # 结构超参
                 num_queries=256,
                 vit_input_size=448,
                 max_length=2048,
                 proj_type='enc_proj',
                 transencoder=None,
                 # 训练开关
                 pretrained_pth=None,
                 freeze_lmm=True,
                 freeze_transformer=True,
                 use_activation_checkpointing=True,
                 lora_modules=None,
                 lora_rank=8,
                 lora_alpha=8,
                 # 精度与输入选择
                 dtype='bf16',                  # 'bf16' | 'fp32'
                 clip_target_source='pixels',   # 'pixels' | 'latents_decode'
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

        # image context token
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            prompt_template['IMG_CONTEXT_TOKEN']
        )
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std',  torch.tensor(IMAGENET_STD),  persistent=False)

        # ---- connector & projector（保留占位，当前训练路径未使用）----
        self.num_queries = num_queries
        self.connector = ConnectorEncoder(ConnectorConfig(**connector))
        self.proj_type = proj_type
        self.projector = nn.Identity()

        # meta queries
        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size)
        )
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))

        # 激活检查点
        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        # 预训练权重
        if pretrained_pth is not None:
            state = guess_load_checkpoint(pretrained_pth)
            _ = self.load_state_dict(state, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        # ---- 对比学习头（图像侧）；温度参数 ----
        img_feat_dim = getattr(self.llm.config, 'hidden_size', 1536)
        # 将 VLM 视觉 token 池化后的向量映射到对齐空间（与 z_text 的维度保持一致）
        self.img_clip_head = nn.Linear(img_feat_dim, 8192, bias=False)
        self.clip_logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        self.clip_loss = ClipLoss()

        # ---- 文本 VE ----
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
        self.latent_channels = 8  # 与你现有实现对齐
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
        self.fm_transformers = FMTransFormers(
            latent_channels=self.latent_channels, stage_configs=stage_configs
        )
        self.fm_transformers.set_cfgs(stage_configs)
        self.fm_transformers.to(self.device, dtype=self._dtype)

        # ---- FrozenAutoencoderKL（用于 encode_moments 监督）----
        config_autoencoder = {
            'pretrained_path': '/vepfs/DI/yaqi/understand_gen/models/stable-diffusion/autoencoder_kl.pth',
            'scale_factor': 0.23010
        }
        ddconfig = dict(
            double_z=True, z_channels=4, resolution=256,
            in_channels=3, out_ch=3, ch=128, ch_mult=[1, 2, 4, 4],
            num_res_blocks=2, attn_resolutions=[], dropout=0.0
        )
        self.autoencoder = FrozenAutoencoderKL(
            ddconfig, 4, config_autoencoder['pretrained_path'], config_autoencoder['scale_factor']
        ).to(self.device)

        # ---- Flow path 超参 ----
        self.time_step_sampler = LogitNormalSampler()
        self.sigma_min: float = 1e-5
        self.sigma_max: float = 1.0
        self.timescale: float = 1.0

        # ---- 输入选择 ----
        self.clip_target_source = clip_target_source  # 'pixels' or 'latents_decode'

        self.to(self._dtype)

    # ---------------- Properties ----------------
    @property
    def llm(self):
        return self.lmm.language_model

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self._dtype

    # ---------------- Train / Grad-ckpt ----------------
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

    # ---------------- VAE utils ----------------
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

    # ---------------- LLM utils ----------------
    def prepare_forward_input(self, x, inputs_embeds=None, input_ids=None,
                              attention_mask=None, past_key_values=None):
        """将查询向量拼接到 token embeddings 末尾，并构造 mask/position。"""
        b, l, _ = x.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        if past_key_values is not None:
            inputs_embeds = x
            position_ids = position_ids[:, -l:]
        else:
            if inputs_embeds is None:
                input_ids = input_ids.to(self.device)
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        return dict(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values)

    # ---------------- VLM vision ----------------
    @torch.no_grad()
    def get_semantic_features(self, x_img: torch.Tensor):
        """
        输入像素值范围 [-1,1]，归一化到 VLM 视觉分布并 resize 到 vit_input_size。
        返回 VLM 的视觉 token 特征 [B, L_img, D].
        """
        x = (x_img + 1.0) / 2.0
        x = x - self.vit_mean.view(1, 3, 1, 1)
        x = x / self.vit_std.view(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.vit_input_size, self.vit_input_size), mode='bilinear')
        v_img_tokens = self.lmm.extract_feature(x)
        return v_img_tokens

    # ---------------- Loss blocks ----------------
    def compute_clip_loss(self, z_text: torch.Tensor, v_img: torch.Tensor, logit_scale: torch.Tensor):
        """
        z_text: [B, L, D] 或 [B, D]（来自 text_ve_encoder）
        v_img : [B, D]（VLM 视觉 token 池化后线性映射）
        """
        z_text_vec = masked_mean_pool(z_text)  # [B, D]
        image_features = v_img  / (v_img.norm(dim=-1, keepdim=True)  + 1e-6)
        text_features  = z_text_vec / (z_text_vec.norm(dim=-1, keepdim=True) + 1e-6)
        return self.clip_loss(image_features, text_features, logit_scale)

    def compute_kl_loss(self, mu, log_var):
        return kl_loss(mu, log_var)  # [B]

    def text_ve_encoder(self, q_tokens: torch.Tensor, q_mask: torch.Tensor):
        """
        q_tokens: [B, Lq, D]; q_mask: [B, Lq] (bool/0-1)
        return: z_text [B, Lq, D/2], mu [B, Lq, D/2], log_var [B, Lq, D/2]
        """
        q_tokens = q_tokens.to(dtype=self._dtype)
        q_mask   = q_mask.to(dtype=self._dtype, device=self.device)
        out = self.context_encoder(q_tokens, q_mask)
        mu, log_var = torch.chunk(out, 2, dim=-1)

        # reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z_text = eps * std + mu
        return z_text, mu, log_var

    def flow_matching_loss(self, z_text: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
        """
        使用 encode_moments 得到目标分布 x1，构造路径 ψ(t,x0,x1)，
        用 FMTransformers 预测速度，与目标速度 Dtψ 计算 MSE。
        """
        x_img_256 = F.interpolate(x_img, size=(256, 256), mode='bilinear', align_corners=False)
        x1 = self.autoencoder(
            x_img_256.to(self.device, dtype=self._dtype), fn='encode_moments'
        ).squeeze(0)
        x1 = x1.to(self._dtype, self.device)

        # 连续时间采样 + 路径
        t = self.time_step_sampler.sample_time(x1)
        log_snr = 4.0 - 8.0 * t  # 目前未直接使用，但保留给 fm_transformers
        x0 = z_text.reshape(x1.shape).to(self._dtype, self.device)

        x_t = psi(t, x0, x1, self.sigma_min, self.sigma_max).to(self._dtype, self.device)
        v_target = Dt_psi(t, x0, x1, self.sigma_min, self.sigma_max)

        # classifier-free 类似的 null 指示
        null_indicator = (torch.rand(x1.shape[0], device=x1.device) < 0.1)
        if null_indicator.sum() > 1:
            target_null = x1[null_indicator]
            target_null = torch.cat((target_null[1:], target_null[:1]))
            x1[null_indicator] = target_null

        preds = self.fm_transformers(x_t, log_snr=log_snr, null_indicator=null_indicator)
        loss = mse_mean_over_spatial(preds[-1] - v_target).mean()
        return loss

    # ---------------- Main T2I loss ----------------
    def text2image_loss(self, batch: dict) -> torch.Tensor:
        """
        三段式：
          1) 图像路径：像素 -> VLM 视觉 token -> 池化+线性 -> v_img
          2) 文本路径：meta queries + LLM -> q_tokens -> TransEncoder -> z_text(重参数)
          3) 三项损失：Flow Matching(z_text vs x1)、CLIP 对齐(z_text vs v_img)、KL 正则(mu/logvar)
        """
        # A. 图像像素 / latents
        if 'pixel_values' in batch:
            x_img = batch['pixel_values'].to(self.device, dtype=self._dtype)   # [-1, 1]
            z_img = self.pixels_to_latents(x_img)
        elif self.clip_target_source == 'latents_decode' and 'image_latents' in batch:
            z_img = batch['image_latents'].to(self.device, dtype=self._dtype)
            x_img = self.latents_to_pixels(z_img)
        else:
            raise ValueError("text2image_loss: need 'pixel_values' or 'image_latents'(with latents_decode).")

        # B. 图像向量 v_img（来自 VLM）
        v_img_tokens = self.get_semantic_features(x_img)              # [B, L_img, D_vlm]
        v_img = masked_mean_pool(v_img_tokens)                        # [B, D_vlm]
        v_img = self.img_clip_head(v_img).to(self._dtype)             # [B, D_align]
        logit_scale = self.clip_logit_scale.exp()

        # C. 文本 queries -> LLM -> VE -> z_text
        b = z_img.shape[0]
        input_ids = batch['input_ids'].to(self.device)
        attn_mask = batch['attention_mask'].to(self.device)

        q_meta = self.meta_queries[None].expand(b, self.num_queries, -1)  # [B, Q, D]
        llm_in = self.prepare_forward_input(x=q_meta, input_ids=input_ids, attention_mask=attn_mask)
        llm_out = self.llm.model(**llm_in, return_dict=True).last_hidden_state
        q_tokens = llm_out[:, -self.num_queries:]                          # [B, Q, D]
        q_mask   = attn_mask.new_ones(q_tokens.shape[:2])                  # [B, Q]

        z_text, mu, log_var = self.text_ve_encoder(q_tokens, q_mask)       # [B, Q, *]

        # D. 三项损失
        loss_flow = self.flow_matching_loss(z_text, x_img)
        loss_clip = self.compute_clip_loss(z_text, v_img, logit_scale)
        loss_kl   = self.compute_kl_loss(mu, log_var).mean()

        kl_weight = 1e-2
        total = loss_flow + loss_clip + kl_weight * loss_kl
        return total

    # ---------------- Runner entry ----------------
    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        raise NotImplementedError

    def compute_loss(self, data_dict):
        # 兼容两种输入：{'text2image': batch} 或直接 batch
        if 'text2image' in data_dict:
            return {'loss_text2image': self.text2image_loss(data_dict['text2image'])}
        if 'pixel_values' in data_dict or 'image_latents' in data_dict:
            return {'loss_text2image': self.text2image_loss(data_dict)}
        raise NotImplementedError("Only text2image path is implemented after refactor.")


```
