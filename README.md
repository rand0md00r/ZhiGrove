# stunning-palm-tree
---
2025年8月10日20:31:29

## configs/pretrain/crossuni_unified_debug.py

# configs/pretrain/crossuni_unified_debug.py
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.dataset import InfiniteSampler
from xtuner.engine.runner import TrainLoop
from xtuner.dataset import ConcatDataset

# ========= Global switches =========
precision = 'bf16'            # 'bf16' | 'fp32'
use_deepspeed = True          # train.py 会用 --deepspeed 控制，但这里留一个标识方便管理
multi_resolution = False      # 是否启用多分辨率随机训练
image_size = 256              # Debug: 256；生产可切 512/1024
image_size_choices = [256]    # multi_resolution=True 时生效，如 [256, 384, 512]
global_seed = 42

# ========= Paths / Models =========
# 模型权重与路径
internvl3_model_name_or_path = "/vepfs/DI/model-public/InternVL3-2B"
sana_model_name_or_path = f"Efficient-Large-Model/SANA1.5_1.6B_{image_size}px_diffusers"
vae_name_or_path = "/vepfs/DI/model-public/dc-ae-f32c32-sana-1.1-diffusers"
open_clip_model_path = "/vepfs/DI/model-public/ViT-L-16-SigLIP-256/open_clip_pytorch_model.bin"  # 交由代码读取

# 训练/数据路径（Debug 10K）
data_json   = "/vepfs/group03/wky/OpenUni/data/text-to-image-2M/data/data_1024_10K.json"
cap_folder  = "/vepfs/group03/wky/OpenUni/data/text-to-image-2M/raw/data_1024_10K"
image_root  = "/vepfs/group03/wky/OpenUni/data/text-to-image-2M/raw/data_1024_10K"
latents_dir_tmpl = "/vepfs/group03/wky/OpenUni/data/text-to-image-2M/raw/data_1024_10K_dc32_{image_size}"

# ========= Tokenizer / Prompt template =========
from transformers import AutoTokenizer, AutoImageProcessor
prompt_template = dict(
    IMG_START_TOKEN='<img>',
    IMG_END_TOKEN='</img>',
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n'),
    GENERATION='{input}',
    CFG='',
    IMG_START_TOKEN_FOR_GENERATION=True,
)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=internvl3_model_name_or_path,
    trust_remote_code=True,
    padding_side='right'
)
image_processor = dict(
    type=AutoImageProcessor.from_pretrained,
    pretrained_model_name_or_path=internvl3_model_name_or_path
)
pad_index = 0
max_length = 128

# ========= Model =========
from diffusers import (AutoencoderDC, SanaTransformer2DModel,
                       DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler)
from src.models.openuni.internvl3_sana_hf import OpenUniInternVL3SANAHF
from src.models.internvl3.modeling_internvl_chat import InternVLChatModel

model = dict(
    type=OpenUniInternVL3SANAHF,
    # === precision control ===
    dtype=precision,                         # 需要在模型 __init__ 新增该参数
    # === InternVL LMM ===
    lmm=dict(
        type=InternVLChatModel.from_pretrained,
        pretrained_model_name_or_path=internvl3_model_name_or_path,
        torch_dtype=None,                    # 由 dtype 统一控制
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ),
    # === VAE / Denoiser ===
    vae=dict(
        type=AutoencoderDC.from_pretrained,
        pretrained_model_name_or_path=vae_name_or_path,
        torch_dtype=None,                    # 由 dtype 统一控制
    ),
    transformer=dict(
        type=SanaTransformer2DModel.from_pretrained,
        pretrained_model_name_or_path=sana_model_name_or_path,
        subfolder="transformer",
        torch_dtype=None,                    # 由 dtype 统一控制
    ),
    train_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sana_model_name_or_path,
        subfolder="scheduler",
    ),
    test_scheduler=dict(
        type=DPMSolverMultistepScheduler.from_pretrained,
        pretrained_model_name_or_path=sana_model_name_or_path,
        subfolder="scheduler",
    ),
    # === Connector / Projection ===
    connector=dict(
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=6,
        num_attention_heads=24,
        _attn_implementation='flash_attention_2',
    ),
    proj_type='enc_proj',
    num_queries=256,
    vit_input_size=448,
    max_length=2048,
    # === Train-time switches ===
    lora_modules=None,
    lora_rank=8,
    lora_alpha=8,
    freeze_lmm=True,
    freeze_transformer=True,

    # === CrossFlow-style encoder head ===
    transencoder=dict(
        d_model=1536,
        num_blocks=11,
        num_token=256,
        num_attention_heads=8,
        hidden_dim=1024,
        # 由 image_size 决定 latent 大小；如果后续多分辨率，这个参数应当从前向时动态计算或在数据里传入
        latten_size=4 * 32 * 32 * 4 if image_size == 256 else 4 * 64 * 64 * 2,
        down_sample_block=3,
        dropout_prob=0.1,
        last_norm=False,
    ),

    # === CLIP/OpenCLIP 相关（放到配置）===
    open_clip_path=open_clip_model_path,        # 需要在模型里读取
    clip_target_source='pixels',                # 'pixels' | 'latents_decode'（无像素图时解码 latents 到像素作为 CLIP 目标）

    # === Tokenizer & prompt ===
    tokenizer=tokenizer,
    prompt_template=prompt_template,
)

# ========= Dataset (CaptionDataset) =========
from src.datasets.text2image.caption_datasets import CaptionDataset
from src.datasets.collate_functions import (
    collate_func_gen_latents_with_prompt, collate_func_gen_latents
)

# 注意：为了 CLIP loss，建议在读取 latents 的同时也读取 pixel_values（读图或改用 latents 解码）
t2i_10k = dict(
    type=CaptionDataset,
    data_path=data_json,
    cap_folder=cap_folder,
    image_folder=image_root,
    image_size=image_size,
    min_image_size=80,
    cap_source='prompt',
    unconditional=0.1,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    max_length=max_length,
    # 预编码 latents
    image_latents_folder=latents_dir_tmpl.format(image_size=image_size),
    # 新增：可选同时加载像素图给 CLIP（需要在 CaptionDataset 实现，见“代码改动点”）
    load_pixels_along_latents=True,
    # 新增：多分辨率（需要在 CaptionDataset 实现）
    multi_resolution=multi_resolution,
    image_size_choices=image_size_choices,
    # 新增：是否把原始 prompt 文本也返回（给 CLIP 编码）
    return_prompt=True,
)

dataset = dict(
    type=ConcatDataset,
    datasets=[t2i_10k]
)

# ========= Dataloader =========
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    # 如果已经能同时返回 pixel_values + prompt，使用带 prompt 的 collate；
    # 否则先用 collate_func_gen_latents，再在模型侧按 clip_target_source='latents_decode' 解码 latents 给 CLIP
    collate_fn=dict(type=collate_func_gen_latents_with_prompt, pad_index=pad_index)
    # collate_fn=dict(type=collate_func_gen_latents, pad_index=pad_index)
)

# ========= Optim / Sched =========
max_iters = 10_000
accumulative_counts = 1
lr = 1e-4
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.01

optim_type = 'src.optimisers.custom_adamw.CustomAdamW'  # 或 'torch.optim.AdamW'
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16" if precision == 'bf16' else "float32",
)

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-5, by_epoch=False, begin=0, end=warmup_ratio * max_iters),
    dict(type=CosineAnnealingLR, T_max=max_iters - int(warmup_ratio * max_iters), by_epoch=False,
         begin=int(warmup_ratio * max_iters), end=max_iters)
]

train_cfg = dict(type=TrainLoop, max_iters=max_iters)

# ========= Runtime / Hooks =========
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=5_000, max_keep_ckpts=1),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=global_seed, deterministic=False)
log_processor = dict(by_epoch=False)

# Runner：train.py 默认会用 CustomRunner.from_cfg
# 这里不强制写 runner_type，保持 train.py 的逻辑




-----
2025年8月11日 10点19分
src/models/openuni/internvl3_sana_hf.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from mmengine.logging import print_log
from torch.nn.utils.rnn import pad_sequence
from xtuner.model.utils import guess_load_checkpoint
from diffusers.pipelines.sana.pipeline_sana import SanaPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from src.models.connector import ConnectorConfig, ConnectorEncoder
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from src.models.encoder import TransEncoder, Adaptor
from src.models.encoder import FrozenCLIPEmbedder

import open_clip

from src.models.cliploss import ClipLoss
from src.models.diffusion import Stage, FMTransFormers
from src.models.encoder.autoencoder import FrozenAutoencoderKL

from timm.models.layers import trunc_normal_, Mlp

import numpy as np
import random

from ml_collections import ConfigDict

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


class TimeStepSampler:
    """
    Abstract class to sample timesteps for flow matching.
    """

    def sample_time(self, x_start):
        # In flow matching, time is in range [0, 1] and 1 indicates the original image; 0 is pure noise
        # this convention is *REVERSE* of diffusion
        raise NotImplementedError


class LogitNormalSampler(TimeStepSampler):
    def __init__(self, normal_mean: float = 0, normal_std: float = 1):
        # follows https://arxiv.org/pdf/2403.03206.pdf
        # sample from a normal distribution
        # pass the output through standard logistic function, i.e., sigmoid
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)

    @torch.no_grad()
    def sample_time(self, x_start):
        x_normal = torch.normal(
            mean=self.normal_mean,
            std=self.normal_std,
            size=(x_start.shape[0],),
            device=x_start.device,
        )
        x_logistic = torch.nn.functional.sigmoid(x_normal)
        return x_logistic


class OpenUniInternVL3SANAHF(BaseModel):
    def __init__(self,
                 lmm,
                 transformer,
                 train_scheduler,
                 test_scheduler,
                 vae,
                 tokenizer,
                 prompt_template,
                 connector,
                 num_queries=256,
                 pretrained_pth=None,
                 use_activation_checkpointing=True,
                 lora_modules=None,  # ["to_k", "to_q", "to_v"],
                 lora_rank=8,
                 lora_alpha=8,
                 freeze_lmm=True,
                 freeze_transformer=True,
                 vit_input_size=448,
                 max_length=2048,
                 proj_type='enc_proj',
                 transencoder=None,
                 dtype='bf16',
                 open_clip_path=None,
                 clip_target_source='pixels',
                 ):
        super().__init__()
        self._dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
        self.use_activation_checkpointing = use_activation_checkpointing

        self.lmm = BUILDER.build(lmm)
        if freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = freeze_lmm

        # self.train_scheduler = BUILDER.build(train_scheduler)
        # self.test_scheduler = BUILDER.build(test_scheduler)

        self.transformer = BUILDER.build(transformer)   # 弃用，仅供跑通
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer
        if lora_modules is not None:
            if lora_modules == 'auto':
                lora_modules = find_all_linear_names(self.transformer)
            # import pdb; pdb.set_trace()
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)

        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.vit_input_size = vit_input_size
        self.max_length = max_length
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)

        self.num_queries = num_queries
        self.connector = ConnectorEncoder(ConnectorConfig(**connector))

        self.proj_type = proj_type
        if self.proj_type == 'proj_enc':
            assert self.connector.config.hidden_size == self.transformer.config.caption_channels
            self.projector = nn.Linear(
                self.llm.config.hidden_size, self.connector.config.hidden_size)
        elif self.proj_type == 'enc_proj':
            assert self.connector.config.hidden_size == self.llm.config.hidden_size
            self.projector = nn.Linear(
                self.connector.config.hidden_size, self.transformer.config.caption_channels)
        elif self.proj_type == 'proj_enc_proj':
            self.projector = nn.ModuleList([
                nn.Linear(self.llm.config.hidden_size, self.connector.config.hidden_size),
                nn.Linear(self.connector.config.hidden_size, self.transformer.config.caption_channels)
            ])
        else:
            raise ValueError(f'Unknown proj type: {self.proj_type}')

        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))

        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        self.resizer = transforms.Resize(256) # for clip

        self.clip_model = CLIPModel.from_pretrained("/vepfs/DI/model-public/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("/vepfs/DI/model-public/clip-vit-base-patch32")
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

        self.frozen_clip = FrozenCLIPEmbedder()
        self.frozen_clip.eval()

        # 初始化 TransEncoder
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

        self.open_clip_path = open_clip_path
        self.open_clip, _, self.open_clip_preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained=self.open_clip_path)

        # self.open_clip_output = Adaptor(input_dim=256 * 1536, tar_dim=4 * 64 * 64)
        self.open_clip_output = Mlp(in_features=1024, 
                            hidden_features=4*32*32, 
                            out_features=8*32*32, 
                            norm_layer=nn.LayerNorm,
                        )

        self.clip_loss = ClipLoss()

        # 估计 latent 通道数（例如 SD VAE 通常4；也可以通过一次encode真值图像拿shape）
        # self.latent_channels = getattr(self.autoencoder, "latent_channels", 4)
        self.latent_channels = 8
        
        # CrossFlow 风格的三阶段配置（与你给的一致）
        stage_configs = [
            ConfigDict({
                "block_type": "TransformerBlock",
                "dim": 1024,
                "hidden_dim": 2048,
                "num_attention_heads": 16,
                "num_blocks": 65,
                "max_height": 16,
                "max_width": 16,
                "image_input_ratio": 1,
                "input_feature_ratio": 2,
                "final_kernel_size": 3,
                "dropout_prob": 0.0,
                "pe_type": "sinusoidal",
                "norm_type": "TDRMSN",
                "gradient_checking": True
            }),
            ConfigDict({
                "block_type": "ConvNeXtBlock",
                "dim": 512,
                "hidden_dim": 1024,
                "kernel_size": 7,
                "num_blocks": 33,
                "max_height": 32,
                "max_width": 32,
                "image_input_ratio": 1,
                "input_feature_ratio": 1,
                "final_kernel_size": 3,
                "dropout_prob": 0.0,
                "pe_type": "sinusoidal",
                "norm_type": "TDRMSN",
                "gradient_checking": True
            })
        ]

        # 三阶段主干
        self.fm_transformers = FMTransFormers(latent_channels=self.latent_channels, stage_configs=stage_configs)
        self.fm_transformers.set_cfgs(stage_configs)
        self.fm_transformers.to(self.device, dtype=self._dtype)


        config_autoencoder = {
            'pretrained_path': '/vepfs/DI/yaqi/understand_gen/models/stable-diffusion/autoencoder_kl.pth',
            'scale_factor': 0.23010
        }

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
        self.autoencoder = FrozenAutoencoderKL(ddconfig, 4, config_autoencoder['pretrained_path'], config_autoencoder['scale_factor'])  # embed_dim设为2，不行
        self.autoencoder.to(self.device)

        self.time_step_sampler = LogitNormalSampler()
        self.sigma_min: float = 1e-5
        self.sigma_max: float = 1.0
        self.timescale: float = 1.0

        self.clip_target_source = clip_target_source # 'pixels' or 'latents_decode'

        self.clip_logit_scale = nn.Parameter(torch.tensor(1/0.07).log())
        
        img_feat_dim = getattr(self.llm.config, 'hidden_size', 1536)
        self.img_clip_head = nn.Linear(img_feat_dim, 8192, bias=False)
        

        self.to(self._dtype)
        
    def _masked_mean_pool(self, x, mask=None):
        """
        x: [B, L, D] or [B, D]
        mask: [B, L] (bool) 1/True means valid
        returns: [B, D]
        """
        if x.ndim == 2:
            return x
        
        if mask is None:
            return x.mean(dim=1)
        
        # 兼容True/False 或 0/1
        m = mask.to(x.dtype)
        m = m / (m.sum(dim=1, keepdim=True) + 1e-6)
        return (x * m.unsqueeze(-1)).sum(dim=1)

    def psi(self, t, x, x1):
        assert (
            t.shape[0] == x.shape[0]
        ), f"Batch size of t and x does not agree {t.shape[0]} vs. {x.shape[0]}"
        assert (
            t.shape[0] == x1.shape[0]
        ), f"Batch size of t and x1 does not agree {t.shape[0]} vs. {x1.shape[0]}"
        assert t.ndim == 1
        t = self.expand_t(t, x)
        return (t * (self.sigma_min / self.sigma_max - 1) + 1) * x + t * x1

    def Dt_psi(self, t: torch.Tensor, x: torch.Tensor, x1: torch.Tensor):
        assert x.shape[0] == x1.shape[0]
        return (self.sigma_min / self.sigma_max - 1) * x + x1

    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_expanded = t
        while t_expanded.ndim < x.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        return t_expanded.expand_as(x)

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


    def llm2dit(self, x):
        if self.proj_type == 'proj_enc':
            return self.connector(self.projector(x))
        elif self.proj_type == 'enc_proj':
            return self.projector(self.connector(x))
        elif self.proj_type == 'proj_enc_proj':
            return self.projector[1](self.connector(self.projector[0](x)))
        else:
            raise ValueError(f'Unknown proj type: {self.proj_type}')

    @property
    def llm(self):
        return self.lmm.language_model

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector.gradient_checkpointing = False

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()

        return self

    @torch.no_grad()
    def pixels_to_latents(self, x):
        scaling_factor = self.vae.config.scaling_factor
        z = self.vae.encode(x)[0] * scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        scaling_factor = self.vae.config.scaling_factor
        x_rec = self.vae.decode(z / scaling_factor)[0]
        return x_rec

    def prepare_forward_input(self,
                              x,
                              inputs_embeds=None,
                              input_ids=None,
                              attention_mask=None,
                              past_key_values=None):
        b, l, _ = x.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat([
            attention_mask, attention_mask.new_ones(b, l)
        ], dim=1)
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # prepare context
        if past_key_values is not None:
            inputs_embeds = x
            position_ids = position_ids[:, -l:]
        else:
            if inputs_embeds is None:
                input_ids = input_ids.to(self.device)
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        inputs = dict(inputs_embeds=inputs_embeds,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values)

        return inputs

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type in ['text2image', 'image2image']:
            if data_type in data_dict:
                losses[f'loss_{data_type}'] = getattr(self, f'{data_type}_loss')(data_dict[data_type])
        if len(losses) == 0:
            if 'pixel_values_src' in data_dict:
                losses[f'loss_image2image'] = self.image2image_loss(data_dict)
            else:
                losses[f'loss_text2image'] = self.text2image_loss(data_dict)

        return losses

    @torch.no_grad()
    def get_semantic_features(self, pixel_values):
        # pixel_values: [-1, 1]
        pixel_values = (pixel_values + 1.0) / 2     # [0, 1]
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

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

        all_prompts = [
            self.prompt_template['INSTRUCTION'].format(input=prompt) + self.prompt_template['IMG_START_TOKEN'],
            self.prompt_template['INSTRUCTION'].format(input=cfg_prompt) + self.prompt_template['IMG_START_TOKEN'],
        ]

        input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
                     for p in all_prompts]
        valid_lens = [len(input_ids_) for input_ids_ in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in range(len(input_ids)):
            attention_mask[i, :valid_lens[i]] = True

        return dict(input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device))

    def text2image_loss(self, data_dict):

        if 'pixel_values' in data_dict:
            # debug 阶段，需要pixel_values，需要image_latents，但只有pixel_values
            pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
            image_latents = self.pixels_to_latents(pixel_values)
        elif self.clip_target_source == 'latents_decode' and 'image_latents' in data_dict:
            # train 阶段，预处理得到image_latents计算diff loss，调用lmm从picel_value得到image_emb(+ mlp)
            pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
            image_latents = data_dict['image_latents'].to(dtype=self._dtype, device=self.device)
        else:
            raise ValueError("No pixel_values found and latents_decode not enabled.")

        vit_embeds = self.get_semantic_features(pixel_values)       # [B, L_img, D_vit]
        img_vec = self._masked_mean_pool(vit_embeds)                # [B, D_vit]
        img_vec = self.img_clip_head(img_vec).to(self._dtype)    # [B, D_clip] 统一维度

        logit_scale = self.clip_logit_scale.exp()                   # 温度

        b, _, height, weight = image_latents.shape

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(x=hidden_states,
                                            input_ids=input_ids,
                                            attention_mask=attention_mask)

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        # hidden_states = self.llm2dit(hidden_states)
        B, L, _= hidden_states.shape
        attention_mask_queries = attention_mask.new_ones(B, L)

        x0, mu, log_var = self.text_ve_encoder(hidden_states, attention_mask_queries)  # hidden_states.shape=[bs, 256, 1536]

        loss_clip = self.compute_clip_loss(x0, img_vec, logit_scale)


        loss_kl = self.compute_kl_loss(mu, log_var)
        kl_loss_weight = 1e-2 # 0.0005

        loss_mlp = loss_clip + loss_kl * kl_loss_weight

        loss_diff = self.diff_loss(x0, data_dict, img_vec)


        print(f"clip: {loss_clip}")

        print(f"kl: {loss_kl}")

        print(f"diff: {loss_diff}")


        return loss_diff + loss_mlp
    
    def text_ve_encoder(self, token_embedding, token_mask):
        token_embedding = token_embedding.to(dtype=self._dtype)
        token_mask = token_mask.to(dtype=self._dtype, device=self.device)
        
        output = self.context_encoder(token_embedding, token_mask)
        mu, log_var = torch.chunk(output, 2, dim=-1)

        def _reparameterize(mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        z = _reparameterize(mu, log_var)

        return [z, mu, log_var]

    def compute_clip_loss(self, x0, recon_gt_clip, logit_scale):

        image_features = recon_gt_clip / recon_gt_clip.norm(dim=-1, keepdim=True)
        text_features = x0 / x0.norm(dim=-1, keepdim=True)
        recons_loss = self.clip_loss(image_features, text_features, logit_scale)

        return recons_loss

    def compute_kl_loss(self, mu, log_var):
        kld_loss = -0.5 * torch.sum(1 + log_var - (0.3 * mu) ** 6 - log_var.exp(), dim = 1) # slightly different KL loss function: mu -> 0 [(0.3*mu) ** 6] and var -> 1
        return kld_loss
    
    def diff_loss(self, x0, data_dict, recon_gt_clip, indicator=None):
        # image_features = recon_gt_clip / recon_gt_clip.norm(dim=-1, keepdim=True)

        # x1 目标图像的分布, 调用autoencoder进行VE编码

        pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
        pixel_values = F.interpolate(pixel_values, size=(256, 256), mode='bilinear', align_corners=False)
        x1 = self.autoencoder(pixel_values, fn='encode_moments').squeeze(0)
        x1 = x1.to(dtype=self._dtype, device=self.device)
        B = x1.shape[0]

        # 采样连续时间 计算 log-SNR / a
        t = self.time_step_sampler.sample_time(x1)
        log_snr = 4.0 - 8.0 * t
        alpha = torch.sigmoid(log_snr)
        sqrt_a = torch.sqrt(alpha)
        sqrt_lma = torch.sqrt(1.0 - alpha)

        # 构造路径与目标速度
        x0 = x0.reshape(x1.shape)
        # x_t = sqrt_a * x1 + sqrt_lma * x0
        x_t = self.psi(t, x=x0, x1=x1)
        x_t = x_t.to(dtype=self._dtype, device=self.device)

        target_velocity = self.Dt_psi(t, x=x0, x1=x1)

        null_indicator = torch.from_numpy(np.array([random.random() < 0.1 for _ in range(x1.shape[0])])).to(x1.device)
        if null_indicator.sum()<=1:
            null_indicator[null_indicator==True] = False
            assert null_indicator.sum() == 0
            pass
        else:
            target_null = x1[null_indicator]
            target_null = torch.cat((target_null[1:], target_null[:1]))
            x1[null_indicator] = target_null


        # ===== 4) 前向：FM transformer 网络 =====
        preds = self.fm_transformers(x_t, log_snr=log_snr, null_indicator=null_indicator)
        # if not isinstance(preds, (list, tuple)):
        #     preds = [preds]

        loss_diff = self.mos(preds[-1] - target_velocity)
        return loss_diff


    def image2image_loss(self, data_dict):

        pixel_values_src = data_dict['pixel_values_src'].to(dtype=self.dtype, device=self.device)
        vit_embeds = self.get_semantic_features(pixel_values_src)
        vit_embeds.requires_grad = True

        pixel_values = data_dict['pixel_values'].to(dtype=self._dtype, device=self.device)
        image_latents = self.pixels_to_latents(pixel_values)

        b, _, height, weight = image_latents.shape

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)

        inputs_embeds = vit_embeds.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == self.image_token_id] = vit_embeds.flatten(0, 1)
        inputs_embeds[input_ids != self.image_token_id] = self.llm.get_input_embeddings()(
            input_ids[input_ids != self.image_token_id]
        )

        max_length = self.max_length
        if inputs_embeds.shape[1] > max_length:
            inputs_embeds = inputs_embeds[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]

        hidden_states = self.meta_queries[None].expand(b, self.num_queries, -1)

        inputs = self.prepare_forward_input(x=hidden_states,
                                            inputs_embeds=inputs_embeds,
                                            attention_mask=attention_mask)

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]
        hidden_states = self.llm2dit(hidden_states)

        loss_diff = self.diff_loss(model_input=image_latents,
                                   prompt_embeds=hidden_states,
                                   prompt_attention_mask=None)

        return loss_diff

    @torch.no_grad()
    def generate(self,
                 input_ids=None,
                 inputs_embeds=None,
                 attention_mask=None,
                 cfg_scale=4.5,
                 num_steps=20,
                 generator=None,
                 height=512,
                 width=512,
                 progress_bar=True,
                 **kwargs):

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        bsz = attention_mask.shape[0]

        assert bsz % 2 == 0

        hidden_states = self.meta_queries[None].expand(bsz, self.num_queries, -1)
        inputs = self.prepare_forward_input(x=hidden_states,
                                            inputs_embeds=inputs_embeds,
                                            attention_mask=attention_mask)

        output = self.llm.model(**inputs, return_dict=True)
        hidden_states = output.last_hidden_state[:, -self.num_queries:]

        hidden_states = self.llm2dit(hidden_states)
        attention_mask = torch.ones(bsz, self.num_queries, device=self.device, dtype=torch.bool)

        pipeline = SanaPipeline(transformer=self.transformer,
                                scheduler=self.test_scheduler,
                                vae=self.vae, text_encoder=None, tokenizer=None
                                )
        pipeline.set_progress_bar_config(disable=not progress_bar)

        samples = pipeline(
            negative_prompt=None,
            height=height,
            width=width,
            prompt_embeds=hidden_states[:bsz // 2],
            prompt_attention_mask=attention_mask[:bsz // 2],
            negative_prompt_embeds=hidden_states[bsz // 2:],
            negative_prompt_attention_mask=attention_mask[bsz // 2:],
            num_inference_steps=num_steps,
            generator=generator,
            complex_human_instruction=None,
            output_type='latent',
            use_resolution_binning=False,
            guidance_scale=cfg_scale,
        ).images.to(self._dtype)

        return self.latents_to_pixels(samples)

    # def diff_loss(self, model_input, prompt_embeds, prompt_attention_mask):
    #     # Sample noise that we'll add to the latents
    #     noise = torch.randn_like(model_input)
    #     bsz = model_input.shape[0]

    #     # Sample a random timestep for each image
    #     # for weighting schemes where we sample timesteps non-uniformly
    #     u = compute_density_for_timestep_sampling(
    #         weighting_scheme="none",
    #         batch_size=bsz,
    #     )
    #     indices = (u * self.train_scheduler.config.num_train_timesteps).long()
    #     timesteps = self.train_scheduler.timesteps[indices].to(device=model_input.device)

    #     # Add noise according to flow matching.
    #     # zt = (1 - texp) * x + texp * z1
    #     sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim)
    #     noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

    #     # Predict the noise residual
    #     model_pred = self.transformer(
    #         hidden_states=noisy_model_input,
    #         encoder_hidden_states=prompt_embeds,
    #         encoder_attention_mask=prompt_attention_mask,
    #         timestep=timesteps,
    #         return_dict=False,
    #     )[0]

    #     # these weighting schemes use a uniform timestep sampling
    #     # and instead post-weight the loss
    #     weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

    #     # flow matching loss
    #     target = noise - model_input

    #     # Compute regular loss.
    #     loss = torch.mean(
    #         (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
    #         1,
    #     )
    #     loss = loss.mean()

    #     return loss

    # def get_sigmas(self, timesteps, n_dim=4):
    #     sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self._dtype)
    #     schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
    #     timesteps = timesteps.to(self.device)
    #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma
