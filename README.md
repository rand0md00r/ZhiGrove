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
