# ===================================================================================
# 实验配置：internvl_query_5TransEncoder_sdvae_23m_256pix_bs4096_0903.py
# -----------------------------------------------------------------------------------
# InternVL3 Query 5TransEncoder SDVAE 23M 配置
# 
# 基础配置：
# 本配置用于训练 InternVL3-Query-CrossFlow 架构，采用 0 层 TransEncoder 作为文本特征压缩器，
# 搭配 SDVAE 23M 图像自编码器，输入分辨率为 256 像素，批量大小 4096。
# - LLM Backbone: InternVL3-2B
# - 图像自编码器: dc-ae-f32c32-sana-1.1-diffusers
# - 图像分辨率: 256x256
# - 批量大小: 4096
# - 支持多种损失与训练策略（包括Flow Matching、KL 正则、DINO v2）
# - 适配 deepspeed 分布式训练
# 
# 实验参数：
# - 文本编码器: 0 层 TransEncoder，3层 ReductionLayer
# - LR : 5e-5
# - batch size = 64
# - global batch = 4096
# - max_iterations: 120000(20 epochs)
# - DINO v2 loss (only global)
#  
# ===================================================================================

