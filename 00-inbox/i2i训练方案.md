# MetaQuery

## 1）训练方法（Training Method）

- 总体范式：用一组可学习的 MetaQueries 直接喂给冻结的多模态 LLM（MLLM），取其输出隐表征作为条件，经一个可训练的 Connector对齐到扩散式生成模型（DiT）的条件输入空间；全流程只用标准去噪目标（diffusion denoising objective）在图文配对数据上训练。可选再加入图像重建目标与去噪目标混合训练以获得重建/编辑能力。

- 模块是否训练：默认冻结 MLLM；训练 MetaQuery（可学习查询）+ Connector；扩散解码器（DiT）既可冻结也可微调，论文给出 ablation：只冻 LLM 时已可达高水准；进一步微调 DiT能继续提升画质。

- Connector 设计：两种结构对比——

  - Proj‑Enc：先投到 DiT 条件维度再过 Transformer Encoder；

  - Enc‑Proj：先在与 MLLM 隐维相同的维度用 Transformer Encoder 对齐，再投到 DiT 条件维度。
实验更推荐 Enc‑Proj（更省参、效果更好）。Connector 采用双向注意力的 Transformer Encoder。

- 训练目标对比：只用 T2I 去噪目标最佳；纯重建目标最差；混合（T2I+重建）可在不明显伤害 T2I 的前提下获得重建/编辑能力。

## 2）训练参数（Training Hyperparameters）

### 预训练（Pre‑training）

- 数据规模：25M 公开图文对。
- 轮数：8 epochs（3.2 概述段落曾写 4 epochs，以 4 节为准）。
- 全局 batch size：4096。
- 学习率与调度：初始 1e‑4，cosine decay；4000 步 warm‑up；最终衰减到 1e‑5。
- 是否冻结：MLLM 冻结；训练 MetaQueries + Connector；DiT 可冻或微调（微调更好）。
- 分辨率 / 生成头：文中对齐实验多用 Sana‑0.6B@512 分辨率；COCO FID 评测使用 Stable Diffusion v1.5。

### 指令微调（Instruction Tuning）

- 数据规模：构建 2.4M “成对图像 + 指令” 样本（见下文数据结构）。
- 轮数：3 epochs。
- batch size：2048。
- 学习率调度：与预训练相同（cosine + 4k warm‑up）.

### 架构/Token 侧关键超参

- MetaQuery 数量（#tokens）：系统性标度实验给出 64 已较好，更大量能进一步提升对齐；论文在模型家族实验中统一设为 256 tokens（博客页明确载明）。
- Connector 层数与维度：推荐 Enc‑Proj，24 层；示例维度 896（Enc‑Proj） vs 2304（Proj‑Enc）；24 层 Enc‑Proj 参数量约 316M。
- 背骨与生成头组合：MLLM 采用 LLaVA‑OneVision‑0.5B / Qwen2.5‑VL‑3B / Qwen2.5‑VL‑7B；生成头测试 SD‑v1.5 与 Sana‑1.6B。

## 3）数据结构（Data Structure / Dataset Construction）

### A. 预训练数据（统一建模的基础对齐）
- 标准 (image, caption) 图文对，规模 25M；用于把冻结的 MLLM 条件通过 Connector 对齐到扩散解码器，仅用去噪目标训练。

### B. 指令微调数据（编辑/主体驱动等进阶能力）

- 来源：从 mmc4 fewer‑faces 子集拿到“图像 + caption”，用 SigLIP 进行按 caption 相似度聚类（每簇 ≤6，阈值 0.5）；每簇中与其他图像平均相似度最低者设为 target，其余为 source；得到 2.4M 组 (sources, target)。随后用 Qwen2.5‑VL‑3B 为每对生成开放式指令。

- 样本格式：
(source_images: 1..N, target_image, instruction_text) —— instruction 要同时描述一条与 sources 的笼统相似点（如“同款上衣/相似斧头/相似建筑”）以及 target 独有的全部差异；不得包含足以单独重建 target 的具体细节（避免泄露/投机），鼓励简洁。

- 用途：在保持 MLLM 冻结的前提下，对 MetaQueries + Connector +（可选）DiT 进行指令微调，获得图像编辑、主体驱动、“视觉联想”“Logo 设计”等能力。

### 补充
- 用 可学习查询（MetaQueries） 不仅在画质与对齐上可与“最后一层嵌入”相当/更好，更关键是保留了 MLLM 的“在上下文学习/推理/知识迁移”能力，在需要世界知识与推理的生成上显著更强。



# OpenUni

## 训练框架概览
- 模型结构
OpenUni 建立在**冻结的多模态大语言模型（InternVL3）**与 **扩散模型（SANA DiT）**之间，通过 可学习查询（256 tokens） + 轻量连接器（6 层 Transformer） 实现语义桥接，从而统一理解与图像生成功能。

## 一、Stage 1：预训练（Pre-training）
- 目标：仅训练 learnable queries + lightweight connector（LLM 与扩散模型权重均冻结），让连接模块学习将 LLM 输出映射为图像生成条件
- 数据来源：共计约 23M 图像-文本对，来自多个公开数据集（text‑to‑image‑2M、LAION‑Aesthetic‑6M、Megalith‑10M、RedCaps‑5M），所有 caption 由 LLM 重写生成。
| 超参数           | 数值                 |
| ------------- | ------------------ |
| 容器 Batch Size | 512                |
| 优化器           | AdamW              |
| 学习率           | $1 \times 10^{-4}$ |
| 权重衰减          | 0.05               |
| 梯度裁剪          | 1.0                |
| Betas         | (0.9, 0.95)        |
| 学习率调度         | Cosine decay       |
| Warm‑up Steps | 1,000              |
| 总训练步数         | 100,000 steps      |

## 二、Stage 2：高质量微调（High-Quality Fine-tuning）
- 目标：解冻扩散模型，让 connector 和 diffusion 模型一起进一步优化，以提升生成质量、对指令的响应度及鲁棒性。
- 数据来源：使用 BLIP3‑o 提供的 60,000 条高质量 instruction-image 样本，这些样本基于 GPT‑4o + DALL‑E3 / Midjourney 生成。
- 训练超参数：
| 超参数           | 数值                 |
| ------------- | ------------------ |
| Batch Size    | 256                |
| 优化器           | AdamW              |
| 学习率           | $1 \times 10^{-5}$ |
| 权重衰减          | 0.05               |
| 梯度裁剪          | 1.0                |
| Betas         | (0.9, 0.95)        |
| 学习率调度         | Cosine decay       |
| Warm‑up Steps | 100                |
| 总训练步数         | 10,000 steps       |

  


# CrossFlow

## 1. 方法概览（Training Approach）

- 核心创新
CrossFlow 完全打破传统扩散/流匹配模型必须从随机噪声开始的限制。它将源模态（如文本、低分辨率图像）分布直接映射到目标模态（如高分辨率图像、图像描述等），无需噪声输入或条件机制（如 cross-attention）。

- 关键技术突破
  - Variational Encoder：用于将源模态编码成与目标模态相同维度与空间结构，解决模态间数据形状不一致的问题，同时引入正则化效果。
  - Classifer-Free Guidance（CFG）：通过在训练中引入二值指示变量，实现无需条件架构也能控制生成质量的引导机制。
- 架构简洁高效
使用最普通的 Transformer（无 cross-attention）、统一处理输入与输出编码的 Token，展现跨模态生成的普适性。无需为特定任务额外设计结构。

## 2. 应用范例及任务覆盖（Use Cases & Performance）
CrossFlow 在多个任务上与主流方法表现旗鼓相当或更优：
- 主推任务：文本转图像生成（Text-to-Image）。
  - CrossFlow 在该任务中，即使不采用条件机制，也略微优于常规 flow matching 模型，同时在大规模训练与模型扩展下更具优势。
- 扩展任务：
  - 图像描述（Image captioning）
  - 单目深度估计（Monocular depth estimation）
  - 图像超分辨率（Super-resolution）
在这些任务中，CrossFlow 均与或优于现有专用架构方法，体现其通用架构的潜力。

- Latent Arithmetic
得益于 Variational Encoder 编码出的源分布具有语义结构，可在潜空间中进行有意思的编辑运算，实现对输出模态的语义控制。


## 3. 精炼训练策略（Training Scheme）


## 4. 训练参数
| 任务类别       | 模型规模   | Epoch / Steps   | Batch Size | Learning Rate       | Warm-up          | 备注               |
| ---------- | ------ | --------------- | ---------- | ------------------- | ---------------- | ---------------- |
| 图像描述       | 351M   | 100 epochs      | 256        | 2 × 10⁻⁴            | 5 epochs         | –                |
| 深度估计       | 527M   | 50 epochs       | 64         | 1 × 10⁻⁴ → 1 × 10⁻⁸ | cosine annealing | –                |
| 图像超分辨率     | 505M   | 1,000,000 steps | 512        | 1 × 10⁻⁴            | 5,000 steps      | –                |
| 文本→图像（T2I） | \~950M | \~300K steps    | 未指出        | 未指出                 | 未指出              | 同步 baseline，性能略优 |

- Image Captioning

“We train a 351M model for 100 epochs with a batch size of 256 and a learning rate of 2e-4 with 5 warm-up epochs.”【CVPR2025 补充材料】

- Depth Estimation

“We train a 527M model for 50 epochs with a batch size of 64 and a learning rate decayed from 1e-4 to 1e-8 with cosine annealing.”

- Super-Resolution

“We train a 505M model for 1M steps with a batch size of 512, a learning rate of 1e-4, and 5k warm-up steps.”

- Text-to-Image

“We train a ~0.95B model for 300K steps with the same training budget as baseline (FID 10.13 vs 10.79).”




# 图像编辑/图生图 - 数据训练表

| 用途                       | 数据/项目                            |       开源 | 规模/形式                                                  | 许可                 | 获取/备注                                                                                      |
| ------------------------ | -------------------------------- | -------: | ------------------------------------------------------ | ------------------ | ------------------------------------------------------------------------------------------ |
| 指令编辑 (Instruction-based) | **UltraEdit**                    |        ✅ | ≈ **4M** 编辑样本（含自由编辑与区域编辑）                              | **CC-BY-4.0**      | 代码、模型与**数据集**均提供，HF 数据集条目：*BleachNick/UltraEdit*。([GitHub][1], [Hugging Face][2])          |
| 指令编辑                     | **MagicBrush**                   |        ✅ | **10K** 三元组（源图、指令、目标图），含单/多轮、带/不带掩码                    | **CC-BY-4.0**      | GitHub 与 HF 提供**训练/验证集下载**；测试集需单独压缩包下载。([GitHub][3], [Hugging Face][4])                    |
| 指令编辑                     | **HQ-Edit**                      |        ✅ | **197,350** 次编辑，高分辨率，含（正向/反向）编辑文本                      | **CC-BY-NC-4.0**   | HF 数据集条目与项目页均公开（**非商用**）。([Hugging Face][5])                                               |
| 指令编辑（合成）                 | **InstructPix2Pix 生成数据**         |        ✅ | **451,990**（随机采样）/ **313,010**（CLIP 过滤）对，附下载脚本         | *未明示*（随项目/源数据）     | 官方仓库提供两版数据规模与下载方法（已做 NSFW 过滤）。([GitHub][6])                                                |
| 聚合编辑集                    | **GPT-Image-Edit-1.5M**（编辑基准 V2） |        ✅ | **1.5M**，聚合自 UltraEdit、HQ-Edit、OmniEdit、Complex-Edit 等 | 依**来源**而异          | 提供统一格式与子集说明；使用时需遵守各源集许可。([Hugging Face][7])                                                |
| 结构/控制条件                  | **ControlNet（示例集 fill50k）**      |        ✅ | **fill50k** 示范数据包（教学/验证）                               | *未明示*              | 官方未发布完整训练语料；推荐**用检测器在公有图集上自建条件**（姿态、边缘、深度等）。([GitHub][8], [Hugging Face][9], [MMagic][10]) |
| 结构/控制条件                  | **T2I-Adapter**                  | ✅（代码/权重） | *无官方固定数据集*                                             | —                  | 官方建议**自备数据**（如 COCO/LAION 派生并生成控制信号）进行训练。([GitHub][11], [arXiv][12])                       |
| 图像提示/风格参照                | **IP-Adapter**                   | ✅（代码/权重） | *无官方固定数据集*                                             | **Apache-2.0**（仓库） | 提供**训练脚本**与数据 JSON 规范，需自建（参考图像 ↔ 文本/目标）样本。([GitHub][13])                                   |
| 示例参照编辑                   | **Paint-by-Example**             | ✅（代码/权重） | *方法型*（基于自监督裁剪/遮挡构造对）                                   | —                  | 论文/代码公开，训练对通常从通用图集**自生成**（非独立发布数据）。([GitHub][14], [CVF开放获取][15])                           |
| 通用 I2I 任务（复原/修补等）        | **Palette**                      | 论文/项目页公开 | *任务可复现*（在 ImageNet/COCO 等上**合成退化/掩码**）                 | —                  | 无独立数据发布；按论文流程**自建合成任务**（上色、补全、去 JPEG 等）。([arXiv][16], [SR3][17])                           |

[1]: https://github.com/HaozheZhao/UltraEdit "GitHub - HaozheZhao/UltraEdit"
[2]: https://huggingface.co/datasets/BleachNick/UltraEdit "BleachNick/UltraEdit · Datasets at Hugging Face"
[3]: https://github.com/OSU-NLP-Group/MagicBrush "GitHub - OSU-NLP-Group/MagicBrush: [NeurIPS'23] \"MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing\"."
[4]: https://huggingface.co/datasets/osunlp/MagicBrush "osunlp/MagicBrush · Datasets at Hugging Face"
[5]: https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit "UCSC-VLAA/HQ-Edit · Datasets at Hugging Face"
[6]: https://github.com/timothybrooks/instruct-pix2pix "GitHub - timothybrooks/instruct-pix2pix"
[7]: https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M?utm_source=chatgpt.com "UCSC-VLAA/GPT-Image-Edit-1.5M · Datasets at ..."
[8]: https://github.com/lllyasviel/ControlNet?utm_source=chatgpt.com "lllyasviel/ControlNet: Let us control diffusion models!"
[9]: https://huggingface.co/blog/train-your-controlnet?utm_source=chatgpt.com "Train your ControlNet with diffusers"
[10]: https://mmagic.readthedocs.io/en/latest/autoapi/mmagic/datasets/controlnet_dataset/index.html?utm_source=chatgpt.com "mmagic.datasets.controlnet_dataset"
[11]: https://github.com/TencentARC/T2I-Adapter?utm_source=chatgpt.com "TencentARC/T2I-Adapter"
[12]: https://arxiv.org/abs/2302.08453?utm_source=chatgpt.com "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models"
[13]: https://github.com/tencent-ailab/IP-Adapter "GitHub - tencent-ailab/IP-Adapter: The image prompt adapter is designed to enable a pretrained text-to-image diffusion model to generate images with image prompt."
[14]: https://github.com/Fantasy-Studio/Paint-by-Example?utm_source=chatgpt.com "Paint by Example: Exemplar-based Image Editing with ..."
[15]: https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Paint_by_Example_Exemplar-Based_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf?utm_source=chatgpt.com "Exemplar-Based Image Editing With Diffusion Models"
[16]: https://arxiv.org/abs/2111.05826?utm_source=chatgpt.com "Palette: Image-to-Image Diffusion Models"
[17]: https://iterative-refinement.github.io/palette/ "Palette: Image-to-Image Diffusion Models"



