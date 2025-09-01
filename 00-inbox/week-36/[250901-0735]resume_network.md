# 王亚齐｜多模态大模型算法工程师
**手机** 183-0517-2953  ·  **邮箱** [yqwang\_2008@163.com](mailto:yqwang_2008@163.com) 
---
### Highlights
* 聚焦 **多模态生成/理解**：在 *OpenUni × CrossFlow* 融合框架上完成从零到一的模型设计、数据与训练落地（面向 ICLR/NeurIPS）。
* **大规模分布式训练**：24M 图文对，A100×128，DeepSpeed ZeRO、bf16、梯度检查点、数据流水线优化。
* **后训练与落地**：SFT、DPO/RLAIF、Function-Call(JSON Schema)、评测与回归体系；支撑复杂车控助手量产链路。

### 核心技能
- **模型/算法**：VLM(MetaQuery)、VAE、Flow Matching/Diffusion、CLIP、KL、指令微调、偏好对齐
- **训练/加速**：PyTorch 2.x、DeepSpeed(ZeRO-2/3)、DDP、mmengine；bf16/AMP、Grad Acc、Gradient Checkpointing
- **推理/系统**：vLLM、Function-Call/JSON Schema、RAG、Docker、Jenkins、Prometheus/ELK
- **编程**：Python / C++；数据工程与评测脚本自动化

### 经历｜吉利汽车中央研究院·人工智能中心（2024.07–至今）
**跨模态生成架构（OpenUni × CrossFlow）**（2024.12–至今）

* **目标**：以 MetaQuery 为语义中介，构建“VLM 潜变量 → 图像 VAE 潜变量”的分布传输通道；无噪声 Flow Matching 对齐，统一 Text→Image 与 Image→Image/编辑，可规模化训练。
* **贡献**：① 设计 *MetaQuery→三层线性对齐→VAE 潜空间* 架构，移除 trans-encoder；② 构建 *FM+CLIP+KL* 多目标与 *log-SNR* 调度，自适应权重稳定训练；③ 完成 24M、A100×128、bf16 的分布式训练（DeepSpeed+mmengine）与统一评测/消融。
* **难点突破**：修复 *encode\_moments vs. sample* 潜变量偏移；缓解多目标梯度竞争；实现大规模可复现与故障回放（配置/日志/检查点一致）。
* **结果**：内部基准 **mFID↓、CLIPScore↑、编辑一致性↑**，同一套架构覆盖生成与编辑，**参数与维护成本更低**。
* **论文计划**：**第一作者（第一贡献人）**；拟投 **ICLR 2026**；阶段性结果超越 SOTA。

**领克 900 语音助手 · 复杂车控后训练（量产）**（2024.10–2025.06）

* **目标**：提升复杂/模糊/多轮语音指令理解与澄清，稳定输出**标准化车控指令**（空调/座椅/灯光/多媒体等），满足量产稳定与安全要求。
* **贡献**：① 车控后训练：沉淀 *schema/槽位/约束* 与澄清模板；*SFT→偏好对齐(DPO/RLAIF)*；Function-Call **JSON Schema** 强校验与拒答策略；② **RAG 注入**：**BGE→Qwen-Embedding** 迁移，重嵌入与 **FAISS/HNSW** 索引重建；二阶段检索(召回+重排)+权限校验；③ **高质量数据**：困难样本/负例/对抗样本；日常回归与多轮脚本。
* **难点突破**：模糊/组合意图解析（schema约束+槽位补全+置信门控/多轮澄清）；安全与合规（二次确认、白/黑名单、边界冲突检测）；工程与可观测性（vLLM 服务化、灰度回滚与异常回放）。
* **结果（量产）**：内部回归 **指令准确率 ≥ 98%**；误触发与越权显著下降；完成 **领克 900** 线上量产并稳定运行。

### 发表/投稿
* **NeurIPS 2026（在审）**：**共同第一作者（Equal Contribution，第2位）**（[OpenReview](https://openreview.net/forum?id=lqdE0W6mFx)）。

### 教育背景
* **东南大学** · 仪器科学与工程学院 · 电子信息 · 工学硕士（2021–2024）
* **嘉兴学院** · 机电工程学院 · 电气工程及其自动化 · 工学学士（2017–2021）

### 奖项
* 全国研究生数学建模竞赛 **全国二等奖**（第一位次）
* 国家智能网联汽车创新中心算法攻关任务 **贡献奖**
