# 王亚齐｜多模态大模型算法工程师
**手机** 18305172953 · **邮箱** yqwang_2008@163.com · **GitHub** github.com/rand0md00r  
**求职意向** 多模态/视觉-语言模型（VLM/LMM）算法研发 · 预训练与后训练 · 评测与落地

---

## 个人简介（Highlights）
- 聚焦**多模态生成/理解**，在 **OpenUni × CrossFlow** 融合框架上完成**从零到一**的模型设计、数据与训练落地，面向 ICLR 征稿方向。  
- 具备**大规模分布式训练**经验：24M 图文对预训练，**A100×128** 多机多卡，DeepSpeed（ZeRO）、混合精度（bf16）、梯度检查点、数据流水线优化。  
- 熟悉**对齐与后训练**：SFT、DPO/RLAIF、提示工程与few-shot模板、Function Call（JSON Schema），可从0到1搭建**评测与回归体系**。  
- 兼具工程化与产品化能力：vLLM 服务化、协议对接、可观测性（Prometheus/ELK），支撑**复杂车控助手**内测上线链路。

---

## 核心技能
- **模型与算法**：VLM（MetaQuery 机制）、VAE、Flow Matching/Diffusion、对比学习（CLIP）、KL 正则、指令微调、偏好对齐  
- **训练与加速**：PyTorch 2.x、DeepSpeed（ZeRO-2/3）、DDP、mmengine；bf16/AMP、Grad Acc、Gradient Checkpointing、数据并行/流水线并行实践  
- **推理与系统**：vLLM、JSON Schema/Function Call、Kafka/REST；Docker、Jenkins；监控与日志（Prometheus/ELK）  
- **编程**：Python/C++；数据工程与评测脚本自动化

---

## 工作经历
**吉利汽车中央研究院 · 人工智能中心｜算法工程师**  _2024.07–至今_

- **多模态生成对齐研究（ICLR 征稿）**  _2024.12–至今_  
  以 **OpenUni × CrossFlow** 为基础，构建**文本→图像潜空间**的对齐与生成路径：  
  - **架构**：提出**去除 TransEncoder** 的轻量化方案；设计 **Text VE Encoder**；将 **MetaQuery(last hidden state)** 经三层线性映射对齐至 **VAE** 潜空间；引入 **CLIP + KL + Flow Matching** 复合损失。  
  - **任务与数据**：统一 **Text-to-Image / Image-to-Image** 训练与评测流水线；构建 24M 图文对数据配置、数据清洗与采样策略。  
  - **训练**：DeepSpeed ZeRO、bf16、多机多卡（**A100×128**）、分布式日志与监控；关键 **ablation**（是否保留 TransEncoder、损失权重、分辨率/步数等）。  
  - **产出**：形成可复用的训练脚本与评测基线；论文撰写与实验完善中（跨模态生成 + Flow Matching 方向）。

- **领克 900 语音助手 · 复杂车控后训练**  _2024.10–2025.06_  
  构建座舱域**复杂控车**能力（空调/座椅/灯光/多媒体/窗帘等）的后训练与服务化链路：  
  - **数据与协议**：沉淀设备/功能 **schema** 与槽位校验，覆盖多轮澄清与歧义处理；统一 **system/few-shot** 模板与日常评测集。  
  - **训练与对齐**：SFT → 偏好对齐（DPO/RLAIF）；函数调用规范化（JSON Schema）；负例挖掘与安全拒答策略。  
  - **上线与可观测性**：串联 **语音 → NLU → 意图映射 → ECU/座舱域** 控制链路；vLLM 服务化；接入 **Prometheus/ELK** 实现端到端回归。

---

## 教育背景
- **东南大学** · 仪器科学与工程学院 · 电子信息 · 工学硕士  _2021–2024_  
- **嘉兴学院** · 机电工程学院 · 电气工程及其自动化 · 工学学士  _2017–2021_

---

## 论文与专利（精选）
- Yaqi, W., et al. *Dynamic Error-Based Leader-Follower Control System using Nonlinear MPC.* **ICAUR 2023**, Springer.  
- **发明专利（受理）**：基于分层混合代价地图的改进 Q-learning 路径规划方法（2022）

---

## 早期研究与实习（压缩保留）
- **路径规划/控制方向（研究生阶段）**：维诺图节点优化、分层混合代价地图、Frenet 局部重规划、MPC 跟踪控制（1 篇会议论文、1 项发明专利受理）。  
- **蔚来汽车 · Logsim 运维平台（Django） 实习** _2022.12–2023.03_：重构 Views、完善单测与 Jenkins 流水，新增数据质检与后验流程，提升稳定性；具备后端工程与CI/CD实践。

---

## 奖项（代表性）
- 全国研究生数学建模竞赛 **全国二等奖**（第一位次）  
- 国家智能网联汽车创新中心算法攻关任务 **贡献奖**
