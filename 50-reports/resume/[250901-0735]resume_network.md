<span id="resume-mode"></span>
# 王亚齐｜多模态大模型算法工程师
**手机** 183-0517-2953  ·  **邮箱** [yqwang\_2008@163.com](mailto:yqwang_2008@163.com) 
---

### Highlights

* **统一多模态生成架构**：提出“**模态Query → 稀疏 TransEncoder → VAE 潜空间**”并用 **Noise-Free Flow Matching** 训练；A100×128 规模稳定收敛，内部 **mFID↓ / CLIPScore↑ / 编辑一致性↑**；论文筹备（CVPR 2026）。
* **复杂车控量产落地**：RAG + 双阶段检索 + JSON Schema 约束，多轮澄清/拒识/越权防护；**领克 900 上线**，复杂指令**准确率 ≥98%**，**Top-5 重排 99.8%**。
* **数据与工程闭环**：多智能体**合成多轮/模糊语料**与困难/对抗集；**vLLM 服务化、灰度回滚、可观测性**完善。

### 核心技能

* **模型/算法**：主流LLM/VLM、VAE、Flow Matching、VICReg；Function-Call 规划与参数约束
* **训练/加速**：PyTorch、DeepSpeed、DDP、mmengine、LLaMa-Factory；
* **推理/系统**：**vLLM**、JSON Schema 约束解码、K8S/Docker/Jenkins；Jira项目跟踪；


<h3 class="title-line">吉利汽车中央研究院·人工智能中心 | 大模型算法研究岗 <span class="time"> 2024.07–至今 </span></h3>

___

<h4 class="title-line"><strong>（研究项目）通用跨模态生成架构</strong> <span class="time"> 2025.3–至今 </span></h4>

- **目标**：实现一种可扩展的通用跨模态信息融合生成架构，对文本/图像(/音频等模态)实现最优特征传输与Noise-Free Flow Matching生成。
* **负责工作**：提出**模态Query** + **稀疏TransEncoder** + **跨模态监督**，**完成A100 * 128集群训练**；
    - ① 设计 **模态Query** 对各自模态进行cross-attn，**ShareQuery**聚合共享特征避免特征重排导致的槽位歧义；
    - ② 通过 **精简的MoE TransEncoder** 将特征压缩映射到VAE潜空间，施加**KL 约束**确保分布可迁移；
    - ③ **VICReg监督** 稳定了z_init <--> z_end语义结构，防止z_init方差坍缩；
* **结果**：系列Bench**超越SOTA**，内部基准 mFID↓、CLIPScore↑、编辑一致性↑；同算力下参数成本更低。
* **论文计划**：**第一作者（第一贡献人）**；拟投 **CVPR 2026**。

---

<h4 class="title-line"><strong>（量产项目）领克 900 语音助手 · 复杂车控后训练</strong> <span class="time">2024.10–2025.06</span></h4>

* **目标**：用户 Query → **RAG** 在≈600条车控协议中召回 Top-5 function → 重排/校验 → **输出 Top-1 标准化指令**；支持**多轮**承接、**极端模糊澄清**、**非车控拒识**。
* **负责工作**：
    - ① **后训练**：Schema-conditioned **SFT** → 拒识/澄清**提示工程**；JSON Schema 约束解码与参数修复。
    - ② **RAG 优化**：语料 **Query-first** 重写+字段加权；双阶段召回/重排+权限校验；Recall@5/Top-1↑；BGE→Qwen 迁移。
    - ③ **数据/评测**：**多智能体**生成**多轮/模糊**数据与**困难/对抗样本**；
* **结果（量产）**：内部回归 **复杂指令准确率 ≥ 98%**；重排Top-5**准确率99.8%**；完成 **领克 900** 线上量产并稳定运行。
* **论文（在审）**：[（NeurIPS 2026）AutoControl-Bench: A Multi-Agent Knowledge Distillation Framework for Complex Vehicle Function Call Understanding](https://openreview.net/forum?id=lqdE0W6mFx#discussion)  [AutoBench-Data](https://github.com/HangerAmber/AutoBench-Data)

---


### 教育背景
* **东南大学** · 仪器科学与工程学院 · 电子信息（时序信息理解与生成）    · 工学硕士（2021–2024）
* **嘉兴学院** · 机电工程学院      · 电气工程及其自动化（优秀毕业生）   · 工学学士（2016–2020）

### 奖项（在校期间）
* 全国研究生数学建模竞赛 **全国二等奖**（第一位次）
* 国家智能网联汽车创新中心算法攻关任务 **贡献奖**
* 优秀毕业生
