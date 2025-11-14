# 论文记录模板（复制用）
> 每篇论文按此模板填写，并放到合适分区（可在多个分区留“短条目”交叉引用）
## 标题（年份，会议/期刊）
- **Citation**：作者，题目，会议/期刊，年份
- **链接**：论文 | 代码 | 项目页（若有）
- **任务**：操作 / 导航 / 混合；场景（家庭/工业/车载…）
- **架构**：Planner-Executor / 端到端；是否工具调用；是否记忆模块
- **动作范式**：AR / 扩散 / Flow Matching / 混合；动作空间（离散/连续；关节/技能/程序）
- **感知输入**：RGB / 深度 / 语言 / 多视角 / 视频长度
- **训练**：预训练数据（规模/来源）、SFT、偏好/强化（DPO/RLAIF/RLHF）、奖励设计
- **数据**：自建 / 公共；真机/仿真；合成/蒸馏
- **评测**：基准与指标；关键结果（可列 1–3 个数字）
- **开源程度**：权重 / 训练代码 / 推理代码 / 数据（许可证）
- **部署**：推理硬件、时延、吞吐、边缘可行性
- **亮点**：3–5 条要点
- **局限**：2–3 条要点
- **个人笔记**：你的理解、与现有系统的可复用性、TODO

---

## π₀․₅: a Vision-Language-Action Model with Open-World Generalization（2025，CoRL）

* **Citation**：Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, *et al.* “π₀․₅: a Vision-Language-Action Model with Open-World Generalization.” *Proceedings of the 9th Conference on Robot Learning (CoRL)*, PMLR 305:17–40, 2025. ([Proceedings of Machine Learning Research][1])

* **链接**：论文（arXiv | PMLR）| 代码（openpi）| 项目页/博文

  * arXiv：([arXiv][2])
  * PMLR页面与PDF：([Proceedings of Machine Learning Research][1])
  * 博文解读：([Physical Intelligence][3])
  * 开源仓库（Apache-2.0）：([GitHub][4])
  * HF 权重（LeRobot 转换）：([Hugging Face][5])

* **任务**：操作（家居场景的长任务，如收拾餐具、擦除污渍、整理卧室）；强调“陌生家庭”零样本泛化。 ([Physical Intelligence][3])

* **架构**：单一 VLA 模型同时做高层语义决策与低层连续控制；推理时先自述式高层子任务（AR 文本），再生成低层“动作块”（Flow Matching 连续控制）。非工具调用范式；无专门记忆模块描述。 ([Physical Intelligence][3])

* **动作范式**：**混合**＝高层**自回归**（文本子任务）+ 低层**Flow Matching**（连续关节控制）；低层以约 1s/50 步的“action chunk”输出连续关节指令；动作空间为**连续**（关节级）。另含约 3e8 参数的“action expert”。 ([Physical Intelligence][3])

* **感知输入**：RGB 视觉 + 语言指令；训练期采用混合多模态信号（检测框、子任务标签、网页多模态等）。([Physical Intelligence][3])

* **训练**：核心是**异构协同训练（co-training）**与混合监督：

  * 来源：移动操作真机数据、跨环境静态/移动机器人数据、跨载体（cross-embodiment）数据、网页多模态任务（VQA/Caption/Detection）、“口头教练”逐步指令等；文中对**取消某一来源**的消融。([Physical Intelligence][3])
  * 规模：开源仓库给出 openpi 基座模型“**10k+ 小时机器人数据**”预训练（π₀/π₀-FAST/π₀․₅均提供 base ckpt）；文中移动操作自采数据在某些消融中约 **400 小时**。([GitHub][4])
  * 相关技术：知识绝缘（Knowledge Insulation）作为 π₀․₅ 训练配方升级的一部分在 repo/白皮书中提及。([GitHub][4])

* **数据**：**真机**为主，家庭/办公室等多环境；**跨载体/跨环境**整合 + **网页多模态**；并在开源模型中提供与 **DROID/LIBERO** 等**公共数据**相关的变体与微调示例。([GitHub][4])

* **评测**：两大设置——**完整清洁任务**与 **OOD 指定物体入抽屉**；指标为**语言跟随率**与**成功率**。关键数字（博文给出）：

  * **IID 成功率**：83%（π₀․₅） vs 57%（无多环境数据 ME）/67%（无跨载体 CE）。
  * **OOD 成功率**：**94%**（π₀․₅） vs 31%（无 ME）/49%（无 CE）/74%（无网页数据 WD）。
  * **结论**：WD 对 OOD 类别识别助益大；ME/CE 对整体泛化关键。([Physical Intelligence][3])

* **开源程度**：**权重**（π₀․₅ base 与若干任务专家）、**训练与推理代码**（现已含 PyTorch 训练流程）、**数据接口/示例**均开放（Apache-2.0）；HF 提供 PyTorch safetensors 转换。([GitHub][4])

* **部署**：官方 openpi README 建议**单卡 RTX 4090（>8GB）可推理**；LoRA 微调约 >22.5GB；全参微调需 A100/H100 级显存。已给出 Docker/UV 环境与多 GPU FSDP 选项。([GitHub][4])

* **亮点**：

  1. **高/低层一体化**：同一模型先“想”（AR 文本）再“做”（Flow连续控制），贴近“内在推理→动作”的链式流程。([Physical Intelligence][3])
  2. **协同训练配方**显著提升陌生环境泛化，**WD/ME/CE** 各司其职。([Physical Intelligence][3])
  3. **开放生态**：代码、基座与专家权重、训练脚本、与 DROID/LIBERO 的适配齐备，易于再训练与复现。([GitHub][4])
  4. **连续动作的 Flow Matching** + **动作专家**，在灵活性与稳定性间取得平衡。([Physical Intelligence][3])
  5. **可扩展数据配方**与环境数缩放实验，显示“100+ 环境”后接近在测试环境训练的上限。([Physical Intelligence][3])

* **局限**：

  1. 官方明确**非追求极致灵巧度**，在新家务场景仍有失败案例；成功并非稳定 100%。([Physical Intelligence][3])
  2. 训练依赖**大规模多源数据**与较重 GPU 资源；低资源/新平台迁移仍需微调与工程适配。([GitHub][4])
  3. 文中未强调外部工具/记忆模块，**长时任务的显式记忆与可解释规划**仍有提升空间（研究社区正在探索）。〔基于文献对比推断；论文未主打该点〕

* **个人笔记**：

  * **与现有系统复用性**：对车内/家居“多阶段流程”的**高层语言-子任务**再到**低层控制**非常契合“Planner（文本）→ Executor（连续动作）”的工程拆分；你现有 VLA-Cabin/工业巡检可复用其**协同训练配方**与**Flow 低层头**，并将**WD（网页/合成）+ CE/ME（跨载体/跨环境）**纳入数据管线。
  * **落地建议/TODO**：

    1. 以 **π₀․₅ base** 为起点，在自家数据上做 **KI（知识绝缘）配置 + LoRA** 微调；优先接入 **DROID/LIBERO** 公共数据做对齐；桌面→移动基座分阶段蒸馏。([GitHub][4])
    2. 保持**高层 AR 子任务链**可观测（日志化），同时评估低层 **1s/50步动作块**的时延与稳定性，必要时在仿真中做 action-chunk horizon 的灵敏度实验。([Physical Intelligence][3])
    3. 建立**WD/ME/CE** 的**可插拔数据开关**，复现实验室里的 ablation 曲线，作为泛化“健康度”回归测试。([Physical Intelligence][3])

[1]: https://proceedings.mlr.press/v305/black25a.html "$\pi_0.5$: a Vision-Language-Action Model with Open-World Generalization"
[2]: https://arxiv.org/abs/2504.16054?utm_source=chatgpt.com "[2504.16054] $π_{0.5}$: a Vision-Language-Action Model ..."
[3]: https://www.physicalintelligence.company/blog/pi05 "A VLA with Open-World Generalization"
[4]: https://github.com/Physical-Intelligence/openpi "GitHub - Physical-Intelligence/openpi"
[5]: https://huggingface.co/lerobot/pi05_base?utm_source=chatgpt.com "lerobot/pi05_base"


---

## RoboOmni: Proactive Robot Manipulation in Omni-modal Context（2025，arXiv）
- **Citation**：Siyin Wang, Jinlan Fu, Feihong Liu, et al. RoboOmni: Proactive Robot Manipulation in Omni-modal Context. arXiv:2510.23763v3 [cs.RO], 2025
- **链接**：论文 https://arxiv.org/pdf/2510.23763 | 代码 https://github.com/OpenMOSS/RoboOmni | 项目页 https://OpenMOSS.github.io/RoboOmni | Hugging Face https://huggingface.co/collections/fnlp/roboomni
- **任务**：操作；场景（家庭）
- **架构**：Perceiver-Thinker-Talker-Executor 端到端；无工具调用；无记忆模块
- **动作范式**：离散令牌（FAST+ tokenizer）；动作空间（离散2048个令牌，映射7-DoF连续关节级控制向量）
- **感知输入**：RGB、语言（文本）、音频（语音+环境声音）；多模态时序输入
- **训练**：预训练数据（OmniAction数据集141k episodes + Open-X Embodiment子集；规模140k+ episodes、来源合成构建）、SFT（下游任务微调）、无偏好/强化训练、奖励设计（自回归最大似然目标，融合对话与动作生成损失）
- **数据**：自建（OmniAction）+ 扩展公共（LIBERO）；真机+仿真；合成（文本脚本+听觉实现+验证）
- **评测**：基准（OmniAction-LIBERO-TTS、OmniAction-LIBERO-Real）；指标（成功率、意图识别准确率、推理时延）；关键结果（平均成功率85.6%、意图识别准确率88.9%、推理时延为ASR+OpenVLA的0.49倍）
- **开源程度**：权重（Hugging Face） / 训练代码（开源） / 推理代码（开源） / 数据（开源，许可证未知）
- **部署**：推理硬件（RTX 4090）、时延（0.49×相对基线）、吞吐（未明确）、边缘可行性（潜力较高，端到端效率优）
- **亮点**：
  1. 提出跨模态上下文指令新范式，无需显式指令，从视、听、文本多模态推断用户意图
  2. 端到端框架统一意图识别、交互确认、动作执行，支持直接语音交互（无需ASR）
  3. 自建大规模OmniAction数据集，覆盖6类上下文指令、5k+说话人、多场景声学环境
  4. 推理速度远超基线，时延仅为传统ASR+VLA pipeline的一半
  5. 真机与仿真实验均验证优势，意图识别与主动协助能力突出
- **局限**：
  1. 非语言指令任务成功率相对较低（~82%），仍是核心挑战
  2. 预训练依赖大规模硬件资源（64 A100 GPUs训练10天），训练成本高
  3. 真机实验仅基于WidowX 250S机械臂，复杂场景与多机器人适配性待验证
- **个人笔记**：RoboOmni的核心突破是打破了机器人对显式指令的依赖，多模态端到端设计既保留了细粒度语义信息，又提升了执行效率，OmniAction数据集的构建流程（文本脚本→听觉实现→验证）可复用在多模态机器人任务中。其主动交互逻辑（推断意图→确认→执行）贴近真实人机协作场景，可复用至家庭服务机器人系统。TODO：关注其在工业场景的适配方案，以及轻量化模型的部署可能性。

---

