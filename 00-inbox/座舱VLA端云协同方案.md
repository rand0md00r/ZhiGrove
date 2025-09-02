# 云端VLA方案：快速落地 vs 最优路线 v1.0

> 聚焦**云端 VLA 模型研发**（多源输入→理解→决策），提供两条路径：
>
> * **T0：快速落地（4–6 周）**——复用现有开源全模态/多模态模型 + 函数调用/工具路由，最小改造即可出“Top‑K 决策选项”。
> * **T1：最优方案（10–12 周）**——“**SOTA 理解能力** × **离散扩散（或并行掩码）实现的**高速**动作序列生成**”的两段式架构，支持偏好/安全对齐与低延迟并发。

---

## 0. 一页图（云端主流程）

```
[多源输入 Edge→Cloud]
 image / audio(ASR) / dialog / vehicle signals / user profile / history
        │
        ▼
[Context Builder]
  - 时间对齐 + 轻量事件抽取 + 场景摘要(可复用Edge caption)
  - 统一表示：Observation JSON + Tokenized Context
        │
        ▼
[理解(Teacher)：SOTA VLM/Omni]
  - 任务识别 / 约束解析 / 风险识别 / 工具规划(函数调用意图)
        │
        ├── T0：直接产出 Top‑K 方案(JSON) + 解释(LLM生成)
        │
        └── T1：产出“**目标行动计划 y**”(教师标签/或草案) →
               [高速生成器(Student)] **离散扩散/并行掩码** 生成最终动作序列
                + 解释头(蒸馏的小LLM)
        │
        ▼
[Safety & Constraints]
  - 合规/权限/行车安全/童乘偏好
  - 规则校验 + 反事实模拟 + 回滚方案
        │
        ▼
[Top‑K OptionCards]
  - title, action_plan[], expected_effects, risks, undo_plan, explanation
        │
        ▼
[日志与偏好学习]
  - 曝光→点击/拒绝/撤销→RLAIF/DPO 蒸馏到生成器与解释头
```

---

## 1. 输入与统一表示（Multi‑Source Fusion）

* **输入**：

  * 视觉：车内/外关键帧（脱敏/裁剪/低清或潜特征）
  * 语音：ASR 文本 + 置信度 + 说话人/情绪（可选）
  * 车况：车速、档位、空调/灯光/媒体、环境温湿度/PM2.5、儿童在座等
  * 用户画像：温度/亮度/音量偏好、勿扰时段、历史点击偏好摘要
  * 会话历史：近 K 轮对话摘要与已执行动作
* **统一Observation（示例）**：

```json
{
  "ts": 1735912345123,
  "scene": {"driving": true, "child_present": true, "time_of_day": "night"},
  "signals": {"speed": 60, "cabin_temp": 28.5, "volume": 10, "media_state": "pause"},
  "intent": {"text": "有点热，调舒服点", "conf": 0.86},
  "profile": {"temp_pref": 24, "volume_pref": 8, "quiet_hours": [22,7]},
  "vision": {"latent": [1024], "caption": "后排小孩睡着"},
  "history": [{"op": "set_temp", "val": 25}]
}
```

* **时序与记忆**：滑窗缓冲(30–120s)→事件抽取（如“刚入隧道”“儿童入睡”）→轻量记忆键值（FAISS/向量DB）→上下文拼接。

---

## 2. 决策语义与动作 DSL（Action Ontology）

* **目标**：从开放自然语言/感知到**受约束的动作 Token 序列**（可回放、可审核）。
* **动作DSL**（例）：

  * `set_temp(zone, val)`，`set_fan(zone, val)`，`set_mode(auto/eco/defog)`，
  * `set_seat(pos_id/heat_level/massage_mode)`，`set_light(zone, on/off, brightness, cct)`，
  * `set_window(zone, percent)`，`set_sunshade(zone, percent)`，
  * `media(play/pause/next, volume±n, source)`，`scene_activate(name, params)`。
* **Token 设计**：`[OP] [ARG1] [ARG2] ... [SEP]`；离散词表≤2–5k；长度≤32 tokens；支持占位/掩码。

---

## 3. T0 快速落地（开源模型直连）

### 3.1 基线模型与路线

* **候选**：

  * **Qwen2.5‑Omni**（全模态）、**InternVL2/3**、**GLM‑4V**、**LLaVA‑OneVision**、**MiniCPM‑V 2.6**（根据可得性/中文表现选择）。
* **方式**：

  1. 将 Observation 摘要化后作为 **系统Prompt + 工具Schema** 输入；
  2. 通过 **函数调用/JSON Schema** 直接产出 Top‑K `action_plan[]` 与 `explanation`；
  3. 规则层进行**安全校验与修正**；
  4. 返回 **OptionCards**。

### 3.2 函数调用 Schema（示例）

```json
{
  "name": "propose_options",
  "description": "根据Observation生成3个可执行的候选方案",
  "parameters": {
    "type": "object",
    "properties": {
      "cards": {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {
          "type": "object",
          "properties": {
            "title": {"type": "string"},
            "action_plan": {"type": "array", "items": {"type": "string"}},
            "expected_effects": {"type": "object"},
            "risk_tags": {"type": "array", "items": {"type": "string"}},
            "undo_plan": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"}
          },
          "required": ["title","action_plan","explanation"]
        }
      }
    },
    "required": ["cards"]
  }
}
```

### 3.3 系统提示词模板（片段）

```
你是车载云端VLA决策引擎。目标是在**安全优先/最小变更/可撤销**的前提下，
基于Observation生成3个可执行的候选方案(JSON)。
硬性约束：夜间音量≤10；行车中禁影响驾驶注意的动作；儿童在座优先安静。
输出须包含：title, action_plan(动作DSL字符串序列), explanation, undo_plan。
```

### 3.4 数据与轻量微调

* 构造 **模板化合成数据**（Observation→3候选）+ **规则引擎自动标注**（风险/约束/撤销）。
* 对大模型进行 **LoRA/SFT**（2–3M 例）以稳定函数调用格式与动作DSL一致性。
* 线上日志用于 **RLAIF/DPO**：以**曝光→点击/拒绝/撤销**为偏好对，周期性蒸馏。

### 3.5 工程与SLA

* **Serving**：vLLM/TensorRT‑LLM；KV缓存+提示裁剪；Speculative/Medusa/Parrot 解码。
* **延迟**：P95 700–900ms（base‑7B/omni‑small）。
* **优点**：最快上线；**缺点**：动作生成仍受自回归限制，延迟与稳定性受模板依赖。

---

## 4. T1 最优方案（理解×高速生成）

### 4.1 两段式结构

* **段A：理解/规划（Teacher）**

  * 选择 **SOTA VLM/Omni**（同 T0）作为**理解与规划教师**：解析任务、抽取约束、给出**行动草案 y\_draft**与解释草案。
* **段B：高速生成（Student）**

  * 以 **离散扩散/并行掩码(MaskGIT/MaskLLM)** 为核心，**并行生成**动作Token序列；
  * **步数**：4–8 步（远少于自回归）；
  * **指导**：Classifier‑Free Guidance + 代价/风险引导（见4.3）；
  * **解释头**：蒸馏到**小型LLM**（1–3B）以生成自然语言解释（可并行/延后）。

### 4.2 离散扩散训练（D3PM/掩码并行）

* **表示**：将动作DSL序列 `y` 映射为离散词表 `V` 上的 token 序列；
* **前向噪声**：`q(y_t | y_{t-1})` 为**置换/掩码/均匀替换**的马尔可夫核；
* **目标**：学习 `p_θ(y_{t-1} | y_t, c)`（条件 `c` 为 Observation 表示与 y\_draft）；
* **损失**：交叉熵/KL + 辅助一致性 loss（与教师草案对齐）；
* **推理**：并行去噪（或 MaskGIT 风格**并行填充+置信度调度**）→ 4–8 步收敛；
* **替代实现**：将离散 token **连续化嵌入**，用**Rectified Flow/Flow Matching** 学习**向量场**，再 `argmax/采样` 回离散词表（ST‑Gumbel）。

### 4.3 约束与偏好作为“引导”

* **规则/安全引导**：在每步去噪时，对不合法 token 施加惩罚（mask‑out 或 logit penalty）；
* **偏好引导**：学习一个**Reward/Preference 模型** `R(y,c)`，采样时做 **energy‑guided decoding**：最大化 `log p_θ - λ·E(y)`；
* **成本引导**：惩罚“多步骤/高变更幅度”方案，优先最小变更。

### 4.4 训练数据与蒸馏

* **教师数据**：T0/线上日志的 Observation→y\_draft→最终被点击方案（正样本）；
* **合成扩充**：规则引擎变体、扰动噪声、反事实（违反约束的负样本）；
* **蒸馏**：将教师产生的**解释**蒸馏到小LLM；将选择偏好蒸馏为 Reward 模型。

### 4.5 推理并行与SLA

* **动作序列**：非自回归并行，**4–8步**即可得到**32 tokens**以内的计划，

  * 预计**模型时延 60–150ms**（不含视觉编码），端到端 P95 **≤ 500–700ms**；
* **解释**：异步/流式生成（不阻塞动作落地）。

---

## 5. 安全与合规护栏（云端）

* **符号规则层**：硬约束白名单/黑名单；时间/场景/权限；
* **一致性校验**：动作DSL解析→API参数合法性检查→冲突检测（例如“夜间音量上限”）；
* **反事实模拟**：对候选进行简易**效果模拟**（如温度/音量预估），筛掉高风险方案；
* **回滚**：每个方案必须给出 `undo_plan`；执行后允许 **10s 撤销**。

---

## 6. 评测指标

* **理解**：意图/约束识别F1、风险召回、工具规划正确率；
* **生成**：Action DSL 语法通过率、API执行成功率、最小变更率、步数；
* **体验**：Top‑K CTR、一击成功率、撤销率、解释满意度；
* **延迟/成本**：端到端P95、GPU‑s/请求、显存占用、上云字节数；
* **安全**：违规动作拦截率、冲突率。

---

## 7. 里程碑（建议）

* **W1–W2（T0）**：接入 Qwen2.5‑Omni/InternVL3；完成 Observation→OptionCards 函数调用；规则校验；离线用例100条通过。
* **W3–W4（T0）**：数据模板扩充+LoRA；A/B 小流量；看板与日志闭环。
* **W5–W8（T1）**：构建离散扩散生成器(Mask/D3PM) + 蒸馏解释头；离线对比自回归。
* **W9–W12（T1）**：上线并行生成器灰度，偏好引导与RLAIF；延迟和成功率达标。

---

## 8. 工程实现要点

* **微服务**：Context Builder / Teacher(LMM) / Student(离散扩散) / Safety / Ranker / Logger；
* **Serving**：Teacher 用 vLLM；Student 用 Torch/TensorRT 如果是并行掩码；
* **缓存**：Prompt裁剪、KV缓存、Observation模板化、视觉特征缓存；
* **可观测性**：每步去噪置信度直方图、规则命中分布、点击/撤销漏斗；
* **成本**：Teacher 降采样+蒸馏；Student 小模型多副本并发。

---

## 9. 风险与对策

* **离散扩散不稳定/步数过多** → 先用 **MaskGIT并行填充** 做过渡；
* **数据稀缺** → 规则/模拟器合成 + 线上日志蒸馏；
* **安全覆盖不足** → 规则先行 + 反事实压力测试 + 执行前静态校验；
* **中文场景理解偏差** → 选型偏中文SOTA，必要时做中文领域LoRA。

---

## 10. 附：Observation→输出示例

**输入Observation（摘要）**：夜间行车，后排儿童入睡，车内28.5℃，用户说“有点热，调舒服点”。

**T0输出（概念）**：

* Card1(推荐)：`set_temp(all,24) → set_fan(front,2) → media(volume,-2)`；解释“优先安静舒适”。
* Card2：`set_temp(all,24)`；解释“最小变更”。
* Card3：`保持不变`。

**T1输出（概念）**：

* 并行去噪 6 步得到同等三卡；模型时延≈100ms（不含视觉编码），解释异步补齐。
