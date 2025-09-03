````markdown
# 今日训练速记（2025-09-03）

**概览**：24M 图文对 · A100×64 · 分支 `wyq_dev`  
**今日目标**：A) 3→5 层 transencoder，`metaquery=256`（不解耦），**LR×5**；观察 loss/平台 loss  
B) 完成后做 **解耦模态 query** 的对照实验

---

## 关键配置（填数即可）
- LR：基线 → **×5** 后：`_____`；Warmup：`_____`；Grad Clip：`_____`
- Batch（全局/每卡）：`_____ / _____`；分辨率：`_____`
- Optim/WD/β：`_____`；ZeRO/wgrad_ckpt：`_____`
- 随机种子：`_____`；Commit：`<hash>`

---

## 实验 A：5 层 transencoder + 256 metaquery（不解耦）+ LR×5
**启动命令（占位）**
```bash
bash scripts/train.sh --config xxx.yaml \
  --trans-layers 5 --metaquery 256 \
  --lr <×5值> --warmup <steps> --grad-clip <val> --zero-stage <0/1/2/3> --bf16
````

**快速记录表（填关键迭代点）**

| iter | total | platform | clip | fm | kl | 备注 |
| ---: | ----: | -------: | ---: | -: | -: | -- |
|   1k |       |          |      |    |    |    |
|   5k |       |          |      |    |    |    |
|  10k |       |          |      |    |    |    |

**结论（一句话/条目）**

* 曲线：*****；平台 loss：*****；稳定性：\_\_\_\_\_

---

## 实验 B：解耦模态 query（A 完成后）

**设置**：`metaquery=256 = T + V`（例：T=128, V=128）
**变体**：B-0 不解耦（A 最优）；B-1 解耦+共享头；B-2 解耦+独立头

**命令占位**

```bash
# B-1
bash scripts/train.sh ... --metaquery_text 128 --metaquery_vis 128 --share_head true
# B-2
bash scripts/train.sh ... --metaquery_text 128 --metaquery_vis 128 --share_head false
```

**对比速记**

| 方案  | platform\@10k | best total | 备注 |
| --- | ------------: | ---------: | -- |
| A   |               |            |    |
| B-1 |               |            |    |
| B-2 |               |            |    |

---

## 产物

* 配置：`configs/...`
* 日志/面板：`<链接>`
* 最优 ckpt：`<path@step>`

---

## 观察 / TODO（勾选）

* [ ] LR×5 是否需更长 warmup / 更强 grad clip
* [ ] 梯度/激活异常？nan/爆炸
* [ ] 解耦是否带来平台 loss 改善
* [ ] 吞吐/通信是否成瓶颈

---

## 明日计划（一句话）

* 若 A 成功：固定最优超参，跑 B 的网格（T/V 比例、共享/独立头）
* 若 A 不理想：回退 LR→×3，延长 warmup，检查 norm/初始化/正则

```
```
