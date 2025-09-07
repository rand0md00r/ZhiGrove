---
title: 多轮对话拓展idea
created: 2025-09-07 16:57
updated: 2025-09-07
origin: week-36
type: idea
status: draft
tags: []
links: []
---

## 问题
- 

## 原因/洞察
- 

## 下一步行动
- [ ] 
- [ ] 


## Raw Notes





如果我们对图文query分组的话，vision query的高斯分布 应该是受到 文本条件的影响 在均值和方差上产生一些轻微偏移 这样也可以  或者FiLM adanorm之类的

另外我昨天在想 要避免高斯分布是纯噪声，不用构建正负样本 其实也可以换成siglip的sigmoid损失 最终结果其实更加近似对比学习。另外也可以得到高斯分布后往回预测。反正很多方法，跑通之后都一个个试一下，作为分析章节的一部分。


后面我们还可以做多轮对话那种  存储metaquery的KV矩阵做矩阵 然后利用ODE的可逆性   这个可能也是后面能拓展的一点
