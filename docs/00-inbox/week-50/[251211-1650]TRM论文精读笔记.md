

# 1. Introduction

1. 分层推理模型（HRM）在（如數独求解、迷宫寻路和 ARC-AGI）等任务上实现了更好的效果，主要贡献为：
    - 递归式分层推理：
        两个小型网络（高频递归的\(f_{L}\)和低频递归的\(f_{H}\)）进行多次递归，以实现答案预测。
    - 深度监督：
        - 通过多个监督步骤对答案进行改进；
        - 将潜在特征从计算图中分离，构建了残差连接，可模拟极深神经网络；

# 2. Background

1. 分层推理模型HRM 包含四个组件；
    - 输入嵌入层\(f_{I}(\cdot ; \theta_{I})\)：将原始输入\(\tilde{x}\)转换为嵌入向量；
    - 低层级循环网络\(f_{L}(\cdot ; \theta_{L})\)：高频递归，输出潜在特征\(z_H\)；
    - 高层级循环网络\(f_{H}(\cdot ; \theta_{H})\)：低频递归，输出潜在特征\(z_L\)；
    - 输出头\(f_{O}(\cdot ; \theta_{O})\)：将高层级潜在特征映射为预测结果。
    - 经过嵌入处理后，其形状变为\([B, L, D]\)，其中D为嵌入维度；
    - 每个网络均采用 4 层 Transformer 架构，并集成了以下技术细节：RMS 归一化（RMSNorm）（Zhang 与 Sennrich，2019）、无偏置设计（Chowdhery 等人，2023）、旋转嵌入（rotary embeddings）（Su 等人，2024）以及 SwiGLU 激活函数（Hendrycks 与 Gimpel，2016；Shazeer，2020）。

2. 关键超参数：
    - 每轮包含\(n=2\)次\(f_L\)调用、1 次\(f_H\)调用，该过程共执行\(T=2\)次；
    - 潜在变量\(z_L\)（低层级潜在特征）、\(z_H\)（高层级潜在特征）：初始为嵌入向量，或继承前一深度监督步骤的分离后特征。

3. HRM 前向传播完整步骤
。。。。。。

4. 基于一步梯度近似的不动点递归
    - 假设低层级潜在特征\(z_L\)与高层级潜在特征\(z_H\)通过\(f_L\)（低层级循环网络）和\(f_H\)（高层级循环网络）的递归运算达到不动点\((z_{L}^{*}, z_{H}^{*})\)
    - 采用隐函数定理（Krantz & Parks, 2002）结合一步梯度近似（Bai 等人，2019）来估算梯度（仅对最后一次\(f_L\)运算和\(f_H\)运算进行反向传播。大幅降低了模型的内存需求）；

5. 深度监督：将前一轮的潜在特征作为下一轮的初始化信息。研究中最多使用\(N_{sup}=16\)个监督步骤。

6. 自适应计算时间（Adaptive Computational Time, ACT）
    - 

# 3. Target for improvements in Hierarchical Reasoning Models

# 4. Tiny Recursion Models

1. 先执行\(T-1\)次无梯度的递归过程以优化（\(z_L, z_H\)），之后再执行 1 次带反向传播的递归过程。

2. TRM 优化了 HRM的层级结构，不再区分HL两个层级，简化为：输入x，候选解y（之前的z_H），推理潜在特征z（之前的z_L）；

3. 在给定输入问题x、当前候选解y与当前推理潜在特征z的情况下，模型会通过递归优化其推理潜在特征z；随后，基于当前的推理潜在特征z与此前的候选解y，模型会生成一个新的候选解y（若当前候选解已足够优，则保持不变）。

4. 将特征z拆分为多个不同的特征，会导致模型性能的下降。
    - 问题：多特征是否是多token的预测？

5. 将f_L和f_H 简化为单个模型，同时完成两项任务： \(z \leftarrow f_{L}(x+y+z)\) 和 \(y \leftarrow f_{H}(y+z)\)；

6. 增加层数会因过拟合导致模型泛化能力下降。反而，在减少层数的同时，按比例增加递归次数n（以确保计算量和模拟深度大致保持不变），使用 2 层网络（而非 4 层网络）时，模型的泛化能力达到最优。

7. 如果上下文长度较长（L >> D, D 为特征纬度），自注意力机制对序列建模更优，需要一个纬度为[D, 3D]的参数矩阵。如果上下文长度满足L <= D, 线性层的成本更低，仅需一个纬度为[L, L] 的参数矩阵；

8. 去掉（来自 Q 学习的）持续损失（continue loss），仅通过 “已得到正确解” 的二元交叉熵（Binary-Cross-Entropy）损失来学习停止概率（halting probability）。

9. 引入了权重的指数移动平均（Exponential Moving Average，简称 EMA）技术 —— 这是生成对抗网络（GANs）和扩散模型中常用于提升稳定性的经典技术；

10. 

TRM 伪代码：

``` python
def latent_recursion(x, y, z, n=6):  # n默认6，对应论文最优值
    for i in range(n):  # 循环n次，完成一轮推理迭代（论文中n=6）
        # 1. 第一步：更新候选解y——用当前y和z精炼解（对应论文y ← f_H(y+z)）
        # 这里用同一个net（单网络替代HRM双网络），通过输入不含x区分“更新y”任务
        y = net(y, z)  
        # 2. 第二步：更新推理特征z——用x（问题）、当前y（解）、z（旧推理）迭代推理（对应论文z ← f_L(x+y+z)）
        # 输入包含x，明确“迭代z”任务（论文核心：用输入是否含x区分双任务）
        z = net(x, y, z)  
    return y, z  # 返回迭代后的新y（更优解）和新z（更完整推理）

def deep_recursion(x, y, z, n=6, T=3):  # T默认3，对应论文最优值
    # 第一部分：T-1轮“无梯度递归”——只迭代y和z，不计算梯度（节省显存+加速训练）
    # 论文逻辑：前T-1轮是“粗糙逼近解”，无需梯度；仅最后1轮需梯度用于参数更新
    with torch.no_grad():  # 禁用梯度计算，避免显存爆炸（TRM轻量化关键）
        for j in range(T−1):  # 循环T-1次（如T=3时，循环2次）
            # 每次调用latent recursion，完成n=6次基础迭代
            y, z = latent_recursion(x, y, z, n)  
    
    # 第二部分：1轮“有梯度递归”——正常计算梯度，用于后续反向传播
    y, z = latent_recursion(x, y, z, n)  
    
    # 返回结果：
    # 1. (y.detach(), z.detach())：剥离梯度的y/z（用于后续早停判断，不影响梯度流）
    # 2. output_head(y)：解的预测头（输出y_hat，对应论文“候选解的概率分布”）
    # 3. Q_head(y)：早停判断头（输出q_hat，对应论文“停止概率halting probability”）
    return (y.detach(), z.detach()), output_head(y), Q_head(y)

# 遍历训练数据（x_input：原始输入，y_true：真实标签/正确解）
for x_input, y_true in train_dataloader:  
    # 初始化y和z：y_init（初始解，如全零向量）、z_init（初始推理特征，如随机向量）
    y, z = y_init, z_init  
    
    # 深度监督：循环N_supervision步，每步递归后都计算损失并更新参数（论文核心优化）
    # 论文逻辑：传统递归只在最后一步算损失，易梯度消失；深度监督每步更新，稳定训练
    for step in range(N_supervision):  
        # 1. 输入嵌入：将原始输入x_input转为模型可处理的向量x（如CNN嵌入数独网格）
        x = input_embedding(x_input)  
        
        # 2. 调用深度递归：得到3个结果
        # (y, z)：无梯度的更新后解和推理特征；y_hat：解的预测；q_hat：早停概率
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)  
        
        # 3. 计算损失：双损失函数（对应论文“仅用BCE损失学习早停”）
        # 3.1 解的分类损失：用交叉熵计算预测y_hat与真实y_true的差距（核心任务损失）
        loss = softmax_cross_entropy(y_hat, y_true)  
        # 3.2 早停判断损失：用二元交叉熵（BCE）学习“解是否正确”（替代HRM的Q-learning持续损失）
        # 标签是(y_hat == y_true)：预测正确则为1，错误则为0（引导q_hat判断是否早停）
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))  
        
        # 4. 反向传播与参数更新：每步监督都更新，避免梯度消失
        loss.backward()  # 计算梯度
        opt.step()       # 优化器更新参数（如Adam）
        opt.zero_grad()  # 清空梯度，避免累积
        
        # 5. 早停机制：若q_hat>0（模型判断当前解已正确），则停止当前样本的迭代（节省时间）
        # 论文逻辑：避免无效递归，提升训练效率
        if q_hat > 0:  # early-stopping
            break

```