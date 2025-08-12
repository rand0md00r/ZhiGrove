# stunning-palm-tree
---
2025年8月11日10:25:24

``` python
def compute_clip_loss(self, z_text: torch.Tensor, v_img: torch.Tensor):
    """
    分布式稳定版 CLIP 对齐损失：
      - 文本/图像各自单位化
      - 关闭 autocast、强制 fp32 计算 logits
      - 全局 all-gather negatives + rank 偏移标签
      - 双向交叉熵（i->t, t->i）
    期望：z_text 池化后的维度 == v_img 的维度（否则请把一侧线性投到一致维度）。
    """
    # A) 池化文本向量并单位化
    t_vec = masked_mean_pool(z_text)                               # [B, D]
    t_feat = F.normalize(t_vec.float(), dim=-1, eps=1e-6)          # fp32

    # B) 图像向量单位化
    i_feat = F.normalize(v_img.float(),  dim=-1, eps=1e-6)         # fp32

    # C) 分布式全局 negatives
    i_all = self._all_gather_cat(i_feat)                           # [B_g, D]
    t_all = self._all_gather_cat(t_feat)                           # [B_g, D]

    B_local = i_feat.size(0)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        labels = rank * B_local + torch.arange(B_local, device=i_feat.device)
    else:
        labels = torch.arange(B_local, device=i_feat.device)

    # D) 关闭 autocast，fp32 计算 logits
    with torch.cuda.amp.autocast(enabled=False):
        logit_scale = torch.clamp(self.clip_logit_scale.exp(), max=100.0).to(torch.float32)

        # 局部 × 全局（两个方向）
        logits_i2t = logit_scale * (i_feat @ t_all.t())            # [B_l, B_g]
        logits_t2i = logit_scale * (t_feat @ i_all.t())            # [B_l, B_g]

        loss_i = F.cross_entropy(logits_i2t, labels)
        loss_t = F.cross_entropy(logits_t2i, labels)
        loss   = 0.5 * (loss_i + loss_t)

    # E) 监控指标（不参与反传）
    with torch.no_grad():
        acc_i = (logits_i2t.argmax(dim=1) == labels).float().mean()
        acc_t = (logits_t2i.argmax(dim=1) == labels).float().mean()
        self._clip_debug = dict(
            clip_acc_i2t=acc_i,
            clip_acc_t2i=acc_t,
            clip_logit_scale_val=logit_scale.detach().item(),
            b_local=B_local,
            b_global=i_all.size(0),
        )

    return loss


``` 
