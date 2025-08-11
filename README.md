# stunning-palm-tree
---
2025年8月11日10:25:24

``` python
def compute_clip_loss(self, z_text: torch.Tensor, v_img: torch.Tensor):
    """
    对齐损失：fp32 计算 + 可学习温度 + 双向 CE。
    z_text: [B, L, D] or [B, D]
    v_img : [B, D]
    """
    # 1) 池化文本向量
    t_vec = masked_mean_pool(z_text)              # [B, D]

    # 2) 全部转 fp32，单位向量化
    i_feat = F.normalize(v_img.float(),  dim=-1)  # [B, D]
    t_feat = F.normalize(t_vec.float(),  dim=-1)  # [B, D]

    # 3) 学习温度（clamp 防炸；保持在计算图里）
    logit_scale = self.clip_logit_scale.exp()
    logit_scale = torch.clamp(logit_scale, max=100.0)

    # 4) 双向 logits（图->文，文->图），再做 CE
    logits_per_image = logit_scale * (i_feat @ t_feat.t())  # [B,B]
    logits_per_text  = logits_per_image.t()

    labels = torch.arange(i_feat.size(0), device=i_feat.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)

    return 0.5 * (loss_i + loss_t)

``` 
