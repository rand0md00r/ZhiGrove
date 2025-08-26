@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    progress_bar: bool = False,
    cfg_scale: float = 4.5,
    num_steps: int = 50,
    generator: torch.Generator = None,
    height: int = 512,
    width: int = 512,
    **kwargs
) -> torch.Tensor:
    """
    与评测脚本兼容的生成函数（fp32 积分累加，避免 bf16 量化吃掉细小步进）。
    返回 [B_cond, 3, H, W] in [-1,1]
    """
    device, dtype = self.device, self._dtype
    self.eval()

    # ---- 0) 拆分 cond / uncond（此处先不做 CFG，引导关掉更易定位） ----
    assert input_ids.dim() == 2 and attention_mask.dim() == 2, "inputs应为 [B, L]"
    B_total = input_ids.size(0)
    assert B_total % 2 == 0, "需要 cond/uncond 数量相同"
    B_cond = B_total // 2

    ids_c, mask_c = input_ids[:B_cond].to(device), attention_mask[:B_cond].to(device)

    # ---- 1) LLM+TransEncoder -> z_text ----
    def _z_text_from(ids, mask):
        q_meta = self.meta_queries[None].expand(ids.size(0), self.num_queries, -1).to(device=device, dtype=dtype)
        llm_in = self.prepare_forward_input(x=q_meta, input_ids=ids, attention_mask=mask)
        llm_out = self.llm.model(**llm_in, return_dict=True).last_hidden_state
        q_tokens = llm_out[:, -self.num_queries:]
        q_mask   = mask.new_ones(q_tokens.shape[:2])

        out = self.context_encoder(q_tokens.to(dtype), q_mask)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) if generator is None else torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
        z  = eps * std + mu
        if z.dim() == 3:
            z = z.reshape(z.size(0), -1)
        return z

    z0 = _z_text_from(ids_c, mask_c)   # 纯条件基线；CFG 先关闭更稳

    # ---- 2) moments 形状，与训练一致的 reshape ----
    dummy_img = torch.zeros(B_cond, 3, height, width, device=device, dtype=dtype)
    mom_dummy = self.autoencoder(dummy_img, fn='encode_moments')
    if isinstance(mom_dummy, (list, tuple)):
        mom_dummy = mom_dummy[0]
    _, C, Hh, Ww = mom_dummy.shape

    x = einops.rearrange(z0, 'b (c h w) -> b c h w', c=C, h=Hh, w=Ww).to(dtype=dtype, device=device)

    # ---- 3) 时间网格（logit-normal 分位点） ----
    def _build_logit_normal_schedule(K: int, device, *, mean=0.0, std=1.0, eps: float = 1e-6):
        normal = torch.distributions.Normal(torch.tensor(mean, device=device), torch.tensor(std, device=device))
        qs = torch.linspace(0.0 + eps, 1.0 - eps, K + 1, device=device)
        z_edges = normal.icdf(qs)                # (K+1,)
        t_edges = torch.sigmoid(z_edges).clamp(eps, 1.0 - eps)
        t_mid   = 0.5 * (t_edges[:-1] + t_edges[1:])   # (K,)
        dt      = (t_edges[1:] - t_edges[:-1])         # (K,)
        return t_mid, dt

    m = getattr(self.time_step_sampler, "normal_mean", 0.0)
    s = getattr(self.time_step_sampler, "normal_std", 1.0)
    t_mid, dt = _build_logit_normal_schedule(num_steps, device=self.device, mean=m, std=s)

    # ---- 4) 核心：fp32 累加做 Euler ODE ----
    x_fp32 = x.float()  # master copy in fp32
    iters = range(num_steps)
    if progress_bar:
        try:
            from tqdm import tqdm as _tqdm
            iters = _tqdm(iters, leave=False)
        except Exception:
            pass

    for k in iters:
        val = 4.0 - 8.0 * float(t_mid[k].item())  # 标量
        # fm_transformers 断言 1D -> 构造 [B] 的 float32
        log_snr = torch.full((B_cond,), val, device=self.device, dtype=torch.float32)

        # 前向仍用模型精度（bf16/float16），但输出提升到 fp32 后再积分
        preds = self.fm_transformers(x_fp32.to(dtype), log_snr=log_snr, null_indicator=None)
        v = preds[-1] if isinstance(preds, (list, tuple)) else preds
        v = v.float()

        if kwargs.get("debug_print", False):
            print(f"step{k}: |x|={x_fp32.abs().mean().item():.4f}, |v|={v.abs().mean().item():.4f}, dt={float(dt[k]):.4f}")

        x_fp32.add_(v, alpha=float(dt[k].item()))

    x = x_fp32.to(dtype)  # 回到模型精度，后续进入 autoencoder

    # ---- 5) moments -> sample -> decode ----
    moments = self.autoencoder.sample(x)      # x 是 moments(μ, logvar) 拼接
    images  = self.autoencoder.decode(moments)
    images  = images.clamp(-1, 1).to(dtype=dtype)

    if images.shape[-2:] != (height, width):
        images = F.interpolate(images, size=(height, width), mode='bilinear', align_corners=False)

    return images
