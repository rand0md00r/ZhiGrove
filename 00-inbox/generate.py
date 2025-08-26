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
        与评测脚本兼容的生成函数：
        - inputs 前一半为 cond prompts（已重复4次），后一半为等量 cfg_prompt（仅用于CFG引导，不单独出图）
        - 返回 [B_cond, 3, H, W] in [-1,1]，供后续 rearrange 成 2x2 拼图
        """
        device, dtype = self.device, self._dtype
        self.eval()

        # ---- 0) 拆分 cond / uncond ----
        assert input_ids.dim() == 2 and attention_mask.dim() == 2, "inputs应为 [B, L]"
        B_total = input_ids.size(0)
        assert B_total % 2 == 0, "需要 cond/uncond 数量相同"
        B_cond = B_total // 2

        ids_c, mask_c = input_ids[:B_cond].to(device), attention_mask[:B_cond].to(device)
        ids_u, mask_u = None, None
        use_cfg = (cfg_scale is not None and cfg_scale > 0)
        if use_cfg:
            ids_u  = input_ids[B_cond:].to(device)
            mask_u = attention_mask[B_cond:].to(device)

        # ---- 1) helper：LLM+TransEncoder -> z_text ----
        def _z_text_from(ids, mask):
            q_meta = self.meta_queries[None].expand(ids.size(0), self.num_queries, -1).to(device=device, dtype=dtype)
            llm_in = self.prepare_forward_input(x=q_meta, input_ids=ids, attention_mask=mask)
            llm_out = self.llm.model(**llm_in, return_dict=True).last_hidden_state
            q_tokens = llm_out[:, -self.num_queries:]
            q_mask   = mask.new_ones(q_tokens.shape[:2])

            out = self.context_encoder(q_tokens.to(dtype), q_mask)
            mu, log_var = torch.chunk(out, 2, dim=-1)
            std = torch.exp(0.5 * log_var)
            if generator is None:
                eps = torch.randn_like(std)
            else:
                eps = torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
            z  = eps * std + mu     # reparam
            if z.dim() == 3:
                z = z.reshape(z.size(0), -1)     # [B, Z]
            return z

        z_c = _z_text_from(ids_c, mask_c)                      # 条件分支
        if use_cfg:
            z_u = _z_text_from(ids_u, mask_u)                  # 无条件（cfg_prompt）分支
            w = float(cfg_scale)
            z0 = (1.0 + w) * z_c - w * z_u                     # 在初态上做“CFG式”引导
        else:
            z0 = z_c

        # ---- 2) 用 VAE 确定 latent 形状，并把 z0 投到 [B,C,h,w] 初态 ----
        dummy_img = torch.zeros(B_cond, 3, height, width, device=device, dtype=dtype)
        # latent_shape = self.pixels_to_latents(dummy_img).shape          # [B, C, h, w]
        latent_shape = self.autoencoder(                                                                      # VAE编码器使用的是autoencoder，解码也使用autoencoder
            dummy_img.to(dtype=self._dtype, device=self.device), fn='encode_moments').squeeze(0).shape
        _, C, Hh, Ww = latent_shape
        target_dim = C * Hh * Ww

        # 加载线性映射到 VAE latent 维度
        # if not hasattr(self, "_zproj_to_latent") or self._zproj_to_latent.out_features != target_dim:
        #     in_dim = z0.shape[1]
        #     self._zproj_to_latent = nn.Linear(in_dim, target_dim, bias=False).to(device=device, dtype=dtype)

        # x = self._zproj_to_latent(z0).reshape(B_cond, C, Hh, Ww).to(dtype=dtype)
        x = einops.rearrange(
            z0, 
            'b (c h w) -> b c h w', 
            c=C, h=Hh, w=Ww
        ).to(dtype=self._dtype, device=z0.device)       # NOTE: 修改维度变换方式

        # ---- 3) 用 FMTransFormers 的速度场做 Euler ODE 积分 ----
        t0, t1 = 0.0, 1.0
        dt = (t1 - t0) / max(1, num_steps)
        iters = range(num_steps)
        if progress_bar:
            try:
                from tqdm import tqdm as _tqdm
                iters = _tqdm(iters, leave=False)
            except Exception:
                pass

        def _build_logit_normal_schedule(
            K: int, device, *, mean=0.0, std=1.0, deterministic: bool = True,
            generator: torch.Generator = None, eps: float = 1e-6
        ):
            """
            返回:
            t_edges: (K+1,)  时间区间边界 (单调)
            t_mid:   (K,)    每步的中点时间  (单调)
            dt:      (K,)    每步步长 = t_edges[k+1]-t_edges[k] (>=0, 求和≈1)
            """
            normal = torch.distributions.Normal(
                torch.tensor(mean, device=device), torch.tensor(std, device=device)
            )
            if deterministic:
                qs = torch.linspace(0.0 + eps, 1.0 - eps, K + 1, device=device)
                z_edges = normal.icdf(qs)                          # (K+1,)
            else:
                z_edges = torch.normal(mean=mean, std=std, size=(K+1,), device=device, generator=generator)
                z_edges, _ = torch.sort(z_edges)

            t_edges = torch.sigmoid(z_edges).clamp(eps, 1.0 - eps) # (K+1,)
            t_mid   = 0.5 * (t_edges[:-1] + t_edges[1:])           # (K,)
            dt      = (t_edges[1:] - t_edges[:-1])                 # (K,)
            return t_edges, t_mid, dt

        # 取 sampler 配置，保持与训练一致
        m = getattr(self.time_step_sampler, "normal_mean", 0.0)
        s = getattr(self.time_step_sampler, "normal_std", 1.0)

        # 构造单调时间网格（推荐 deterministic=True；要随机可改 False 并传 generator）
        _, t_mid, dt = _build_logit_normal_schedule(
            K=num_steps, device=self.device, mean=m, std=s, deterministic=True, generator=generator
        )

        for k in range(num_steps):
            # 训练同式: log_snr = 4 - 8 * t
            val = 4.0 - 8.0 * float(t_mid[k].item())   # 标量
            # if use_cfg:                                 # 若你拼了 2B 做 CFG
            #     log_snr = torch.full((2*B_cond,), val, device=self.device, dtype=torch.float32)  # 一维
            #     preds = self.fm_transformers(x_cat, log_snr=log_snr, null_indicator=null_indicator)
            #     v_all = preds[-1] if isinstance(preds, (list, tuple)) else preds
            #     v = (1.0 + cfg_scale) * v_all[:B_cond] - cfg_scale * v_all[B_cond:]
            # else:
            #     log_snr = torch.full((B_cond,), val, device=self.device, dtype=torch.float32)     # 一维
            #     preds = self.fm_transformers(x, log_snr=log_snr, null_indicator=None)
            #     v = preds[-1] if isinstance(preds, (list, tuple)) else preds

            log_snr = torch.full((B_cond,), val, device=self.device, dtype=self.dtype)     # 一维
            preds = self.fm_transformers(x, log_snr=log_snr, null_indicator=None)
            v = preds[-1] if isinstance(preds, (list, tuple)) else preds

            print(f"step{k}: |x|={x.abs().mean():.4f}, |v|={v.abs().mean():.4f}, dt={float(dt[k]):.4f}")

            x = x + v * float(dt[k].item())            # 非均匀步长，更稳

        # ---- 4) 用 VAE 解码回像素，并规整到 [-1,1] ----
        moments = self.autoencoder.sample(x)
        images = self.autoencoder.decode(moments)
        images = images.clamp(-1, 1).to(dtype=dtype)
        if images.shape[-2:] != (height, width):
            images = F.interpolate(images, size=(height, width), mode='bilinear', align_corners=False)

        return images
