    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        progress_bar: bool = False,
        cfg_scale: float = 4.5,
        num_steps: int = 20,
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
            eps = torch.randn_like(std) if generator is None else torch.randn_like(std, generator=generator)
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
        latent_shape = self.pixels_to_latents(dummy_img).shape          # [B, C, h, w]
        _, C, Hh, Ww = latent_shape
        target_dim = C * Hh * Ww

        # 懒加载线性映射到 VAE latent 维度
        if not hasattr(self, "_zproj_to_latent") or self._zproj_to_latent.out_features != target_dim:
            in_dim = z0.shape[1]
            self._zproj_to_latent = nn.Linear(in_dim, target_dim, bias=False).to(device=device, dtype=dtype)

        x = self._zproj_to_latent(z0).reshape(B_cond, C, Hh, Ww).to(dtype=dtype)

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

        for _ in iters:
            t = torch.full((B_cond, 1, 1, 1), fill_value=t0, device=device, dtype=dtype)
            log_snr = (4.0 - 8.0 * t).to(dtype=torch.float32).squeeze(-1).squeeze(-1)
            # 这里模型不显式接收文本条件；文本条件通过初态 x 已经注入
            preds = self.fm_transformers(x, log_snr=log_snr, null_indicator=None)
            v = preds[-1] if isinstance(preds, (list, tuple)) else preds
            x = (x + v * dt).to(dtype=dtype)
            t0 += dt

        latents = x  # [B_cond, C, Hh, Ww]

        # ---- 4) 用 VAE 解码回像素，并规整到 [-1,1] ----
        images = self.latents_to_pixels(latents)               # [B_cond, 3, H, W]（VAE内部已做尺度还原）
        images = images.clamp(-1, 1).to(dtype=dtype)
        if images.shape[-2:] != (height, width):
            images = F.interpolate(images, size=(height, width), mode='bilinear', align_corners=False)

        return images
