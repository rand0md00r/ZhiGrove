    @torch.no_grad()
    def generate(
        self,
        batch: dict = None,
        prompts=None,                      # 也可传入字符串或字符串列表（需要能拿到tokenizer）
        num_steps: int = 40,
        guidance_scale: float = 3.0,       # CFG系数；<=0则关闭CFG
        height: int = 256,                 # 目标输出分辨率
        width: int = 256,
        stochastic: bool = True,           # 采样z_text时是否加入随机项
        seed: int = None,
        return_intermediates: bool = False,
        decode_pixels: bool = True,        # True则返回解码后的像素；False仅返回latent
    ):
        """
        文本->图像生成（评测用）
        返回: dict(images=[B,3,H,W] in [-1,1], latents=[B,C,h,w], steps=[T, ...](可选), misc={...})
        """
        device = self.device
        dtype  = self._dtype
        self.eval()

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # --------- 1) 准备输入 token ---------
        if batch is None:
            batch = {}

        if ('input_ids' not in batch) or ('attention_mask' not in batch):
            # 简易tokenize（尽可能从lmm里挖tokenizer；否则报错）
            if prompts is None:
                raise ValueError("generate: 请提供 batch['input_ids']/['attention_mask']，或传入 prompts。")
            if isinstance(prompts, str):
                prompts = [prompts]
            tok = getattr(self, "tokenizer", None)
            if tok is None:
                tok = getattr(self.lmm, "tokenizer", None)
            if tok is None:
                raise ValueError("找不到 tokenizer（self.tokenizer 或 self.lmm.tokenizer），无法从 prompts 构造输入。")

            enc = tok(
                [self.prompt_template.format(x=p) if self.prompt_template else p for p in prompts],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )
            batch['input_ids'] = enc['input_ids'].to(device)
            batch['attention_mask'] = enc['attention_mask'].to(device)

        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        B = input_ids.shape[0]

        # --------- 2) LLM生成 queries -> TransEncoder -> z_text ---------
        q_meta = self.meta_queries[None].expand(B, self.num_queries, -1).to(device=device, dtype=dtype)
        llm_in = self.prepare_forward_input(x=q_meta, input_ids=input_ids, attention_mask=attn_mask)
        llm_out = self.llm.model(**llm_in, return_dict=True).last_hidden_state
        q_tokens = llm_out[:, -self.num_queries:]                         # [B, Q, D]
        q_mask   = attn_mask.new_ones(q_tokens.shape[:2])                 # [B, Q]

        # context_encoder 输出被你训练为 [B, dim] 或 [B, Q, dim] 的高维向量，这里与训练一致
        z_text, mu, log_var = self.text_ve_encoder(q_tokens, q_mask)      # reparam
        if not stochastic:
            # 评测可设为确定性（均值）
            z_text = mu

        # 如果是 [B, Q, D]，先做池化或展平；与你训练中 flow_matching_loss 的假设保持一致
        if z_text.dim() == 3:
            # 优先按你训练时的方式：将 [B, Q, D] => [B, Q*D]
            z_text = z_text.reshape(B, -1)

        # --------- 3) 构造 AE latent 形状，并把 z_text 投到 [B,C,h,w] 初态 ---------
        # 用 AE 的 encode_moments 推导目标 latent 形状（与训练时相同：对 256x256）
        dummy_img = torch.zeros(B, 3, height, width, device=device, dtype=dtype)
        try:
            x1_shape = self.autoencoder(dummy_img, fn='encode_moments').shape  # [B,C,h,w]
        except TypeError:
            # 某些实现是方法而非 __call__
            x1_shape = self.autoencoder.encode_moments(dummy_img).shape

        _, C, Hh, Ww = x1_shape
        target_dim = C * Hh * Ww

        # 若维度不匹配，懒加载一个线性层把 z_text 投到 C*H*W
        if not hasattr(self, "_zproj_to_latent") or self._zproj_to_latent.out_features != target_dim:
            in_dim = z_text.shape[1]
            self._zproj_to_latent = nn.Linear(in_dim, target_dim, bias=False).to(device=device, dtype=dtype)

        x = self._zproj_to_latent(z_text).reshape(B, C, Hh, Ww).to(dtype=dtype)

        # --------- 4) ODE 积分（Euler），支持 CFG ---------
        t0, t1 = 0.0, 1.0
        ts = torch.linspace(t0, t1, num_steps+1, device=device, dtype=dtype)  # 0, ..., 1
        dt = (t1 - t0) / num_steps

        intermediates = [] if return_intermediates else None

        for k in range(num_steps):
            t = ts[k].expand(B, 1, 1, 1)                         # [B,1,1,1]
            log_snr = (4.0 - 8.0 * t).to(dtype=torch.float32)    # 模型里按fp32计算log_snr更稳

            if guidance_scale is None or guidance_scale <= 0:
                # 纯条件
                preds = self.fm_transformers(x, log_snr=log_snr.squeeze(-1).squeeze(-1), null_indicator=None)
                v = preds[-1] if isinstance(preds, (list, tuple)) else preds
            else:
                # 组个 2B batch：前半uncond，后半cond
                x_cat = torch.cat([x, x], dim=0)
                log_snr_cat = torch.cat([log_snr, log_snr], dim=0).squeeze(-1).squeeze(-1)
                null_indicator = torch.zeros(2*B, device=device, dtype=torch.bool)
                null_indicator[:B] = True   # 前B为uncond

                preds = self.fm_transformers(x_cat, log_snr=log_snr_cat, null_indicator=null_indicator)
                v_all = preds[-1] if isinstance(preds, (list, tuple)) else preds
                v_uncond, v_cond = v_all[:B], v_all[B:]

                # CFG: v = (1+w)*v_cond - w*v_uncond
                w = guidance_scale
                v = (1.0 + w) * v_cond - w * v_uncond

            # Euler 前向积分：x_{k+1} = x_k + dt * v(t, x_k)
            x = (x + v * dt).to(dtype=dtype)

            if return_intermediates and (k == num_steps - 1 or (k+1) % max(1, num_steps // 10) == 0):
                # 采样少量中间态用于观测
                intermediates.append(x.detach().clone())

        latents = x  # [B,C,Hh,Ww]

        # --------- 5) 解码到像素（[-1,1]） ---------
        images = None
        if decode_pixels:
            # 优先尝试 autoencoder 的 decode_moments
            decoded = None
            try:
                decoded = self.autoencoder(latents, fn='decode_moments')
            except TypeError:
                if hasattr(self.autoencoder, "decode_moments"):
                    decoded = self.autoencoder.decode_moments(latents)
            except Exception:
                decoded = None

            if decoded is None:
                # 兜底：尝试 VAE 解码（注意可能不是同一空间，谨慎用于快速可视化）
                try:
                    scaling = getattr(self.vae.config, 'scaling_factor', 1.0)
                    images = self.vae.decode(latents / scaling)[0]
                except Exception:
                    # 最后兜底：直接tanh压回像素域（仅可视化用途）
                    images = torch.tanh(latents)

            else:
                images = decoded

            # 规范到 [-1,1]
            images = images.clamp(-1, 1)

            # 若需要特定输出分辨率，做一次上采样（保持与训练一致的 256 更稳）
            if images.shape[-2:] != (height, width):
                images = F.interpolate(images, size=(height, width), mode='bilinear', align_corners=False)

        # --------- 6) 打包返回 ---------
        out = {
            "images": images,                # [B,3,H,W] in [-1,1] or None
            "latents": latents,              # [B,C,h,w]
            "misc": {
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "stochastic": stochastic,
                "latent_shape": (C, Hh, Ww),
            }
        }
        if return_intermediates:
            out["intermediates"] = intermediates  # list of [B,C,h,w]

        return out
