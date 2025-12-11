##################################################################
# 文本通路测试
##################################################################

        # @torch.no_grad()
        # def dbg_text_cos(model, prompts=("a cat","a car","a red apple")):
        #     zs=[]
        #     for p in prompts:
        #         tok = self.tokenizer(p, return_tensors="pt").to(model.device)
        #         q_meta = self.meta_queries[None].expand(1, model.num_queries, -1).to(model.device, model.dtype)
        #         llm_in = self.prepare_forward_input(x=q_meta, input_ids=tok.input_ids, attention_mask=torch.ones_like(tok.input_ids))
        #         hs = model.llm.model(**llm_in, return_dict=True).last_hidden_state
        #         q_tokens = hs[:, -model.num_queries:]
        #         q_mask = torch.ones(q_tokens.shape[:2], device=model.device, dtype=torch.bool)
        #         z_text, _, _ = model.text_ve_encoder(q_tokens, q_mask)   # (1,4096)
        #         zs.append(z_text)
        #     import torch.nn.functional as F
        #     for i in range(len(zs)):
        #         for j in range(i+1,len(zs)):
        #             c = F.cosine_similarity(zs[i], zs[j], dim=-1).mean().item()
        #             print(f"[text] cos({prompts[i]}, {prompts[j]}) = {c:.3f}")

##################################################################
# 单步对齐测试
##################################################################

        # import torch
        # @torch.no_grad()
        # def dbg_fm_single_step(model, pixel_values, input_ids, attention_mask, n_t=4):
        #     device, dtype = model.device, model.dtype
        #     # x1: 图像latent（与训练一致 256×256）
        #     x_img = F.interpolate(pixel_values.to(device, dtype), size=(256,256), mode='bilinear', align_corners=False)
        #     x1 = model.pixels_to_latents(x_img)                # (B,4,32,32)
        #     # x0: 文本latent
        #     q_meta = model.meta_queries[None].expand(input_ids.size(0), model.num_queries, -1).to(device, dtype)
        #     llm_in = model.prepare_forward_input(x=q_meta, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        #     hs = model.llm.model(**llm_in, return_dict=True).last_hidden_state
        #     q_tokens = hs[:, -model.num_queries:]
        #     q_mask = torch.ones(q_tokens.shape[:2], device=device, dtype=torch.bool)
        #     z_text, _, _ = model.text_ve_encoder(q_tokens, q_mask)      # (B,4096)
        #     x0 = einops.rearrange(z_text, 'b (c h w)-> b c h w', c=4,h=x1.shape[-2], w=x1.shape[-1]).to(dtype)

        #     # 采样几个t
            
        #     from src.models.openuni.fm_utils import psi, Dt_psi
        #     sigma_min, sigma_max = model.sigma_min, model.sigma_max
        #     B = x1.size(0)
        #     ts = torch.linspace(0.1, 0.9, n_t, device=device, dtype=dtype)

        #     for t in ts:
        #         t_vec = torch.full((B,), float(t), device=device, dtype=dtype)
        #         x_t = psi(t_vec, x0, x1, sigma_min, sigma_max).to(dtype)
        #         v_tgt = Dt_psi(t_vec, x0, x1, sigma_min, sigma_max)      # 目标速度
        #         log_snr = 4.0 - 8.0 * t_vec
        #         v_pred = model.fm_transformers(x_t, log_snr=log_snr, null_indicator=torch.zeros(B, dtype=torch.bool, device=device))[-1]

        #         # 计算对齐度（cos & 比例）
        #         a = v_pred.flatten(1).float(); b = v_tgt.flatten(1).float()
        #         cos = (F.normalize(a,dim=1) * F.normalize(b,dim=1)).sum(1).mean().item()
        #         ratio = (a.norm(dim=1) / (b.norm(dim=1)+1e-8)).mean().item()
        #         print(f"[fm] t={float(t):.2f}  cos(v_pred, v_tgt)={cos:.3f}  ||pred||/||tgt||={ratio:.3f}")


        # # 输入本地图像和prompt，构建 dbg_fm_single_step 的输入
        # from PIL import Image
        # import torchvision.transforms as T

        # # 假设本地图像路径和prompt如下
        # local_image_path = "/vepfs/group03/wyq/ug_uni/CrossUni-do/tests/2.png"  # 请替换为你的本地图像路径
        # prompt = "a cat with a red hat."  # 请替换为你的prompt

        # # 图像预处理
        # image = Image.open(local_image_path).convert("RGB")
        # transform = T.Compose([
        #     T.Resize((self.vit_input_size, self.vit_input_size)),
        #     T.ToTensor(),
        #     T.Normalize(mean=self.vit_mean.tolist(), std=self.vit_std.tolist())
        # ])
        # x_img = transform(image).unsqueeze(0)  # (1, 3, H, W)

        # # 文本tokenize
        # inputs = self.tokenizer(
        #     [prompt], add_special_tokens=True, return_tensors='pt', padding=True
        # )
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # dbg_fm_single_step(self, x_img, input_ids, attention_mask)

##################################################################
# 真实图像flow测试
##################################################################

        # @torch.no_grad()
        # def dbg_shoot_to_x1(model, pixel_values, input_ids, attention_mask, num_steps=100, dt_scale=1.0):
        #     device, dtype = model.device, model.dtype
        #     x_img = F.interpolate(pixel_values.to(device, dtype), size=(256,256), mode='bilinear', align_corners=False)
        #     x1 = model.pixels_to_latents(x_img)  # target
        #     # x0
        #     q_meta = model.meta_queries[None].expand(input_ids.size(0), model.num_queries, -1).to(device, dtype)
        #     llm_in = model.prepare_forward_input(x=q_meta, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        #     hs = model.llm.model(**llm_in, return_dict=True).last_hidden_state
        #     q_tokens = hs[:, -model.num_queries:]; q_mask = torch.ones(q_tokens.shape[:2], device=device, dtype=torch.bool)
        #     z_text,_,_ = model.text_ve_encoder(q_tokens, q_mask)
        #     z = einops.rearrange(z_text, 'b (c h w)-> b c h w', c=4,h=x1.shape[-2], w=x1.shape[-1]).to(dtype)

        #     # 最简欧拉（与训练同：log_snr=4-8t，0->1）
        #     N=max(1,int(num_steps)); dt=(1.0/N)*float(dt_scale)
        #     B=z.size(0); null=torch.zeros(B, dtype=torch.bool, device=device)
        #     for i in range(N):
        #         t = torch.full((B,), (i+0.5)/N, device=device, dtype=dtype)
        #         log_snr = 4.0 - 8.0 * t
        #         v = model.fm_transformers(z, log_snr=log_snr, null_indicator=null)[-1]
        #         z = z + v * dt
        #     # 看和 x1 的距离
        #     err = F.mse_loss(z.float(), x1.float()).item()
        #     print(f"[shoot] MSE(z_end, x1) = {err:.6f}")

        # # 输入本地图像和prompt，构建 dbg_fm_single_step 的输入
        # from PIL import Image
        # import torchvision.transforms as T

        # # 假设本地图像路径和prompt如下
        # local_image_path = "/vepfs/group03/wyq/ug_uni/CrossUni-do/tests/2.png"  # 请替换为你的本地图像路径
        # prompt = "a cat with a red hat."  # 请替换为你的prompt

        # # 图像预处理
        # image = Image.open(local_image_path).convert("RGB")
        # transform = T.Compose([
        #     T.Resize((self.vit_input_size, self.vit_input_size)),
        #     T.ToTensor(),
        #     T.Normalize(mean=self.vit_mean.tolist(), std=self.vit_std.tolist())
        # ])
        # x_img = transform(image).unsqueeze(0)  # (1, 3, H, W)

        # # 文本tokenize
        # inputs = self.tokenizer(
        #     [prompt], add_special_tokens=True, return_tensors='pt', padding=True
        # )
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # dbg_shoot_to_x1(self, x_img, input_ids, attention_mask)


##################################################################
# 用v_target进行积分
##################################################################

        @torch.no_grad()
        def check_teacher_euler(model, x0, x1, num_steps=20):
            # x0,x1: (B,4,H,W) 同 batch/shape/dtype
            device, dtype = model.device, model.dtype
            import torch.nn.functional as F
            from src.models.openuni.fm_utils import Dt_psi

            N = max(1,int(num_steps)); dt = 1.0 / N
            B = x0.size(0)
            z = x0.clone()
            for i in range(N):
                t = torch.full((B,), (i+0.5)/N, device=device, dtype=dtype)
                v_tgt = Dt_psi(t, x0, x1, model.sigma_min, model.sigma_max)
                z = z + v_tgt * dt
            mse = F.mse_loss(z.float(), x1.float()).item()
            print(f"[teacher-euler] N={N}, MSE(zN, x1)={mse:.6e}")
            return mse


        # 输入本地图像和prompt，构建 dbg_fm_single_step 的输入
        from PIL import Image
        import torchvision.transforms as T

        # 假设本地图像路径和prompt如下
        local_image_path = "/vepfs/group03/wyq/ug_uni/CrossUni-do/tests/2.png"  # 请替换为你的本地图像路径
        prompt = "a cat with a red hat."  # 请替换为你的prompt

        # 图像预处理
        image = Image.open(local_image_path).convert("RGB")
        transform = T.Compose([
            T.Resize((self.vit_input_size, self.vit_input_size)),
            T.ToTensor(),
            T.Normalize(mean=self.vit_mean.tolist(), std=self.vit_std.tolist())
        ])
        x_img = transform(image).unsqueeze(0)  # (1, 3, H, W)

        # 文本tokenize
        inputs = self.tokenizer(
            [prompt], add_special_tokens=True, return_tensors='pt', padding=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        device, dtype = self.device, self.dtype
        x_img = F.interpolate(x_img.to(device, dtype), size=(256,256), mode='bilinear', align_corners=False)
        x1 = self.pixels_to_latents(x_img)  # target
        # x0
        q_meta = self.meta_queries[None].expand(input_ids.size(0), self.num_queries, -1).to(device, dtype)
        llm_in = self.prepare_forward_input(x=q_meta, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        hs = self.llm.model(**llm_in, return_dict=True).last_hidden_state
        q_tokens = hs[:, -self.num_queries:]; q_mask = torch.ones(q_tokens.shape[:2], device=device, dtype=torch.bool)
        z_text,_,_ = self.text_ve_encoder(q_tokens, q_mask)
        z = einops.rearrange(z_text, 'b (c h w)-> b c h w', c=4,h=x1.shape[-2], w=x1.shape[-1]).to(dtype)

        check_teacher_euler(self, z, x1, num_steps=20)
        check_teacher_euler(self, z, x1, num_steps=10)
        check_teacher_euler(self, z, x1, num_steps=5)



##################################################################
