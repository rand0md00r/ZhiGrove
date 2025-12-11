import{_ as t,c as n,o,ag as i}from"./chunks/framework.CQuhCYrb.js";const c=JSON.parse('{"title":"文本通路测试","description":"","frontmatter":{},"headers":[],"relativePath":"00-inbox/week-38/250920-0022-check_code.md","filePath":"00-inbox/week-38/250920-0022-check_code.md"}'),s={name:"00-inbox/week-38/250920-0022-check_code.md"};function a(_,e,r,m,d,p){return o(),n("div",null,[...e[0]||(e[0]=[i(`<p>##################################################################</p><h1 id="文本通路测试" tabindex="-1">文本通路测试 <a class="header-anchor" href="#文本通路测试" aria-label="Permalink to &quot;文本通路测试&quot;">​</a></h1><p>##################################################################</p><pre><code>    # @torch.no_grad()
    # def dbg_text_cos(model, prompts=(&quot;a cat&quot;,&quot;a car&quot;,&quot;a red apple&quot;)):
    #     zs=[]
    #     for p in prompts:
    #         tok = self.tokenizer(p, return_tensors=&quot;pt&quot;).to(model.device)
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
    #             print(f&quot;[text] cos({prompts[i]}, {prompts[j]}) = {c:.3f}&quot;)
</code></pre><p>##################################################################</p><h1 id="单步对齐测试" tabindex="-1">单步对齐测试 <a class="header-anchor" href="#单步对齐测试" aria-label="Permalink to &quot;单步对齐测试&quot;">​</a></h1><p>##################################################################</p><pre><code>    # import torch
    # @torch.no_grad()
    # def dbg_fm_single_step(model, pixel_values, input_ids, attention_mask, n_t=4):
    #     device, dtype = model.device, model.dtype
    #     # x1: 图像latent（与训练一致 256×256）
    #     x_img = F.interpolate(pixel_values.to(device, dtype), size=(256,256), mode=&#39;bilinear&#39;, align_corners=False)
    #     x1 = model.pixels_to_latents(x_img)                # (B,4,32,32)
    #     # x0: 文本latent
    #     q_meta = model.meta_queries[None].expand(input_ids.size(0), model.num_queries, -1).to(device, dtype)
    #     llm_in = model.prepare_forward_input(x=q_meta, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    #     hs = model.llm.model(**llm_in, return_dict=True).last_hidden_state
    #     q_tokens = hs[:, -model.num_queries:]
    #     q_mask = torch.ones(q_tokens.shape[:2], device=device, dtype=torch.bool)
    #     z_text, _, _ = model.text_ve_encoder(q_tokens, q_mask)      # (B,4096)
    #     x0 = einops.rearrange(z_text, &#39;b (c h w)-&gt; b c h w&#39;, c=4,h=x1.shape[-2], w=x1.shape[-1]).to(dtype)

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

    #         # 计算对齐度（cos &amp; 比例）
    #         a = v_pred.flatten(1).float(); b = v_tgt.flatten(1).float()
    #         cos = (F.normalize(a,dim=1) * F.normalize(b,dim=1)).sum(1).mean().item()
    #         ratio = (a.norm(dim=1) / (b.norm(dim=1)+1e-8)).mean().item()
    #         print(f&quot;[fm] t={float(t):.2f}  cos(v_pred, v_tgt)={cos:.3f}  ||pred||/||tgt||={ratio:.3f}&quot;)


    # # 输入本地图像和prompt，构建 dbg_fm_single_step 的输入
    # from PIL import Image
    # import torchvision.transforms as T

    # # 假设本地图像路径和prompt如下
    # local_image_path = &quot;/vepfs/group03/wyq/ug_uni/CrossUni-do/tests/2.png&quot;  # 请替换为你的本地图像路径
    # prompt = &quot;a cat with a red hat.&quot;  # 请替换为你的prompt

    # # 图像预处理
    # image = Image.open(local_image_path).convert(&quot;RGB&quot;)
    # transform = T.Compose([
    #     T.Resize((self.vit_input_size, self.vit_input_size)),
    #     T.ToTensor(),
    #     T.Normalize(mean=self.vit_mean.tolist(), std=self.vit_std.tolist())
    # ])
    # x_img = transform(image).unsqueeze(0)  # (1, 3, H, W)

    # # 文本tokenize
    # inputs = self.tokenizer(
    #     [prompt], add_special_tokens=True, return_tensors=&#39;pt&#39;, padding=True
    # )
    # input_ids = inputs[&#39;input_ids&#39;]
    # attention_mask = inputs[&#39;attention_mask&#39;]
    # dbg_fm_single_step(self, x_img, input_ids, attention_mask)
</code></pre><p>##################################################################</p><h1 id="真实图像flow测试" tabindex="-1">真实图像flow测试 <a class="header-anchor" href="#真实图像flow测试" aria-label="Permalink to &quot;真实图像flow测试&quot;">​</a></h1><p>##################################################################</p><pre><code>    # @torch.no_grad()
    # def dbg_shoot_to_x1(model, pixel_values, input_ids, attention_mask, num_steps=100, dt_scale=1.0):
    #     device, dtype = model.device, model.dtype
    #     x_img = F.interpolate(pixel_values.to(device, dtype), size=(256,256), mode=&#39;bilinear&#39;, align_corners=False)
    #     x1 = model.pixels_to_latents(x_img)  # target
    #     # x0
    #     q_meta = model.meta_queries[None].expand(input_ids.size(0), model.num_queries, -1).to(device, dtype)
    #     llm_in = model.prepare_forward_input(x=q_meta, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    #     hs = model.llm.model(**llm_in, return_dict=True).last_hidden_state
    #     q_tokens = hs[:, -model.num_queries:]; q_mask = torch.ones(q_tokens.shape[:2], device=device, dtype=torch.bool)
    #     z_text,_,_ = model.text_ve_encoder(q_tokens, q_mask)
    #     z = einops.rearrange(z_text, &#39;b (c h w)-&gt; b c h w&#39;, c=4,h=x1.shape[-2], w=x1.shape[-1]).to(dtype)

    #     # 最简欧拉（与训练同：log_snr=4-8t，0-&gt;1）
    #     N=max(1,int(num_steps)); dt=(1.0/N)*float(dt_scale)
    #     B=z.size(0); null=torch.zeros(B, dtype=torch.bool, device=device)
    #     for i in range(N):
    #         t = torch.full((B,), (i+0.5)/N, device=device, dtype=dtype)
    #         log_snr = 4.0 - 8.0 * t
    #         v = model.fm_transformers(z, log_snr=log_snr, null_indicator=null)[-1]
    #         z = z + v * dt
    #     # 看和 x1 的距离
    #     err = F.mse_loss(z.float(), x1.float()).item()
    #     print(f&quot;[shoot] MSE(z_end, x1) = {err:.6f}&quot;)

    # # 输入本地图像和prompt，构建 dbg_fm_single_step 的输入
    # from PIL import Image
    # import torchvision.transforms as T

    # # 假设本地图像路径和prompt如下
    # local_image_path = &quot;/vepfs/group03/wyq/ug_uni/CrossUni-do/tests/2.png&quot;  # 请替换为你的本地图像路径
    # prompt = &quot;a cat with a red hat.&quot;  # 请替换为你的prompt

    # # 图像预处理
    # image = Image.open(local_image_path).convert(&quot;RGB&quot;)
    # transform = T.Compose([
    #     T.Resize((self.vit_input_size, self.vit_input_size)),
    #     T.ToTensor(),
    #     T.Normalize(mean=self.vit_mean.tolist(), std=self.vit_std.tolist())
    # ])
    # x_img = transform(image).unsqueeze(0)  # (1, 3, H, W)

    # # 文本tokenize
    # inputs = self.tokenizer(
    #     [prompt], add_special_tokens=True, return_tensors=&#39;pt&#39;, padding=True
    # )
    # input_ids = inputs[&#39;input_ids&#39;]
    # attention_mask = inputs[&#39;attention_mask&#39;]
    # dbg_shoot_to_x1(self, x_img, input_ids, attention_mask)
</code></pre><p>##################################################################</p><h1 id="用v-target进行积分" tabindex="-1">用v_target进行积分 <a class="header-anchor" href="#用v-target进行积分" aria-label="Permalink to &quot;用v_target进行积分&quot;">​</a></h1><p>##################################################################</p><pre><code>    @torch.no_grad()
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
        print(f&quot;[teacher-euler] N={N}, MSE(zN, x1)={mse:.6e}&quot;)
        return mse


    # 输入本地图像和prompt，构建 dbg_fm_single_step 的输入
    from PIL import Image
    import torchvision.transforms as T

    # 假设本地图像路径和prompt如下
    local_image_path = &quot;/vepfs/group03/wyq/ug_uni/CrossUni-do/tests/2.png&quot;  # 请替换为你的本地图像路径
    prompt = &quot;a cat with a red hat.&quot;  # 请替换为你的prompt

    # 图像预处理
    image = Image.open(local_image_path).convert(&quot;RGB&quot;)
    transform = T.Compose([
        T.Resize((self.vit_input_size, self.vit_input_size)),
        T.ToTensor(),
        T.Normalize(mean=self.vit_mean.tolist(), std=self.vit_std.tolist())
    ])
    x_img = transform(image).unsqueeze(0)  # (1, 3, H, W)

    # 文本tokenize
    inputs = self.tokenizer(
        [prompt], add_special_tokens=True, return_tensors=&#39;pt&#39;, padding=True
    )
    input_ids = inputs[&#39;input_ids&#39;]
    attention_mask = inputs[&#39;attention_mask&#39;]

    device, dtype = self.device, self.dtype
    x_img = F.interpolate(x_img.to(device, dtype), size=(256,256), mode=&#39;bilinear&#39;, align_corners=False)
    x1 = self.pixels_to_latents(x_img)  # target
    # x0
    q_meta = self.meta_queries[None].expand(input_ids.size(0), self.num_queries, -1).to(device, dtype)
    llm_in = self.prepare_forward_input(x=q_meta, input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    hs = self.llm.model(**llm_in, return_dict=True).last_hidden_state
    q_tokens = hs[:, -self.num_queries:]; q_mask = torch.ones(q_tokens.shape[:2], device=device, dtype=torch.bool)
    z_text,_,_ = self.text_ve_encoder(q_tokens, q_mask)
    z = einops.rearrange(z_text, &#39;b (c h w)-&gt; b c h w&#39;, c=4,h=x1.shape[-2], w=x1.shape[-1]).to(dtype)

    check_teacher_euler(self, z, x1, num_steps=20)
    check_teacher_euler(self, z, x1, num_steps=10)
    check_teacher_euler(self, z, x1, num_steps=5)
</code></pre><p>##################################################################</p>`,17)])])}const u=t(s,[["render",a]]);export{c as __pageData,u as default};
