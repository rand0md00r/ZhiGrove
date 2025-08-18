# 先决定是否打乱 x1
null_indicator = (torch.rand(x1.shape[0], device=x1.device) < 0.1)
if null_indicator.any():
    x1_perm = x1.clone()
    x1_perm[null_indicator] = torch.roll(x1_perm[null_indicator], shifts=1, dims=0)
    x1_eff = x1_perm
else:
    x1_eff = x1

# 用有效的 x1 重算路径与目标速度
x_t = psi(t, x0, x1_eff, self.sigma_min, self.sigma_max).to(self._dtype)
v_target = Dt_psi(t, x0, x1_eff, self.sigma_min, self.sigma_max)

preds = self.fm_transformers(x_t, log_snr=log_snr, null_indicator=null_indicator)
loss = mse_mean_over_spatial(preds[-1] - v_target).mean()
