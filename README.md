names = dict(self.named_parameters())
opt_param_ids = {id(p) for g in self.runner.optim_wrapper.optimizer.param_groups for p in g['params']}

for key in ['clip_logit_scale', 'img_clip_head.weight']:
    p = names.get(key, None)
    print(f'[opt/cover] {key}: present={p is not None}, requires_grad={getattr(p, "requires_grad", None)}, in_optim={(id(p) in opt_param_ids) if p is not None else None}')
