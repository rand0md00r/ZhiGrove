import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class EnsureClipParamsInOptimHook(Hook):
    def _get_opt(self, runner):
        ow = runner.optim_wrapper
        if hasattr(ow, 'optimizer') and hasattr(ow.optimizer, 'param_groups'):
            return ow.optimizer
        if hasattr(ow, '_optimizer') and hasattr(ow._optimizer, 'param_groups'):
            return ow._optimizer
        return None

    def before_train(self, runner):
        m = runner.model.module if hasattr(runner.model, 'module') else runner.model
        opt = self._get_opt(runner)
        if opt is None:
            print('[ensure] cannot locate optimizer')
            return

        # 待确保的参数对象
        targets = {
            'clip_logit_scale': m.clip_logit_scale,
            'img_clip_head.weight': m.img_clip_head.weight,
        }

        # 拉直已有的 param 引用集合
        existing = set()
        for g in opt.param_groups:
            for p in g['params']:
                existing.add(p)

        # 对每个目标，若不在 optimizer 中，则 add_param_group
        added = []
        for name, p in targets.items():
            if p not in existing:
                # 独立设置一个合理的 lr / wd（会在 paramwise_cfg 之上生效）
                opt.add_param_group({
                    'params': [p],
                    'lr': opt.param_groups[0].get('lr', 1e-4),  # 复用组0的lr
                    'weight_decay': 0.0 if name.startswith('clip_logit_scale') else opt.param_groups[0].get('weight_decay', 0.0),
                })
                added.append(name)

        print(f'[ensure] added to optimizer: {added}' if added else '[ensure] all present')




# __init__ 里
import math, torch.nn as nn, torch
self.clip_temp = nn.Linear(1, 1, bias=False)  # 参数形状 [1,1]
with torch.no_grad():
    self.clip_temp.weight.copy_(torch.tensor([[math.log(1/0.07)]], dtype=torch.float32))
self.clip_temp.weight.requires_grad_(True)
self.clip_temp.weight.data = self.clip_temp.weight.data.to(torch.float32)  # FP32 保精度

# compute_clip_loss 里
logit_scale_log = self.clip_temp.weight[0, 0]
logit_scale = torch.exp(logit_scale_log.float().clamp(min=0.0, max=math.log(100.0)))
logit_scale_scalar = float(logit_scale.detach().item())

# 调整 paramwise_cfg 里的 key：'clip_temp' 或 'clip_temp.weight'







print([n for n,_ in m.named_parameters() if 'clip' in n])
