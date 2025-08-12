from mmengine.hooks import Hook
import torch
import torch.distributed as dist

class ClipHeadDebugHook(Hook):
    def _get_param_groups(self, runner):
        # 尽可能兼容不同 wrapper/deepspeed 形态
        optw = runner.optim_wrapper
        opt  = getattr(optw, 'optimizer', None)
        if hasattr(opt, 'param_groups'):
            return opt.param_groups
        if hasattr(opt, 'optimizer') and hasattr(opt.optimizer, 'param_groups'):
            return opt.optimizer.param_groups
        opt2 = getattr(optw, '_optimizer', None)
        if hasattr(opt2, 'param_groups'):
            return opt2.param_groups
        return None

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # 只在前几步打印，避免刷屏
        if batch_idx not in (0, 1, 10):
            return

        # DeepSpeed 下 model 被包裹，取真实模块
        m = runner.model.module if hasattr(runner.model, 'module') else runner.model

        # 世界大小与 b_*（如果你在 loss 里返回了它们）
        ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        b_local  = float(outputs.get('b_local',  float('nan')))
        b_global = float(outputs.get('b_global', float('nan')))
        print(f'[clipdbg] world_size={ws}  b_local={b_local}  b_global={b_global}')

        # requires_grad & grad 均值（证明有反传）
        def gmean(x):
            return None if x is None else float(x.detach().abs().mean().cpu())
        print('[clipdbg] requires_grad:',
              'logit_scale', getattr(m.clip_logit_scale, 'requires_grad', None),
              'img_head',    getattr(m.img_clip_head.weight, 'requires_grad', None))
        print('[clipdbg] grad_mean:',
              'logit_scale', gmean(getattr(m.clip_logit_scale, 'grad', None)),
              'img_head',    gmean(getattr(m.img_clip_head.weight, 'grad', None)))

        # optimizer 覆盖 & 学习率/权重衰减
        groups = self._get_param_groups(runner)
        if groups is None:
            print('[clipdbg] param_groups not found on optimizer')
            return
        pid = id(m.clip_logit_scale)
        hid = id(m.img_clip_head.weight)
        found_p = found_h = False
        for gi, g in enumerate(groups):
            ids = {id(p) for p in g['params']}
            has_p = pid in ids
            has_h = hid in ids
            if has_p or has_h:
                print(f'[clipdbg] group#{gi} lr={g.get("lr")} wd={g.get("weight_decay")}  '
                      f'contains logit:{has_p} head:{has_h}')
            found_p |= has_p
            found_h |= has_h
        if not (found_p and found_h):
            print(f'[clipdbg][WARN] not covered by optimizer: logit={found_p} head={found_h}')







custom_hooks = [
    dict(type='ClipHeadDebugHook', priority=50),
]




paramwise_cfg=dict(
    custom_keys={
        'clip_logit_scale': dict(lr_mult=1.0, decay_mult=0.0),
        'img_clip_head':    dict(lr_mult=1.0),
        'context_encoder':  dict(lr_mult=1.0),
    },
    bias_decay_mult=0.0,
    norm_decay_mult=0.0,
)




