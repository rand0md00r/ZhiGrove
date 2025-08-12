# src/optimisers/custom_adamw.py
from typing import Iterable, List, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW

def _flatten_params(params) -> List[nn.Parameter]:
    """把各种形态的 params 统一成 List[nn.Parameter]，过滤掉非 Parameter 和不需要训练的。"""
    out: List[nn.Parameter] = []
    if params is None:
        return out
    # list/tuple of dicts -> 已分组，直接提取
    if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
        for g in params:
            for p in g.get('params', []):
                if isinstance(p, nn.Parameter) and p.requires_grad:
                    out.append(p)
        return out
    # Mapping（包括 mmengine 的 ConfigDict）
    if hasattr(params, 'values'):
        for p in params.values():
            if isinstance(p, nn.Parameter) and p.requires_grad:
                out.append(p)
        return out
    # Iterable of Parameters
    if isinstance(params, Iterable):
        for p in params:
            if isinstance(p, nn.Parameter) and p.requires_grad:
                out.append(p)
        return out
    # 兜底：单个 Parameter
    if isinstance(params, nn.Parameter) and params.requires_grad:
        out.append(params)
    return out

def _split_decay_groups(trainable: List[nn.Parameter], weight_decay: float) -> List[Dict[str, Any]]:
    """把 2D 及以上张量进 decay 组，bias/LayerNorm 这类 1D 进 no-decay 组。"""
    decay_params = [p for p in trainable if p.dim() >= 2]
    nodecay_params = [p for p in trainable if p.dim() < 2]
    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({'params': decay_params, 'weight_decay': weight_decay})
    if nodecay_params:
        groups.append({'params': nodecay_params, 'weight_decay': 0.0})
    # 小打印方便确认
    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"[CustomAdamW] decayed tensors: {len(decay_params)} ({num_decay:,} params) | "
          f"non-decayed tensors: {len(nodecay_params)} ({num_nodecay:,} params)")
    return groups

class CustomAdamW(AdamW):
    """
    稳健版：
    - 接受 Iterable[Parameter] / List[Dict] / Mapping(ConfigDict)
    - 自动按维度切分 decay / no-decay
    - 不再对 dict/ConfigDict 直接做 .requires_grad 访问
    """
    def __init__(self, params, weight_decay=0.0, *args, **kwargs):
        # 如果已经是 List[Dict]（已分好组），就仅过滤无效参数并补 weight_decay 默认值
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            groups: List[Dict[str, Any]] = []
            for g in params:
                p_list = [p for p in g.get('params', []) if isinstance(p, nn.Parameter) and p.requires_grad]
                if not p_list:
                    continue
                new_g = dict(g)
                new_g['params'] = p_list
                # 若组里没给 weight_decay，则继承传入的 weight_decay
                if 'weight_decay' not in new_g:
                    new_g['weight_decay'] = weight_decay
                groups.append(new_g)
            super().__init__(params=groups, weight_decay=weight_decay, *args, **kwargs)
            return

        # 其他情形：统一扁平化后自动切 decay/no-decay
        trainable = _flatten_params(params)
        groups = _split_decay_groups(trainable, weight_decay)
        super().__init__(params=groups, weight_decay=weight_decay, *args, **kwargs)

class ParamWiseAdamW(AdamW):
    """
    如果你真的想手工传入 param groups，就用这个类：
    - 要求传入 List[Dict]，每个 Dict 里 'params' 是 List[nn.Parameter]
    - 对 1D 参数自动置 weight_decay=0.0（若未设置）
    """
    def __init__(self, params, *args, **kwargs):
        assert isinstance(params, list), "ParamWiseAdamW expects a list of param groups."
        groups: List[Dict[str, Any]] = []
        for g in params:
            assert isinstance(g, dict) and 'params' in g
            plist = [p for p in g['params'] if isinstance(p, nn.Parameter) and p.requires_grad]
            if not plist:
                continue
            new_g = dict(g)
            new_g['params'] = plist
            # 如果该组只有 1D 参数且未显式指定 weight_decay，则自动置 0
            if 'weight_decay' not in new_g:
                only_1d = all((p.dim() == 1) for p in plist)
                if only_1d:
                    new_g['weight_decay'] = 0.0
            groups.append(new_g)
        super().__init__(params=groups, *args, **kwargs)


-----------------------------------------------------------------------------------------------------------------------------


optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='CustomAdamW',   # 或 'ParamWiseAdamW' / 'AdamW'
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(      # 可选：这里做 lr/wd 的细粒度配置
        custom_keys={
            'clip_logit_scale': dict(lr_mult=0.1, decay_mult=0.0),
            'img_clip_head':    dict(lr_mult=1.0),
            'context_encoder':  dict(lr_mult=1.0),
        },
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=False),
    loss_scale='dynamic',
    dtype='bfloat16',
)
