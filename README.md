# text2image_loss 返回前自检

for k, v in ret.items():
    if isinstance(v, torch.Tensor):
        assert v.dim() >= 0, f'{k} is weird tensor'
        assert v.dtype.is_floating_point or v.is_complex(), f'{k} must be float/complex tensor'
    elif isinstance(v, (list, tuple)):
        assert all(isinstance(x, torch.Tensor) for x in v), f'{k} must be list of tensors'



# 统一把所有非 Tensor 的标量转成 float32 Tensor；把整型 Tensor 转 float32
for k, v in list(ret.items()):
    if isinstance(v, torch.Tensor):
        if not (v.is_floating_point() or v.is_complex()):
            ret[k] = v.to(torch.float32)
    elif isinstance(v, (int, float)):
        ret[k] = torch.tensor(float(v), device=self.device, dtype=torch.float32)
    elif isinstance(v, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in v):
        ret[k] = [x.to(torch.float32) if not (x.is_floating_point() or x.is_complex()) else x for x in v]
    # 其他类型就别返回到 losses 里了
