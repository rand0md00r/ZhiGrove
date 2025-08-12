[clip/debug] dist_initialized=True, world_size=1 
tensor(2.7869, device='cuda:0')
[clipdbg] world_size=1  b_local=16.0  b_global=16.0
[clipdbg] requires_grad: logit_scale True img_head True
[clipdbg] grad_mean: logit_scale None img_head None
[clipdbg][WARN] not covered by optimizer: logit=False head=False
/usr/local/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:745: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  return fn(*args, **kwargs)
/usr/local/lib/python3.10/site-packages/torch/utils/checkpoint.py:87: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
/vepfs/DI/yaqi/understand_gen/CrossUni-do/src/models/diffusion/sigmoid_kernel.py:30: UserWarning: Using slower tdp_torch implementation for a tensor with shape torch.Size([1024])
  warnings.warn(f'Using slower tdp_torch implementation for a tensor with shape {param0.shape}')
/vepfs/DI/yaqi/understand_gen/CrossUni-do/src/models/diffusion/sigmoid_kernel.py:30: UserWarning: Using slower tdp_torch implementation for a tensor with shape torch.Size([67584])
  warnings.warn(f'Using slower tdp_torch implementation for a tensor with shape {param0.shape}')
/vepfs/DI/yaqi/understand_gen/CrossUni-do/src/models/diffusion/sigmoid_kernel.py:30: UserWarning: Using slower tdp_torch implementation for a tensor with shape torch.Size([512])
  warnings.warn(f'Using slower tdp_torch implementation for a tensor with shape {param0.shape}')
tensor(2.7681, device='cuda:0')
[clipdbg] world_size=1  b_local=16.0  b_global=16.0
[clipdbg] requires_grad: logit_scale True img_head True
[clipdbg] grad_mean: logit_scale None img_head None
[clipdbg][WARN] not covered by optimizer: logit=False head=False
tensor(2.7742, device='cuda:0')
tensor(2.7830, device='cuda:0')
tensor(2.7906, device='cuda:0')
tensor(2.7615, device='cuda:0')
tensor(2.7685, device='cuda:0')
tensor(2.7720, device='cuda:0')
tensor(2.7797, device='cuda:0')
tensor(2.7784, device='cuda:0')
08/12 15:28:46 - mmengine - INFO - Iter(train) [   10/10000]  base_lr: 9.0918e-06 lr: 9.0918e-06  eta: 20:09:02  time: 7.2615  data_time: 0.0495  memory: 36388  loss: 437.3079  loss_flow: 152.9000  loss_clip: 277.6297  loss_kl: 6.7781  kl_raw: 640.0000  clip_logit_scale: 14.2500  clip_acc_i2t: 0.0625  clip_acc_t2i: 0.0625  b_local: 16.0000  b_global: 16.0000
tensor(2.7476, device='cuda:0')
[clipdbg] world_size=1  b_local=16.0  b_global=16.0
[clipdbg] requires_grad: logit_scale True img_head True
[clipdbg] grad_mean: logit_scale None img_head None
[clipdbg][WARN] not covered by optimizer: logit=False head=False
