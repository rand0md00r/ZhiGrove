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






# 在你的 Hook 里加一个方法（或把原来的 after_train_iter 拷贝到这个回调里）：
def after_train_backward(self, runner, batch_idx, data_batch=None, outputs=None):
    m = runner.model.module if hasattr(runner.model, 'module') else runner.model
    def gmean(x): return None if x is None else float(x.detach().abs().mean().cpu())
    print('[clipdbg/backward] grad_mean:',
          'logit_scale', gmean(getattr(m.clip_logit_scale, 'grad', None)),
          'img_head',    gmean(getattr(m.img_clip_head.weight, 'grad', None)))






if getattr(self, '_debug_autograd_steps', 0) < 2 and self.training:
    g_logit, g_head = torch.autograd.grad(
        loss, [self.clip_logit_scale, self.img_clip_head.weight],
        retain_graph=True, allow_unused=True
    )
    print('[autograd] clip_logit_scale grad None?', g_logit is None,
          ' img_head grad None?', g_head is None,
          ' norms:', None if g_logit is None else float(g_logit.abs().mean().cpu()),
                    None if g_head is None else float(g_head.abs().mean().cpu()))
    self._debug_autograd_steps = getattr(self, '_debug_autograd_steps', 0) + 1



# 在 Hook.before_train_iter 里缓存一次
def before_train_iter(self, runner, batch_idx, data_batch=None):
    m = runner.model.module if hasattr(runner.model, 'module') else runner.model
    self._snap = dict(
        logit_scale = m.clip_logit_scale.detach().clone(),
        img_head    = m.img_clip_head.weight.detach().clone()
    )

# 在 after_train_iter 里比较更新幅度
def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
    if batch_idx not in (0,1,10): return
    m = runner.model.module if hasattr(runner.model, 'module') else runner.model
    d_logit = float((m.clip_logit_scale.detach() - self._snap['logit_scale']).abs().max().cpu())
    d_head  = float((m.img_clip_head.weight.detach() - self._snap['img_head']).abs().max().cpu())
    print(f'[clipdbg/update] Δlogit_scale={d_logit:e}  Δimg_head={d_head:e}')
