
optw = runner.optim_wrapper
opt = getattr(optw, 'optimizer', None)
optw
{'type': 'DeepSpeedOptimWrapper', 'optimizer': {'type': 'src.optimisers.custom_adamw.CustomAdamW', 'lr': 0.0001, 'betas': (...), 'weight_decay': 0.05}, 'constructor': 'DefaultOptimWrapperConstructor', 'paramwise_cfg': {'custom_keys': {...}, 'bias_decay_mult': 0.0, 'norm_decay_mult': 0.0}}
opt
{'type': 'src.optimisers.custom_adamw.CustomAdamW', 'lr': 0.0001, 'betas': (0.9, 0.95), 'weight_decay': 0.05}
