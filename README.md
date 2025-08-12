[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.10/runpy.py", line 196, in _run_module_as_main
[rank0]:     return _run_code(code, main_globals, None,
[rank0]:   File "/usr/local/lib/python3.10/runpy.py", line 86, in _run_code
[rank0]:     exec(code, run_globals)
[rank0]:   File "/root/.vscode-server/extensions/ms-python.debugpy-2025.10.0/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 71, in <module>
[rank0]:     cli.main()
[rank0]:   File "/root/.vscode-server/extensions/ms-python.debugpy-2025.10.0/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 501, in main
[rank0]:     run()
[rank0]:   File "/root/.vscode-server/extensions/ms-python.debugpy-2025.10.0/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 351, in run_file
[rank0]:     runpy.run_path(target, run_name="__main__")
[rank0]:   File "/root/.vscode-server/extensions/ms-python.debugpy-2025.10.0/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 310, in run_path
[rank0]:     return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
[rank0]:   File "/root/.vscode-server/extensions/ms-python.debugpy-2025.10.0/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 127, in _run_module_code
[rank0]:     _run_code(code, mod_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)
[rank0]:   File "/root/.vscode-server/extensions/ms-python.debugpy-2025.10.0/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 118, in _run_code
[rank0]:     exec(code, run_globals)
[rank0]:   File "/vepfs/DI/yaqi/understand_gen/CrossUni-do/scripts/train.py", line 357, in <module>
[rank0]:     main()
[rank0]:   File "/vepfs/DI/yaqi/understand_gen/CrossUni-do/scripts/train.py", line 353, in main
[rank0]:     runner.train()
[rank0]:   File "/usr/local/lib/python3.10/site-packages/mmengine/runner/_flexible_runner.py", line 1200, in train
[rank0]:     model = self.train_loop.run()  # type: ignore
[rank0]:   File "/usr/local/lib/python3.10/site-packages/mmengine/runner/loops.py", line 289, in run
[rank0]:     self.run_iter(data_batch)
[rank0]:   File "/usr/local/lib/python3.10/site-packages/mmengine/runner/loops.py", line 313, in run_iter
[rank0]:     outputs = self.runner.model.train_step(
[rank0]:   File "/usr/local/lib/python3.10/site-packages/mmengine/_strategy/deepspeed.py", line 134, in train_step
[rank0]:     parsed_loss, log_vars = self.model.module.parse_losses(losses)
[rank0]:   File "/usr/local/lib/python3.10/site-packages/mmengine/model/base_model/base_model.py", line 171, in parse_losses
[rank0]:     raise TypeError(
[rank0]: TypeError: clip_logit_scale is not a tensor or list of tensors
[rank0]:[W812 10:32:28.401062990 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
