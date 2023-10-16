amount of memory:128G
	Python3 anaconda is now loaded in your environment. 
	For all commands you should now do the following: 
	source $condaDotFile

Traceback (most recent call last):
  File "/jmain02/home/J2AD003/txk66/jac88-txk66/repos/end2end_arg/llama2_demo.py", line 12, in <module>
    pipeline = transformers.pipeline(
  File "/jmain02/home/J2AD003/txk66/jac88-txk66/.conda/envs/summarize_thread/lib/python3.10/site-packages/transformers/pipelines/__init__.py", line 788, in pipeline
    framework, model = infer_framework_load_model(
  File "/jmain02/home/J2AD003/txk66/jac88-txk66/.conda/envs/summarize_thread/lib/python3.10/site-packages/transformers/pipelines/base.py", line 270, in infer_framework_load_model
    model = model_class.from_pretrained(model, **kwargs)
  File "/jmain02/home/J2AD003/txk66/jac88-txk66/.conda/envs/summarize_thread/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 467, in from_pretrained
    return model_class.from_pretrained(
  File "/jmain02/home/J2AD003/txk66/jac88-txk66/.conda/envs/summarize_thread/lib/python3.10/site-packages/transformers/modeling_utils.py", line 2192, in from_pretrained
    raise ImportError(
ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`
