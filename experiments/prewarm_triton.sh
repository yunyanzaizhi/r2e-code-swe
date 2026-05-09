#!/bin/bash
cd /home/caiting/verl-agent-exp
source .venv/bin/activate
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0

echo "=== Pre-warming triton cache ==="
python3 -c "
import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
os.environ['VLLM_USE_V1'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-1.5B-Instruct', dtype='float16', enforce_eager=True, 
          tensor_parallel_size=2, gpu_memory_utilization=0.5,
          max_model_len=4608)
prompts = [
    'Hello ' * 5,
    'You are a helpful AI assistant. ' * 50,
    'In this task, you need to ' * 100,
    'Step 1: ' * 200,
]
output = llm.generate(prompts, SamplingParams(max_tokens=50))
for o in output:
    print(o.outputs[0].text[:50])
print('PREWARM SUCCESS')
del llm
import gc
gc.collect()
import torch
torch.cuda.empty_cache()
"
echo "=== Pre-warming done ==="
