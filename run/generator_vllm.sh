
# generator server llama3-8b-baseline
pixi run CUDA_VISIBLE_DEVICES=0 vllm serve ./model/Llama3-8B-baseline --gpu-memory-utilization 0.2 --max-logprobs 2000