
# gasket server llama3.1
pixi run CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001 \
  --served_model_name llama3.1 \
  --gpu-memory-utilization 0.7 \
  --enable-lora \
  --lora-modules llama3.1=out/kto_llama3.1_17k