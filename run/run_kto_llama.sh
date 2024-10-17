#!/bin/bash

pixi run accelerate launch kto_llama.py \
    --model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --dataset_name='data/labelled_training_data/all_train.jsonl' \
    --desirable_weight 1.5 \
    --weight \
    --per_device_train_batch_size 2 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --save_steps 200 \
    --output_dir='out/kto_llama3.1_17k' \
    --warmup_ratio 0.1 \
    --bf16 \
    --logging_first_step \
    --max_length=3000 \
    --max_prompt_length=2800 \
    --remove_unused_columns=False \
    --use_peft \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16