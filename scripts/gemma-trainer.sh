#!/bin/bash

identifier="gemma-2-9b-it-double-prompts-1024-full"
fs_dir="/root/autodl-fs"
data_disk_dir="/root/autodl-tmp"
model_name_or_path="${fs_dir}/pretrained/gemma-2-9b-it"
tokenizer_name_or_path="${fs_dir}/pretrained/gemma-2-9b-it"
dataset_dir="${fs_dir}/datasets/llm-classification-full"
data_dir="${data_disk_dir}/train-${identifier}"
logs_dir="/root/tf-logs"
scripts_dir=$(dirname $(readlink -f "$0"))
batch_size=8

export CUDA_VISIBLE_DEVICES="0,1"

accelerate launch --config_file ${scripts_dir}/default_deepseepd_config.yaml ${scripts_dir}/lstm_text_classifier.py \
    --base_model_name_or_path "${model_name_or_path}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --dataset_dir "${dataset_dir}" \
    --num_classes 3 \
    --max_seq_length 1024 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --learning_rate 4e-4 \
    --num_train_epochs 1 \
    --data_dir "${data_dir}" \
    --logs_dir "${logs_dir}" \
    --identifier "${identifier}" \
    --eval_steps 0.2 \
    --logging_steps 50 \
    --warmup_steps 100 
