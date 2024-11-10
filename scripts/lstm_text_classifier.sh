#!/bin/bash

identifier="llama-3-8b-text-classifier"
base_dir="/root/autodl-fs"
model_name_or_path="${base_dir}/pretrained/llama-3-8b"
tokenizer_name_or_path="${base_dir}/pretrained/llama-3-8b"
dataset_dir="${base_dir}/datasets/llm-classification"
data_dir="${base_dir}/train-${identifier}"
logs_dir="/root/tf-logs/logs/${identifier}"
scripts_dir=$(dirname $(readlink -f "$0"))
batch_size=4

export CUDA_VISIBLE_DEVICES="0,1"

accelerate launch --config_file ${scripts_dir}/default_deepseepd_config.yaml ${scripts_dir}/lstm_text_classifier.py \
    --base_model_name_or_path "${model_name_or_path}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --dataset_dir "${dataset_dir}" \
    --num_classes 3 \
    --max_seq_length 512 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --data_dir "${data_dir}" \
    --logs_dir "${logs_dir}" \
    --identifier "${identifier}" \
    --eval_steps 0.1 \
    --mini_dataset_test_size 0.1
