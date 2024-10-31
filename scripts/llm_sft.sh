#!/bin/bash

SCRIPTS_DIR=$(dirname $(readlink -f "$0"))
DATA_DIR=$1
DATASET_DIR="$DATA_DIR/dataset_text"
OUTPUT_DIR="$DATA_DIR/output"
LOGS_DIR="$DATA_DIR/logs"
PYTHON_SCRIPT="$SCRIPTS_DIR/text_classifier_train.py"
lr=7e-5
batch_size=8
lora_rank=16
lora_alpha=16
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="score"
num_epochs=2

gradient_accumulation_steps=8

model_name="FacebookAI/roberta-large"

torchrun --nnodes 1 --nproc_per_node 1 $PYTHON_SCRIPT \
    --model_name_or_path $model_name \
    --tokenizer_name_or_path $model_name \
    --dataset_dir $DATASET_DIR \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --num_train_epochs $num_epochs \
    --lr_scheduler_type cosine \
    --learning_rate $lr \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 0.01 \
    --logging_dir $LOGS_DIR \
    --save_strategy steps \
    --save_steps 0.2 \
    --eval_strategy steps \
    --eval_steps 0.2 \
    --save_total_limit 3 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --report_to tensorboard \
    --deepspeed "$SCRIPTS_DIR/ds_config.json" \
    --tokenizer_padding max_length \
    --max_seq_length 512 \
    --tokenizer_truncation True \
    --do_train \
    --label_names labels \
    --num_labels 3 \
    --full_tunning \
    --tokenizer_add_special_tokens
