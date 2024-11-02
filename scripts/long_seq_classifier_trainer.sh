#!/bin/bash

SCRIPTS_DIR=$(dirname $(readlink -f "$0"))
PY_SCRIPT=long_seq_classifier_trainer.py
DATA_DIR=${SCRIPTS_DIR}/../data/llm-classification-finetuning
DATASET_DIR="$DATA_DIR/dataset_dialog"

base_model_name=meta-llama/Meta-Llama-3-8B
identifier=Meta-Llama-3-8B-4bit
batch_size=8

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=9994
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export WANDB_PROJECT=long_seq_classifier_train
export WANDB_DIR=$DATA_DIR

torchrun --nnodes 1 --nproc_per_node 1 ${PY_SCRIPT} \
    --base_model_name=${base_model_name} \
    --num_classes=3 \
    --dataset_path=${DATASET_DIR} \
    --run_name="run-${identifier}" \
    --output_dir="${DATA_DIR}/output-${identifier}" \
    --logging_dir="${DATA_DIR}/logs-${identifier}" \
    --eval_strategy="steps" \
    --eval_steps=0.2 \
    --save_only_model=True \
    --save_steps=0.2 \
    --save_strategy="steps" \
    --save_total_limit=3 \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=${batch_size} \
    --num_train_epochs=1 \
    --weight_decay=0.01 \
    --logging_steps=100 \
    --logging_strategy="steps" \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=200 \
    --report_to="wandb" \
    --overwrite_output_dir=True \
    --load_best_model_at_end=True \
    --greater_is_better=False \
    --bf16=True \
    --ddp_find_unused_parameters=False \
    --label_names="labels" \
