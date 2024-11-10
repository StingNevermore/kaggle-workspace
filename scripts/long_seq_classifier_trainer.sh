#!/bin/bash

# ENV_NAME: local, tencent, virtai, autodl, default: local
ENV_NAME=$1
if [ -z "${ENV_NAME}" ]; then
    ENV_NAME=local
fi

identifier=llama-3.2-3b-lora
WANDB_API_KEY=68b48b5fbbb9808a21858dca257cd4d4f699643c
SCRIPTS_DIR=$(dirname $(readlink -f "$0"))
PY_SCRIPT=${SCRIPTS_DIR}/long_seq_classifier_trainer.py

if [ "${ENV_NAME}" == "local" ]; then
    DATA_DIR=${SCRIPTS_DIR}/../data/llm-classification-finetuning
    DATA_PATH="$DATA_DIR/data_csv/train.csv"
    BASE_MODEL_NAME=meta-llama/Llama-3.2-3B
    OUTPUT_DIR=${DATA_DIR}/output-${identifier}
    LOGS_DIR=${DATA_DIR}/logs-${identifier}
    batch_size=4
elif [ "${ENV_NAME}" == "virtai" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    OUTPUT_BASE_DIR=/gemini/output
    DATA_PATH=${GEMINI_DATA_IN1}/train.csv
    LOGS_DIR=${OUTPUT_BASE_DIR}/logs-${identifier}
    BASE_MODEL_NAME=${GEMINI_PRETRAIN}
    TOKENIZER_NAME=${GEMINI_PRETRAIN2}
    OUTPUT_DIR=${OUTPUT_BASE_DIR}/output-${identifier}
    batch_size=8
elif [ "${ENV_NAME}" == "tencent" ]; then
    DATA_PATH=/opt/ml/datasets/train.csv
    LOGS_DIR=/opt/ml/logs
    BASE_MODEL_NAME=/opt/ml/pretrained/llama-3-8b
    TOKENIZER_NAME=/opt/ml/pretrained/llama-3-8b-tokenizer
    OUTPUT_DIR=/opt/ml/output
    batch_size=4
elif [ "${ENV_NAME}" == "autodl" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    OUTPUT_BASE_DIR=/root/autodl-fs
    DATA_PATH=${OUTPUT_BASE_DIR}/datasets/llm-classification/train.csv
    LOGS_DIR=/root/tf-logs/${identifier}
    BASE_MODEL_NAME=${OUTPUT_BASE_DIR}/pretrained/llama-3-8b
    TOKENIZER_NAME=${OUTPUT_BASE_DIR}/pretrained/llama-3-8b
    OUTPUT_DIR=${OUTPUT_BASE_DIR}/output/${identifier}
    batch_size=4
fi

export LOCAL_RANK=0
export WORLD_SIZE=1
export WANDB_PROJECT=long_seq_classifier_train
export WANDB_DIR=${LOGS_DIR}

accelerate launch --config_file ${SCRIPTS_DIR}/default_deepseepd_config.yaml ${PY_SCRIPT} \
    --do_train=True \
    --base_model_name=${BASE_MODEL_NAME} \
    --tokenizer_name=${TOKENIZER_NAME} \
    --num_classes=3 \
    --dataset_path=${DATA_PATH} \
    --run_name="run-${identifier}" \
    --output_dir=${OUTPUT_DIR} \
    --logging_dir="${LOGS_DIR}" \
    --evaluation_strategy="steps" \
    --eval_steps=0.2 \
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
    --report_to=tensorboard \
    --report_to=wandb \
    --overwrite_output_dir=True \
    --load_best_model_at_end=True \
    --greater_is_better=False \
    --bf16=True \
    --ddp_find_unused_parameters=False \
    --label_names="labels" \
    --max_seq_length=512 \
    --use_4bit=False \
    --use_lora=True \
    --deepspeed_config_file=${SCRIPTS_DIR}/zero_stage3_config.json
