import json
from functools import partial
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

id2label = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}
label2id = {v: k for k, v in id2label.items()}


def preprocess_function(examples):
    prompts = examples["prompt"]
    response_as = examples["response_a"]
    response_bs = examples["response_b"]
    ids = examples["id"]

    samples = []
    for (
        prompt,
        response_a,
        response_b,
        id,
    ) in zip(
        prompts,
        response_as,
        response_bs,
        ids,
    ):
        prompt = json.loads(prompt)
        response_a = json.loads(response_a)
        response_b = json.loads(response_b)

        prompt = "".join(prompt)
        response_a = "".join([r if r is not None else "" for r in response_a])
        response_b = "".join([r if r is not None else "" for r in response_b])

        sentences = [prompt + "\n" + response_a, prompt + "\n" + response_b]
        samples.append((id, sentences))
    result = {
        "id": [id for id, _ in samples],
        "sentences": [text for _, text in samples],
    }

    if "winner_model_a" in examples.keys():
        labels = []
        winner_model_a = examples["winner_model_a"]
        winner_model_b = examples["winner_model_b"]
        winner_tie = examples["winner_tie"]
        for model_a, model_b, tie in zip(winner_model_a, winner_model_b, winner_tie):
            if model_a is None or model_b is None or tie is None:
                break
            if model_a == 1:
                label = "winner_model_a"
            elif model_b == 1:
                label = "winner_model_b"
            elif tie == 1:
                label = "winner_tie"
            else:
                raise ValueError("Invalid label")
            labels.append(label2id[label])
        if len(labels) == len(result["id"]):
            result["labels"] = labels
    return result


def tokenize_function(examples, tokenizer, max_seq_length=512):
    """Tokenize function"""
    encodings = []
    for sentence in examples["sentences"]:
        encoding = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        encodings.append(encoding)
    result = {}
    for key in encodings[0].keys():
        result[key] = torch.stack([encoding[key] for encoding in encodings])
    if "labels" in examples.keys():
        result["labels"] = examples["labels"]
    return result


def preprocess_dataset(
    dataset: Dataset,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    accelerator=None,
):
    """Preprocess the dataset"""
    t = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    tokenizer = partial(
        tokenize_function,
        tokenizer=t,
        max_seq_length=max_seq_length,
    )

    if accelerator is not None:
        with accelerator.main_process_first():
            return do_preprocess(dataset, tokenizer, max_seq_length)
    else:
        return do_preprocess(dataset, tokenizer, max_seq_length)


def do_preprocess(dataset: Dataset, tokenizer, max_seq_length: int):
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=get_remove_columns(dataset),
        disable_nullable=True,
        desc="Preprocessing dataset",
    )
    dataset = dataset.map(
        tokenizer,
        batched=True,
        remove_columns=["id", "sentences"],
        disable_nullable=True,
        num_proc=16,
        desc="Tokenizing dataset",
    )
    return dataset


def get_remove_columns(dataset: Dataset):
    if isinstance(dataset, dict):
        return dataset[list(dataset.keys())[0]].column_names
    else:
        return dataset.column_names


def prepare_dataset(
    dataset_dir: str,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    accelerator=None,
    dataset_name: Optional[str] = None,
):
    """Prepare the dataset"""

    dataset = load_dataset("csv", data_dir=dataset_dir)
    if dataset_name is not None:
        dataset = dataset[dataset_name]
    dataset = preprocess_dataset(
        dataset, tokenizer_name_or_path, max_seq_length, accelerator
    )
    return dataset
