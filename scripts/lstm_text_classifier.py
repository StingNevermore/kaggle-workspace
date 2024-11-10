import json
import os
from functools import partial

import evaluate
import torch
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from args.lstm_text_classifier_args import ModelArguments, TrainingArguments
from datasets import Dataset, load_dataset
from models.LstmTextClassifier import LstmTextClassifier
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

id2label = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}
label2id = {v: k for k, v in id2label.items()}


def preprocess_function(examples):
    prompts = examples["prompt"]
    response_as = examples["response_a"]
    response_bs = examples["response_b"]
    winner_model_a = examples["winner_model_a"]
    winner_model_b = examples["winner_model_b"]
    winner_tie = examples["winner_tie"]
    ids = examples["id"]

    samples = []
    for (
        prompt,
        response_a,
        response_b,
        winner_model_a,
        winner_model_b,
        winner_tie,
        id,
    ) in zip(
        prompts,
        response_as,
        response_bs,
        winner_model_a,
        winner_model_b,
        winner_tie,
        ids,
    ):
        prompt = json.loads(prompt)
        response_a = json.loads(response_a)
        response_b = json.loads(response_b)
        if winner_model_a == 1:
            label = "winner_model_a"
        elif winner_model_b == 1:
            label = "winner_model_b"
        elif winner_tie == 1:
            label = "winner_tie"
        else:
            raise ValueError("Invalid label")

        prompt = "".join(prompt)
        response_a = "".join([r if r is not None else "" for r in response_a])
        response_b = "".join([r if r is not None else "" for r in response_b])

        sentences = [prompt, response_a, response_b]
        samples.append((id, sentences, label))

    return {
        "id": [id for id, _, _ in samples],
        "sentences": [text for _, text, _ in samples],
        "labels": [label2id[label] for _, _, label in samples],
    }


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
    result["labels"] = examples["labels"]
    return result


def preprocess_dataset(
    dataset: Dataset,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    accelerator: Accelerator,
    seed: int,
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
    with accelerator.main_process_first():
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        dataset = dataset.map(
            tokenizer, batched=True, remove_columns=dataset.column_names
        )
        dataset = dataset.shuffle(seed=seed)
    return dataset


def prepare_dataset(
    dataset_dir: str,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    accelerator: Accelerator,
    seed: int,
):
    """Prepare the dataset"""
    dataset = load_dataset("csv", data_files=os.path.join(dataset_dir, "*.csv"))
    dataset = dataset["train"]
    dataset = preprocess_dataset(
        dataset, tokenizer_name_or_path, max_seq_length, accelerator, seed
    )
    return dataset.train_test_split(test_size=0.1, seed=seed)


def prepare_dataloader(dataset: Dataset, train_batch_size: int, eval_batch_size: int):
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    eval_dataloader = DataLoader(
        dataset["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    return train_dataloader, eval_dataloader


def prepare_model(model_args: ModelArguments):
    """Prepare the model"""
    base_model = AutoModel.from_pretrained(
        model_args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    model = LstmTextClassifier(
        base_model,
        model_args.num_classes,
        base_model_require_grad=True,
    )
    lora_config = LoraConfig(
        r=model_args.base_model_lora_r,
        lora_alpha=model_args.base_model_lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=model_args.base_model_lora_dropout,
        modules_to_save=["word_lstm", "sentence_lstm", "classifier"],
    )
    return get_peft_model(model, lora_config)


def training_loop(
    model,
    optimizer,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    lr_scheduler,
    accelerator: Accelerator,
    training_args: TrainingArguments,
):
    metric = evaluate.load("accuracy")
    torch.cuda.empty_cache()
    for epoch in range(training_args.num_train_epochs):
        progress_bar = get_progress_bar(
            total_steps=len(train_dataloader) * training_args.num_train_epochs,
            desc=f"Epoch {epoch + 1}/{training_args.num_train_epochs}",
            accelerator=accelerator,
        )
        model.train()
        total_steps = len(train_dataloader) * training_args.num_train_epochs
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / training_args.gradient_accumulation_steps
            total_train_loss += loss.item()
            accelerator.backward(loss)
            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), training_args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            logging_steps = handle_steps(training_args.logging_steps, total_steps)
            if (step + 1) % logging_steps == 0:
                print(
                    f"Epoch {epoch}, Step {step}, Loss {total_train_loss / logging_steps}"
                )
                accelerator.log(
                    {"train_loss": total_train_loss / logging_steps},
                    step=step,
                )
            progress_bar.update()
            if accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "loss": f"{total_train_loss / logging_steps:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                )
            save_steps = handle_steps(training_args.save_steps, total_steps)
            if (step + 1) % save_steps == 0:
                accelerator.save_state(training_args.output_dir)
            eval_steps = handle_steps(training_args.eval_steps, total_steps)
            if (step + 1) % eval_steps == 0:
                eval_loop(
                    model, eval_dataloader, accelerator, metric, (step + 1) / eval_steps
                )
            if (step + 1) % 100 == 0:
                torch.cuda.empty_cache()
            progress_bar.close()
    accelerator.end_training()


def eval_loop(
    model, eval_dataloader: DataLoader, accelerator: Accelerator, metric, eval_step
):
    model.eval()
    total_eval_loss = 0
    for batch in enumerate(eval_dataloader):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        logits, labels = accelerator.gather_for_metrics((logits, batch["labels"]))
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
        eval_metric = metric.compute()
        total_eval_loss += accelerator.reduce(outputs.loss, reduction="mean").item()
    accelerator.log(
        {
            "eval_accuracy": eval_metric["accuracy"],
            "eval_loss": total_eval_loss / len(eval_dataloader),
        },
        step=eval_step,
    )


def get_progress_bar(total_steps, desc, accelerator):
    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    return tqdm(
        total=total_steps,
        desc=desc,
        disable=not accelerator.is_main_process,
        position=0,
        leave=True,
        ascii=True,
        bar_format=progress_bar_format,
    )


def handle_steps(steps, total_steps):
    return int(steps) if steps > 1 else int(total_steps * steps)


def main():
    """Main function"""
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(project_dir=training_args.logs_dir, log_with=["all"])

    dataset = prepare_dataset(
        training_args.dataset_dir,
        model_args.tokenizer_name_or_path,
        model_args.max_seq_length,
        accelerator,
        training_args.seed,
    )

    train_dataloader, eval_dataloader = prepare_dataloader(
        dataset,
        training_args.per_device_train_batch_size,
        training_args.per_device_eval_batch_size,
    )

    set_seed(training_args.seed)

    model = prepare_model(model_args)
    model = model.to(accelerator.device)

    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=training_args.learning_rate)

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=200,
            num_training_steps=(len(train_dataloader) * training_args.num_train_epochs)
            // training_args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            total_num_steps=(len(train_dataloader) * training_args.num_train_epochs),
            warmup_num_steps=200,
        )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    training_loop(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
        accelerator,
        training_args,
    )
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            training_args.model_save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )


if __name__ == "__main__":
    main()
