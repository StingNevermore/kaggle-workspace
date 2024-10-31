"""
text_classifier_train.py
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import datasets
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    model arguments
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )

    model_module: Optional[str] = field(
        default=None,
        metadata={"help": ("The model module.")},
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The tokenizer for weights initialization.")},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings"
                "when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under "
                "this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    padding_slide: Optional[str] = field(
        default=None,
        metadata={"help": ("The padding slide.")},
    )
    num_labels: Optional[int] = field(
        default=1,
        metadata=({"help": ("The number of labels.")}),
    )
    tokenizer_padding: Optional[str] = field(
        default=None,
        metadata={"help": ("The padding strategy.")},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": ("The maximum sequence length.")},
    )
    tokenizer_truncation: Optional[bool] = field(
        default=None,
        metadata={"help": ("Whether to truncate the tokenizer.")},
    )
    tokenizer_add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to add special tokens.")},
    )


@dataclass
class DatasetArguments:
    """
    dataset arguments
    """

    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("The dataset directory.")},
    )
    validation_split_percentage: Optional[float] = field(
        default=0.1,
        metadata={"help": ("The percentage of the dataset to use for validation.")},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": ("The number of processes to use for the preprocessing.")},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    training arguments
    """

    trainable: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": ("The layers to train.")},
    )
    lora_rank: Optional[int] = field(
        default=16, metadata={"help": ("The rank of the LoRA approximation.")}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": ("The dropout rate.")}
    )
    lora_alpha: Optional[float] = field(default=16.0, metadata={"help": ("The alpha.")})
    modules_to_save: Optional[str] = field(
        default=None, metadata={"help": ("The modules to save.")}
    )
    full_tunning: Optional[bool] = field(
        default=False, metadata={"help": ("Whether to fully tune the model.")}
    )


def prepare_dataset(tokenizer, dataset_args, model_args):
    """
    Prepare the dataset
    """
    dataset = datasets.load_from_disk(dataset_args.dataset_dir)
    tokenize_config = {
        "add_special_tokens": model_args.tokenizer_add_special_tokens,
    }
    if model_args.tokenizer_padding is not None:
        tokenize_config["padding"] = model_args.tokenizer_padding
    if model_args.max_seq_length is not None:
        tokenize_config["max_length"] = model_args.max_seq_length
    if model_args.tokenizer_truncation is not None:
        tokenize_config["truncation"] = model_args.tokenizer_truncation
    tokenize_function = partial(do_tokenize, tokenizer=tokenizer, **tokenize_config)

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=dataset_args.preprocessing_num_workers,
        remove_columns=["text"],
    )
    if isinstance(dataset, datasets.DatasetDict):
        return dataset
    return dataset.train_test_split(test_size=dataset_args.validation_split_percentage)


def load_model_and_tokenizer(model_args, trainning_args):
    """
    Load the model and tokenizer
    """
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    tokenizer_config = {
        "use_fast": True,
    }
    if model_args.padding_slide is not None:
        tokenizer_config["padding_side"] = model_args.padding_slide
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path, **tokenizer_config
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="cuda",
        # pad_token_id=tokenizer.pad_token_id,
        num_labels=model_args.num_labels,
    )
    if trainning_args.gradient_accumulation_steps is not None:
        model.enable_input_require_grads()

    if not trainning_args.full_tunning:
        target_modules = trainning_args.trainable.split(",")
        modules_to_save = trainning_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(",")
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=trainning_args.lora_rank,
            lora_alpha=trainning_args.lora_alpha,
            lora_dropout=trainning_args.lora_dropout,
            modules_to_save=modules_to_save,
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model, tokenizer


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def do_get_checkpoint_from_folder(folder):
    """
    get the last checkpoint
    """
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
    )


def get_checkpoint(training_args):
    """
    Check if a checkpoint is resumed
    """
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = do_get_checkpoint_from_folder(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.",
                last_checkpoint,
            )
        print(f"last checkpoint: {last_checkpoint}")
        return last_checkpoint
    return None


def do_tokenize(examples, tokenizer, **tokenizer_config):
    """
    tokenize function
    """
    batch_encoding = tokenizer(examples["text"], **tokenizer_config)
    return {
        "input_ids": batch_encoding["input_ids"],
        "attention_mask": batch_encoding["attention_mask"],
    }


def train_model(model, train_dataset, eval_dataset, training_args):
    """
    Train the model
    """
    if training_args.do_train:
        last_checkpoint = get_checkpoint(training_args)
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=transformers.default_data_collator,
        )
        trainer.train(resume_from_checkpoint=last_checkpoint)


def main():
    """
    main function
    """
    parser = HfArgumentParser((ModelArguments, DatasetArguments, MyTrainingArguments))
    # pylint: disable-next=unbalanced-tuple-unpacking
    model_args, dataset_args, trainning_args = parser.parse_args_into_dataclasses()

    if trainning_args.logging_dir is not None:
        os.makedirs(trainning_args.logging_dir, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(trainning_args.logging_dir, "train.log")),
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=handlers,
    )
    datasets.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    transformers.utils.logging.add_handler(handlers[1])

    model, tokenizer = load_model_and_tokenizer(model_args, trainning_args)
    dataset = prepare_dataset(tokenizer, dataset_args, model_args)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else dataset["validation"]
    print(f"train dataset: {train_dataset}")
    print(f"eval dataset: {eval_dataset}")

    train_model(model, train_dataset, eval_dataset, trainning_args)

    model.save_pretrained(os.path.join(trainning_args.output_dir, "final_model"))
    print("Training complete!")


if __name__ == "__main__":
    main()
