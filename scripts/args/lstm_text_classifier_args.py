import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Model arguments"""

    base_model_name_or_path: str = field(
        metadata={"help": "Path to the model or model name"}
    )

    tokenizer_name_or_path: str = field(
        metadata={"help": "Path to the tokenizer or tokenizer name"}
    )

    num_classes: int = field(
        metadata={"help": "The number of classes in the dataset"},
    )

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum sequence length to use with the model"},
    )

    base_model_lora_r: Optional[int] = field(
        default=16,
        metadata={"help": "The rank of the LoRA matrices"},
    )

    base_model_lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "The alpha scale of the LoRA matrices"},
    )

    base_model_lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "The dropout probability for the LoRA layers"},
    )


@dataclass
class TrainingArguments:
    """Training arguments"""

    identifier: str = field(metadata={"help": "Identifier for the training run"})

    data_dir: str = field(metadata={"help": "Path to the data directory"})

    dataset_dir: str = field(metadata={"help": "Path to the dataset directory"})

    logs_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the logs directory"},
    )

    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the output directory"},
    )

    model_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model save directory"},
    )

    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to train the model"},
    )

    do_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to evaluate the model"},
    )

    per_device_train_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "The batch size for training per device"},
    )

    per_device_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "The batch size for evaluation per device"},
    )

    learning_rate: Optional[float] = field(
        default=1e-4,
        metadata={"help": "The learning rate for the optimizer"},
    )

    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs"},
    )

    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing"},
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "The number of steps to accumulate gradients"},
    )

    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={"help": "The maximum gradient norm"},
    )

    logging_steps: Optional[int] = field(
        default=100,
        metadata={"help": "The number of steps between logging"},
    )

    eval_steps: Optional[float] = field(
        default=0.2,
        metadata={"help": "The fraction of training steps between evaluations"},
    )

    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The random seed"},
    )

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = os.path.join(
                self.data_dir, "outputs", f"outputs-{self.identifier}"
            )
        if self.model_save_dir is None:
            self.model_save_dir = os.path.join(
                self.output_dir, f"model-{self.identifier}"
            )
        if self.logs_dir is None:
            self.logs_dir = os.path.join(
                self.data_dir, "logs", f"logs-{self.identifier}"
            )
        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.per_device_train_batch_size
