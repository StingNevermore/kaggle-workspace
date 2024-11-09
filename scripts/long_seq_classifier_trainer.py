"""Trainer for LongSeqClassifier"""

import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


@dataclass
class ModelArguments:
    """Arguments for model configuration"""

    base_model_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The model name of the pre-trained model")},
    )

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The tokenizer name of the pre-trained model")},
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": ("The maximum sequence length")},
    )

    num_classes: Optional[int] = field(
        default=None,
        metadata={"help": ("The number of classes for classification")},
    )

    lstm_hidden_size: Optional[int] = field(
        default=256,
        metadata={"help": ("The hidden size for LSTM")},
    )

    dropout: Optional[float] = field(
        default=0.5,
        metadata={"help": ("Dropout rate")},
    )

    set_eval_base_model: Optional[bool] = field(
        default=True,
        metadata={"help": ("Set the base model to eval mode")},
    )

    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The path to the dataset")},
    )

    use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": ("Use LoRA")},
    )

    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": ("Use 4-bit quantization")},
    )


class LongSeqClassifier(nn.Module):
    """
    A long sequence classifier that uses a pre-trained transformer model as the base model
    """

    def __init__(
        self,
        base_model,
        num_classes,
        base_model_require_grad=True,
        lstm_hidden_size=256,
        dropout=0.5,
    ):
        super().__init__()
        hidden_size = base_model.config.hidden_size
        self.model = base_model
        self.base_model_require_grad = base_model_require_grad
        self.word_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.sentence_lstm = nn.LSTM(
            input_size=lstm_hidden_size * 2,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

        def xavier_init(layer):
            for name, param in layer.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    torch.nn.init.zeros_(param.data)
                    # 设置forget gate的偏置为1
                    param.data[lstm_hidden_size : 2 * lstm_hidden_size].fill_(1)

        xavier_init(self.word_lstm)
        xavier_init(self.sentence_lstm)
        nn.init.uniform_(self.classifier.weight, a=-0.1, b=0.1)
        if self.classifier.bias is not None:
            nn.init.uniform_(self.classifier.bias, -0.1, 0.1)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        """
        Forward pass of the model
        """
        batch_size = input_ids.size(0)
        num_sentences = input_ids.size(1)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        inputs = {
            k: v for k, v in locals().items() if k in ["input_ids", "attention_mask"]
        }
        if self.base_model_require_grad:
            transformer_outputs = self.model(**inputs)
        else:
            self.model.eval()
            with torch.no_grad():
                transformer_outputs = self.model(**inputs)

        outputs = transformer_outputs.last_hidden_state

        word_lstm_output, _ = self.word_lstm(
            outputs
        )  # (batch_size * num_sentences, seq_len, lstm_hidden_size * 2)
        sentence_embeddings = []
        for i in range(batch_size * num_sentences):
            mask = attention_mask[i].bool()
            valid_output = word_lstm_output[i][mask]
            if len(valid_output) > 0:
                sentence_embedding = valid_output.mean(dim=0)
            else:
                sentence_embedding = torch.zeros(self.word_lstm.hidden_size * 2).to(
                    word_lstm_output.device
                )
            sentence_embeddings.append(sentence_embedding)

        sentence_embeddings = torch.stack(sentence_embeddings).view(
            batch_size, num_sentences, -1
        )

        sentence_lstm_output, _ = self.sentence_lstm(sentence_embeddings)
        finnal_output = self.dropout(sentence_lstm_output[:, -1, :])
        logits = self.classifier(finnal_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, logits) if loss is not None else logits


def get_model(model_args):
    """ "Get a LongSeqClassifier model"""
    if model_args.use_4bit:
        bit_and_byte_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bit_and_byte_config = None
    base_model = AutoModel.from_pretrained(
        model_args.base_model_name,
        quantization_config=bit_and_byte_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if model_args.use_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = LongSeqClassifier(
        base_model,
        model_args.num_classes,
        base_model_require_grad=True if model_args.use_lora else False,
    )
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            modules_to_save=["word_lstm", "sentence_lstm", "classifier"],
        )
        print("Using LoRA")
        return get_peft_model(model, lora_config)
    print("Using pure model")
    return model


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
        "labels": [label2id[l] for _, _, l in samples],
    }


def prepare_dataset(model_args):
    """Prepare dataset"""
    tokenzier_name = model_args.tokenizer_name or model_args.base_model_name
    dataset = load_dataset("csv", data_files=model_args.dataset_path)["train"]
    dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=dataset.column_names
    )
    t = AutoTokenizer.from_pretrained(
        tokenzier_name,
        use_fast=True,
    )
    if t.pad_token is None:
        t.pad_token = t.eos_token
    tokenizer = partial(
        tokenize_function,
        tokenizer=t,
        max_seq_length=model_args.max_seq_length,
    )
    dataset = dataset.map(tokenizer, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42)
    return dataset.train_test_split(test_size=0.1)


def main():
    """Main function"""
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    # pylint: disable-next=unbalanced-tuple-unpacking
    model_args, training_args = parser.parse_args_into_dataclasses()

    dataset = prepare_dataset(model_args)

    model = get_model(model_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model(os.path.join(training_args.output_dir, "model"))


if __name__ == "__main__":
    main()
