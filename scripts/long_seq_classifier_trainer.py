"""Trainer for LongSeqClassifier"""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from datasets import load_from_disk
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedModel,
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


class LongSeqClassifier(nn.Module):
    """
    A long sequence classifier that uses a pre-trained transformer model as the base model
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_classes,
        lstm_hidden_size=256,
        dropout=0.5,
    ):
        super().__init__()
        hidden_size = base_model.config.hidden_size
        self.base_model = base_model
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

    def forward(self, input_ids, attention_mask, labels=None):
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
        with torch.no_grad():
            outputs = self.base_model(**inputs).last_hidden_state

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
            return loss, logits
        return logits


def get_model(base_model_name, num_classes, **kwargs):
    """ "Get a LongSeqClassifier model"""
    base_model = AutoModel.from_pretrained(base_model_name, **kwargs)
    model = LongSeqClassifier(base_model, num_classes)
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


def prepare_dataset(model_args):
    """Prepare dataset"""
    tokenzier_name = model_args.tokenizer_name or model_args.base_model_name
    dataset = load_from_disk(model_args.dataset_path)
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
    return dataset.map(
        tokenizer, batched=True, remove_columns=dataset["train"].column_names
    )


def main():
    """Main function"""
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    # pylint: disable-next=unbalanced-tuple-unpacking
    model_args, training_args = parser.parse_args_into_dataclasses()

    dataset = prepare_dataset(model_args)

    bit_and_byte_config = BitsAndBytesConfig(load_in_8bit=True)
    model = get_model(
        model_args.base_model_name,
        model_args.num_classes,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        bit_and_byte_config=bit_and_byte_config,
        device_map="auto",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "model"))


if __name__ == "__main__":
    main()
