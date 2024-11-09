import json
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
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

        return logits


@dataclass
class EvalArguments:
    base_model_name_or_path: str = field(
        default=None, metadata={"help": "The base model name or path"}
    )

    lora_path: str = field(default=None, metadata={"help": "The lora path"})

    tokenizer_name_or_path: str = field(
        default=None, metadata={"help": "The tokenizer name or path"}
    )

    dataset_path: str = field(default=None, metadata={"help": "The dataset path"})

    batch_size: Optional[int] = field(default=16, metadata={"help": "The batch size"})


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

        prompt = "".join(prompt)
        response_a = "".join([r if r is not None else "" for r in response_a])
        response_b = "".join([r if r is not None else "" for r in response_b])

        sentences = [prompt, response_a, response_b]
        samples.append((id, sentences))

    return {
        "id": [id for id, _ in samples],
        "sentences": [text for _, text in samples],
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
    return result


def prepare_dataset(tokenizer_name_or_path: str, dataset_path: str):
    t = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    tokenizer = partial(
        tokenize_function,
        tokenizer=t,
    )
    dataset = load_dataset(dataset_path)["test"]
    dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=dataset.column_names
    )
    dataset = dataset.map(tokenizer, batched=True, remove_columns=dataset.column_names)
    return dataset


def load_model(base_model_name_or_path: str, lora_path: str):
    base_model = AutoModel.from_pretrained(
        base_model_name_or_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    classifier = LongSeqClassifier(base_model, num_classes=3)
    classifier = PeftModel.from_pretrained(
        classifier,
        lora_path,
        low_cpu_mem_usage=True,
    )
    return classifier


def eval_loop(model: nn.Module, dataset: Dataset, batch_size: int):
    accelerator = Accelerator()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator
    )
    model, dataloader = accelerator.prepare(model, dataloader)
    logits = []
    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        logits.append(model(input_ids, attention_mask).detach())
    return torch.cat(logits, dim=0)


def write_result(logits: torch.Tensor, dataset: Dataset, output_path: str):
    logits = logits.softmax(dim=1)
    df = pd.DataFrame(
        {
            "id": dataset["id"],
            "winner_model_a": logits[:, 0],
            "winner_model_b": logits[:, 1],
            "winner_tie": logits[:, 2],
        }
    )
    df.to_csv(output_path, index=False)


def main():
    parser = HfArgumentParser((EvalArguments))
    # pylint: disable-next=unbalanced-tuple-unpacking
    (eval_args,) = parser.parse_args_into_dataclasses()
    dataset = prepare_dataset(eval_args.tokenizer_name_or_path, eval_args.dataset_path)
    model = load_model(eval_args.base_model_name_or_path, eval_args.lora_path)
    logits = eval_loop(model, dataset, eval_args.batch_size)
    write_result(logits, dataset, "submission.csv")


if __name__ == "__main__":
    main()
