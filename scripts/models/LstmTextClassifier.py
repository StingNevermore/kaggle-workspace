import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class LstmTextClassifier(nn.Module):
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

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
