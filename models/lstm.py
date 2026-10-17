import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) Network
    for Autoregressive Next-Word Prediction.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.model_name = "LSTM"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] = None):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.lstm(embedded, hidden)
        last_hidden = out[:, -1, :]
        logits = self.fc(self.dropout(last_hidden))
        return logits, hidden
