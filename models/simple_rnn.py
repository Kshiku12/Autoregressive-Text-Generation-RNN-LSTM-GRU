import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    """
    Standard Unidirectional Elman Recurrent Neural Network (RNN)
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
        self.model_name = "Simple RNN"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, embed_dim)
        out, hidden = self.rnn(embedded, hidden)   # out: (batch_size, seq_len, hidden_dim)
        # Extract representation at final sequence position for next-word prediction
        last_hidden = out[:, -1, :]                # (batch_size, hidden_dim)
        logits = self.fc(self.dropout(last_hidden))# (batch_size, vocab_size)
        return logits, hidden
