import torch
import torch.nn as nn

class BidirectionalRNNModel(nn.Module):
    """
    Bidirectional Recurrent Neural Network (BiRNN)
    for Autoregressive Next-Word Prediction.

    Encodes prefix sequence context from both forward and backward directions
    to predict the subsequent token sequentially without future information leakage.
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
        self.model_name = "Bidirectional RNN"
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
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional concatenation doubles the hidden feature dimension
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))   # (batch_size, seq_len, embed_dim)
        out, hidden = self.rnn(embedded, hidden)    # out: (batch_size, seq_len, hidden_dim * 2)

        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        # hidden[-2] is the final forward state, hidden[-1] is the final backward state
        forward_hidden = hidden[-2, :, :]           # (batch_size, hidden_dim)
        backward_hidden = hidden[-1, :, :]          # (batch_size, hidden_dim)
        combined_repr = torch.cat([forward_hidden, backward_hidden], dim=-1) # (batch_size, hidden_dim * 2)

        logits = self.fc(self.dropout(combined_repr)) # (batch_size, vocab_size)
        return logits, hidden
