# positional_encoding.py
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the positional encodings as a learnable parameter
        self.pe = nn.Parameter(torch.randn(max_len, 1, embedding_dim))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]

        Returns:
            Tensor with positional encoding added to the input tensor.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEncodingSine(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncodingSine, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]

        Returns:
            Tensor with positional encoding added to the input tensor.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
