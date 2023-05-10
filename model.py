# model.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, bottleneck_size):
        super().__init__()

        # Encoder
        self.linear_in = nn.Linear(d_model, dim_feedforward)
        self.pos_encoder = PositionalEncoding(dim_feedforward)
        encoder_layers = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Bottleneck
        self.linear_bottleneck = nn.Linear(dim_feedforward, bottleneck_size)

        # Decoder
        self.linear_out = nn.Linear(bottleneck_size, d_model)

    def forward(self, src):
        src = self.linear_in(src)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.linear_bottleneck(src)
        src = self.linear_out(src)
        return src
