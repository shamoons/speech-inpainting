# model.py
import torch
import torch.nn as nn
import math


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, bottleneck_size, dropout=0.5):
        super(TransformerAutoencoder, self).__init__()

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_layers
        )

        self.bottleneck = nn.Linear(d_model, bottleneck_size)
        self.bottleneck_expansion = nn.Linear(bottleneck_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.relu = nn.ReLU()

    def forward(self, src):
        num_time_frames = src.size(1)

        # Generate sinusoidal position embeddings
        position_embeddings = self._get_sinusoidal_position_embeddings(num_time_frames, self.d_model).to(src.device)

        # Add position embeddings to input, shape: (batch_size, num_time_frames, d_model)
        src = src + position_embeddings

        # Pass the input through the encoder, shape: (batch_size, num_time_frames, d_model)
        encoded = self.encoder(src)

        # Pass the encoded output through the bottleneck layer, shape: (batch_size, num_time_frames, bottleneck_size)
        bottleneck_output = self.bottleneck(encoded)
        bottleneck_output = self.dropout(bottleneck_output)

        # Expand the bottleneck output back to the original dimension, shape: (batch_size, num_time_frames, d_model)
        expanded = self.bottleneck_expansion(bottleneck_output)
        expanded = self.dropout(expanded)

        # Pass the expanded output through the decoder, shape: (batch_size, num_time_frames, d_model)
        decoded = self.decoder(expanded, encoded)

        # Apply the ReLU activation to the decoded output
        decoded = self.relu(decoded)

        return decoded, bottleneck_output

    def _get_sinusoidal_position_embeddings(self, num_positions, d_model):
        position_embeddings = torch.zeros(num_positions, d_model)
        positions = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        position_embeddings[:, 0::2] = torch.sin(positions * div_term)
        position_embeddings[:, 1::2] = torch.cos(positions * div_term)
        position_embeddings = position_embeddings.unsqueeze(0)

        return position_embeddings
