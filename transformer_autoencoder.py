# transformer_autoencoder.py
import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_length):
        super(TransformerAutoencoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

    def forward(self, src):
        # Positional encoding
        positions = torch.arange(0, src.size(0), device=src.device).unsqueeze(1)
        src = src + self.position_embedding(positions)
        # Encoding
        encoded = self.encoder(src)
        # Pooling to obtain fixed-size embeddings
        embeddings = encoded.mean(dim=0)
        # Repeat embeddings to match the sequence length of the input
        repeated_embeddings = embeddings.repeat(src.size(0), 1, 1)
        # Decoding
        decoded = self.decoder(self.fc(repeated_embeddings), encoded)
        return decoded
