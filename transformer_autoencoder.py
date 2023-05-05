# transformer_autoencoder.py
import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_length):
        super(TransformerAutoencoder, self).__init__()
        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, dim_feedforward, batch_first=True
        )
        # Ensure that the position_embedding layer has enough embeddings for the maximum sequence length
        self.position_embedding = nn.Embedding(max_length, d_model)

    def forward(self, src):
        # Get the batch size and sequence length from the input tensor
        batch_size, seq_length, _ = src.size()

        # Generate position indices for each position in the sequence
        positions = torch.arange(0, seq_length).unsqueeze(0).repeat(batch_size, 1).to(src.device)

        # Convert position indices into position embeddings
        position_embeddings = self.position_embedding(positions)

        # Add position embeddings to the input tensor
        src = src + position_embeddings

        # Pass the input tensor through the transformer
        output = self.transformer(src, src)
        return output
