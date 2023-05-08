# transformer_autoencoder.py
import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_length, bottleneck_dim, dropout=0.5):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        self.fc_out = nn.Linear(d_model, d_model)
        self.bottleneck = nn.Linear(d_model, bottleneck_dim)
        self.dropout = nn.Dropout(dropout)  # Add dropout layer
        self.bottleneck_expansion = nn.Linear(bottleneck_dim, d_model)

    def forward(self, src):
        # Dynamically generate position embeddings based on the number of time frames
        num_time_frames = src.size(1)
        positions = torch.arange(0, num_time_frames).unsqueeze(1).to(src.device)
        position_embeddings = self.position_embedding(positions).transpose(0, 1)

        # Add position embeddings to the input Mel spectrograms
        src = src + position_embeddings

        # Pass the input through the transformer
        output = self.transformer(src, src)

        # Pass the output through the bottleneck layer
        bottleneck_output = self.bottleneck(output)
        bottleneck_output = self.dropout(bottleneck_output)  # Apply dropout

        # Expand the bottleneck output back to the original dimension
        output = self.bottleneck_expansion(bottleneck_output)
        output = self.dropout(output)  # Apply dropout

        # Pass the output through the final linear layer
        output = self.fc_out(output)

        return output, bottleneck_output
