# model.py
import torch
import torch.nn as nn
import math


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.0):
        super(TransformerAutoencoder, self).__init__()

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_layers,
        )
        self.relu = nn.ReLU()

        self.bottleneck = nn.Linear(d_model, d_model)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_layers
        )

        self.d_model = d_model

    def forward(self, src, tgt=None):
        num_time_frames = src.size(1)

        # Generate sinusoidal position embeddings
        position_embeddings_src = self._get_sinusoidal_position_embeddings(num_time_frames, self.d_model).to(src.device)

        # Add position embeddings to input
        src = src + position_embeddings_src

        src = src.transpose(0, 1)  # shape: (T, batch_size, n_mels)

        # Pass the input through the encoder
        memory = self.encoder(src).transpose(0, 1)  # shape: (batch_size, T, n_mels)
        memory = self.relu(memory)

        # Pass the output of the encoder through the bottleneck
        bottleneck = self.bottleneck(memory)  # shape: (batch_size, T, n_mels)
        bottleneck = self.relu(bottleneck)
        bottleneck = bottleneck.mean(dim=1)  # shape: (batch_size, n_mels)

        if tgt is not None:
            # In training mode, we have the target sequence
            # Prepend the bottleneck to the target sequence
            tgt = torch.cat((bottleneck.unsqueeze(1), tgt), dim=1)  # shape: (batch_size, T + 1, n_mels)

            # Generate position embeddings for the new target sequence
            position_embeddings_tgt = self._get_sinusoidal_position_embeddings(
                num_time_frames + 1, self.d_model).to(tgt.device)  # +1 to account for the bottleneck

            tgt = tgt + position_embeddings_tgt

            tgt = tgt.transpose(0, 1)  # shape: (T + 1, batch_size, n_mels)
            output = self.decoder(tgt, memory.transpose(0, 1))  # shape: (T + 1, batch_size, n_mels)

        else:
            # In inference mode, we generate the target sequence step by step
            output = self._generate_sequence(bottleneck, memory.transpose(0, 1), num_time_frames)

        # Transpose output back to (batch_size, T, n_mels)
        output = output.transpose(0, 1)

        return output

    def _generate_sequence(self, bottleneck, memory, max_length):
        # Initialize output with the bottleneck
        output = bottleneck.unsqueeze(0)  # shape: (1, batch_size, n_mels)
        for _ in range(max_length):
            output_step = self.decoder(output, memory)
            output = torch.cat((output, output_step[-1:, :, :]), dim=0)
        return output

    def _get_sinusoidal_position_embeddings(self, num_positions, d_model):
        position_embeddings = torch.zeros(num_positions, d_model)
        positions = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        position_embeddings[:, 0::2] = torch.sin(positions * div_term)
        position_embeddings[:, 1::2] = torch.cos(positions * div_term)
        position_embeddings = position_embeddings.unsqueeze(0)

        return position_embeddings
