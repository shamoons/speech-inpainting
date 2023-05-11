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
        self.EOS_token = -1.0  # Define the EOS token as a constant

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
        if self.training or True:
            decoded = self.decoder(expanded, encoded)
        else:
            decoded = self._decode_token_by_token(expanded, encoded)

        # # Apply the ReLU activation to the decoded output
        # decoded = self.relu(decoded)

        return decoded, bottleneck_output

    def _decode_token_by_token(self, expanded, encoded):
        batch_size, num_time_frames, _ = expanded.size()
        max_seq_length = num_time_frames  # Set a maximum sequence length to avoid infinite loops
        decoded = torch.full((batch_size, max_seq_length, _), self.EOS_token, device=expanded.device)
        eos_reached = torch.zeros(batch_size, dtype=torch.bool).to(
            expanded.device)  # Track which sequences have reached the EOS token

        # Pad the encoded (memory) input with zeros to match the maximum sequence length
        encoded_padded = torch.cat([encoded, torch.zeros(batch_size, max_seq_length -
                                                         num_time_frames, _, device=expanded.device)], dim=1)

        # Create a mask to ignore future tokens in the decoder's input
        mask = torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool().to(expanded.device)

        t = 0
        while not eos_reached.all() and t < max_seq_length:
            if t == 0:
                decoder_input = expanded[:, :1]
            else:
                decoder_input = torch.cat([expanded[:, :1], decoded[:, :t]], dim=1)

            # Transpose the dimensions of decoder_input and encoded_padded
            decoder_input = decoder_input.transpose(0, 1)
            encoded_padded_transposed = encoded_padded[:, :t+1].transpose(0, 1)

            decoded_output = self.decoder(decoder_input, encoded_padded_transposed, tgt_mask=mask[:t+1, :t+1])
            decoded[:, t] = decoded_output[-1]  # Use the last output from the decoder

            # Check if the EOS token has been reached
            eos_reached = eos_reached | (decoded[:, t] == self.EOS_token).all(dim=-1)

            t += 1

        return decoded[:, :t, :]  # Return only the part of the tensor that was filled

    def _get_sinusoidal_position_embeddings(self, num_positions, d_model):
        position_embeddings = torch.zeros(num_positions, d_model)
        positions = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        position_embeddings[:, 0::2] = torch.sin(positions * div_term)
        position_embeddings[:, 1::2] = torch.cos(positions * div_term)
        position_embeddings = position_embeddings.unsqueeze(0)

        return position_embeddings
