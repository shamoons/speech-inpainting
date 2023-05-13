import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, max_len, dropout=0.0):
        super(TransformerAutoencoder, self).__init__()

        self.sos_embedding = nn.Parameter(torch.randn(d_model))  # [d_model]
        self.eos_embedding = nn.Parameter(torch.randn(d_model))  # [d_model]

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, trg):
        # src: [batch_size, src_len, d_model]
        # trg: [batch_size, trg_len, d_model]
        src = src.transpose(0, 1)  # [src_len, batch_size, d_model]
        trg = trg.transpose(0, 1)  # [trg_len, batch_size, d_model]

        sos_tensor = self.sos_embedding.repeat(1, src.size(1), 1).to(src.device)  # [1, batch_size, d_model]
        eos_tensor = self.eos_embedding.repeat(1, src.size(1), 1).to(src.device)  # [1, batch_size, d_model]

        # src = torch.log1p(src)  # [src_len, batch_size, d_model]
        # trg = torch.log1p(trg)  # [trg_len, batch_size, d_model]
        src_normalized = (src - src.mean()) / src.std()  # [src_len, batch_size, d_model]
        trg_normalized = (trg - trg.mean()) / trg.std()  # [trg_len, batch_size, d_model]

        src_scaled = torch.log1p(src_normalized)  # [src_len, batch_size, d_model]
        trg_scaled = torch.log1p(trg_normalized)  # [trg_len, batch_size, d_model]

        print(f"src_normalized.mean: {src_normalized.mean()}")
        print(f"src_normalized.max: {src_normalized.max()}")
        print(f"src_normalized.min: {src_normalized.min()}")
        print(f"src_normalized.std: {src_normalized.std()}")
        print("-" * 50)
        print(f"src_scaled.max: {src_scaled.max()}")
        print(f"src_scaled.min: {src_scaled.min()}")
        print(f"src_scaled.std: {src_scaled.std()}")

        trg = torch.cat([sos_tensor, trg, eos_tensor], dim=0)  # [trg_len+2, batch_size, d_model]

        src = self.pos_encoder(src).to(src.device)  # [src_len, batch_size, d_model]
        trg = self.pos_decoder(trg).to(trg.device)  # [trg_len+2, batch_size, d_model]

        latent_representation = self.encoder(src)  # [src_len, batch_size, d_model]
        latent_representation = self.relu(latent_representation)

        output = self.decoder(trg, latent_representation)  # [trg_len+2, batch_size, d_model]

        # Remove sos and eos from output
        output = output[1:-1, :, :]  # [trg_len, batch_size, d_model]

        # output = torch.exp(output)  # [trg_len, batch_size, d_model]
        denormalized_output = (output * trg.std()) + trg.mean()  # [trg_len, batch_size, d_model]

        print(f"denormalized_output.mean: {denormalized_output.mean()}")
        print(f"denormalized_output.max: {denormalized_output.max()}")
        print(f"denormalized_output.min: {denormalized_output.min()}")
        print(f"denormalized_output.std: {denormalized_output.std()}")

        output = self.fc_out(output).transpose(0, 1)  # [batch_size, trg_len, d_model]
        output = self.relu(output)  # [batch_size, trg_len, d_model]

        return output, latent_representation, sos_tensor, eos_tensor

    def inference(self, sos_tensor, eos_tensor, latent_representation, max_len=50):
        device = latent_representation.device  # No specific shape, this is a device type

        # Initialize output tensor with zeros
        # Shape: [max_len, batch_size, d_model]
        outputs = torch.zeros(max_len, latent_representation.size(1), latent_representation.size(-1)).to(device)

        # Set the first output as the SOS tensor
        # Shape: [max_len, batch_size, d_model]
        outputs[0, :] = sos_tensor.to(device)

        # Loop over the maximum length
        for i in range(1, max_len):
            # Get positional encoding for current outputs
            # Shape of trg_tmp: [i, batch_size, d_model]
            trg_tmp = self.pos_decoder(outputs[:i])

            # Run decoder
            # Shape of out: [i, batch_size, d_model]
            out = self.decoder(trg_tmp, latent_representation)

            # Apply linear layer to the output
            # Shape of out: [i, batch_size, d_model]
            out = self.fc_out(out)

            # Set the current output
            # Shape of outputs: [max_len, batch_size, d_model]
            outputs[i] = out[-1]

            # If all values in the current output equal the EOS tensor, end the loop
            if torch.all(torch.eq(outputs[i], eos_tensor.to(device))):
                # Return outputs until current index
                # Shape: [i, batch_size, d_model]
                return outputs[:i]

        # If EOS was not hit, return all outputs
        return outputs.transpose(0, 1)  # [batch_size, max_len, d_model]


# class TransformerAutoencoder2(nn.Module):
#     def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.0):
#         super(TransformerAutoencoder2, self).__init__()

#         self.encoder = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
#             num_layers=num_layers,
#         )
#         self.relu = nn.ReLU()

#         self.bottleneck = nn.Linear(d_model, d_model)

#         self.decoder = nn.TransformerDecoder(
#             decoder_layer=nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
#             num_layers=num_layers
#         )

#         self.d_model = d_model

#     def forward(self, src, tgt=None):
#         num_time_frames = src.size(1)

#         # Generate sinusoidal position embeddings
#         position_embeddings_src = self._get_sinusoidal_position_embeddings(num_time_frames, self.d_model).to(src.device)

#         # Add position embeddings to input
#         src = src + position_embeddings_src

#         src = src.transpose(0, 1)  # shape: (T, batch_size, n_mels)

#         # Pass the input through the encoder
#         memory = self.encoder(src).transpose(0, 1)  # shape: (batch_size, T, n_mels)
#         memory = self.relu(memory)

#         # Pass the output of the encoder through the bottleneck
#         bottleneck = self.bottleneck(memory)  # shape: (batch_size, T, n_mels)
#         bottleneck = self.relu(bottleneck)
#         bottleneck = bottleneck.mean(dim=1)  # shape: (batch_size, n_mels)

#         if tgt is not None:
#             # In training mode, we have the target sequence
#             # Prepend the bottleneck to the target sequence
#             tgt = torch.cat((bottleneck.unsqueeze(1), tgt), dim=1)  # shape: (batch_size, T + 1, n_mels)

#             # Generate position embeddings for the new target sequence
#             position_embeddings_tgt = self._get_sinusoidal_position_embeddings(
#                 num_time_frames + 1, self.d_model).to(tgt.device)  # +1 to account for the bottleneck

#             tgt = tgt + position_embeddings_tgt

#             tgt = tgt.transpose(0, 1)  # shape: (T + 1, batch_size, n_mels)
#             output = self.decoder(tgt, memory.transpose(0, 1))  # shape: (T + 1, batch_size, n_mels)

#         else:
#             # In inference mode, we generate the target sequence step by step
#             output = self._generate_sequence(bottleneck, memory.transpose(0, 1), num_time_frames)

#         # Transpose output back to (batch_size, T, n_mels)
#         output = output.transpose(0, 1)

#         return output, bottleneck

#     def inference(self, bottleneck, max_length):


#     def _generate_sequence(self, bottleneck, memory, max_length):
#         # Initialize output with the bottleneck
#         output = bottleneck.unsqueeze(0)  # shape: (1, batch_size, n_mels)
#         for _ in range(max_length):
#             output_step = self.decoder(output, memory)
#             output = torch.cat((output, output_step[-1:, :, :]), dim=0)
#         return output

#     def _get_sinusoidal_position_embeddings(self, num_positions, d_model):
#         position_embeddings = torch.zeros(num_positions, d_model)
#         positions = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

#         position_embeddings[:, 0::2] = torch.sin(positions * div_term)
#         position_embeddings[:, 1::2] = torch.cos(positions * div_term)
#         position_embeddings = position_embeddings.unsqueeze(0)

#         return position_embeddings
