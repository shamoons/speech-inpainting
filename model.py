# model.py
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
        x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        # Add positional encoding to the input
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, max_len, embedding_dim, dropout=0.0):
        super(TransformerAutoencoder, self).__init__()

        # Initialize start of sequence and end of sequence embeddings
        self.sos_embedding = nn.Parameter(torch.randn(d_model))
        self.eos_embedding = nn.Parameter(torch.randn(d_model))

        # Initialize input and target encoders
        self.input_encoder = nn.Linear(d_model, embedding_dim)
        self.target_encoder = nn.Linear(d_model, embedding_dim)

        # Initialize transformer encoder and decoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead), num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead), num_layers=num_layers
        )

        # Initialize final fully connected layer
        self.fc_out = nn.Linear(embedding_dim, d_model)

        # Initialize positional encoders for input and target
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)

        self.d_model = d_model
        self.device = None

    def forward(self, src, trg, src_lengths, trg_lengths):
        """
        src: Tensor of shape [batch_size, src_len, d_model]
        trg: Tensor of shape [batch_size, trg_len, d_model]
        src_lengths: Tensor of shape [batch_size]
        trg_lengths: Tensor of shape [batch_size]
        """
        self.device = src.device

        # Transpose source and target tensors for transformer
        src = src.transpose(0, 1)  # [src_len, batch_size, d_model]
        trg = trg.transpose(0, 1)  # [trg_len, batch_size, d_model]

        # Generate masks based on sequence lengths
        src_mask = torch.arange(src.size(0) + 1).unsqueeze(1) < (src_lengths +
                                                                 1).unsqueeze(0)  # [src_len+1, batch_size]
        src_mask = src_mask.transpose(0, 1).to(self.device)  # [batch_size, src_len+1]
        src_key_padding_mask = ~src_mask

        # Generate target mask
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg.size(0), self.device)  # [trg_len, trg_len]

        # Repeat sos and eos tensor for each instance in the batch
        sos_tensor = self.sos_embedding.repeat(1, src.size(1), 1).to(self.device)  # [1, batch_size, d_model]
        eos_tensor = self.eos_embedding.repeat(1, src.size(1), 1).to(self.device)  # [1, batch_size, d_model]

        # Normalize source and target tensors
        src_mean = src.mean()
        src_std = src.std()
        trg_mean = trg.mean()
        trg_std = trg.std()

        src_normalized = (src - src_mean) / src_std  # [src_len, batch_size, d_model]
        trg_normalized = (trg - trg_mean) / trg_std  # [trg_len, batch_size, d_model]

        # Apply logarithm plus 1 for further stability
        src_scaled = torch.log1p(src_normalized)  # [src_len, batch_size, d_model]
        trg_scaled = torch.log1p(trg_normalized)  # [trg_len, batch_size, d_model]

        # Concatenate start of sequence tensors with source
        src_sos = torch.cat([sos_tensor, src_scaled], dim=0)  # [src_len+1, batch_size, d_model]

        # Insert end of sequence tensors in target before padding
        trg_eos = self._insert_eos_before_pad(trg_scaled, eos_tensor, trg_lengths)  # [trg_len+1, batch_size, d_model]

        embedding_scaling_factor = torch.sqrt(torch.tensor(self.d_model).float())

        # Apply input and target encoders and scale the output by square root of d_model
        # [src_len + 1, batch_size, embedding_dim]
        src_embedding = self.input_encoder(src_sos) * embedding_scaling_factor
        # [trg_len+1, batch_size, embedding_dim]
        tgt_embedding = self.target_encoder(trg_eos) * embedding_scaling_factor

        # Apply positional encoding to the source and target embeddings
        src_with_pe = self.pos_encoder(src_embedding).to(self.device)  # [src_len + 1, batch_size, embedding_dim]
        trg_with_pe = self.pos_decoder(tgt_embedding).to(self.device)  # [trg_len+1, batch_size, embedding_dim]

        # Pass the source embeddings through the transformer encoder
        latent_representation = self.transformer_encoder(
            src_with_pe, src_key_padding_mask=src_key_padding_mask)  # [src_len+1, batch_size, embedding_dim]

        # Pass the target embeddings and the encoder output through the transformer decoder
        # [trg_len+1, batch_size, embedding_dim]
        output = self.transformer_decoder(trg_with_pe, latent_representation, tgt_mask=trg_mask,
                                          memory_key_padding_mask=src_key_padding_mask)

        # Remove eos from the output
        output_without_eos = self._remove_eos(output, trg_lengths)  # [trg_len, batch_size, embedding_dim]

        # Pass the decoder output through the final layer
        output_spectrogram = self.fc_out(output_without_eos).transpose(0, 1)  # [batch_size, trg_len, d_model]

        # Reverse the earlier applied transformations to get the final output
        unscaled_output = torch.exp(output_spectrogram)  # [batch_size, trg_len, d_model]
        denormalized_output = (unscaled_output * src_std) + src_mean  # [batch_size, trg_len, d_model]

        return denormalized_output, latent_representation, sos_tensor, eos_tensor

    def _remove_eos(self, tensor, lengths):
        """
        tensor: a Tensor of shape [seq_len, batch_size, d_model]
        lengths: a Tensor of shape [batch_size]
        """
        output = torch.zeros_like(tensor)

        # For each sequence in the batch, copy until the actual length, effectively removing the eos token
        for i in range(tensor.size(1)):
            length = lengths[i]
            output[:length, i, :] = tensor[:length, i, :]

        return output

    def _insert_eos_before_pad(self, tensor, eos_tensor, lengths):
        """
        tensor: a Tensor of shape [seq_len, batch_size, d_model]
        eos_tensor: a Tensor of shape [1, batch_size, d_model]
        lengths: a Tensor of shape [batch_size]
        """
        seq_len, batch_size, _ = tensor.shape
        output = torch.zeros(seq_len + 1, batch_size, tensor.shape[2]).to(tensor.device)
        for i in range(batch_size):
            length = lengths[i]
            output[:length, i, :] = tensor[:length, i, :]
            output[length, i, :] = eos_tensor[0, i, :]  # insert eos at the end of sequence or before padding
        return output
