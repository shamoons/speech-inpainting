# model.py
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


class TransformerCompressionAutoencoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, max_len, embedding_dim, dropout=0.0):
        """
        Initialize the Transformer autoencoder.

        Parameters:
        d_model: The dimension of the input and output vectors.
        num_layers: The number of transformer layers.
        nhead: The number of heads in the multihead attention models.
        max_len: The maximum length of the input sequence.
        embedding_dim: The dimension of the embeddings.
        dropout: The dropout value.
        """
        super(TransformerCompressionAutoencoder, self).__init__()

        # Initialize input encoders
        self.input_encoder = nn.Linear(d_model, embedding_dim)
        self.target_encoder = nn.Linear(d_model, embedding_dim)

        # Initialize transformer encoder and decoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout), num_layers=num_layers
        )

        # Initialize positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)

    def forward(self, src, src_length):
        """
        Forward pass of the Transformer Compression Autoencoder.

        Parameters:
        src: The input sequence of shape [batch_size, src_len, d_model].
        src_length: The length of the input sequence. Shape: [batch_size]
        """

        embedding_scaling_factor = torch.sqrt(torch.tensor(self.d_model).float())

        # Transpose and scale source tensor for transformer
        src = torch.log1p(src).transpose(0, 1)  # [src_len, batch_size, d_model]

        # Create eos tensor
        eos_tensor = self.eos_embedding.repeat(1, src.size(1), 1).to(self.device)  # [1, batch_size, embedding_dim]

        # Apply input encoder and scale the output by square root of d_model
        src_embedding = self.input_encoder(src) * embedding_scaling_factor  # [src_len, batch_size, embedding_dim]

        # Create target from source by adding eos before padding
        # [src_len+1, batch_size, embedding_dim]
        trg_eos = self._insert_eos_before_pad(src_embedding, eos_tensor, src_length)
        trg_embedding = self.target_encoder(trg_eos) * embedding_scaling_factor

        # Apply positional encoding to the source and target embeddings
        src_with_pe = self.pos_encoder(src_embedding)  # [src_len, batch_size, embedding_dim]
        trg_with_pe = self.pos_encoder(trg_embedding)  # [src_len+1, batch_size, embedding_dim]

        # Pass the source embeddings through the transformer encoder
        encoder_output = self.transformer_encoder(src_with_pe)  # [src_len, batch_size, embedding_dim]

        # Compute the mean of the encoder output across the sequence length
        mean_encoder_output = torch.mean(encoder_output, dim=0)  # [batch_size, embedding_dim]

        # Pass the mean encoder output through the transformer decoder
        # [src_len+1, batch_size, embedding_dim]
        decoder_output = self.transformer_decoder(trg_with_pe, mean_encoder_output)
