# compression_model.py
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


class TransformerCompressionAutoencoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, max_len, embedding_dim, dim_feedforward, use_layer_norm=False, dropout=0.0):
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

        # Initialize start and end of sequence embedding
        self.eos_embedding = nn.Parameter(torch.randn(embedding_dim))
        self.sos_embedding = nn.Parameter(torch.randn(embedding_dim))

        # Initialize input encoders
        self.input_encoder = nn.Linear(d_model, embedding_dim)
        self.target_encoder = nn.Linear(d_model, embedding_dim)

        # Initialize transformer encoder and decoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout),
            num_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout),
            num_layers=num_layers,
            dim_feedforward=dim_feedforward
        )

        # Initialize positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)
        # self.pos_compression = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)

        # Initialize final fully connected layer
        self.output_linear = nn.Linear(embedding_dim, d_model)

        # Initialize LayerNorm
        self.layer_norm_enc_input = nn.LayerNorm(embedding_dim)
        self.layer_norm_enc_output = nn.LayerNorm(embedding_dim)
        self.layer_norm_dec_input = nn.LayerNorm(embedding_dim)

        # Initialize an additional transformer layer with a single output position
        # self.compression_transformer = nn.Transformer(
        #     d_model=embedding_dim,
        #     nhead=nhead,
        #     num_encoder_layers=1,
        #     num_decoder_layers=0,
        #     dim_feedforward=embedding_dim,
        #     dropout=dropout
        # )
        # self.compression_transformer_out_pos = nn.Parameter(torch.zeros(embedding_dim))

        self.device = 'cpu'
        self.embedding_dim = embedding_dim
        self.use_layer_norm = use_layer_norm

    def forward(self, src, src_length):
        """
        Forward pass of the Transformer Compression Autoencoder.

        Parameters:
        src: The input sequence of shape [batch_size, src_len, d_model].
        src_length: The length of the input sequence. Shape: [batch_size]
        """

        self.device = src.device

        # Scale the embeddings by square root of embedding dimension as a Python int, not a float
        embedding_scaling_factor = torch.sqrt(torch.tensor(self.embedding_dim).float().to(self.device))
        embedding_scaling_factor = int(embedding_scaling_factor)

        # Transpose and scale source tensor for transformer
        scaled_src = torch.log1p(src).transpose(0, 1)  # [src_len, batch_size, d_model]
        scaled_trg = torch.log1p(src).transpose(0, 1)  # [src_len, batch_size, d_model]

        # Create sos and eos tensor
        sos_tensor = self.sos_embedding.repeat(1, scaled_src.size(1), 1).to(
            self.device)  # [1, batch_size, embedding_dim]

        # Apply input encoder and scale the output by square root of d_model
        # [src_len, batch_size, embedding_dim]
        src_embedding = self.input_encoder(scaled_src) * embedding_scaling_factor

        trg_embedding = self.target_encoder(scaled_trg) * embedding_scaling_factor
        # Add sos to beginning of target embedding and eos to end of target embedding
        # [src_len+1, batch_size, embedding_dim]
        trg_eos = self._insert_eos_before_pad(trg_embedding, src_length)
        trg_sos_eos = torch.cat([sos_tensor, trg_eos], dim=0)  # [src_len+2, batch_size, embedding_dim]

        # Apply positional encoding to the source and target embeddings
        src_with_pe = self.pos_encoder(src_embedding)  # [src_len, batch_size, embedding_dim]
        trg_with_pe = self.pos_decoder(trg_sos_eos)  # [src_len+2, batch_size, embedding_dim] with sos and eos

        # Apply LayerNorm before transformer encoder and decoder
        src_layer_norm = self.layer_norm_enc_input(src_with_pe) if self.use_layer_norm else src_with_pe
        trg_layer_norm = self.layer_norm_dec_input(trg_with_pe) if self.use_layer_norm else trg_with_pe

        # Pass the source embeddings through the transformer encoder
        # Then when you call the transformer encoder:
        padding_mask = self._create_padding_mask(seq_lengths=src_length)  # [batch_size, src_len]
        encoder_output = self.transformer_encoder(
            src_layer_norm, src_key_padding_mask=padding_mask)  # [src_len, batch_size, embedding_dim]

        encoder_output_norm = self.self.layer_norm_enc_output(encoder_output) if self.use_layer_norm else encoder_output

        # Pass the mean encoder output through the transformer decoder
        compressed_vector = encoder_output_norm.mean(dim=0).unsqueeze(0)  # [1, batch_size, embedding_dim]
        # compressed_vector = self.pos_compression(mean_encoder_output)  # [1, batch_size, embedding_dim]

        # Pass the mean encoder output through the transformer decoder
        # [src_len+2, batch_size, embedding_dim]
        decoder_output = self.transformer_decoder(trg_layer_norm, compressed_vector)

        # Apply final linear layer to get the output
        output_spectrogram = self.output_linear(decoder_output).transpose(0, 1)  # [batch_size, src_len+2, d_model]

        # Expand the output spectrogram to the original range
        output_spectrogram = torch.exp(output_spectrogram)

        return output_spectrogram

    def _insert_eos_before_pad(self, trg, lengths):
        """
        Insert end of sequence tensors in the input before padding.

        Parameters:
        trg: Tensor of shape [trg_len, batch_size, embedding_dim]. The input sequence.
        lengths: Tensor of shape [batch_size]. The lengths of the sequences in the batch.

        Returns:
        A tensor of shape [trg_len+1, batch_size, embedding_dim] with the eos inserted before padding.
        """
        # Adjust the shape of the eos tensor
        eos = self.eos_embedding.unsqueeze(0).expand(trg.size(1), -1)  # [batch_size, embedding_dim]

        trg_list = []
        for i, length in enumerate(lengths):
            trg_sequence = trg[:length.item(), i, :]  # Get the non-padded part of the sequence
            trg_sequence = torch.cat([trg_sequence, eos[i].unsqueeze(0)], dim=0)  # Insert the EOS token
            if length.item() < trg.size(0):  # If there was padding in the original sequence
                trg_sequence = torch.cat([trg_sequence, trg[length.item():, i, :]], dim=0)  # Add the padding back in
            trg_list.append(trg_sequence.unsqueeze(1))  # Add the new sequence to the list of sequences
        trg_eos = torch.cat(trg_list, dim=1)  # Concatenate all sequences along the batch dimension

        return trg_eos

    def _create_padding_mask(self, seq_lengths):
        """
        Creates a mask from the sequence lengths.

        Parameters:
        seq_lengths (torch.Tensor): tensor containing sequence lengths of shape (batch_size)

        Returns:
        mask (torch.Tensor): mask of shape (batch_size, max_len) where True indicates a padding token
        """
        batch_size = seq_lengths.size(0)
        max_len = seq_lengths.max().item()
        mask = torch.arange(max_len).expand(batch_size, max_len).to(seq_lengths.device)
        mask = mask >= seq_lengths.unsqueeze(1)
        return mask
