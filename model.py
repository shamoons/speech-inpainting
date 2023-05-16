# model.py
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding


class TransformerAutoencoder(nn.Module):
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
        super(TransformerAutoencoder, self).__init__()

        # Initialize start of sequence and end of sequence embeddings
        self.sos_embedding = nn.Parameter(torch.randn(embedding_dim))
        self.eos_embedding = nn.Parameter(torch.randn(embedding_dim))

        # Initialize input and target encoders
        self.input_encoder = nn.Linear(d_model, embedding_dim)
        self.target_encoder = nn.Linear(d_model, embedding_dim)

        # Initialize transformer encoder and decoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout), num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout), num_layers=num_layers
        )

        # Initialize final fully connected layer
        self.fc_out = nn.Linear(embedding_dim, d_model)

        # Initialize positional encoders for input and target
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)

        self.d_model = d_model
        self.device = None

    def forward(self, src, src_lengths):
        """
        Perform a forward pass of the Transformer autoencoder.

        Parameters:
        src: Tensor of shape [batch_size, src_len, d_model]. The input sequence.
        src_lengths: Tensor of shape [batch_size]. The lengths of the sequences in the batch.

        Returns:
        output_spectrogram: The output tensor after processing. [batch_size, src_len+1, d_model].
        latent_representation: The output of the Transformer encoder. [src_len+1, batch_size, embedding_dim].
        sos_embedding: The start of sequence tensor in the embedding space. [1, batch_size, embedding_dim].
        eos_embedding: The end of sequence tensor in the embedding space. [1, batch_size, embedding_dim].
        """

        self.device = src.device

        # Transpose source tensor for transformer
        src = src.transpose(0, 1)  # [src_len, batch_size, d_model]

        # Scale the input by log1p
        src = torch.log1p(src)

        # Repeat sos and eos tensor for each instance in the batch
        sos_tensor = self.sos_embedding.repeat(1, src.size(1), 1).to(self.device)  # [1, batch_size, embedding_dim]
        eos_tensor = self.eos_embedding.repeat(1, src.size(1), 1).to(self.device)  # [1, batch_size, embedding_dim]

        embedding_scaling_factor = torch.sqrt(torch.tensor(self.d_model).float())

        # Apply input encoder and scale the output by square root of d_model
        src_embedding = self.input_encoder(src) * embedding_scaling_factor  # [src_len, batch_size, embedding_dim]

        # Concatenate start of sequence tensors with source embeddings
        src_sos = torch.cat([sos_tensor, src_embedding], dim=0)  # [src_len+1, batch_size, embedding_dim]

        # Create target from source by adding eos before padding
        # [src_len+1, batch_size, embedding_dim]
        trg_eos = self._insert_eos_before_pad(src_embedding, eos_tensor, src_lengths)

        # Update sequence lengths considering the addition of sos and eos tokens
        src_lengths = src_lengths + 1

        # Generate masks based on sequence lengths
        src_mask = torch.arange(src_sos.size(0)).unsqueeze(1).to(
            self.device) < src_lengths.unsqueeze(0)  # [src_len+1, batch_size]
        src_mask = src_mask.transpose(0, 1).to(self.device)  # [batch_size, src_len+1]
        src_key_padding_mask = ~src_mask

        # Generate target mask
        trg_mask = nn.Transformer.generate_square_subsequent_mask(
            trg_eos.size(0)).to(self.device)  # [src_len+1, src_len+1]

        # Apply positional encoding to the source and target embeddings
        src_with_pe = self.pos_encoder(src_sos).to(self.device)  # [src_len+1, batch_size, embedding_dim]
        # [src_len+1, batch_size, embedding_dim]
        trg_with_pe = self.pos_decoder(trg_eos) * embedding_scaling_factor.to(self.device)
        trg_with_pe = self.pos_decoder(trg_with_pe).to(self.device)  # [src_len+1, batch_size, embedding_dim]

        # Pass the source embeddings through the transformer encoder
        latent_representation = self.transformer_encoder(
            src_with_pe, src_key_padding_mask=src_key_padding_mask)  # [src_len+1, batch_size, embedding_dim]

        # Pass the target embeddings and the encoder output through the transformer decoder
        output = self.transformer_decoder(trg_with_pe, latent_representation, tgt_mask=trg_mask,
                                          memory_key_padding_mask=src_key_padding_mask)

        # Pass the decoder output through the final layer
        output_spectrogram = self.fc_out(output).transpose(0, 1)  # [batch_size, src_len, d_model]

        # Get the output in the original range by applying the exponential function
        output_spectrogram = torch.exp(output_spectrogram)

        return output_spectrogram, latent_representation, sos_tensor, eos_tensor

    def _insert_eos_before_pad(self, trg, eos, lengths):
        """
        Insert end of sequence tensors in the input before padding.

        Parameters:
        trg: Tensor of shape [src_len, batch_size, embedding_dim]. The input sequence.
        eos: Tensor of shape [1, batch_size, embedding_dim]. The end of sequence tensor.
        lengths: Tensor of shape [batch_size]. The lengths of the sequences in the batch.

        Returns:
        A tensor of shape [src_len+1, batch_size, embedding_dim] with the eos inserted before padding.
        """
        trg_eos = torch.cat([trg, eos], dim=0)  # [src_len+1, batch_size, embedding_dim]
        for i, length in enumerate(lengths):
            trg_eos[length, i] = eos[0, i]
        return trg_eos

    def inference(self, latent_representation, sos_tensor, eos_tensor, max_len):
        """
        Perform inference with the model.

        Parameters:
        latent_representation: The output of the Transformer encoder. Shape: [batch_size, src_len+1, embedding_dim]
        sos_tensor: The start of sequence tensor. Shape: [batch_size, 1, d_model]
        eos_tensor: The end of sequence tensor. Shape: [batch_size, 1, d_model]
        max_len: The maximum length of the sequence.

        Returns:
        The generated sequence. Shape: [src_len+1, batch_size, d_model]
        """
        self.device = latent_representation.device

        # The first input to the decoder will be the sos_tensor
        decoder_input = sos_tensor

        # Initialize the sequence with the sos_tensor
        generated_sequence = sos_tensor

        # Cosine similarity
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        latent_representation = latent_representation.transpose(0, 1)  # [src_len+1, batch_size embedding_dim]
        decoder_input = decoder_input.transpose(0, 1)  # [1, batch_size,  d_model]
        generated_sequence = generated_sequence.transpose(0, 1)  # [1, batch_size, d_model]
        sos_tensor = sos_tensor.transpose(0, 1)  # [1, batch_size, d_model]
        eos_tensor = eos_tensor.transpose(0, 1)  # [1, batch_size, d_model]

        print(f"decoder_input: {decoder_input.shape}")
        print(f"latent_representation: {latent_representation.shape}")
        print(f"eos_tensor: {eos_tensor.shape}")

        for _ in range(max_len):
            # Pass the decoder input through the transformer decoder
            output = self.transformer_decoder(decoder_input, latent_representation)
            print(f"generated_sequence: {generated_sequence.shape}")

            # Get the last output token from the transformer
            output_token = output[-1].unsqueeze(0)

            # Check if the output token is similar to the eos_tensor
            if cos(output_token, eos_tensor).item() > 0.99:
                break

            # Append the output token to the generated sequence
            generated_sequence = torch.cat((generated_sequence, output_token), dim=0)

            # Use the output token as the next input to the decoder
            decoder_input = output_token

        # Pass the generated sequence through the final layer
        output_spectrogram = self.fc_out(generated_sequence).transpose(0, 1)  # [batch_size, src_len, d_model]
        output_spectrogram = torch.exp(output_spectrogram)

        return output_spectrogram
