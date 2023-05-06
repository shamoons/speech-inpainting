# reconstruct.py
import os
import random
import torch
from transformer_autoencoder import TransformerAutoencoder
from mel_spectrogram_dataset import MelSpectrogramDataset
import torchaudio.transforms as transforms
import argparse
import torchaudio


def reconstruct_and_save(checkpoint_path, output_dir):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    d_model = 80
    # Define the bottleneck dimension
    bottleneck_dim = 40

    # Instantiate the TransformerAutoencoder
    model = TransformerAutoencoder(d_model=d_model, nhead=4, num_layers=2,
                                   dim_feedforward=512, max_length=1600, bottleneck_dim=bottleneck_dim).to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Instantiate the MelSpectrogramDataset
    mel_train_dataset = MelSpectrogramDataset(n_mels=d_model)
    random_sample_index = torch.randint(0, len(mel_train_dataset), (1,))
    mel_specgram, filename = mel_train_dataset[random_sample_index.item()]

    # Preprocess the waveform using MelSpectrogram
    mel_transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=d_model, hop_length=160)
    mel_specgram = mel_transform(mel_specgram)

    # Convert the mel spectrogram to tensor
    mel_specgram = mel_specgram.unsqueeze(0).to(device)

    # Run the model and perform reconstruction
    with torch.no_grad():
        output, _ = model(mel_specgram)

    # Convert the output tensor to the waveform
    reconstructed_waveform = output.squeeze().transpose(0, 1).cpu()

    # Scale the waveform back to the original range
    reconstructed_waveform = (reconstructed_waveform * 0.5) + 0.5

    # Convert the waveform to mono if needed
    if reconstructed_waveform.ndim == 3:
        reconstructed_waveform = reconstructed_waveform.mean(dim=0)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the reconstructed waveform as an audio file
    file_name = os.path.splitext(os.path.basename(filename))[0]
    reconstructed_file_path = os.path.join(output_dir, f"reconstructed_{file_name}.wav")
    torchaudio.save(reconstructed_file_path, reconstructed_waveform, 16000)

    # Save the original waveform as an audio file
    original_file_path = os.path.join(output_dir, f"original_{file_name}.wav")
    waveform, sample_rate = torchaudio.load(filename)
    torchaudio.save(original_file_path, waveform, sample_rate)


# Create the argument parser
parser = argparse.ArgumentParser(description='Reconstruct audio using a trained model.')

# Add the arguments
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--output-dir', type=str, help='Directory to save the reconstructed file')

# Parse the command-line arguments
args = parser.parse_args()

# Call the reconstruct_and_save function with the provided arguments
reconstruct_and_save(args.checkpoint, args.output_dir)
