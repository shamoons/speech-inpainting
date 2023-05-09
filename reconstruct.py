# reconstruct.py
import os
import torch
import matplotlib.pyplot as plt
from transformer_autoencoder import TransformerAutoencoder
from mel_spectrogram_dataset import MelSpectrogramDataset
from torch.utils.data import DataLoader
import argparse
import torchaudio
import numpy as np
from matplotlib.ticker import FuncFormatter
from torchaudio.transforms import InverseMelScale, GriffinLim


# Define the parameters for the InverseMelScale and GriffinLim transforms
n_mels = 128
n_fft = 256
n_iter = 128
hop_length = n_fft // 2
n_stft = n_fft // 2 + 1
sample_rate = 16000


def plot_spectrogram(spec, output_file, waveform_length):
    fig, ax = plt.subplots()

    # Calculate the time for each frame
    time_per_frame = hop_length / sample_rate
    actual_frames = waveform_length // hop_length + 1

    # Slice the spectrogram to keep only the frames corresponding to the specified length
    spec = spec[:, :actual_frames]

    img = plt.imshow(spec, aspect='auto', origin='lower', cmap='inferno')
    colorbar = plt.colorbar(img, ax=ax)

    # Set the colorbar labels to have two decimal precision
    colorbar.formatter = FuncFormatter(lambda x, pos: f"{x:.2f}")
    colorbar.update_ticks()

    # Set the x-axis ticks and labels for every 0.2 seconds
    tick_interval = 0.2
    num_ticks = int(actual_frames * time_per_frame / tick_interval)
    x_ticks = np.arange(0, actual_frames, actual_frames / num_ticks)
    x_tick_labels = [f"{i * tick_interval:.1f}" for i in range(len(x_ticks))]

    # Set the x-axis ticks and labels
    plt.xticks(x_ticks, x_tick_labels)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def reconstruct_and_save(checkpoint_path, output_dir):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    d_model = n_mels
    # Define the bottleneck dimension
    bottleneck_dim = 128

    # Instantiate the TransformerAutoencoder
    model = TransformerAutoencoder(d_model=d_model, nhead=4, num_layers=2,
                                   dim_feedforward=512, bottleneck_dim=bottleneck_dim).to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Instantiate the MelSpectrogramDataset
    mel_train_dataset = MelSpectrogramDataset(n_mels=d_model, subset='validation')
    train_loader = DataLoader(mel_train_dataset, batch_size=1,
                              shuffle=True, collate_fn=mel_train_dataset.collate_fn)

    for i, (mel_specgrams, labels, waveform_lengths) in enumerate(train_loader):
        mel_specgrams = mel_specgrams.transpose(0, 1).to(device)
        # Receive both the final output and the bottleneck output from the model
        outputs, bottleneck_output = model(mel_specgrams)

        # Save the spectrogram of the original audio and the reconstructed audio
        original_spec_path = os.path.join(output_dir, f"original_{labels[0]}_spec.png")
        reconstructed_spec_path = os.path.join(output_dir, f"reconstructed_{labels[0]}_spec.png")

        # Squeeze the tensors, detach them, and move them to the CPU
        original_mel_specgram = mel_specgrams.squeeze().detach().cpu().numpy()
        reconstructed_mel_specgram = outputs.squeeze().detach().cpu().numpy()

        # Plot and save the spectrograms
        plot_spectrogram(original_mel_specgram, original_spec_path, waveform_lengths[0])
        plot_spectrogram(reconstructed_mel_specgram, reconstructed_spec_path,
                         waveform_lengths[0])

        # Instantiate the InverseMelScale transform
        inverse_mel_scale = InverseMelScale(n_stft=n_stft, n_mels=n_mels, sample_rate=sample_rate)

        # Convert the numpy array to a PyTorch tensor and add a batch dimension
        original_mel_specgram_tensor = torch.from_numpy(original_mel_specgram).unsqueeze(0)
        reconstructed_mel_specgram_tensor = torch.from_numpy(reconstructed_mel_specgram).unsqueeze(0)

        # Convert the Mel spectrogram to a linear-frequency spectrogram
        original_linear_specgram = inverse_mel_scale(original_mel_specgram_tensor)
        reconstructed_linear_specgram = inverse_mel_scale(reconstructed_mel_specgram_tensor)

        # Instantiate the Griffin-Lim transform
        griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length,
                                 n_iter=n_iter)

        # Use the Griffin-Lim algorithm to reconstruct the waveform from the linear-frequency spectrogram
        original_waveform = griffin_lim(original_linear_specgram)
        reconstructed_waveform = griffin_lim(reconstructed_linear_specgram)

        # Save the waveform as a .wav file
        original_audio_path = os.path.join(output_dir, f"original_{labels[0]}_audio.wav")
        reconstructed_audio_path = os.path.join(output_dir, f"reconstructed_{labels[0]}_audio.wav")

        torchaudio.save(original_audio_path, original_waveform, sample_rate)
        torchaudio.save(reconstructed_audio_path, reconstructed_waveform, sample_rate)

        break
# Create the argument parser


parser = argparse.ArgumentParser(description='Reconstruct audio using a trained model.')

# Add the arguments

parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--output-dir', type=str, help='Directory to save the reconstructed file')

# Parse the command-line arguments

args = parser.parse_args()

# Call the reconstruct_and_save function with the provided arguments

reconstruct_and_save(args.checkpoint, args.output_dir)
