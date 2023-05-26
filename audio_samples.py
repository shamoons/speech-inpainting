# audio_samples.py
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torchaudio
import torch
import torchaudio.transforms as T
import random

from data_loader import SpeechCommandsDataset


def save_audio(path, audio, sr):
    torchaudio.save(path, audio, sr)


def save_spectrogram(path, spec):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, ax=ax, x_axis='time', y_axis='mel')
    fig.colorbar(img, ax=ax)
    plt.savefig(path)
    plt.close()


def create_noisy_audio_and_save_spectrogram(dataset, output_dir, noise_levels, n_samples):
    os.makedirs(output_dir, exist_ok=True)

    # Choose a random subset of indices
    indices = random.sample(range(len(dataset)), n_samples)

    for i in indices:
        clean_spec, noisy_spec, _ = dataset[i]

        for noise_level in noise_levels:
            prefix = os.path.join(output_dir, f"sample_{i}_noise_{noise_level}")

            save_spectrogram(f"{prefix}_clean.png", clean_spec.numpy())
            save_spectrogram(f"{prefix}_noisy.png", noisy_spec.numpy())

            # save_audio(f"{prefix}_clean.wav", clean_waveform, sr=16000)
            # save_audio(f"{prefix}_noisy.wav", noisy_waveform, sr=16000)

            # save_spectrogram(f"{prefix}_clean.png", clean_spec.numpy())
            # save_spectrogram(f"{prefix}_noisy.png", noisy_spec.numpy())


if __name__ == "__main__":
    n_mels = 128
    dataset = SpeechCommandsDataset(n_mels=n_mels, root_dir='./data', subset='training', noise_factor=0.0)
    noise_levels = [1, 0.01, 0.05, 0.1]
    n_samples = 100  # Number of random samples to process
    create_noisy_audio_and_save_spectrogram(dataset, output_dir='output',
                                            noise_levels=noise_levels, n_samples=n_samples)
