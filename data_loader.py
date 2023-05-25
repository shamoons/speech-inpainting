# data_loader.py
import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, n_mels, root_dir='./data', subset='training', noise_factor=0.0):
        # Initialize SPEECHCOMMANDS dataset
        super().__init__(root_dir, download=True, subset=subset)

        # Define MelSpectrogram transformation
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)

        # Noise factor for additive noise
        self.noise_factor = noise_factor

    def __getitem__(self, idx):
        # Get waveform and meta data
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(idx)

        # Apply MelSpectrogram transformation
        mel_specgram = self.transform(waveform)

        # Add random white noise
        noisy_waveform = waveform + self.noise_factor * torch.randn(waveform.shape)

        # Transform noisy waveform to mel spectrogram
        noisy_mel_specgram = self.transform(noisy_waveform)

        # Squeeze unnecessary dimension
        mel_specgram = torch.squeeze(mel_specgram, dim=0)  # [n_mels, T]
        noisy_mel_specgram = torch.squeeze(noisy_mel_specgram, dim=0)  # [n_mels, T]
        seq_length = mel_specgram.shape[-1]

        return mel_specgram, noisy_mel_specgram, seq_length


def pad_collate(batch):
    """
    Collate function for padding melspectrograms to have the same length in a batch.
    """
    # Extract waveforms and their sequence lengths
    clean_waveforms, noisy_waveforms, seq_lengths = zip(*batch)

    # Find the maximum length in the batch
    max_len = max(seq_lengths)

    # Pad all items in the batch to max_len with zeros
    clean_waveforms = [torch.cat([item, item.new_zeros((item.shape[0], max_len - item.shape[1]))], dim=-1)
                       for item in clean_waveforms]  # list of tensors of shape [n_mels, T']

    noisy_waveforms = [torch.cat([item, item.new_zeros((item.shape[0], max_len - item.shape[1]))], dim=-1)
                       for item in noisy_waveforms]  # list of tensors of shape [n_mels, T']

    # Stack all padded waveforms in the batch and transpose last two dimensions
    # [batch_size, T', n_mels], list of sequence lengths
    return torch.stack(clean_waveforms, dim=0).transpose(1, 2), torch.stack(noisy_waveforms, dim=0).transpose(1, 2), torch.tensor(seq_lengths)


def get_dataloader(root_dir, n_mels, batch_size, subset='training', lite=None, noise_factor=0.005):
    """
    Returns a DataLoader for the SpeechCommandsDataset.
    """
    # Initialize SpeechCommandsDataset for clean and possibly noisy datasets
    clean_dataset = SpeechCommandsDataset(n_mels, root_dir, subset=subset, noise_factor=0.0)

    if noise_factor > 0:
        noisy_dataset = SpeechCommandsDataset(n_mels, root_dir, subset=subset, noise_factor=noise_factor)
        dataset = ConcatDataset([clean_dataset, noisy_dataset])
    else:
        dataset = clean_dataset

    # If lite flag is set, only use a subset of the data
    if lite is not None:
        lite = min(lite, len(dataset))  # Make sure not to exceed total dataset size
        if subset == 'training':
            dataset = torch.utils.data.Subset(dataset, range(lite))
        elif subset == 'validation':
            val_size = lite // 4
            val_size = min(val_size, len(dataset))  # Make sure not to exceed total dataset size
            dataset = torch.utils.data.Subset(dataset, range(val_size))

    # Return DataLoader with appropriate parameters
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=pad_collate), len(dataset)
