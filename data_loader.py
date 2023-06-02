# data_loader.py
import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, n_mels, root_dir='./data', subset='training',
                 noise_factor=0.0, noise_to_spec=False, expand_factor=0.0):
        # Initialize SPEECHCOMMANDS dataset
        super().__init__(root_dir, download=True, subset=subset)

        # Define MelSpectrogram transformation
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)

        # Noise factor for additive noise
        self.noise_factor = noise_factor

        # Flag to determine where to add noise
        self.noise_to_spec = noise_to_spec

        # Expansion factor for time-stretching/shrinking
        self.expand_factor = expand_factor

    def __getitem__(self, idx):
        # Get waveform and meta data
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(idx)

        # Apply time stretching or shrinking
        if self.expand_factor > 0:
            # random number between 1 +/- expand_factor
            stretch_factor = 1.0 + (2 * torch.rand(1) - 1) * self.expand_factor
            resampler = torchaudio.transforms.Resample(sample_rate, int(sample_rate * stretch_factor))
            waveform = resampler(waveform)

        if self.noise_to_spec:
            # If noise_to_spec flag is True, add noise to spectrogram
            mel_specgram = self.transform(waveform)
            noisy_mel_specgram = mel_specgram + self.noise_factor * torch.randn(mel_specgram.shape)
            mel_specgram = torch.squeeze(mel_specgram, dim=0)  # [n_mels, T]
            noisy_mel_specgram = torch.squeeze(noisy_mel_specgram, dim=0)  # [n_mels, T]

        else:
            # If noise_to_spec flag is False, add noise to waveform
            noisy_waveform = waveform + self.noise_factor * torch.randn(waveform.shape)
            mel_specgram = self.transform(waveform)
            noisy_mel_specgram = self.transform(noisy_waveform)
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
    clean_waveforms_stack = torch.stack(clean_waveforms, dim=0).transpose(1, 2)
    noisy_waveforms_stack = torch.stack(noisy_waveforms, dim=0).transpose(1, 2)
    return clean_waveforms_stack, noisy_waveforms_stack, torch.tensor(seq_lengths)


def get_dataloader(root_dir, n_mels, batch_size, subset='training', lite=None,
                   noise_factor=0.005, noise_to_spec=False, expand_factor=0.0):
    """
    Returns a DataLoader for the SpeechCommandsDataset.
    """
    # Always initialize SpeechCommandsDataset for clean datasets
    datasets = [SpeechCommandsDataset(n_mels, root_dir, subset=subset, noise_factor=0.0,
                                      noise_to_spec=noise_to_spec, expand_factor=0.0)]

    # If noise_factor > 0, initialize SpeechCommandsDataset for noisy datasets
    if noise_factor > 0:
        datasets.append(SpeechCommandsDataset(n_mels, root_dir, subset=subset,
                                              noise_factor=noise_factor, noise_to_spec=noise_to_spec, expand_factor=0.0))
    # If expand_factor > 0, initialize SpeechCommandsDataset for expanded datasets
    if expand_factor > 0:
        datasets.append(SpeechCommandsDataset(n_mels, root_dir, subset=subset,
                                              noise_factor=0.0,
                                              noise_to_spec=noise_to_spec, expand_factor=expand_factor))

        # If both noise_factor and expand_factor are > 0, initialize SpeechCommandsDataset for expanded noisy datasets
        if noise_factor > 0:
            datasets.append(SpeechCommandsDataset(n_mels, root_dir, subset=subset,
                                                  noise_factor=noise_factor,
                                                  noise_to_spec=noise_to_spec, expand_factor=expand_factor))

    # Concatenate all datasets together
    dataset = ConcatDataset(datasets)

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
