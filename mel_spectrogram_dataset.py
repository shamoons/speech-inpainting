# mel_spectrogram_dataset.py

import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.datasets import SPEECHCOMMANDS


class MelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, n_mels=80, subset='training'):
        self.dataset = SPEECHCOMMANDS('./data', download=True, subset=subset)
        self.mel_spectrogram = MelSpectrogram(n_mels=n_mels)
        self.n_mels = n_mels

    def __getitem__(self, index):
        waveform, sample_rate, label, _, _ = self.dataset[index]
        mel_specgram = self.mel_spectrogram(waveform)
        return mel_specgram.squeeze(0), label, waveform.shape[-1]  # return waveform length

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        def pad_tensor(tensor, target_length, target_width):
            padding = (0, target_width - tensor.size(1), 0, target_length - tensor.size(0))
            return torch.nn.functional.pad(tensor, padding)

        mel_specgrams, labels, waveform_lengths = zip(*batch)

        max_length = max([specgram.size(0) for specgram in mel_specgrams])
        max_width = self.n_mels  # Use n_mels as the target width

        mel_specgrams = [pad_tensor(specgram, max_length, max_width) for specgram in mel_specgrams]
        mel_specgrams = torch.stack(mel_specgrams)

        return mel_specgrams, labels, waveform_lengths
