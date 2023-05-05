# mel_spectrogram_dataset.py

import torch
from torchaudio.transforms import MelSpectrogram


class MelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_mels=80):
        self.dataset = dataset
        self.mel_spectrogram = MelSpectrogram(n_mels=n_mels)

    def __getitem__(self, index):
        waveform, sample_rate, label, _, _ = self.dataset[index]
        mel_specgram = self.mel_spectrogram(waveform)
        return mel_specgram.squeeze(0), label

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    def pad_tensor(tensor, target_length, target_width):
        padding = (0, target_width - tensor.size(1), 0, target_length - tensor.size(0))
        return torch.nn.functional.pad(tensor, padding)

    mel_specgrams, labels = zip(*batch)
    max_length = max([specgram.size(0) for specgram in mel_specgrams])
    target_width = 80  # Fixed number of mel bins (d_model)

    mel_specgrams = [pad_tensor(specgram, max_length, target_width) for specgram in mel_specgrams]
    mel_specgrams = torch.stack(mel_specgrams)

    return mel_specgrams, labels
