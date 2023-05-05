# mel_spectrogram_dataset.py

import torch
import torchaudio
from torch.utils.data import Dataset


class MelSpectrogramDataset(Dataset):
    def __init__(self, speechcommands_dataset, n_mels=128):
        self.speechcommands_dataset = speechcommands_dataset
        self.n_mels = n_mels

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.speechcommands_dataset[index]
        mel_specgram = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)(waveform)
        return mel_specgram.squeeze(0).transpose(0, 1), label

    def __len__(self):
        return len(self.speechcommands_dataset)


def collate_fn(batch):
    mel_specgrams, labels = zip(*batch)
    mel_specgrams = torch.nn.utils.rnn.pad_sequence(
        mel_specgrams, batch_first=True, padding_value=0.0)
    return mel_specgrams, labels
