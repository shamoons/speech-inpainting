# data_loader.py
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, n_mels, root_dir='./data', subset='training'):
        super().__init__(root_dir, download=True, subset=subset)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)
        self.EOS_token = -1.0  # Define the EOS token as a constant

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(idx)

        if self.transform:
            waveform = self.transform(waveform)
            waveform = torch.squeeze(waveform, dim=0)  # shape: (n_mels, T)

        return waveform


def pad_collate(batch):
    # Padding melspectrograms to have the same length
    max_len = max([item.shape[-1] for item in batch])
    waveforms = [torch.cat([item, item.new_zeros(item.shape[:-1] + (max_len - item.shape[-1],))], dim=-1)
                 for item in batch]
    return torch.stack(waveforms, dim=0).transpose(1, 2)  # shape: (batch_size, T, n_mels)


def get_dataloader(root_dir, n_mels, batch_size, subset='training', lite=None):
    dataset = SpeechCommandsDataset(n_mels, root_dir,  subset=subset)

    # If lite flag is set, only use a subset of the data
    if lite is not None:
        if subset == 'training':
            dataset = torch.utils.data.Subset(dataset, range(lite))
        elif subset == 'validation':
            dataset = torch.utils.data.Subset(dataset, range(lite // 4))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate)
