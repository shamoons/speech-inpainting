# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, root_dir='./data', transform=None):
        super().__init__(root_dir, download=True)
        self.transform = transform

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(idx)

        if self.transform:
            waveform = self.transform(waveform)
            waveform = torch.squeeze(waveform, dim=0)

        return waveform


def pad_collate(batch):
    # Padding melspectrograms to have the same length
    max_len = max([item.shape[-1] for item in batch])
    waveforms = [torch.cat([item, item.new_zeros(item.shape[:-1] + (max_len - item.shape[-1],))], dim=-1)
                 for item in batch]
    return torch.stack(waveforms, dim=0)


def get_dataloader(root_dir, batch_size, transform=None):
    dataset = SpeechCommandsDataset(root_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate)
