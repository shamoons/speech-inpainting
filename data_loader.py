# data_loader.py
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

        return waveform


def get_dataloader(root_dir, batch_size, transform=None):
    dataset = SpeechCommandsDataset(root_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
