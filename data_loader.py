# data_loader.py
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, n_mels, root_dir='./data', subset='training'):
        # Initialize SPEECHCOMMANDS dataset
        super().__init__(root_dir, download=True, subset=subset)

        # Define MelSpectrogram transformation
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)

    def __getitem__(self, idx):
        # Get waveform and meta data
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(idx)

        # Apply MelSpectrogram transformation
        waveform = self.transform(waveform)

        # Squeeze unnecessary dimension
        waveform = torch.squeeze(waveform, dim=0)  # [n_mels, T]
        seq_length = waveform.shape[-1]

        return waveform, seq_length


def pad_collate(batch):
    """
    Collate function for padding melspectrograms to have the same length in a batch.
    """

    # Extract waveforms and their sequence lengths
    waveforms, seq_lengths = zip(*batch)

    # Find the maximum length in the batch
    max_len = max(seq_lengths)

    # Pad all items in the batch to max_len with zeros
    waveforms = [torch.cat([item, item.new_zeros((item.shape[0], max_len - item.shape[1]))], dim=-1)
                 for item in waveforms]  # list of tensors of shape [n_mels, T']

    # Stack all padded waveforms in the batch and transpose last two dimensions
    # [batch_size, T', n_mels], list of sequence lengths
    return torch.stack(waveforms, dim=0).transpose(1, 2), torch.tensor(seq_lengths)


def get_dataloader(root_dir, n_mels, batch_size, subset='training', lite=None):
    """
    Returns a DataLoader for the SpeechCommandsDataset.
    """

    # Initialize SpeechCommandsDataset
    dataset = SpeechCommandsDataset(n_mels, root_dir,  subset=subset)

    # If lite flag is set, only use a subset of the data
    if lite is not None:
        if subset == 'training':
            dataset = torch.utils.data.Subset(dataset, range(lite))
        elif subset == 'validation':
            dataset = torch.utils.data.Subset(dataset, range(lite // 4))

    # Return DataLoader with appropriate parameters
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate)
