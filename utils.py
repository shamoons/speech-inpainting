# utils.py
import torch
import torchaudio
import argparse


def melspectrogram_transform(n_mels):
    return torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)


def save_checkpoint(state, filepath):
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['train_loss']


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Transformer Autoencoder for SpeechCommands Dataset')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands in melspectrogram')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint file')
    parser.add_argument('--use_cuda', action=argparse.BooleanOptionalAction, help='Use CUDA if available')
    parser.add_argument('--use_mps', action=argparse.BooleanOptionalAction, help='Use MPS if available')
    parser.add_argument('--use_layer_norm', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Use layer normalization')
    parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--base_lr', type=float, default=0.1, help='Base learning rate')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of the the input and output embeddings')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of the feedforward network')
    parser.add_argument('--warmup_steps', type=float, default=0.05,
                        help='Percent of total steps to warmup for learning rate scheduler')
    parser.add_argument('--lite', type=int, default=None, help='Lite mode for debugging')
    return parser
