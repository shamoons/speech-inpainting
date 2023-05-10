# utils.py
import torch
import torchaudio
import argparse


def melspectrogram_transform(n_mels):
    return torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels)


def save_checkpoint(state, filepath):
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Transformer Autoencoder for SpeechCommands Dataset')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands in melspectrogram')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Path to save/load checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load checkpoint and resume training')
    return parser
