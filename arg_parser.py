# arg_parser.py
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train Transformer Autoencoder')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to resume training')
    parser.add_argument('--output_dir', type=str, default='./data/output',
                        help='Output directory for reconstructed WAV files')
    return parser.parse_args()
