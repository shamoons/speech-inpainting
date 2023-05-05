# arg_parser.py
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train Transformer Autoencoder')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to resume training')
    parser.add_argument('--output_dir', type=str, default='./data/output',
                        help='Output directory for reconstructed WAV files')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Interval (in batches) at which to log training statistics')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--use_cuda', type=bool, default=False, help='Use CUDA if available')
    parser.add_argument('--use_mps', type=bool, default=False, help='Use MPS if available')
    return parser.parse_args()
