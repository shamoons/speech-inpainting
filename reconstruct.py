# reconstruct.py
import torch
import matplotlib.pyplot as plt
from data_loader import get_dataloader
from model import TransformerAutoencoder
from utils import melspectrogram_transform, load_checkpoint, get_arg_parser


def main():
    args = get_arg_parser().parse_args()

    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    transform = melspectrogram_transform(args.n_mels)
    dataloader = get_dataloader(args.data_path, args.batch_size, transform)

    model = TransformerAutoencoder(d_model=args.n_mels, nhead=2, num_layers=2,
                                   dim_feedforward=512, bottleneck_size=128).to(device)

    if args.checkpoint_path:
        _, _ = load_checkpoint(args.checkpoint_path, model)

    model.eval()

    with torch.no_grad():
        for batch_idx, mel_specgrams in enumerate(dataloader):
            mel_specgrams = mel_specgrams.transpose(1, 2).to(device)
            output, _ = model(mel_specgrams)
            break

    # Convert tensors to numpy arrays for plotting
    original = mel_specgrams[0].cpu().numpy()
    reconstructed = output[0].cpu().numpy()

    # Plot original and reconstructed melspectrograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original, aspect='auto', origin='lower')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, aspect='auto', origin='lower')
    plt.title('Reconstructed')
    plt.tight_layout()
    plt.savefig('reconstruction.png')


if __name__ == "__main__":
    main()
