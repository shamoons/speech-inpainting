# reconstruct.py
import torch
import matplotlib.pyplot as plt
import torchaudio
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

    # Inverse transformations to restore audio
    inv_mel_scale = torchaudio.transforms.InverseMelScale(n_stft=201, n_mels=args.n_mels, sample_rate=16000).to(device)
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=400, hop_length=160, win_length=400, power=2).to(device)

    # Convert numpy arrays back to tensors, add an extra dimension, and transpose the last two dimensions
    original_tensor = torch.tensor(original)[None, ...].transpose(-1, -2)
    reconstructed_tensor = torch.tensor(reconstructed)[None, ...].transpose(-1, -2)

    original_audio = griffin_lim(inv_mel_scale(original_tensor))
    reconstructed_audio = griffin_lim(inv_mel_scale(reconstructed_tensor))

    # Save original and reconstructed audio
    torchaudio.save("original_audio.wav", original_audio.cpu(), sample_rate=16000)
    torchaudio.save("reconstructed_audio.wav", reconstructed_audio.cpu(), sample_rate=16000)


if __name__ == "__main__":
    main()
