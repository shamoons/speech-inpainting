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
    dataloader = get_dataloader(args.data_path, 100, transform)

    model = TransformerAutoencoder(d_model=args.n_mels, nhead=args.nhead, num_layers=args.num_layers,
                                   dim_feedforward=512, bottleneck_size=128).to(device)

    if args.checkpoint_path:
        _, _ = load_checkpoint(args.checkpoint_path, model)

    model.eval()

    with torch.no_grad():
        for batch_idx, mel_specgrams in enumerate(dataloader):  # shape: (batch_size, n_mels, T)
            mel_specgrams = mel_specgrams.transpose(1, 2).to(device)  # shape: (batch_size, T, n_mels)
            output, _ = model(mel_specgrams)  # shape: (batch_size, T, n_mels)
            break

    # Find the index of the EOS token in the original tensor
    EOS_token = -1.0  # Assuming this is your EOS token
    eos_idx = (mel_specgrams[0] == EOS_token).nonzero(as_tuple=True)[0][0]

    # Slice the tensors to remove the EOS token and following timesteps
    print(f"Original shape: {mel_specgrams.shape}")
    original = mel_specgrams[0, :eos_idx, :].cpu().numpy()  # shape: (Te, n_mels) where Te is T truncated by eos_idx
    # original = mel_specgrams[0].cpu().numpy()  # shape: (Te, n_mels) where Te is T truncated by eos_idx
    print(f"Stripped shape: {original.shape}")
    reconstructed = output[0, :eos_idx, :].cpu().numpy()  # shape: (Te, n_mels) where Te is T truncated by eos_idx
    # reconstructed = output[0].cpu().numpy()  # shape: (Te, n_mels) where Te is T truncated by eos_idx

    # Plot original and reconstructed melspectrograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original.T, aspect='auto', origin='lower')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.T, aspect='auto', origin='lower')
    plt.title('Reconstructed')
    plt.tight_layout()
    plt.savefig('./data/reconstructed/reconstruction.png')

    # Inverse transformations to restore audio
    inv_mel_scale = torchaudio.transforms.InverseMelScale(n_stft=201, n_mels=args.n_mels, sample_rate=16000).to(device)
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=400, hop_length=160, win_length=400, power=2).to(device)

    original_tensor = torch.tensor(original)[None, ...].transpose(-1, -2)  # shape: (1, n_mels, T)
    reconstructed_tensor = torch.tensor(reconstructed)[None, ...].transpose(-1, -2)  # shape: (1, n_mels, T)

    original_audio = griffin_lim(inv_mel_scale(original_tensor))  # shape: (1, T')
    reconstructed_audio = griffin_lim(inv_mel_scale(reconstructed_tensor))  # shape: (1, T')

    # Save original and reconstructed audio
    torchaudio.save("./data/reconstructed/original_audio.wav", original_audio.cpu(), sample_rate=16000)
    torchaudio.save("./data/reconstructed/reconstructed_audio.wav", reconstructed_audio.cpu(), sample_rate=16000)


if __name__ == "__main__":
    main()
