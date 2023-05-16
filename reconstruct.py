# reconstruct.py
import torch
import matplotlib.pyplot as plt
import torchaudio
from data_loader import get_dataloader
from model import TransformerAutoencoder
from utils import load_checkpoint, get_arg_parser


def main():
    args = get_arg_parser().parse_args()

    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    device = torch.device(device)

    dataloader = get_dataloader(args.data_path, args.n_mels, 10)

    model = TransformerAutoencoder(d_model=args.n_mels, num_layers=args.num_layers,
                                   nhead=args.nhead, max_len=200, embedding_dim=args.embedding_dim,
                                   dropout=args.dropout).to(device)

    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        _, _, latent_representation, sos_tensor, eos_tensor = load_checkpoint(args.checkpoint_path, model)

    model.eval()
    with torch.no_grad():
        for _, (mel_specgrams, _) in enumerate(dataloader):
            mel_specgrams = mel_specgrams[0].unsqueeze(0).to(device)  # shape: (batch_size, T, n_mels)

            latent_representation = latent_representation.transpose(0, 1)  # shape: (batch_size, T, d_model)
            latent_representation = latent_representation[0].unsqueeze(0)  # shape: (batch_size, T, d_model)

            sos_tensor = sos_tensor.transpose(0, 1)  # shape: (batch_size, T, n_mels)
            sos_tensor = sos_tensor[0].unsqueeze(0)  # shape: (batch_size, T, n_mels)

            eos_tensor = eos_tensor.transpose(0, 1)  # shape: (batch_size, T, n_mels)
            eos_tensor = eos_tensor[0].unsqueeze(0)  # shape: (batch_size, T, n_mels)

            print(f"mel_specgrams shape: {mel_specgrams.shape}")
            print(f"latent_representation shape: {latent_representation.shape}")
            print(f"sos_tensor shape: {sos_tensor.shape}")
            print(f"eos_tensor shape: {eos_tensor.shape}")

            # Remove the first timestep from the predicted spectrograms
            output = model.inference(latent_representation=latent_representation, sos_tensor=sos_tensor,
                                     eos_tensor=eos_tensor, max_len=100)  # shape: (batch_size, T, n_mels)

            print("mel_specgrams", mel_specgrams.size(), mel_specgrams[0])
            print("output", output.size(), output[0])
            # quit()
            break

    original = mel_specgrams[0].cpu().numpy()
    reconstructed = output[0].cpu().numpy()

    # Find the index of the EOS token in the original tensor
    EOS_token = torch.tensor(-1.0)  # Assuming this is your EOS token
    eos_indices = torch.isclose(mel_specgrams[0], EOS_token, atol=1e-6).all(dim=1).nonzero(as_tuple=True)[0]
    if eos_indices.size(0) > 0 and False:
        original = mel_specgrams[0, :eos_indices[0], :].cpu().numpy()
        reconstructed = output[0, :eos_indices[0], :].cpu().numpy()
    else:
        original = mel_specgrams[0].cpu().numpy()
        reconstructed = output[0].cpu().numpy()

    # Plot original and reconstructed melspectrograms
    print("Saving plot")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original.T, aspect='auto', origin='lower')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.T, aspect='auto', origin='lower')
    plt.title('Reconstructed')
    plt.tight_layout()
    plt.savefig('./data/reconstructed/reconstruction.png')

    quit()

    # Inverse transformations to restore audio
    inv_mel_scale = torchaudio.transforms.InverseMelScale(n_stft=201, n_mels=args.n_mels, sample_rate=16000).to(device)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=400, hop_length=160, win_length=400, power=2, n_iter=64
    ).to(device)

    original_tensor = torch.tensor(original)[None, ...].transpose(-1, -2).to(device)  # shape: (1, n_mels, T)
    reconstructed_tensor = torch.tensor(reconstructed)[None, ...].transpose(-1, -2).to(device)  # shape: (1, n_mels, T)

    original_audio = griffin_lim(inv_mel_scale(original_tensor))  # shape: (1, T')
    reconstructed_audio = griffin_lim(inv_mel_scale(reconstructed_tensor))  # shape: (1, T')

    # Save original and reconstructed audio
    torchaudio.save("./data/reconstructed/original_audio.wav", original_audio.cpu(), sample_rate=16000)
    torchaudio.save("./data/reconstructed/reconstructed_audio.wav", reconstructed_audio.cpu(), sample_rate=16000)


if __name__ == "__main__":
    main()
