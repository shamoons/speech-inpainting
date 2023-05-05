import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from transformer_autoencoder import TransformerAutoencoder
from mel_spectrogram_dataset import MelSpectrogramDataset, collate_fn
from arg_parser import parse_args
from reconstruction import reconstruct_and_save


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        print("MPS is available and will be used.")
        device = torch.device("mps")

    args = parse_args()

    train_dataset = SPEECHCOMMANDS('./data', download=True)
    n_mels = 128
    mel_train_dataset = MelSpectrogramDataset(train_dataset, n_mels=n_mels)
    mel_train_dataset = MelSpectrogramDataset(train_dataset)
    train_loader = DataLoader(mel_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = TransformerAutoencoder(d_model=128, nhead=4, num_layers=2, dim_feedforward=512, max_length=1600).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for i, (mel_specgrams, labels) in enumerate(train_loader):
            mel_specgrams = mel_specgrams.transpose(0, 1).to(device)
            optimizer.zero_grad()
            outputs = model(mel_specgrams)
            outputs = outputs.transpose(0, 1)
            loss = criterion(outputs, mel_specgrams.transpose(0, 1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Print statistics every log_interval batches
            if (i + 1) % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Average Loss: {avg_loss:.4f}')
                total_loss = 0  # Reset total_loss for the next log_interval batches

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_epoch_{epoch + 1}.pth')
        # Print summary statistics at the end of each epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Attempt reconstruction after each epoch
        with torch.no_grad():
            for mel_specgrams, labels in train_loader:
                mel_specgrams = mel_specgrams.transpose(0, 1).to(device)
                outputs = model(mel_specgrams)
                break
        # Along with the reconstructed file, save the original file
        reconstruct_and_save(outputs, mel_specgrams, args.output_dir, epoch + 1, n_mels=n_mels)

    # Note: The reconstructed Mel spectrograms ('outputs') can be converted back
    # to raw audio using an inverse Mel-spectrogram transformation if needed.


if __name__ == '__main__':
    main()
