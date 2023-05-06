# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_autoencoder import TransformerAutoencoder
from mel_spectrogram_dataset import MelSpectrogramDataset
from arg_parser import parse_args
from reconstruction import reconstruct_and_save
from batch_train_loop import train_epoch


def main():
    args = parse_args()

    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    d_model = 80
    mel_train_dataset = MelSpectrogramDataset(n_mels=d_model)
    train_loader = DataLoader(mel_train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=mel_train_dataset.collate_fn)

    # Define the bottleneck dimension
    bottleneck_dim = 40

    # Instantiate the TransformerAutoencoder with the bottleneck_dim parameter
    model = TransformerAutoencoder(d_model=d_model, nhead=4, num_layers=2,
                                   dim_feedforward=512, max_length=1600, bottleneck_dim=bottleneck_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    num_epochs = 10 + start_epoch
    for epoch in range(start_epoch, num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.log_interval)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_epoch_{epoch + 1}.pth')

        # Attempt reconstruction after each epoch
        # with torch.no_grad():
        #     for mel_specgrams, labels in train_loader:
        #         mel_specgrams = mel_specgrams.transpose(0, 1).to(device)
        #         # Receive both the final output and the bottleneck output from the model
        #         outputs, bottleneck_output = model(mel_specgrams)
        #         break
        # # Along with the reconstructed file, save the original file
        # reconstruct_and_save(outputs, mel_specgrams, args.output_dir, epoch + 1, n_mels=d_model)


if __name__ == '__main__':
    main()
