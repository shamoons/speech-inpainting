# main.py
import torch
import torch.optim as optim
from data_loader import get_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TransformerAutoencoder
from tqdm import tqdm
import wandb
from utils import melspectrogram_transform, save_checkpoint, load_checkpoint, get_arg_parser


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validate', leave=False)

    with torch.no_grad():
        for batch_idx, mel_specgrams in progress_bar:
            # Transpose to match model input shape: (batch_size, T, n_mels)
            mel_specgrams = mel_specgrams.transpose(1, 2).to(device)
            output, _ = model(mel_specgrams)  # Output shape: (batch_size, T, n_mels)
            loss = criterion(output, mel_specgrams)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

    return epoch_loss / len(dataloader)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train', leave=False)

    for batch_idx, mel_specgrams in progress_bar:
        # Transpose to match model input shape: (batch_size, T, n_mels)
        mel_specgrams = mel_specgrams.transpose(1, 2).to(device)
        optimizer.zero_grad()
        output, _ = model(mel_specgrams)  # Output shape: (batch_size, T, n_mels)
        loss = criterion(output, mel_specgrams)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

        # Calculate average for each time step for original and reconstructed
        original_avg_per_timestep = mel_specgrams[0].mean(axis=1)
        reconstructed_avg_per_timestep = output[0].mean(axis=1)

        # Print averages
        print("Average value for each time step in the original tensor:")
        print(mel_specgrams.size(), original_avg_per_timestep)
        print("Average value for each time step in the reconstructed tensor:")
        print(output.size(), reconstructed_avg_per_timestep)

    return epoch_loss / len(dataloader)


def msle_loss(y_pred, y_true):
    return torch.nn.MSELoss()(y_pred, y_true)


def main():
    args = get_arg_parser().parse_args()

    # Set device
    device = "cpu"
    if args.use_mps and torch.backends.mps.is_available():
        device = "mps"
    elif args.use_cuda and torch.cuda.is_available():
        device = "cuda"

    print(f"Using device: {device}")
    torch.device(device)

    # Initialize wandb
    wandb_run = wandb.init(project="speech-inpainting", config=args.__dict__)

    transform = melspectrogram_transform(args.n_mels)
    train_dataloader = get_dataloader(args.data_path, args.batch_size, transform, lite=args.lite, add_eos=True)
    val_dataloader = get_dataloader(args.data_path, args.batch_size, transform,
                                    subset='validation', lite=args.lite, add_eos=True)

    model = TransformerAutoencoder(d_model=args.n_mels, nhead=args.nhead, num_layers=args.num_layers,
                                   dim_feedforward=512, bottleneck_size=128).to(device)
    criterion = msle_loss
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, verbose=True)

    start_epoch = 0
    if args.checkpoint_path:
        start_epoch, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_dataloader, criterion, device)

        # Logging losses to console and to wandb
        print(f"Epoch {epoch + 1}/{total_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        wandb_run.log({"train_loss": train_loss, "val_loss": val_loss})

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Saving model and optimizer state
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, f"./checkpoint_{epoch + 1}.pt")

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
