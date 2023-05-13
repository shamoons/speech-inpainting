# main.py
import torch
import torch.optim as optim
from data_loader import get_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TransformerAutoencoder
from tqdm import tqdm
import wandb
import os
from utils import save_checkpoint, load_checkpoint, get_arg_parser

torch.manual_seed(0)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validate', leave=False)

    with torch.no_grad():
        for batch_idx, mel_specgrams in progress_bar:
            # mel_specgrams shape: (batch_size, T, n_mels)
            mel_specgrams = mel_specgrams.to(device)
            output, _, _, _ = model(mel_specgrams, mel_specgrams)  # Output shape: (batch_size, T, n_mels)
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
        # mel_specgrams shape: (batch_size, T, n_mels)
        mel_specgrams = mel_specgrams.to(device)

        optimizer.zero_grad()
        output, latent_representation, sos_tensor, eos_tensor = model(
            mel_specgrams, mel_specgrams)  # Output shape: (batch_size, T, n_mels)
        # print(f"mel_specgrams.mean: {mel_specgrams.mean()}")
        # print(f"mel_specgrams.max: {mel_specgrams.max()}")
        # print(f"mel_specgrams.min: {mel_specgrams.min()}")
        # print(f"output.mean: {output.mean()}")
        # print(f"output.max: {output.max()}")
        # print(f"output.min: {output.min()}")

        loss = criterion(output, mel_specgrams)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

    return epoch_loss / len(dataloader), latent_representation, sos_tensor, eos_tensor


def loss_fn(y_pred, y_true):
    print(f"y_pred.mean: {y_pred.mean()}")
    print(f"y_pred.max: {y_pred.max()}")
    print(f"y_pred.min: {y_pred.min()}")
    print(f"y_pred.std: {y_pred.std()}")
    print("====================================")
    print(f"y_true.mean: {y_true.mean()}")
    print(f"y_true.max: {y_true.max()}")
    print(f"y_true.min: {y_true.min()}")
    print(f"y_true.std: {y_true.std()}")
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
    device = torch.device(device)

    # Initialize wandb
    wandb_run = wandb.init(project="speech-inpainting", config=args.__dict__)
    print("wandb dir:", wandb.run.dir)

    train_dataloader = get_dataloader(args.data_path, args.n_mels, args.batch_size, lite=args.lite)
    val_dataloader = get_dataloader(args.data_path, args.n_mels, args.batch_size,
                                    subset='validation', lite=args.lite)

    # model = TransformerAutoencoder(d_model=args.n_mels, nhead=args.nhead, num_layers=args.num_layers,
    #                                dim_feedforward=args.dim_feedforward, dropout=args.dropout).to(device)
    model = TransformerAutoencoder(d_model=args.n_mels, num_layers=args.num_layers,
                                   nhead=args.nhead, max_len=1000).to(device)
    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, verbose=True)

    start_epoch = 0
    if args.checkpoint_path:
        start_epoch, _, _, _, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        train_loss, latent_representation, sos_tensor, eos_tensor = train_epoch(
            model, train_dataloader, criterion, optimizer, device)
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
            'val_loss': val_loss,
            'latent_representation': latent_representation,
            'sos_tensor': sos_tensor,
            'eos_tensor': eos_tensor
            # }, os.path.join(wandb.run.dir, f"./checkpoint_{epoch + 1}.pt"))
        }, os.path.join("./", f"checkpoint_{epoch + 1}.pt"))

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
