# main.py
import torch
import torch.optim as optim
from data_loader import get_dataloader
from torch.optim.lr_scheduler import LambdaLR
from model import TransformerAutoencoder
from tqdm import tqdm
import wandb
import os
from utils import save_checkpoint, load_checkpoint, get_arg_parser


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validate', leave=False)

    with torch.no_grad():
        for batch_idx, (mel_specgrams, seq_lengths) in progress_bar:
            # mel_specgrams shape: (batch_size, T, n_mels)
            mel_specgrams = mel_specgrams.to(device)
            output, _, _, _ = model(mel_specgrams, mel_specgrams, seq_lengths,
                                    seq_lengths)  # Output shape: (batch_size, T, n_mels)
            loss = criterion(output, mel_specgrams)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

    return epoch_loss / len(dataloader)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train', leave=False)

    for batch_idx, (mel_specgrams, seq_lengths) in progress_bar:
        # mel_specgrams shape: (batch_size, T, n_mels)
        mel_specgrams = mel_specgrams.to(device)

        optimizer.zero_grad()
        output, latent_representation, sos_tensor, eos_tensor = model(
            mel_specgrams, mel_specgrams, seq_lengths, seq_lengths)  # Output shape: (batch_size, T, n_mels)
        loss = criterion(output, mel_specgrams)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

    return epoch_loss / len(dataloader), latent_representation, sos_tensor, eos_tensor


def loss_fn(y_pred, y_true):
    # print(
    #     f"y_pred: mean: {y_pred.mean()}\tstd: {y_pred.std()}\tmax: {y_pred.max()}\tmin: {y_pred.min()}")
    # print(
    #     f"y_true: mean: {y_true.mean()}\tstd: {y_true.std()}\tmax: {y_true.max()}\tmin: {y_true.min()}")

    return torch.nn.MSELoss()(y_pred, y_true)
    # return torch.mean((torch.log1p(y_pred) - torch.log1p(y_true)) ** 2)


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

    model = TransformerAutoencoder(d_model=args.n_mels, num_layers=args.num_layers,
                                   nhead=args.nhead, max_len=200, embedding_dim=args.embedding_dim,
                                   dropout=args.dropout).to(device)
    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    def lr_lambda(epoch): return args.n_mels**-0.5 * min((epoch + 1)
                                                         ** -0.5, (epoch + 1) * (args.epochs // 20)**-1.5)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 0
    if args.checkpoint_path:
        start_epoch, _, _, _, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        train_loss, latent_representation, sos_tensor, eos_tensor = train_epoch(
            model, train_dataloader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_dataloader, criterion, device)

        # Logging losses to console and to wandb
        print(
            f"Epoch {epoch + 1}/{total_epochs}\tTrain Loss: {train_loss}\tVal Loss: {val_loss}\tLearning Rate: {scheduler.get_last_lr()[0]}")
        wandb_run.log({"train_loss": train_loss, "val_loss": val_loss, "learning_rate": scheduler.get_last_lr()[0]})

        # Adjust learning rate based on validation loss
        scheduler.step()

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
