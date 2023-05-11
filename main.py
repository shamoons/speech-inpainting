# main.py
import torch
import torch.optim as optim
from data_loader import get_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TransformerAutoencoder
from tqdm import tqdm
import wandb
from utils import melspectrogram_transform, save_checkpoint, load_checkpoint, get_arg_parser


def log_weights(model):
    for name, param in model.named_parameters():
        wandb.log({f"{name}_mean": param.data.mean()})
        wandb.log({f"{name}_stddev": param.data.std()})


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                            desc='Validate', leave=False)  # create a progress bar
        for batch_idx, mel_specgrams in progress_bar:
            mel_specgrams = mel_specgrams.transpose(1, 2).to(device)
            output, bottleneck_output = model(mel_specgrams)
            loss = criterion(output, mel_specgrams)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})  # update the progress bar

    return epoch_loss / len(dataloader)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                        desc='Train', leave=False)  # create a progress bar
    for batch_idx, mel_specgrams in progress_bar:
        mel_specgrams = mel_specgrams.transpose(1, 2).to(device)
        optimizer.zero_grad()
        output, bottleneck_output = model(mel_specgrams)
        loss = criterion(output, mel_specgrams)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})  # update the progress bar

    return epoch_loss / len(dataloader)


def msle_loss(y_pred, y_true):
    return torch.nn.MSELoss()(torch.log1p(torch.relu(y_pred)), torch.log1p(torch.relu(y_true)))
    # return torch.nn.MSELoss()(y_pred, y_true)


def main():
    args = get_arg_parser().parse_args()

    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="speech-inpainting",

        # track hyperparameters and run metadata
        config={
            "initial_learning_rate": args.initial_lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "n_mels": args.n_mels,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
        }
    )

    transform = melspectrogram_transform(args.n_mels)
    train_dataloader = get_dataloader(args.data_path, args.batch_size, transform, lite=args.lite)
    val_dataloader = get_dataloader(args.data_path, args.batch_size, transform, subset='validation', lite=args.lite)

    model = TransformerAutoencoder(d_model=args.n_mels, nhead=args.nhead, num_layers=args.num_layers,
                                   dim_feedforward=512, bottleneck_size=128).to(device)
    criterion = msle_loss
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)  # Change this line
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, verbose=True)  # Add this line

    start_epoch = 0
    if args.checkpoint_path:
        start_epoch, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch, total_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch + 1}/{total_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        scheduler.step(val_loss)
        wandb_run.log({"train_loss": train_loss, "val_loss": val_loss})
        # log_weights(model)

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, f"./checkpoint_{epoch + 1}.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
