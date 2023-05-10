# main.py
import torch
import torch.optim as optim
from data_loader import get_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TransformerAutoencoder
from tqdm import tqdm
from utils import melspectrogram_transform, save_checkpoint, load_checkpoint, get_arg_parser


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
    train_dataloader = get_dataloader(args.data_path, args.batch_size, transform)
    val_dataloader = get_dataloader(args.data_path, args.batch_size, transform, subset='validation')

    model = TransformerAutoencoder(d_model=args.n_mels, nhead=2, num_layers=2,
                                   dim_feedforward=512, bottleneck_size=128).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)  # Change this line
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # Add this line

    start_epoch = 0
    if args.checkpoint_path:
        start_epoch, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_dataloader, criterion, device)  # Add this line
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        scheduler.step(val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, f"./checkpoint_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
