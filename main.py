# main.py
import torch
import torch.optim as optim
from data_loader import get_dataloader
from model import TransformerAutoencoder
from utils import melspectrogram_transform, save_checkpoint, load_checkpoint, get_arg_parser


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for batch_idx, waveform in enumerate(dataloader):
        waveform = waveform.to(device)
        optimizer.zero_grad()
        output = model(waveform)
        loss = criterion(output, waveform)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
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

    model = TransformerAutoencoder(d_model=args.n_mels, nhead=2, num_layers=2,
                                   dim_feedforward=512, bottleneck_size=128).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.load_checkpoint:
        start_epoch, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, args.epochs):
        loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.n_epochs}, Loss: {loss}")

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f"{args.checkpoint_path}/checkpoint_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
