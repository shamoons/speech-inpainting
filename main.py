# main.py
import torch
import torch.optim as optim
from data_loader import get_dataloader
from model import TransformerAutoencoder
from utils import melspectrogram_transform, save_checkpoint, load_checkpoint, get_arg_parser


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for batch_idx, waveform in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(waveform)
        loss = criterion(output, waveform)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def main():
    args = get_arg_parser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = melspectrogram_transform(args.n_mels)
    train_dataloader = get_dataloader(args.data_path, args.batch_size, transform)

    model = TransformerAutoencoder(d_model=args.n_mels, nhead=8, num_layers=3,
                                   dim_feedforward=512, bottleneck_size=128).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.load_checkpoint:
        start_epoch, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, args.n_epochs):
        loss = train_epoch(model, train_dataloader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{args.n_epochs}, Loss: {loss}")

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f"{args.checkpoint_path}/checkpoint_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
