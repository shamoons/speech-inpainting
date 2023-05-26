# main.py
import torch
import time
import torch.optim as optim
from data_loader import get_dataloader
from torch.optim.lr_scheduler import LambdaLR
from compression_model import TransformerCompressionAutoencoder
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
        for batch_idx, (mel_specgrams, noisy_mel_specgrams, seq_lengths) in progress_bar:
            # mel_specgrams shape: (batch_size, T, n_mels)
            mel_specgrams = mel_specgrams.to(device)
            noisy_mel_specgrams = noisy_mel_specgrams.to(device)
            seq_lengths = seq_lengths.to(device)
            output = model(noisy_mel_specgrams, seq_lengths)  # Output shape: (batch_size, T, n_mels)
            loss = criterion(output, mel_specgrams, seq_lengths)
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

    return epoch_loss / len(dataloader)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train', leave=False)

    for batch_idx, (mel_specgrams, noisy_mel_specgrams, seq_lengths) in progress_bar:
        # mel_specgrams shape: (batch_size, T, n_mels)
        mel_specgrams = mel_specgrams.to(device)
        noisy_mel_specgrams = noisy_mel_specgrams.to(device)
        seq_lengths = seq_lengths.to(device)

        optimizer.zero_grad()
        output = model(
            noisy_mel_specgrams, seq_lengths)  # Output shape: (batch_size, T, n_mels)
        loss = criterion(output, mel_specgrams, seq_lengths)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})

    return epoch_loss / len(dataloader)  # , latent_representation, sos_tensor, eos_tensor


def loss_fn(y_pred, y_true, seq_lengths):
    # y_pred shape: (batch_size, T + 2, n_mels)
    # y_true shape: (batch_size, T, n_mels)
    # seq_lengths shape: (batch_size)

    batch_size, _, _ = y_pred.shape

    # Initialize a list to hold the processed sequences
    y_pred_masked_list = []

    # For each item in the batch
    for i in range(batch_size):
        # Remove the first timestep (i.e., the SOS timestep)
        without_sos = y_pred[i, 1:]  # Shape: (T + 1, n_mels)

        # Remove the timestep at seq_lengths[i] (i.e., the EOS timestep)
        pre_eos = without_sos[:seq_lengths[i]]
        post_eos = without_sos[seq_lengths[i] + 1:]

        # Concatenate pre_eos and post_eos
        masked = torch.cat((pre_eos, post_eos), dim=0)

        y_pred_masked_list.append(masked)

    # Stack the processed sequences back into a tensor
    y_pred_masked = torch.stack(y_pred_masked_list).to(y_pred.device)

    return torch.nn.MSELoss()(y_pred_masked, y_true)


def run_main(args):
    batch_size = args.batch_size
    try:

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
        print(f"Using Layer Normalization: {args.use_layer_norm}")

        train_dataloader, train_dataset_size = get_dataloader(
            args.data_path, args.n_mels, batch_size, lite=args.lite, noise_factor=args.noise_factor)
        val_dataloader, val_dataset_size = get_dataloader(args.data_path, args.n_mels, batch_size,
                                                          subset='validation', lite=args.lite,
                                                          noise_factor=args.noise_factor)

        model = TransformerCompressionAutoencoder(d_model=args.n_mels, num_layers=args.num_layers,
                                                  nhead=args.nhead, max_len=200, embedding_dim=args.embedding_dim,
                                                  dim_feedforward=args.dim_feedforward,
                                                  dropout=args.dropout).to(device)
        criterion = loss_fn

        warmup_steps = int(args.epochs * args.warmup_steps)  # % of total steps
        print(f"Training dataset size: {train_dataset_size}, validation dataset size: {val_dataset_size}")

        optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

        def lr_lambda(step):

            lr = args.base_lr * min((step + 1)
                                    ** -0.5, (step + 1) * warmup_steps ** -1.5)
            return lr

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        start_epoch = 0
        if args.checkpoint_path:
            start_epoch, _ = load_checkpoint(args.checkpoint_path, model, optimizer)

        total_epochs = start_epoch + args.epochs
        for epoch in range(start_epoch, total_epochs):
            train_loss = train_epoch(
                model, train_dataloader, criterion, optimizer, dwandb agent shamoons/speech-inpainting/m2ecz7hfevice)
            val_loss = validate_epoch(model, val_dataloader, criterion, device)

            # Logging losses to console and to wandb
            print(
                f"Epoch {epoch + 1}/{total_epochs}\tTrain Loss: {train_loss}\tVal Loss: {val_loss}\tLearning Rate: {scheduler.get_last_lr()[0]}")
            wandb_run.log({"train_loss": train_loss, "val_loss": val_loss, "learning_rate": scheduler.get_last_lr()[0]})

            # Adjust learning rate based on validation loss
            scheduler.step()

            # Saving model and optimizer state every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    # 'latent_representation': latent_representation,
                    # 'sos_tensor': sos_tensor,
                    # 'eos_tensor': eos_tensor
                }, os.path.join(wandb.run.dir, f"./checkpoint.pt"))
                # }, os.path.join("./", f"checkpoint_{epoch + 1}.pt"))

        # Finish the wandb run
        wandb.finish()
    except RuntimeError as e:
        # If we hit a CUDA out of memory error, cut the batch size in half and retry
        if 'out of memory' in str(e):
            print(f'Batch size of {batch_size} was too large, trying with {batch_size // 2}...')
            args.batch_size = batch_size // 2
            time.sleep(10)
            return run_main(args)
        else:
            # If it's a different kind of RuntimeError, raise it.
            raise e


def main():
    args = get_arg_parser().parse_args()
    run_main(args)


if __name__ == "__main__":
    main()
