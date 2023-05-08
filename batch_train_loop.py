# batch_train_loop.py
import torch


def train_epoch(model, train_loader, optimizer, criterion, device, log_interval, l1_lambda=0.001):
    model.train()
    total_loss = 0
    for i, (mel_specgrams, labels) in enumerate(train_loader):
        mel_specgrams = mel_specgrams.transpose(0, 1).to(device)
        optimizer.zero_grad()
        # Receive both the final output and the bottleneck output from the model
        outputs, bottleneck_output = model(mel_specgrams)
        outputs = outputs.transpose(0, 1)
        mse_loss = criterion(outputs, mel_specgrams.transpose(0, 1))
        # Calculate L1 regularization term
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.norm(param, 1)
        # Combine MSE loss with L1 regularization term
        loss = mse_loss + l1_lambda * l1_loss
        total_loss += mse_loss.item()
        loss.backward()
        optimizer.step()
        # Print statistics every log_interval batches
        if (i + 1) % log_interval == 0:
            avg_loss = total_loss / (i + 1)
            print(f'\tBatch [{i + 1}/{len(train_loader)}], Average Loss: {avg_loss:.4f}')

    # Return average loss for the entire epoch
    return total_loss / len(train_loader)
