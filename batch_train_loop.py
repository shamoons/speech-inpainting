# batch_train_loop.py
import torch
import tqdm
import datetime  # Import the datetime module


def train_epoch(model, train_loader, optimizer, criterion, device, l1_lambda=0.001):
    model.train()
    total_loss = 0
    # Wrap the train_loader with tqdm to create a progress bar
    progress_bar = tqdm(train_loader, desc='Training', unit='batch')
    for i, (mel_specgrams, labels, waveform_lengths) in enumerate(progress_bar):
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
        # Update the description of the progress bar with the average loss and current time
        avg_loss = total_loss / (i + 1)
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        progress_bar.set_description(f'Training (Average Loss: {avg_loss:.4f}, Time: {current_time})')

    # Return average loss for the entire epoch
    return total_loss / len(train_loader)
