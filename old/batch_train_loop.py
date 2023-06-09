# batch_train_loop.py
import torch
import tqdm
import datetime


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    print('\n')
    # Wrap the train_loader with tqdm to create a progress bar
    progress_bar = tqdm.tqdm(train_loader, desc='Training', unit='batch')

    for i, (mel_specgrams, labels, waveform_lengths) in enumerate(progress_bar):
        mel_specgrams = mel_specgrams.transpose(0, 1).to(device)  # Shape: (seq_len, batch_size, d_model)
        optimizer.zero_grad()
        # Receive both the final output and the bottleneck output from the model
        outputs, bottleneck_output = model(mel_specgrams)  # Shape: (seq_len, batch_size, d_model)
        outputs = outputs.transpose(0, 1)  # Shape: (batch_size, seq_len, d_model)

        # Calculate min, max, and average for input and output
        input_min, input_max, input_avg = mel_specgrams.min().item(), mel_specgrams.max().item(), mel_specgrams.mean().item()
        output_min, output_max, output_avg = outputs.min().item(), outputs.max().item(), outputs.mean().item()

        # Print min, max, and average for input and output
        print(f'Input - Min: {input_min:.4f}, Max: {input_max:.4f}, Avg: {input_avg:.4f}')
        print(f'Output - Min: {output_min:.4f}, Max: {output_max:.4f}, Avg: {output_avg:.4f}')

        loss = mse_loss(outputs, mel_specgrams.transpose(0, 1))  # Shape: (batch_size, seq_len, d_model)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Update the description of the progress bar with the average loss and current time
        avg_loss = total_loss / (i + 1)
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        progress_bar.set_description(f'Training (Average Loss: {avg_loss:.4f}, Time: {current_time})')

    # Return average loss for the entire epoch
    return total_loss / len(train_loader)


mse_loss = torch.nn.MSELoss()
