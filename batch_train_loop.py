# batch_train_loop.py

def train_epoch(model, train_loader, optimizer, criterion, device, log_interval):
    model.train()
    total_loss = 0
    for i, (mel_specgrams, labels) in enumerate(train_loader):
        mel_specgrams = mel_specgrams.transpose(0, 1).to(device)
        optimizer.zero_grad()
        # Receive both the final output and the bottleneck output from the model
        outputs, bottleneck_output = model(mel_specgrams)
        outputs = outputs.transpose(0, 1)
        loss = criterion(outputs, mel_specgrams.transpose(0, 1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Print statistics every log_interval batches
        if (i + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            print(f'Batch [{i + 1}/{len(train_loader)}], Average Loss: {avg_loss:.4f}')
            total_loss = 0  # Reset total_loss for the next log_interval batches

    # Return average loss for the entire epoch
    return total_loss / len(train_loader)
