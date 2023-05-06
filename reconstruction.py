# reconstruction.py
import torchaudio
import os


def reconstruct_and_save(outputs, output_dir, epoch, n_mels):
    # Convert the outputs tensor to the waveform
    reconstructed_waveform = outputs.squeeze().transpose(0, 1).cpu()

    # Scale the waveform back to the original range
    reconstructed_waveform = (reconstructed_waveform * 0.5) + 0.5

    # Convert the waveform to mono if needed
    if reconstructed_waveform.ndim == 3:
        reconstructed_waveform = reconstructed_waveform.mean(dim=0)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the reconstructed waveform as an audio file
    torchaudio.save(f'{output_dir}/reconstructed_epoch_{epoch}.wav', reconstructed_waveform, 16000)
