# reconstruction.py
import os
import torchaudio


def reconstruct_and_save(outputs, mel_specgrams, output_dir, epoch, n_mels=128):
    mel_inv = torchaudio.transforms.InverseMelScale(n_mels=n_mels, sample_rate=16000, n_stft=201)
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=400, power=1.0)
    waveform = griffin_lim(mel_inv(outputs.transpose(0, 1).squeeze(0).cpu()))
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(os.path.join(output_dir, f'reconstructed_epoch_{epoch}.wav'), waveform, sample_rate=16000)
    # Save the original Mel spectrogram as a waveform
    original_waveform = griffin_lim(mel_inv(mel_specgrams.transpose(0, 1).squeeze(0).cpu()))
    torchaudio.save(os.path.join(output_dir, f'original_epoch_{epoch}.wav'), original_waveform, sample_rate=16000)
