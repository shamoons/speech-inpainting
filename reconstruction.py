# reconstruction.py
import torchaudio


def reconstruct_and_save(outputs, mel_specgrams, output_dir, epoch, n_mels=80):
    mel_inv = torchaudio.transforms.InverseMelScale(n_mels=n_mels, n_stft=513, sample_rate=16000)
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024, n_iter=32)
    waveform = griffin_lim(mel_inv(outputs.transpose(0, 1).squeeze(0).cpu()))
    torchaudio.save(f'{output_dir}/reconstructed_epoch_{epoch}.wav', waveform, 16000)
