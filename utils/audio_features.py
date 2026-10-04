import torch
import torchaudio

from configs.config import (
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
)


class LogMelFeatureExtractor:
    def __init__(self):
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )

    def __call__(self, waveform):
        """
        Convert waveform into log-Mel spectrogram.

        Input:
            waveform: [1, num_samples]

        Output:
            features: [time, n_mels]
        """

        if waveform.ndim != 2:
            raise ValueError(
                f"Expected waveform shape [channels, samples], "
                f"got {waveform.shape}"
            )

        if waveform.shape[0] != 1:
            raise ValueError(
                f"Expected mono waveform, got {waveform.shape[0]} channels"
            )

        # [1, n_mels, time]
        mel = self.mel_spectrogram(waveform)

        # Numerical stability
        mel = torch.clamp(mel, min=1e-10)

        # Log-Mel
        log_mel = torch.log(mel)

        # [time, n_mels]
        log_mel = log_mel.squeeze(0).transpose(0, 1)

        return log_mel


if __name__ == "__main__":
    print("Testing Log-Mel feature extractor...")

    extractor = LogMelFeatureExtractor()

    # Synthetic waveform for basic sanity check
    waveform = torch.randn(1, SAMPLE_RATE)

    features = extractor(waveform)

    print(f"Input waveform shape : {waveform.shape}")
    print(f"Feature shape         : {features.shape}")
    print(f"Expected feature dim  : {N_MELS}")

    assert features.ndim == 2
    assert features.shape[1] == N_MELS
    assert torch.isfinite(features).all()

    print("\nLOG-MEL FEATURE TEST PASSED")