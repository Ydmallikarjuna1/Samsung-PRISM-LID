import soundfile as sf
import torch
import torchaudio

TARGET_SAMPLE_RATE = 16000


def load_audio(audio_path):
    """
    Load a WAV audio file.

    Returns:
        waveform (Tensor): Shape [1, num_samples]
        sample_rate (int)
    """

    # Read audio using SoundFile
    waveform, sample_rate = sf.read(audio_path)

    # Convert NumPy -> Torch
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Shape: [1, num_samples]
    waveform = waveform.unsqueeze(0)

    # Resample if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            TARGET_SAMPLE_RATE
        )
        waveform = resampler(waveform)
        sample_rate = TARGET_SAMPLE_RATE

    # Peak normalization (avoid division by zero)
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    return waveform, sample_rate