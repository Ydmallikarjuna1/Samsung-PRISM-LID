import torchaudio


def extract_fbank(
    waveform,
    sample_rate=16000,
    num_mel_bins=80
):
    """
    Extract 80-dimensional Kaldi FBank features.

    Args:
        waveform (Tensor): Shape [1, num_samples]

    Returns:
        Tensor: Shape [num_frames, 80]
    """

    features = torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=sample_rate,
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        use_energy=False,
        window_type="hamming"
    )

    return features