from utils.dataset import VoxLinguaDataset
from utils.audio import load_audio

dataset = VoxLinguaDataset(
    "datasets/voxlingua107_wav/train.csv"
)

audio_path, label = dataset[0]

waveform, sample_rate = load_audio(audio_path)

print("=" * 50)
print("Label        :", label)
print("Sample Rate  :", sample_rate)
print("Shape        :", waveform.shape)
print("Duration (s) :", waveform.shape[1] / sample_rate)
print("Min          :", waveform.min().item())
print("Max          :", waveform.max().item())