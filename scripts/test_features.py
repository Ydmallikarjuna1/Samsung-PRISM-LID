from utils.dataset import VoxLinguaDataset
from utils.audio import load_audio
from utils.features import extract_fbank

dataset = VoxLinguaDataset(
    "datasets/voxlingua107_wav/train.csv"
)

audio_path, label = dataset[0]

waveform, sample_rate = load_audio(audio_path)

features = extract_fbank(
    waveform,
    sample_rate
)

print("=" * 60)
print("Label        :", label)
print("Waveform     :", waveform.shape)
print("FBank Shape  :", features.shape)
print("Feature Dim  :", features.shape[1])
print("Frames       :", features.shape[0])
print("=" * 60)