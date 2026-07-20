import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.dataset import VoxLinguaDataset

dataset = VoxLinguaDataset(
    "datasets/voxlingua107_wav/train.csv"
)

print("=" * 50)
print("Dataset Size:", len(dataset))
print("=" * 50)

audio_path, label = dataset[0]

print("First Audio :", audio_path)
print("Label       :", label)