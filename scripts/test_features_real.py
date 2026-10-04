from pathlib import Path

from datasets.voxlingua_dataset import VoxLinguaDataset
from utils.audio_features import LogMelFeatureExtractor


MANIFEST_PATH = Path("datasets/manifests/train.jsonl")


def main():
    print("Loading dataset...")

    dataset = VoxLinguaDataset(MANIFEST_PATH)
    extractor = LogMelFeatureExtractor()

    test_indices = [0, 100, 5000]

    print("\nTesting Log-Mel features on real dataset samples...\n")

    for index in test_indices:
        sample = dataset[index]

        waveform = sample["waveform"]
        features = extractor(waveform)

        print(
            f"Index: {index:5d} | "
            f"Language: {sample['language']:2s} | "
            f"Waveform: {tuple(waveform.shape)} | "
            f"Features: {tuple(features.shape)}"
        )

        assert waveform.shape[0] == 1
        assert features.ndim == 2
        assert features.shape[1] == 80
        assert features.shape[0] > 0
        assert features.dtype == waveform.dtype
        assert features.isfinite().all()

    print("\nREAL AUDIO FEATURE TEST PASSED")


if __name__ == "__main__":
    main()