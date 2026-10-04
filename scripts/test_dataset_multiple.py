from pathlib import Path

from datasets.voxlingua_dataset import VoxLinguaDataset


MANIFEST_PATH = Path("datasets/manifests/train.jsonl")


def main():
    print("Loading dataset...")

    dataset = VoxLinguaDataset(MANIFEST_PATH)

    print(f"Dataset size: {len(dataset)}")

    test_indices = [0, 100, 1000, 5000, 10000]

    print("\nTesting multiple samples...\n")

    for index in test_indices:
        sample = dataset[index]

        waveform = sample["waveform"]

        print(
            f"Index: {index:6d} | "
            f"Language: {sample['language']:2s} | "
            f"Label: {sample['label'].item():2d} | "
            f"Shape: {tuple(waveform.shape)} | "
            f"UTT: {sample['utt_id']}"
        )

        assert waveform.ndim == 2
        assert waveform.shape[0] == 1
        assert sample["label"].item() >= 0
        assert sample["label"].item() < 10

    print("\nMULTIPLE SAMPLE DATASET TEST PASSED")


if __name__ == "__main__":
    main()