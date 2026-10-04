from pathlib import Path

from datasets.voxlingua_dataset import VoxLinguaDataset


MANIFEST_PATH = Path("datasets/manifests/train.jsonl")


def main():
    print("Loading dataset manifest...")

    dataset = VoxLinguaDataset(MANIFEST_PATH)

    print(f"Dataset size: {len(dataset)}")

    print("\nLoading first sample...")

    sample = dataset[0]

    waveform = sample["waveform"]

    print(f"Waveform shape : {waveform.shape}")
    print(f"Sample rate    : 16000 Hz")
    print(f"Label ID       : {sample['label'].item()}")
    print(f"Language       : {sample['language']}")
    print(f"UTT ID         : {sample['utt_id']}")

    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    assert sample["label"].ndim == 0

    print("\nDATASET TEST PASSED")


if __name__ == "__main__":
    main()