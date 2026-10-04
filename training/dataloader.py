from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import SAMPLE_RATE
from datasets.voxlingua_dataset import VoxLinguaDataset
from utils.audio_features import LogMelFeatureExtractor


class LIDDataLoader:
    """
    Creates DataLoaders for Spoken Language Identification.

    Audio is converted to a fixed duration so that samples
    can be efficiently batched.
    """

    def __init__(
        self,
        manifest_path,
        batch_size=8,
        num_workers=0,
        max_duration=3.0,
        shuffle=True,
    ):
        self.dataset = VoxLinguaDataset(
            manifest_path
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = int(
            SAMPLE_RATE * max_duration
        )
        self.shuffle = shuffle

        self.feature_extractor = LogMelFeatureExtractor()

    def _prepare_waveform(self, waveform):
        """
        Convert waveform to exactly max_samples.

        Longer audio:
            randomly crop

        Shorter audio:
            zero-pad
        """

        current_length = waveform.shape[-1]

        if current_length > self.max_samples:
            max_start = (
                current_length - self.max_samples
            )

            start = torch.randint(
                0,
                max_start + 1,
                (1,),
            ).item()

            waveform = waveform[
                :,
                start:start + self.max_samples,
            ]

        elif current_length < self.max_samples:
            padding = (
                self.max_samples - current_length
            )

            waveform = F.pad(
                waveform,
                (0, padding),
            )

        return waveform

    def collate_fn(self, batch):
        """
        Convert a list of dataset samples into
        one training batch.
        """

        features = []
        labels = []
        utt_ids = []
        languages = []

        for sample in batch:
            waveform = sample["waveform"]

            waveform = self._prepare_waveform(
                waveform
            )

            log_mel = self.feature_extractor(
                waveform
            )

            features.append(log_mel)
            labels.append(sample["label"])
            utt_ids.append(sample["utt_id"])
            languages.append(sample["language"])

        features = torch.stack(
            features,
            dim=0,
        )

        labels = torch.stack(
            labels,
            dim=0,
        )

        return {
            "features": features,
            "labels": labels,
            "utt_ids": utt_ids,
            "languages": languages,
        }

    def get_loader(self):
        """
        Return the PyTorch DataLoader.
        """

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )


if __name__ == "__main__":
    print("Testing LID DataLoader...")

    manifest_path = (
        Path("datasets")
        / "manifests"
        / "train.jsonl"
    )

    loader_builder = LIDDataLoader(
        manifest_path=manifest_path,
        batch_size=4,
        num_workers=0,
        max_duration=3.0,
        shuffle=False,
    )

    loader = loader_builder.get_loader()

    print(
        f"Dataset size : "
        f"{len(loader_builder.dataset)}"
    )

    batch = next(iter(loader))

    print(
        f"Features shape : "
        f"{batch['features'].shape}"
    )

    print(
        f"Labels shape   : "
        f"{batch['labels'].shape}"
    )

    print(
        f"Languages      : "
        f"{batch['languages']}"
    )

    print(
        f"Feature dtype  : "
        f"{batch['features'].dtype}"
    )

    assert batch["features"].ndim == 3
    assert batch["features"].shape[0] == 4
    assert batch["features"].shape[2] == 80

    assert batch["labels"].shape == (4,)

    assert torch.isfinite(
        batch["features"]
    ).all()

    print("\nLID DATALOADER TEST PASSED")