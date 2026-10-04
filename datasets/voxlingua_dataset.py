import json
import tarfile
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from configs.config import RAW_DATA_DIR, SAMPLE_RATE


class VoxLinguaDataset(Dataset):
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path)

        self.samples = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    self.samples.append(json.loads(line))

        self._tar_cache = {}

    def __len__(self):
        return len(self.samples)

    def _get_tar(self, shard_path):
        shard_path = str(shard_path)

        if shard_path not in self._tar_cache:
            full_path = RAW_DATA_DIR / shard_path

            if not full_path.exists():
                raise FileNotFoundError(
                    f"Shard not found: {full_path}"
                )

            self._tar_cache[shard_path] = tarfile.open(
                full_path,
                mode="r"
            )

        return self._tar_cache[shard_path]

    def __getitem__(self, index):
        sample = self.samples[index]

        shard_path = sample["shard_path"]
        wav_member = sample["wav_member"]

        tar = self._get_tar(shard_path)

        try:
            member = tar.getmember(wav_member)
        except KeyError:
            raise FileNotFoundError(
                f"WAV member '{wav_member}' not found "
                f"in shard '{shard_path}'"
            )

        extracted = tar.extractfile(member)

        if extracted is None:
            raise RuntimeError(
                f"Could not extract '{wav_member}' "
                f"from '{shard_path}'"
            )

        waveform_bytes = extracted.read()

        import io

        waveform, sample_rate = torchaudio.load(
            io.BytesIO(waveform_bytes),
            format="wav"
        )

        # Convert stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                SAMPLE_RATE
            )

        label = sample["label_id"]

        return {
            "waveform": waveform,
            "label": torch.tensor(label, dtype=torch.long),
            "utt_id": sample["utt_id"],
            "language": sample["language"],
        }