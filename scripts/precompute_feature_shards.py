import argparse
import json
import tarfile
import io
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

from configs.config import RAW_DATA_DIR, SAMPLE_RATE
from utils.audio_features import LogMelFeatureExtractor


def load_audio_from_tar(shard_path, wav_member):
    """Load one WAV file directly from a TAR shard."""

    full_path = RAW_DATA_DIR / shard_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Shard not found: {full_path}"
        )

    with tarfile.open(full_path, mode="r") as tar:

        try:
            member = tar.getmember(wav_member)
        except KeyError:
            raise FileNotFoundError(
                f"WAV member not found: {wav_member}"
            )

        extracted = tar.extractfile(member)

        if extracted is None:
            raise RuntimeError(
                f"Could not extract: {wav_member}"
            )

        waveform_bytes = extracted.read()

    waveform, sample_rate = torchaudio.load(
        io.BytesIO(waveform_bytes),
        format="wav",
    )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            SAMPLE_RATE,
        )

    return waveform


def prepare_waveform(
    waveform,
    max_duration=3.0,
):
    """
    Randomly crop long audio to max_duration.
    Zero-pad short audio.
    """

    max_samples = int(
        SAMPLE_RATE * max_duration
    )

    current_length = waveform.shape[-1]

    if current_length > max_samples:

        max_start = (
            current_length - max_samples
        )

        start = torch.randint(
            0,
            max_start + 1,
            (1,),
        ).item()

        waveform = waveform[
            :,
            start:start + max_samples,
        ]

    elif current_length < max_samples:

        padding = (
            max_samples - current_length
        )

        waveform = F.pad(
            waveform,
            (0, padding),
        )

    return waveform


def save_shard(
    features,
    labels,
    languages,
    utt_ids,
    output_path,
):
    """Save one feature shard."""

    data = {
        "features": torch.stack(
            features
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
        "languages": languages,
        "utt_ids": utt_ids,
    }

    torch.save(
        data,
        output_path,
    )


def precompute(
    manifest_path,
    output_dir,
    shard_size=1000,
    max_samples=None,
    max_duration=3.0,
    seed=42,
):
    manifest_path = Path(
        manifest_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_extractor = (
        LogMelFeatureExtractor()
    )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:

        samples = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if max_samples is not None:
        samples = samples[:max_samples]

    # Make random cropping reproducible.
    torch.manual_seed(seed)

    features = []
    labels = []
    languages = []
    utt_ids = []

    shard_index = 0
    processed = 0
    errors = 0

    for sample in tqdm(
        samples,
        desc="Precomputing feature shards",
    ):

        try:

            waveform = load_audio_from_tar(
                sample["shard_path"],
                sample["wav_member"],
            )

            waveform = prepare_waveform(
                waveform,
                max_duration=max_duration,
            )

            feature = feature_extractor(
                waveform
            )

            # For exactly 3 seconds with our current
            # feature configuration, this should be 301 x 80.
            if feature.shape != (301, 80):
                raise RuntimeError(
                    "Unexpected feature shape: "
                    f"{feature.shape}. "
                    "Expected [301, 80]."
                )

            features.append(
                feature.cpu()
            )

            labels.append(
                sample["label_id"]
            )

            languages.append(
                sample["language"]
            )

            utt_ids.append(
                sample["utt_id"]
            )

            processed += 1

            if len(features) >= shard_size:

                shard_path = (
                    output_dir
                    / f"shard_{shard_index:05d}.pt"
                )

                save_shard(
                    features,
                    labels,
                    languages,
                    utt_ids,
                    shard_path,
                )

                shard_index += 1

                features = []
                labels = []
                languages = []
                utt_ids = []

        except Exception as exc:

            errors += 1

            print(
                f"\nERROR: {sample['utt_id']}"
            )

            print(
                f"Reason: {exc}"
            )

    # Save remaining samples.
    if features:

        shard_path = (
            output_dir
            / f"shard_{shard_index:05d}.pt"
        )

        save_shard(
            features,
            labels,
            languages,
            utt_ids,
            shard_path,
        )

        shard_index += 1

    print("\n" + "=" * 60)
    print("FEATURE SHARD PRECOMPUTATION COMPLETE")
    print("=" * 60)

    print(
        f"Processed : {processed}"
    )

    print(
        f"Errors    : {errors}"
    )

    print(
        f"Shards    : {shard_index}"
    )

    print(
        f"Output    : {output_dir}"
    )

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        default="datasets/manifests/train.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/features/train",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    precompute(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        max_duration=args.max_duration,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()