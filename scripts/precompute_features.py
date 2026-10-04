import argparse
import json
from pathlib import Path
import tarfile
import io

import torch
import torchaudio
from tqdm import tqdm

from configs.config import RAW_DATA_DIR, SAMPLE_RATE
from utils.audio_features import LogMelFeatureExtractor


def load_audio_from_tar(shard_path, wav_member):
    """
    Read one WAV file directly from a TAR shard.
    """

    full_path = RAW_DATA_DIR / shard_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Shard not found: {full_path}"
        )

    with tarfile.open(
        full_path,
        mode="r",
    ) as tar:

        try:
            member = tar.getmember(
                wav_member
            )
        except KeyError:
            raise FileNotFoundError(
                f"WAV member not found: "
                f"{wav_member}"
            )

        extracted = tar.extractfile(member)

        if extracted is None:
            raise RuntimeError(
                f"Could not extract: "
                f"{wav_member}"
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


def process_samples(
    manifest_path,
    output_dir,
    max_samples=None,
):
    """
    Convert manifest audio samples into
    precomputed Log-Mel feature files.
    """

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

    metadata_path = (
        output_dir / "metadata.jsonl"
    )

    processed = 0
    skipped = 0
    errors = 0

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as metadata_file:

        for sample in tqdm(
            samples,
            desc="Precomputing features",
        ):

            utt_id = sample["utt_id"]

            feature_path = (
                output_dir
                / f"{utt_id}.pt"
            )

            try:

                # Skip if already processed
                if feature_path.exists():
                    skipped += 1

                    metadata_file.write(
                        json.dumps(
                            {
                                "utt_id": utt_id,
                                "language": sample[
                                    "language"
                                ],
                                "label_id": sample[
                                    "label_id"
                                ],
                                "feature_path": str(
                                    feature_path
                                ),
                            }
                        )
                        + "\n"
                    )

                    continue

                waveform = load_audio_from_tar(
                    sample["shard_path"],
                    sample["wav_member"],
                )

                features = feature_extractor(
                    waveform
                )

                torch.save(
                    features.cpu(),
                    feature_path,
                )

                metadata_file.write(
                    json.dumps(
                        {
                            "utt_id": utt_id,
                            "language": sample[
                                "language"
                            ],
                            "label_id": sample[
                                "label_id"
                            ],
                            "feature_path": str(
                                feature_path
                            ),
                        }
                    )
                    + "\n"
                )

                processed += 1

            except Exception as exc:

                errors += 1

                print(
                    f"\nERROR: {utt_id}"
                )

                print(
                    f"Reason: {exc}"
                )

    print("\n" + "=" * 60)
    print("FEATURE PRECOMPUTATION COMPLETE")
    print("=" * 60)

    print(
        f"Processed : {processed}"
    )

    print(
        f"Skipped   : {skipped}"
    )

    print(
        f"Errors    : {errors}"
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
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples to process "
            "for testing."
        ),
    )

    args = parser.parse_args()

    process_samples(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()