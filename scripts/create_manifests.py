import json
import random
from collections import defaultdict, Counter
from pathlib import Path
import tarfile

from configs.config import (
    RAW_DATA_DIR,
    LANGUAGES,
    LANGUAGE_TO_ID,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    SEED,
)


MANIFEST_DIR = RAW_DATA_DIR.parent / "manifests"


def collect_samples(language):
    language_dir = RAW_DATA_DIR / "train" / language
    samples = []

    for shard_path in sorted(language_dir.glob("*.tar")):
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()

        json_members = {
            Path(member.name).stem: member
            for member in members
            if member.isfile() and member.name.endswith(".json")
        }

        wav_members = {
            Path(member.name).stem: member
            for member in members
            if member.isfile() and member.name.endswith(".wav")
        }

        common_ids = sorted(set(json_members) & set(wav_members))

        for utt_id in common_ids:
            json_member = json_members[utt_id]
            wav_member = wav_members[utt_id]

            with tarfile.open(shard_path, "r") as tar:
                file_obj = tar.extractfile(json_member)

                if file_obj is None:
                    raise RuntimeError(
                        f"Could not read metadata: {json_member.name}"
                    )

                metadata = json.load(file_obj)

            if metadata.get("lang") != language:
                raise ValueError(
                    f"Language mismatch: expected {language}, "
                    f"got {metadata.get('lang')}"
                )

            if metadata.get("utt_id") != utt_id:
                raise ValueError(
                    f"utt_id mismatch: {metadata.get('utt_id')} != {utt_id}"
                )

            source_id = utt_id.split("__U__", 1)[0]

            samples.append(
                {
                    "utt_id": utt_id,
                    "language": language,
                    "label_id": LANGUAGE_TO_ID[language],
                    "source_id": source_id,
                    "shard_path": str(shard_path.resolve()),
                    "wav_member": wav_member.name,
                    "json_member": json_member.name,
                }
            )

    return samples


def split_by_source(samples):
    grouped = defaultdict(list)

    for sample in samples:
        grouped[sample["source_id"]].append(sample)

    source_ids = list(grouped.keys())

    rng = random.Random(SEED)
    rng.shuffle(source_ids)

    total_sources = len(source_ids)

    train_end = int(total_sources * TRAIN_RATIO)
    valid_end = train_end + int(total_sources * VALID_RATIO)

    train_sources = set(source_ids[:train_end])
    valid_sources = set(source_ids[train_end:valid_end])
    test_sources = set(source_ids[valid_end:])

    train = []
    valid = []
    test = []

    for source_id, source_samples in grouped.items():
        if source_id in train_sources:
            train.extend(source_samples)
        elif source_id in valid_sources:
            valid.extend(source_samples)
        elif source_id in test_sources:
            test.extend(source_samples)
        else:
            raise RuntimeError(f"Source was not assigned: {source_id}")

    return train, valid, test


def write_jsonl(path, samples):
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_summary(split_name, samples):
    language_counts = Counter(sample["language"] for sample in samples)
    source_counts = Counter(sample["source_id"] for sample in samples)

    print(f"\n{split_name.upper()}")
    print("-" * 70)

    for language in LANGUAGES:
        print(
            f"{language:>3} | "
            f"utterances: {language_counts[language]:>7,}"
        )

    print(f"Total utterances : {len(samples):,}")
    print(f"Unique sources   : {len(source_counts):,}")


def main():
    print("=" * 70)
    print("SOURCE-GROUPED MANIFEST GENERATOR")
    print("=" * 70)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_valid = []
    all_test = []

    for language in LANGUAGES:
        print(f"\nProcessing {language}...")

        samples = collect_samples(language)

        print(f"Collected {len(samples):,} samples.")

        train, valid, test = split_by_source(samples)

        print(
            f"Split: train={len(train):,}, "
            f"valid={len(valid):,}, "
            f"test={len(test):,}"
        )

        all_train.extend(train)
        all_valid.extend(valid)
        all_test.extend(test)

    rng = random.Random(SEED)

    rng.shuffle(all_train)
    rng.shuffle(all_valid)
    rng.shuffle(all_test)

    write_jsonl(MANIFEST_DIR / "train.jsonl", all_train)
    write_jsonl(MANIFEST_DIR / "valid.jsonl", all_valid)
    write_jsonl(MANIFEST_DIR / "test.jsonl", all_test)

    print_summary("train", all_train)
    print_summary("valid", all_valid)
    print_summary("test", all_test)

    print("\n" + "=" * 70)
    print("MANIFEST GENERATION COMPLETE")
    print("=" * 70)
    print(f"Train : {MANIFEST_DIR / 'train.jsonl'}")
    print(f"Valid : {MANIFEST_DIR / 'valid.jsonl'}")
    print(f"Test  : {MANIFEST_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    main()
