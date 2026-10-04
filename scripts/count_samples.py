import json
import tarfile
from pathlib import Path

from configs.config import RAW_DATA_DIR, LANGUAGES


def count_language_samples(language):
    language_dir = RAW_DATA_DIR / "train" / language
    total = 0

    shards = sorted(language_dir.glob("*.tar"))

    for shard_path in shards:
        with tarfile.open(shard_path, "r") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue

                file_obj = tar.extractfile(member)

                if file_obj is None:
                    continue

                metadata = json.load(file_obj)

                if metadata.get("lang") != language:
                    raise ValueError(
                        f"Language mismatch in {shard_path.name}: "
                        f"expected {language}, got {metadata.get('lang')}"
                    )

                if not metadata.get("utt_id"):
                    raise ValueError(
                        f"Missing utt_id in {shard_path.name}: {member.name}"
                    )

                total += 1

    return len(shards), total


def main():
    print("=" * 70)
    print("VoxLingua107 Dataset Sample Counter")
    print("=" * 70)

    grand_total = 0

    for language in LANGUAGES:
        shards, samples = count_language_samples(language)

        print(
            f"{language:>3} | "
            f"shards: {shards:>3} | "
            f"samples: {samples:>8}"
        )

        grand_total += samples

    print("-" * 70)
    print(f"Total samples: {grand_total:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
