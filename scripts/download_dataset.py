from pathlib import Path
from huggingface_hub import snapshot_download

from configs.config import RAW_DATA_DIR, LANGUAGES


REPO_ID = "TalTechNLP/voxlingua107_wds"


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    allow_patterns = []

    for language in LANGUAGES:
        allow_patterns.append(f"train/{language}/*.tar")
        allow_patterns.append(f"train/{language}/*.json")

    print("=" * 70)
    print("VoxLingua107 Dataset Download")
    print("=" * 70)
    print(f"Repository : {REPO_ID}")
    print(f"Languages  : {', '.join(LANGUAGES)}")
    print(f"Output     : {RAW_DATA_DIR}")
    print()
    print("Only the selected language shards will be downloaded.")
    print("=" * 70)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW_DATA_DIR),
        allow_patterns=allow_patterns,
    )

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
