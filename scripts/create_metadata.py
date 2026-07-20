import pandas as pd
from pathlib import Path

ROOT = Path("datasets/voxlingua107_wav")

LANGUAGES = {
    "hi": 0,
    "en": 1,
    "zh": 2
}

rows = []

for language, label in LANGUAGES.items():

    folder = ROOT / language

    for wav in folder.glob("*.wav"):

        rows.append({
            "utt_id": wav.stem,
            "language": language,
            "label": label,
            "path": str(wav.resolve())
        })

df = pd.DataFrame(rows)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output = ROOT / "metadata.csv"

df.to_csv(output, index=False)

print("=" * 50)
print("Metadata Created")
print("=" * 50)
print(df.head())
print()
print(f"Total Samples : {len(df)}")
print(f"Saved At      : {output}")