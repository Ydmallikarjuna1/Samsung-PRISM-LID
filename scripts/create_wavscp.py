from pathlib import Path

# Folder containing extracted audio
AUDIO_DIR = Path("datasets/voxlingua107_wav/hi")

# Output file
OUTPUT_FILE = Path("datasets/voxlingua107_wav/wav.scp")

wav_files = sorted(AUDIO_DIR.glob("*.wav"))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for wav in wav_files:
        utt_id = wav.stem
        f.write(f"{utt_id} {wav.resolve()}\n")

print("=" * 50)
print("wav.scp Created Successfully")
print("=" * 50)
print(f"Total Entries : {len(wav_files)}")
print(f"Saved At      : {OUTPUT_FILE}")