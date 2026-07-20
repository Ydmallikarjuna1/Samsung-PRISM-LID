import os
import tarfile
from pathlib import Path
from tqdm import tqdm

# ==========================
# CHANGE ONLY THIS VALUE
# ==========================
LANGUAGE = "zh"
# ==========================

INPUT_DIR = Path(f"datasets/voxlingua107/train/{LANGUAGE}")
OUTPUT_DIR = Path(f"datasets/voxlingua107_wav/{LANGUAGE}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tar_files = sorted(INPUT_DIR.glob("*.tar"))

print(f"Language      : {LANGUAGE}")
print(f"Shard Files   : {len(tar_files)}")
print(f"Output Folder : {OUTPUT_DIR}\n")

total_audio = 0

for tar_path in tqdm(tar_files, desc=f"Extracting {LANGUAGE}"):

    with tarfile.open(tar_path, "r") as tar:

        for member in tar.getmembers():

            if not member.name.endswith(".wav"):
                continue

            output_file = OUTPUT_DIR / Path(member.name).name

            if output_file.exists():
                continue

            wav_file = tar.extractfile(member)

            with open(output_file, "wb") as f:
                f.write(wav_file.read())

            total_audio += 1

print("\n======================================")
print("Extraction Finished")
print("======================================")
print("Language :", LANGUAGE)
print("Extracted:", total_audio)