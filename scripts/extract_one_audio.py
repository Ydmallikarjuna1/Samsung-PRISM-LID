import os
import tarfile

# Input shard
tar_path = "datasets/voxlingua107/train/hi/000000.tar"

# Output folder
output_dir = "datasets/voxlingua107_wav/hi"

os.makedirs(output_dir, exist_ok=True)

with tarfile.open(tar_path, "r") as tar:

    members = tar.getmembers()

    # First WAV file
    wav_member = members[1]

    # Read bytes
    wav_file = tar.extractfile(wav_member)

    # Output path
    output_path = os.path.join(output_dir, wav_member.name)

    with open(output_path, "wb") as f:
        f.write(wav_file.read())

print("Saved successfully!")
print(output_path)