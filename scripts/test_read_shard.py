import tarfile

tar_path = "datasets/voxlingua107/train/hi/000000.tar"

with tarfile.open(tar_path, "r") as tar:
    members = tar.getmembers()

print(f"Total files inside shard: {len(members)}")

print("\nFirst 20 files:\n")

for member in members[:20]:
    print(member.name)