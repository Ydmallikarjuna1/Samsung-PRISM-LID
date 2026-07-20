import tarfile
import json
import io
import soundfile as sf

tar_path = "datasets/voxlingua107/train/hi/000000.tar"

with tarfile.open(tar_path, "r") as tar:

    members = tar.getmembers()

    # Read first JSON
    json_member = members[0]
    json_file = tar.extractfile(json_member)

    metadata = json.load(json_file)

    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    print(metadata)

    # Read corresponding WAV
    wav_member = members[1]
    wav_file = tar.extractfile(wav_member)

    audio_bytes = wav_file.read()

    audio, sr = sf.read(io.BytesIO(audio_bytes))

    print("\n" + "=" * 60)
    print("AUDIO INFORMATION")
    print("=" * 60)

    print("Filename :", wav_member.name)
    print("Shape    :", audio.shape)
    print("Sample Rate :", sr)
    print("Duration :", len(audio) / sr, "seconds")