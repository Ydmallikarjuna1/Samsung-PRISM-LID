#!/usr/bin/env python
"""
VoxLingua107 (Hugging Face WDS mirror)  ->  per-language 16 kHz WAV folders + wav.scp
=====================================================================================

One script, three steps:

    1. DOWNLOAD  only the wanted language shards (train/<lang>/*.tar) plus the dev
       shards from the Hugging Face mirror ``TalTechNLP/voxlingua107_wds``.
    2. DECODE    each sample's audio to a 16 kHz mono array with ``soundfile``
       (no torchcodec / ffmpeg needed) and write ``<out>/<lang>/<key>.wav``.
    3. INDEX     write ``<out>/wav.scp`` — one ``"<utt_id> <abs_path>"`` per line —
       which the Stage-1 session-disjoint splitter reads.

The result matches unzipping the old per-language ``<lang>.zip`` files. Each
sample's WebDataset key (``__key__``) becomes both the filename and the utt_id,
so the video/session grouping the disjoint split relies on is preserved.

Layout / fields are per the dataset card's own usage example:
    train/<lang>/*.tar , dev/*.tar ; each sample = {"wav": audio, "json": {"lang": ...}}

Why not plain ``datasets.load_dataset(...)``? On ``datasets`` >= 3 the WebDataset
builder decodes audio through the ``Audio`` feature and needs ``torchcodec``
(hence ffmpeg) to write it back. Streaming raw bytes with the ``webdataset``
library + decoding with ``soundfile`` avoids that entirely.

Install:
    pip install huggingface_hub webdataset soundfile numpy scipy tqdm

Verify the machinery locally WITHOUT downloading (recommended first run):
    python voxlingua_download_and_extract.py --selftest

Real use (downloads en/hi/zh + dev, then converts):
    python voxlingua_download_and_extract.py
    python voxlingua_download_and_extract.py --skip-download   # reconvert only
    python voxlingua_download_and_extract.py --no-dev          # train shards only
    HF_TOKEN=hf_xxx python voxlingua_download_and_extract.py    # if ever gated

DISK NOTE: English is one of the largest languages in VoxLingua107. Downloaded
tars plus extracted 16 kHz wavs for en+hi+zh can total tens of GB. Make sure the
target volume has room before running.
"""

import argparse
import io
import json
import os
import sys
from glob import glob

import numpy as np
import soundfile as sf
import webdataset as wds
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ID = "TalTechNLP/voxlingua107_wds"
TARGET_LANGS = {"en", "hi", "zh"}          # English, Hindi, Mandarin/Chinese
TARGET_SR = 16000
# audio member extensions webdataset may expose (dataset uses "wav"; others are
# accepted defensively so a future format change won't silently drop samples)
AUDIO_EXTS = ("wav", "flac", "ogg", "opus", "mp3", "m4a", "wave")


# ---------------------------------------------------------------------------
# Step 1 — download
# ---------------------------------------------------------------------------
def download_shards(repo_id, langs, download_root, include_dev=True,
                    token=None, max_workers=8):
    """Fetch only the shards we need. Idempotent (HF caches + resumes)."""
    from huggingface_hub import snapshot_download

    allow = [f"train/{lang}/*.tar" for lang in sorted(langs)]
    if include_dev:
        allow.append("dev/*.tar")

    print(f"Downloading {sorted(langs)}"
          f"{' + dev' if include_dev else ''} from {repo_id} ...")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=download_root,
        token=token or os.environ.get("HF_TOKEN"),
        max_workers=max_workers,
    )
    print(f"Shards are under: {path}")
    return path


# ---------------------------------------------------------------------------
# Step 2 — decode & save
# ---------------------------------------------------------------------------
def to_mono_16k(array, sr):
    """array: (samples, channels) float32 -> (samples,) float32 at TARGET_SR."""
    if array.ndim == 2:
        array = array.mean(axis=1)                     # downmix to mono
    if int(sr) != TARGET_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), TARGET_SR)
        array = resample_poly(array, TARGET_SR // g, int(sr) // g)
    return np.asarray(array, dtype=np.float32)


def parse_meta(meta):
    if isinstance(meta, (bytes, bytearray)):
        try:
            return json.loads(meta)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
    return meta if isinstance(meta, dict) else {}


def lang_of(meta, folder_lang):
    """Language from JSON 'lang' if present, else the train/<lang>/ folder name."""
    m = parse_meta(meta)
    return m.get("lang") or folder_lang


def iter_shard(tar_path):
    """Yield (audio_bytes, meta, key) per sample. Audio field found by extension;
    no built-in audio decoding, so torchcodec/ffmpeg are never touched."""
    ds = wds.WebDataset(tar_path, shardshuffle=False, handler=wds.warn_and_continue)
    for sample in ds:
        key = sample.get("__key__")
        audio_bytes = next((sample[e] for e in AUDIO_EXTS if e in sample), None)
        yield audio_bytes, sample.get("json"), key


def extract_shard(tar_path, out_root, folder_lang, target_langs, overwrite):
    written = skipped = errors = 0
    for audio_bytes, meta, key in iter_shard(tar_path):
        lang = lang_of(meta, folder_lang)
        if audio_bytes is None or key is None:
            errors += 1
            continue
        if lang is None or (target_langs and lang not in target_langs):
            skipped += 1
            continue

        out_dir = os.path.join(out_root, lang)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(key) + ".wav")
        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue

        try:
            array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32",
                                always_2d=True)
            array = to_mono_16k(array, sr)
            tmp = out_path + ".part"
            sf.write(tmp, array, TARGET_SR, subtype="PCM_16", format="WAV")
            os.replace(tmp, out_path)
            written += 1
        except Exception as e:                          # never abort the whole run
            errors += 1
            print(f"  [warn] failed on {key!r} in {os.path.basename(tar_path)}: {e}")
    return written, skipped, errors


def extract_all(download_root, out_root, target_langs, overwrite):
    os.makedirs(out_root, exist_ok=True)
    shards = []
    for lang in sorted(target_langs):                    # train: lang from folder
        shards += [(p, lang) for p in
                   sorted(glob(os.path.join(download_root, "train", lang, "*.tar")))]
    shards += [(p, None) for p in                        # dev: lang from json
               sorted(glob(os.path.join(download_root, "dev", "*.tar")))]
    if not shards:
        raise SystemExit(
            f"No .tar shards under {download_root!r} "
            f"(expected train/<lang>/*.tar and/or dev/*.tar). Did the download run?")

    tot_w = tot_s = tot_e = 0
    for tar_path, folder_lang in tqdm(shards, desc="Converting shards"):
        w, s, e = extract_shard(tar_path, out_root, folder_lang, target_langs, overwrite)
        tot_w += w; tot_s += s; tot_e += e
    print(f"\nConversion done: {tot_w} wav written, {tot_s} skipped, {tot_e} errors.")
    return tot_w, tot_s, tot_e


# ---------------------------------------------------------------------------
# Step 3 — wav.scp
# ---------------------------------------------------------------------------
def build_wav_scp(out_root, target_langs, scp_path=None):
    scp_path = scp_path or os.path.join(out_root, "wav.scp")
    rows, seen, dupes = [], set(), 0
    for lang in sorted(target_langs):
        d = os.path.join(out_root, lang)
        if not os.path.isdir(d):
            print(f"  warning: no folder for '{lang}' (nothing extracted?)")
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".wav"):
                utt = fn[:-4]
                if utt in seen:
                    dupes += 1
                seen.add(utt)
                rows.append(f"{utt} {os.path.abspath(os.path.join(d, fn))}")
    with open(scp_path, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"Wrote {len(rows)} entries to {scp_path}")
    if dupes:
        print(f"  WARNING: {dupes} duplicate utt_ids across languages — "
              f"prefix utt_ids with the language code to disambiguate.")
    return scp_path


# ---------------------------------------------------------------------------
# Optional self-test: exercises decode+save+scp on synthetic shards, no network
# ---------------------------------------------------------------------------
def run_selftest():
    import tarfile, tempfile, shutil, wave, struct
    print("=== SELF-TEST (synthetic shards, no network) ===")
    tmp = tempfile.mkdtemp(prefix="voxselftest_")
    try:
        root = os.path.join(tmp, "dl")
        out = os.path.join(tmp, "out")

        def add_wav(tar, key, lang, sr=16000, n=1600, ch=1):
            buf = io.BytesIO()
            w = wave.open(buf, "wb"); w.setnchannels(ch); w.setsampwidth(2)
            w.setframerate(sr); w.writeframes(struct.pack("<%dh" % (n*ch), *([0]*(n*ch))))
            w.close(); wb = buf.getvalue()
            ti = tarfile.TarInfo(key + ".wav"); ti.size = len(wb)
            tar.addfile(ti, io.BytesIO(wb))
            jb = json.dumps({"lang": lang}).encode()
            tj = tarfile.TarInfo(key + ".json"); tj.size = len(jb)
            tar.addfile(tj, io.BytesIO(jb))

        def add_flac(tar, key, lang, sr=16000, n=1600):
            buf = io.BytesIO()
            sf.write(buf, np.zeros(n, np.float32), sr, format="FLAC", subtype="PCM_16")
            fb = buf.getvalue()
            ti = tarfile.TarInfo(key + ".flac"); ti.size = len(fb)
            tar.addfile(ti, io.BytesIO(fb))
            jb = json.dumps({"lang": lang}).encode()
            tj = tarfile.TarInfo(key + ".json"); tj.size = len(jb)
            tar.addfile(tj, io.BytesIO(jb))

        # train/en: normal + stereo + 48k(resample) + a sample with NO json
        d = os.path.join(root, "train", "en"); os.makedirs(d)
        with tarfile.open(os.path.join(d, "000000.tar"), "w") as t:
            add_wav(t, "vidEN1__seg0", "en")
            add_wav(t, "vidEN1__seg1", "en", ch=2)              # stereo -> mono
            add_wav(t, "vidEN2__seg0", "en", sr=48000, n=4800)  # resample
            # sample missing json -> language must fall back to folder "en"
            key = "vidEN3__seg0"
            buf = io.BytesIO(); w = wave.open(buf, "wb")
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(struct.pack("<1600h", *([0]*1600))); w.close()
            wb = buf.getvalue(); ti = tarfile.TarInfo(key + ".wav"); ti.size = len(wb)
            t.addfile(ti, io.BytesIO(wb))
        # train/hi: flac sample
        d = os.path.join(root, "train", "hi"); os.makedirs(d)
        with tarfile.open(os.path.join(d, "000000.tar"), "w") as t:
            add_flac(t, "vidHI1__seg0", "hi")
        # dev: mixed, incl. an unwanted language
        d = os.path.join(root, "dev"); os.makedirs(d)
        with tarfile.open(os.path.join(d, "000000.tar"), "w") as t:
            add_wav(t, "devEN__u0", "en"); add_wav(t, "devHI__u0", "hi")
            add_wav(t, "devZH__u0", "zh"); add_wav(t, "devES__u0", "es")  # skip es

        langs = {"en", "hi", "zh"}
        w, s, e = extract_all(root, out, langs, overwrite=False)
        scp = build_wav_scp(out, langs)

        # ---- assertions ----
        got = {os.path.relpath(p, out) for p in glob(os.path.join(out, "*", "*.wav"))}
        expect = {
            os.path.join("en", "vidEN1__seg0.wav"), os.path.join("en", "vidEN1__seg1.wav"),
            os.path.join("en", "vidEN2__seg0.wav"), os.path.join("en", "vidEN3__seg0.wav"),
            os.path.join("hi", "vidHI1__seg0.wav"),
            os.path.join("en", "devEN__u0.wav"),   os.path.join("hi", "devHI__u0.wav"),
            os.path.join("zh", "devZH__u0.wav"),
        }
        assert got == expect, f"file set mismatch:\n got={sorted(got)}\n exp={sorted(expect)}"
        assert e == 0, f"unexpected errors: {e}"
        assert not os.path.exists(os.path.join(out, "es")), "es should be skipped"
        # every output is 16k mono PCM_16
        for p in glob(os.path.join(out, "*", "*.wav")):
            i = sf.info(p)
            assert i.samplerate == 16000 and i.channels == 1 and i.subtype == "PCM_16", \
                f"bad format for {p}: {i.samplerate}/{i.channels}/{i.subtype}"
        # resampled 48k/4800 -> ~1600 samples
        i = sf.info(os.path.join(out, "en", "vidEN2__seg0.wav"))
        assert 1580 <= i.frames <= 1620, f"resample frames off: {i.frames}"
        # missing-json sample landed under folder lang 'en'
        assert os.path.exists(os.path.join(out, "en", "vidEN3__seg0.wav"))
        # scp: 8 lines, well formed, files exist
        with open(scp) as f:
            lines = [ln for ln in f.read().splitlines() if ln]
        assert len(lines) == 8, f"scp expected 8 lines, got {len(lines)}"
        for ln in lines:
            utt, path = ln.split(" ", 1)
            assert os.path.isfile(path) and path.endswith(utt + ".wav")

        # idempotency: second pass writes nothing
        w2, s2, e2 = extract_all(root, out, langs, overwrite=False)
        assert w2 == 0 and e2 == 0, f"not idempotent: wrote {w2}, errors {e2}"

        print("\nSELF-TEST PASSED: 8 wavs (wav+flac+stereo+resample+missing-json),"
              " 'es' filtered, 16k/mono/PCM_16, wav.scp valid, idempotent.")
        return 0
    except AssertionError as a:
        print("SELF-TEST FAILED:", a); return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--download-root", default="./voxlingua107",
                    help="where shards are downloaded to / read from")
    ap.add_argument("--out-root", default="./voxlingua107_wav",
                    help="where <lang>/*.wav and wav.scp are written")
    ap.add_argument("--langs", default=",".join(sorted(TARGET_LANGS)),
                    help="comma-separated ISO codes, e.g. en,hi,zh")
    ap.add_argument("--no-dev", action="store_true", help="skip the dev shards")
    ap.add_argument("--skip-download", action="store_true",
                    help="use already-downloaded shards under --download-root")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-decode wavs even if they already exist")
    ap.add_argument("--hf-token", default=None, help="HF token (or env HF_TOKEN)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--selftest", action="store_true",
                    help="run a local synthetic-data test and exit (no network)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    if not args.skip_download:
        download_shards(args.repo_id, langs, args.download_root,
                        include_dev=not args.no_dev,
                        token=args.hf_token, max_workers=args.max_workers)
    else:
        print("Skipping download (using existing shards).")

    extract_all(args.download_root, args.out_root, langs, args.overwrite)
    scp = build_wav_scp(args.out_root, langs)
    print(f"\nAll done. Point Stage 1 at:\n  VOX_SCP = {os.path.abspath(scp)!r}")


if __name__ == "__main__":
    main()
