import os
import io
import json
import re

os.environ['HF_HOME'] = "/storage/ssd1/richtsai1103/hf_cache"

import soundfile as sf
from datasets import load_dataset, Audio
from huggingface_hub import snapshot_download
from tqdm import tqdm

output_dir = "/storage/ssd1/richtsai1103/Jamendo_Test"
audio_dir  = os.path.join(output_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

REPO_ID        = "amaai-lab/JamendoMaxCaps"
num_samples    = 10000
CAPTION_CACHE  = os.path.join(output_dir, "caption_lookup.json")
META_CACHE     = os.path.join(output_dir, "meta_lookup.json")

# ---------------------------------------------------------------
# STEP 1: Download all JSONL files to disk (one-time)
# ---------------------------------------------------------------
print("Downloading all metadata + caption JSONL files...")
local_repo = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    ignore_patterns=["data/*", "*.parquet"],
)
print(f"JSONL files cached at: {local_repo}")

# ---------------------------------------------------------------
# STEP 2: Build or load caption lookup
# ---------------------------------------------------------------
if os.path.exists(CAPTION_CACHE):
    print(f"Loading caption lookup from cache: {CAPTION_CACHE}")
    with open(CAPTION_CACHE, "r") as f:
        caption_lookup = json.load(f)
else:
    print("Building caption lookup...")
    caption_lookup = {}
    caption_path = os.path.join(local_repo, "final_caption30sec.jsonl")
    with open(caption_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            track_id = str(entry["id"])
            if track_id not in caption_lookup:
                caption_lookup[track_id] = entry
    print(f"Saving caption lookup to {CAPTION_CACHE}...")
    with open(CAPTION_CACHE, "w") as f:
        json.dump(caption_lookup, f)

print(f"Captions ready: {len(caption_lookup)} tracks.")

# ---------------------------------------------------------------
# STEP 3: Build or load metadata lookup
# ---------------------------------------------------------------
if os.path.exists(META_CACHE):
    print(f"Loading metadata lookup from cache: {META_CACHE}")
    with open(META_CACHE, "r") as f:
        meta_lookup = json.load(f)
else:
    print("Building metadata lookup (first run only)...")
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")
    meta_files = [
        os.path.join(local_repo, f)
        for f in os.listdir(local_repo)
        if date_pattern.match(f)
    ]
    print(f"Found {len(meta_files)} metadata shards.")

    meta_lookup = {}
    for meta_file in tqdm(meta_files, desc="Loading metadata"):
        with open(meta_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    mid = str(entry["id"])
                    # Store only needed fields — skip waveform to save RAM/disk
                    meta_lookup[mid] = {
                        "artist_name": entry.get("artist_name", ""),
                        "name":        entry.get("name", ""),
                        "musicinfo":   entry.get("musicinfo", {})
                    }
                except json.JSONDecodeError:
                    continue

    print(f"Saving metadata lookup to {META_CACHE}...")
    with open(META_CACHE, "w") as f:
        json.dump(meta_lookup, f)

print(f"Metadata ready: {len(meta_lookup)} tracks.")

# ---------------------------------------------------------------
# STEP 4: Stream audio one-at-a-time and process immediately
# ---------------------------------------------------------------
print(f"\nStreaming and processing {num_samples} samples...")
dataset = load_dataset(REPO_ID, split="train", streaming=True)
dataset = dataset.cast_column("audio", Audio(decode=False))

metadata_records = []

for example in tqdm(dataset.take(num_samples), total=num_samples):
    # Extract track id from audio filename
    audio_path = example['audio'].get('path', '')
    track_id   = os.path.splitext(os.path.basename(audio_path))[0]

    # Decode audio — held in RAM only for this iteration
    audio_bytes = example['audio']['bytes']
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    del audio_bytes

    # Resample if needed
    target_sr = 44100
    if sr != target_sr:
        import librosa
        audio_data = librosa.resample(
            audio_data.T if audio_data.ndim > 1 else audio_data,
            orig_sr=sr, target_sr=target_sr
        )
        if audio_data.ndim > 1:
            audio_data = audio_data.T
        sr = target_sr

    # Look up caption and metadata
    cap  = caption_lookup.get(track_id, {})
    meta = meta_lookup.get(track_id, {})

    start_t = float(cap.get('start_time', 0))
    end_t   = float(cap.get('end_time', start_t + 30))

    # Slice, save, discard
    start_sample = int(start_t * sr)
    end_sample   = int(end_t * sr)
    audio_clip   = audio_data[start_sample:end_sample]
    del audio_data

    file_name = f"{track_id}_{int(start_t)}.wav"
    file_path = os.path.join(audio_dir, file_name)
    sf.write(file_path, audio_clip, sr)
    del audio_clip

    # Parse metadata
    musicinfo   = meta.get('musicinfo', {})
    tags        = musicinfo.get('tags', {})
    genres      = tags.get('genres', [])
    instruments = tags.get('instruments', [])
    moods       = tags.get('vartags', [])
    speed       = musicinfo.get('speed', '')
    duration    = end_t - start_t

    record = {
        "key": "",
        "artist": meta.get("artist_name", ""),
        "sample_rate": sr,
        "file_extension": "wav",
        "description": cap.get("caption", ""),
        "keywords": ", ".join(genres + instruments + moods),
        "duration": duration,
        "bpm": speed,
        "genre": genres[0] if genres else "instrumental",
        "title": meta.get("name", f"Track {track_id}"),
        "name": f"{track_id}_{int(start_t)}",
        "instrument": ", ".join(instruments) if instruments else "Mix",
        "moods": moods,
        "start_time": start_t,
        "end_time": end_t
    }
    metadata_records.append(record)

# Write output metadata
with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
    for record in metadata_records:
        f.write(json.dumps(record) + "\n")

print(f"\nDone! {num_samples} clips saved to {output_dir}")
print(f"Lookup caches saved to {output_dir} for future runs.")