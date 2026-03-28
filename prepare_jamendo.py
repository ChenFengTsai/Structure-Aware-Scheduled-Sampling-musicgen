import json
import os
import random
from pathlib import Path
from tqdm import tqdm

INPUT_JSONL = '/storage/ssd1/richtsai1103/Jamendo/metadata.jsonl'
AUDIO_BASE_DIR = '/storage/ssd1/richtsai1103/Jamendo/audio'
BASE_OUTPUT_DIR = '/storage/ssd1/richtsai1103/Jamendo/egs/jamendo'

def process_split(split_name, dataset_chunk):
    print(f"\nProcessing {split_name} split ({len(dataset_chunk)} items)...")
    split_dir = os.path.join(BASE_OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    jsonl_path = os.path.join(split_dir, 'data.jsonl')
    jsonl_entries = []

    for item in tqdm(dataset_chunk):
        audio_filename = item['name'] + '.wav'
        audio_path = os.path.join(AUDIO_BASE_DIR, audio_filename)

        if not os.path.exists(audio_path):
            continue

        description = item.get('description', '').strip()
        if not description:
            continue

        duration = item.get('duration')
        sample_rate = item.get('sample_rate', 44100)

        hardlink_path = os.path.join(split_dir, audio_filename)
        json_path = os.path.join(split_dir, item['name'] + '.json')

        if not os.path.exists(hardlink_path):
            os.link(audio_path, hardlink_path)  # hardlink instead of symlink

        metadata = {
            "key": item.get('key', ''),
            "artist": item.get('artist', ''),
            "sample_rate": sample_rate,
            "file_extension": "wav",
            "description": description,
            "keywords": item.get('keywords', ''),
            "duration": duration,
            "bpm": item.get('bpm', ''),
            "genre": item.get('genre', ''),
            "title": item.get('title', ''),
            "name": item['name'],
            "instrument": item.get('instrument', ''),
            "moods": item.get('moods', [])
        }

        with open(json_path, 'w') as mf:
            json.dump(metadata, mf)

        jsonl_entries.append({
            "path": hardlink_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "amplitude": None,
            "weight": None
        })

    with open(jsonl_path, 'w') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + '\n')

    return len(jsonl_entries)

def main():
    print(f"Loading dataset from: {INPUT_JSONL}")
    with open(INPUT_JSONL, 'r') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    random.seed(42)
    random.shuffle(dataset)

    total = len(dataset)
    train_end = int(total * 0.90)
    valid_end = int(total * 0.95)

    train_data = dataset[:train_end]
    valid_data = dataset[train_end:valid_end]
    generate_data = dataset[valid_end:]

    train_count = process_split('train', train_data)
    valid_count = process_split('valid', valid_data)
    gen_count = process_split('generate', generate_data)

    print(f"\nDone! Generated entries - Train: {train_count}, Valid: {valid_count}, Generate: {gen_count}")

if __name__ == "__main__":
    main()