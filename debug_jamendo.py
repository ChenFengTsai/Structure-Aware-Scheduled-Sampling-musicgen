import os
os.environ['HF_HOME'] = "/storage/ssd1/richtsai1103/hf_cache"

from huggingface_hub import hf_hub_download
import json

REPO_ID = "amaai-lab/JamendoMaxCaps"

local_path = hf_hub_download(repo_id=REPO_ID, filename="2024-01-02.jsonl", repo_type="dataset")

print("=== First 3 entries of 2024-01-02.jsonl ===\n")
with open(local_path, "r") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        entry = json.loads(line)
        print(json.dumps(entry, indent=2))
        print("---")