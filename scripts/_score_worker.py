#!/usr/bin/env python3
"""Scoring worker subprocess for Best-of-N analysis.

Usage: python _score_worker.py <gpu_id> <work_file>

Each work item: {"audio_path": str, "dramabox_prompt": str, ...}
Writes results to work_file.replace('.json', '_results.json')
"""
import json, os, sys, time, traceback

gpu_id = int(sys.argv[1])
work_file = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
sys.path.insert(0, "/home/deployer/laion/dramabox-pipeline")
sys.path.insert(0, "/home/deployer/laion/voice-acting-pipeline")

import warnings
warnings.filterwarnings("ignore")

from dramabox.scoring import score_audio

with open(work_file) as f:
    work_items = json.load(f)

print(f"[SCORE GPU {gpu_id}] {len(work_items)} items to score", flush=True)

results = []
for i, item in enumerate(work_items):
    try:
        score = score_audio(item["audio_path"], item["dramabox_prompt"], device="cuda")
        results.append({
            "group_key": item["group_key"],
            "item_idx": item["item_idx"],
            "n": item["n"],
            "audio_path": item["audio_path"],
            "wer": score["wer"],
            "content_enjoyment": score["content_enjoyment"],
            "reward": score["reward"],
            "transcription": score.get("transcription", ""),
            "expected_text": score.get("expected_text", ""),
            "status": "ok",
        })
    except Exception as e:
        traceback.print_exc()
        results.append({
            "group_key": item["group_key"],
            "item_idx": item["item_idx"],
            "n": item["n"],
            "audio_path": item["audio_path"],
            "wer": 1.0,
            "content_enjoyment": 0.0,
            "reward": 0.0,
            "status": f"error: {e}",
        })

    if (i + 1) % 20 == 0 or i == len(work_items) - 1:
        print(f"[SCORE GPU {gpu_id}] {i+1}/{len(work_items)} scored", flush=True)

output_file = work_file.replace(".json", "_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"[SCORE GPU {gpu_id}] Done. {len(results)} results saved.", flush=True)
