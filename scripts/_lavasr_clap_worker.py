#!/usr/bin/env python3
"""Combined VoiceCLAP scoring: sanitized + neg_san in one pass.

Usage: python _lavasr_clap_worker.py <gpu_id> <work_file> <model_size>

model_size: "large" or "small"

Each work item: {
    "audio_path": str,
    "sanitized_text": str,
    "group_key": str,
    "item_idx": int,
    "n": int,
}

Scores each audio against:
  1. sanitized_text (per-group, varies)
  2. NEG_SAN_TEXT (fixed negative prompt)

Writes results to work_file.replace('.json', '_results.json')
"""
import json, os, sys, time, traceback

gpu_id = int(sys.argv[1])
work_file = sys.argv[2]
model_size = sys.argv[3]  # "large" or "small"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

NEG_SAN_TEXT = "robotic, distorted, uncanny, distorted, distortion"

with open(work_file) as f:
    work_items = json.load(f)

print(f"[CLAP-{model_size.upper()} GPU {gpu_id}] {len(work_items)} items", flush=True)

# Collect unique sanitized texts
unique_sanitized = list(set(item["sanitized_text"] for item in work_items))

if model_size == "large":
    from sentence_transformers import SentenceTransformer
    import librosa

    model = SentenceTransformer("laion/voiceclap-large", trust_remote_code=True, device="cuda")

    # Text embeddings: all unique sanitized texts + neg_san
    all_texts = unique_sanitized + [NEG_SAN_TEXT]
    text_embs = model.encode(all_texts, normalize_embeddings=True)

    sanitized_emb_map = {}
    for i, s in enumerate(unique_sanitized):
        sanitized_emb_map[s] = text_embs[i : i + 1]  # (1, 3584)
    neg_san_emb = text_embs[-1:]  # (1, 3584)

    def get_audio_emb(audio_path):
        arr, _ = librosa.load(audio_path, sr=16000, mono=True)
        emb = model.encode([{"array": arr, "sampling_rate": 16000}], normalize_embeddings=True)
        return emb  # (1, 3584)

elif model_size == "small":
    from transformers import AutoModel, AutoTokenizer
    import torchaudio

    model = AutoModel.from_pretrained("laion/voiceclap-small", trust_remote_code=True).eval().to("cuda")
    tok = AutoTokenizer.from_pretrained("laion/voiceclap-small")

    def encode_text(texts):
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            emb = model.encode_text(enc.input_ids, enc.attention_mask)
        return emb.cpu().numpy()

    sanitized_emb_map = {}
    for s in unique_sanitized:
        sanitized_emb_map[s] = encode_text([s])  # (1, 768)
    neg_san_emb = encode_text([NEG_SAN_TEXT])  # (1, 768)

    def get_audio_emb(audio_path):
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0)  # mono
        with torch.no_grad():
            emb = model.encode_waveform(wav.to("cuda"))
        return emb.cpu().numpy()  # (1, 768)

else:
    raise ValueError(f"Unknown model_size: {model_size}")

print(f"[CLAP-{model_size.upper()} GPU {gpu_id}] Model loaded, {len(unique_sanitized)} sanitized texts + neg_san encoded", flush=True)

# Score all audio files
results = []
for i, item in enumerate(work_items):
    try:
        audio_emb = get_audio_emb(item["audio_path"])
        clap_sanitized = float(audio_emb @ sanitized_emb_map[item["sanitized_text"]].T)
        neg_san = float(audio_emb @ neg_san_emb.T)

        results.append({
            "group_key": item["group_key"],
            "item_idx": item["item_idx"],
            "n": item["n"],
            "audio_path": item["audio_path"],
            "clap_sanitized": clap_sanitized,
            "neg_san": neg_san,
            "status": "ok",
        })
    except Exception as e:
        traceback.print_exc()
        results.append({
            "group_key": item["group_key"],
            "item_idx": item["item_idx"],
            "n": item["n"],
            "audio_path": item["audio_path"],
            "clap_sanitized": 0.0,
            "neg_san": 0.0,
            "status": f"error: {e}",
        })

    if (i + 1) % 20 == 0 or i == len(work_items) - 1:
        print(f"[CLAP-{model_size.upper()} GPU {gpu_id}] {i+1}/{len(work_items)} scored", flush=True)

output_file = work_file.replace(".json", "_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"[CLAP-{model_size.upper()} GPU {gpu_id}] Done. {len(results)} results saved.", flush=True)
