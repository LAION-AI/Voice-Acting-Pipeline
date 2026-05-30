#!/usr/bin/env python3
"""EmoNet emotion scoring worker — BUD-E-Whisper + 40 Empathic Insight Plus MLPs.

Usage: python _emonet_worker.py <gpu_id> <work_file>

Each work item: {"audio_path": str, "item_idx": int, "n": int, "group_key": str}
Writes results to work_file.replace('.json', '_results.json')

The 40 emotion MLPs use flattened encoder output (1500 x 768 = 1,152,000 dim)
as input, NOT the pooled 3072-dim features used by content_enjoyment.
"""
import json, os, sys, time, traceback

gpu_id = int(sys.argv[1])
work_file = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor
from huggingface_hub import hf_hub_download

EMONET_EMOTIONS = [
    "Affection", "Amusement", "Anger", "Astonishment_Surprise", "Awe",
    "Bitterness", "Concentration", "Confusion", "Contemplation", "Contempt",
    "Contentment", "Disappointment", "Disgust", "Distress", "Doubt",
    "Elation", "Embarrassment", "Emotional_Numbness", "Fatigue_Exhaustion",
    "Fear", "Helplessness", "Hope_Enthusiasm_Optimism",
    "Impatience_and_Irritability", "Infatuation", "Interest",
    "Intoxication_Altered_States_of_Consciousness", "Jealousy_&_Envy",
    "Longing", "Malevolence_Malice", "Pain", "Pleasure_Ecstasy", "Pride",
    "Relief", "Sadness", "Sexual_Lust", "Shame", "Sourness", "Teasing",
    "Thankfulness_Gratitude", "Triumph",
]

# Emotion MLPs input: flattened 1500 frames x 768 dim = 1,152,000
EMONET_INPUT_DIM = 1500 * 768  # 1152000


class _EmoNetMLP(torch.nn.Module):
    """MLP matching Empathic Insight Plus emotion experts.

    Same architecture as _PooledEmbeddingMLP but with input_dim=1152000
    (flattened encoder hidden states) instead of 3072 (pooled).
    """

    def __init__(self, input_dim: int = 1152000):
        super().__init__()
        import torch.nn as nn
        self.proj = nn.Linear(input_dim, 64)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.mlp(self.proj(x))


# ── Load models ──────────────────────────────────────────────────────

print(f"[EMONET GPU {gpu_id}] Downloading 40 emotion MLP weights...", flush=True)
emotion_paths = {}
for name in EMONET_EMOTIONS:
    fname = f"model_{name}_best.pth"
    path = hf_hub_download("laion/Empathic-Insight-Voice-Plus", fname)
    emotion_paths[name] = path
print(f"[EMONET GPU {gpu_id}] All weights downloaded.", flush=True)

print(f"[EMONET GPU {gpu_id}] Loading BUD-E-Whisper encoder...", flush=True)
encoder_id = "laion/BUD-E-Whisper"
feature_extractor = WhisperFeatureExtractor.from_pretrained(encoder_id)
encoder = WhisperModel.from_pretrained(encoder_id, torch_dtype=torch.float32).to("cuda")
encoder.eval()

print(f"[EMONET GPU {gpu_id}] Loading 40 emotion MLPs...", flush=True)
scorers = {}
for name, path in emotion_paths.items():
    model = _EmoNetMLP(EMONET_INPUT_DIM)
    state_dict = torch.load(path, map_location="cuda", weights_only=True)
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model = model.to("cuda")
    model.eval()
    scorers[name] = model
print(f"[EMONET GPU {gpu_id}] All models loaded.", flush=True)

# ── Process work items ───────────────────────────────────────────────

with open(work_file) as f:
    work_items = json.load(f)

print(f"[EMONET GPU {gpu_id}] {len(work_items)} items to score", flush=True)

results = []
for i, item in enumerate(work_items):
    try:
        waveform, sr = torchaudio.load(item["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        inputs = feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to("cuda")

        with torch.no_grad():
            encoder_output = encoder.encoder(input_features)
            hidden_states = encoder_output.last_hidden_state  # (1, 1500, 768)
            # Flatten to (1, 1152000) — emotion MLPs take raw flattened output
            flat = hidden_states.view(1, -1)

            emotions = {}
            for name, scorer in scorers.items():
                score = scorer(flat).item()
                emotions[name] = round(score, 6)

        results.append({
            "group_key": item["group_key"],
            "item_idx": item["item_idx"],
            "n": item["n"],
            "emotions": emotions,
            "status": "ok",
        })
    except Exception as e:
        traceback.print_exc()
        results.append({
            "group_key": item["group_key"],
            "item_idx": item["item_idx"],
            "n": item["n"],
            "emotions": {name: 0.0 for name in EMONET_EMOTIONS},
            "status": f"error: {e}",
        })

    if (i + 1) % 20 == 0 or i == len(work_items) - 1:
        print(f"[EMONET GPU {gpu_id}] {i+1}/{len(work_items)} scored", flush=True)

output_file = work_file.replace(".json", "_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"[EMONET GPU {gpu_id}] Done. {len(results)} results saved.", flush=True)
