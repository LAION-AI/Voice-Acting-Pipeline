#!/usr/bin/env python3
"""LavaSR BWE worker subprocess for parallel GPU processing.

Usage: python _lavasr_worker.py <gpu_id> <work_file>

Each work item: {"input_path": str, "output_path": str}
"""
import json, os, sys, shutil, traceback

gpu_id = int(sys.argv[1])
work_file = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio

# Vocos monkey-patch: MelSpectrogramFeatures needs f_min/f_max/norm/mel_scale
import vocos.feature_extractors as _vfe
_OrigMSF = _vfe.MelSpectrogramFeatures

class _PatchedMSF(_OrigMSF):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256,
                 n_mels=100, padding="center", f_min=None, f_max=None,
                 norm=None, mel_scale=None):
        super(_OrigMSF, self).__init__()
        if padding not in ("center", "same"):
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        mel_kwargs = dict(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                          n_mels=n_mels, center=(padding == "center"), power=1)
        if f_min is not None: mel_kwargs["f_min"] = f_min
        if f_max is not None: mel_kwargs["f_max"] = f_max
        if norm is not None: mel_kwargs["norm"] = norm
        if mel_scale is not None: mel_kwargs["mel_scale"] = mel_scale
        self.mel_spec = torchaudio.transforms.MelSpectrogram(**mel_kwargs)

_vfe.MelSpectrogramFeatures = _PatchedMSF

from LavaSR.model import LavaEnhance2

with open(work_file) as f:
    work_items = json.load(f)

print(f"[LAVASR GPU {gpu_id}] {len(work_items)} items to enhance", flush=True)

model = LavaEnhance2("YatharthS/LavaSR", device="cuda:0")

for i, item in enumerate(work_items):
    try:
        wav, sr = model.load_audio(item["input_path"], input_sr=16000)
        output = model.enhance(wav, enhance=True, denoise=False)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        torchaudio.save(item["output_path"], output.cpu(), 48000)
    except Exception as e:
        traceback.print_exc()
        shutil.copy2(item["input_path"], item["output_path"])

    if (i + 1) % 10 == 0 or i == len(work_items) - 1:
        print(f"[LAVASR GPU {gpu_id}] {i+1}/{len(work_items)} enhanced", flush=True)

print(f"[LAVASR GPU {gpu_id}] Done.", flush=True)
