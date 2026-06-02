#!/usr/bin/env python3
"""
Vocal Bursts Generation — Stable Audio 3 Small SFX
====================================================

Generates vocal burst samples using Stable Audio 3 Small SFX, a
text-to-audio sound effects model from Stability AI.

Prompt format (SA3)
-------------------
SA3 is a sound effects model, NOT a TTS model. It does not understand
performer-framing prompts like DramaBox. Instead, prompts describe the
sound itself:

    "{burst_description}, {age_descriptor}"

Examples:
    "A deep, uncontrollable laugh that involves the whole body, toddler girl"
    "A soft, breathy, undulating sound of physical pleasure, young man"

This is adapted from the DramaBox prompt format:
    DramaBox: "A young man performing Belly Laugh, A deep, uncontrollable laugh..."
    SA3:      "A deep, uncontrollable laugh..., young man"

The description comes first (most important for SFX models), followed by
the age/gender descriptor as a secondary cue.

Model
-----
    cocktailpeanut/stable-audio-3-small-sfx   (ungated copy)
    Library: stable_audio_3.StableAudioModel
    Sample rate: 44100 Hz
    Steps: 8, cfg_scale: 1.0

Usage
-----
    python generate_vocal_bursts_sa3.py --multi-gpu           # all GPUs
    python generate_vocal_bursts_sa3.py --multi-gpu --num-gpus 4

Prerequisites
-------------
    pip install stable-audio-3   # from github.com/Stability-AI/stable-audio-3

Dataset
-------
    https://huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw
"""

import os
import csv
import json
import re
import sys
import time
import traceback
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

# Fix LD_LIBRARY_PATH (conda cuDNN conflicts)
if "LD_LIBRARY_PATH" in os.environ:
    filtered = ":".join(
        p for p in os.environ["LD_LIBRARY_PATH"].split(":")
        if "ml-general" not in p
    )
    if filtered:
        os.environ["LD_LIBRARY_PATH"] = filtered
    else:
        del os.environ["LD_LIBRARY_PATH"]

BASE_DIR = Path(__file__).resolve().parent
MANIFEST = BASE_DIR / "prompts_manifest.csv"
OUTPUT_DIR = BASE_DIR / "output_sa3"


def adapt_prompt_for_sa3(row: dict) -> str:
    """Convert a DramaBox-style prompt to SA3 SFX-style.

    DramaBox: "A young adult woman performing belly_laugh, A deep, uncontrollable laugh..."
    SA3:      "A deep, uncontrollable laugh..., young adult woman"

    The burst description goes first (most salient for SFX models),
    followed by the age/gender descriptor as a secondary modifier.
    """
    desc = row["vocal_burst_description"]
    prompt = row["prompt"]

    # Extract the age descriptor from "A {descriptor} performing ..."
    if " performing " in prompt:
        age_descriptor = prompt.split(" performing ")[0]
        if age_descriptor.startswith("A "):
            age_descriptor = age_descriptor[2:]
    else:
        # Fallback: construct from metadata
        age_descriptor = f"{row['age_group']} {row['gender']}"

    return f"{desc}, {age_descriptor}"


def generate_worker(gpu_id: int, worker_id: int = 0, num_workers: int = 1):
    """Generate SA3 samples on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import soundfile as sf
    from stable_audio_3 import StableAudioModel

    # Load manifest
    with open(MANIFEST) as f:
        samples = list(csv.DictReader(f))

    # Shard across workers
    my_samples = [s for i, s in enumerate(samples) if i % num_workers == worker_id]
    total = len(my_samples)

    print(f"[GPU {gpu_id}] Loading SA3 'small-sfx'...")
    t0 = time.time()
    model = StableAudioModel.from_pretrained("small-sfx", device="cuda")
    sample_rate = model.model_config["sample_rate"]  # 44100
    print(f"[GPU {gpu_id}] Model loaded in {time.time()-t0:.1f}s (sr={sample_rate})")

    generated = 0
    skipped = 0
    errors = 0

    for idx, s in enumerate(my_samples, 1):
        # Mirror output/ -> output_sa3/ directory structure
        outpath = Path(s["output_path"].replace("/output/", "/output_sa3/"))
        if outpath.exists():
            skipped += 1
            continue

        outpath.parent.mkdir(parents=True, exist_ok=True)
        prompt = adapt_prompt_for_sa3(s)
        duration = float(s["duration_s"])

        try:
            # SA3 returns [batch, channels, samples] tensor
            audio = model.generate(
                prompt=prompt,
                duration=duration,
                steps=8,
                cfg_scale=1.0,
                seed=42,
            )
            sf.write(str(outpath), audio[0].cpu().float().numpy().T, sample_rate)
            generated += 1

            if idx % 50 == 0 or idx == total:
                print(f"[GPU {gpu_id}] [{idx}/{total}] generated={generated} skipped={skipped}")

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR on {outpath.name}: {e}")
            traceback.print_exc()
            errors += 1

    print(f"\n[GPU {gpu_id}] Done. generated={generated} skipped={skipped} errors={errors}")


def get_available_gpus() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [int(l.strip()) for l in result.stdout.strip().split("\n") if l.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def launch_multi_gpu(num_gpus: int):
    available = get_available_gpus()
    num_gpus = min(num_gpus, len(available))
    gpu_ids = available[:num_gpus]

    with open(MANIFEST) as f:
        total = sum(1 for _ in csv.DictReader(f))

    print(f"Total samples: {total}")
    print(f"Launching {num_gpus} SA3 workers on GPUs: {gpu_ids}")

    script = str(Path(__file__).resolve())
    processes = []
    log_files = []

    for wid, gid in enumerate(gpu_ids):
        log_path = BASE_DIR / f"sa3_worker_gpu{gid}.log"
        log_f = open(log_path, "w")
        log_files.append(log_f)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["LD_LIBRARY_PATH"] = ""
        cmd = [sys.executable, script, "--gpu", str(gid),
               "--worker-id", str(wid), "--num-workers", str(num_gpus)]
        print(f"  Worker {wid} -> GPU {gid}")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        processes.append((wid, gid, proc))

    while True:
        all_done = all(p.poll() is not None for _, _, p in processes)
        count = sum(1 for _ in OUTPUT_DIR.rglob("*.wav")) if OUTPUT_DIR.exists() else 0
        print(f"  Progress: {count}/{total}", end="\r")
        if all_done:
            break
        time.sleep(10)

    for f in log_files:
        f.close()

    final = sum(1 for _ in OUTPUT_DIR.rglob("*.wav")) if OUTPUT_DIR.exists() else 0
    print(f"\n\nAll workers finished. Total: {final}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    if args.multi_gpu:
        launch_multi_gpu(num_gpus=args.num_gpus)
    else:
        generate_worker(args.gpu, args.worker_id, args.num_workers)


if __name__ == "__main__":
    main()
