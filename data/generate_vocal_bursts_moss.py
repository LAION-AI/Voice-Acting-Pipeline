#!/usr/bin/env python3
"""
Vocal Bursts Generation — MOSS SoundEffect v2.0
=================================================

Generates vocal burst samples using MOSS-SoundEffect-v2.0, a 1.3B
parameter DiT text-to-audio model from OpenMOSS.

Prompt format (MOSS)
--------------------
MOSS is a text-to-audio model similar to SA3. Prompts describe the sound:

    "{burst_description}, {age_descriptor}"

Same adaptation as SA3 — the burst description first, age/gender second.

Examples:
    "A deep, uncontrollable laugh that involves the whole body, toddler girl"
    "A soft, breathy, undulating sound of physical pleasure, young man"

Model
-----
    OpenMOSS-Team/MOSS-SoundEffect-v2.0
    Library: moss_soundeffect_v2.MossSoundEffectPipeline
    Sample rate: 48000 Hz
    Steps: 100, cfg_scale: 4.0, bfloat16

Note: The first call on each GPU triggers torch.compile + Triton CUDA
graph compilation, which takes ~90 seconds. Subsequent calls run at
~14 it/s (100 steps in ~7 seconds per sample).

Usage
-----
    python generate_vocal_bursts_moss.py --multi-gpu
    python generate_vocal_bursts_moss.py --multi-gpu --num-gpus 8

Prerequisites
-------------
    git clone https://github.com/OpenMOSS/MOSS-TTS.git
    pip install MOSS-TTS/moss_soundeffect_v2/ --no-deps
    pip install descript-audiotools ftfy einops

Dataset
-------
    https://huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw
"""

import os
import csv
import json
import sys
import time
import traceback
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

# Fix LD_LIBRARY_PATH
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
TAXONOMY = BASE_DIR / "vocal_bursts_taxonomy.json"
OUTPUT_DIR = BASE_DIR / "output_moss"


def adapt_prompt_for_moss(row: dict) -> str:
    """Convert DramaBox prompt to MOSS SFX style.

    Same approach as SA3: description first, age/gender second.
    """
    desc = row["vocal_burst_description"]
    prompt = row["prompt"]
    if " performing " in prompt:
        age_descriptor = prompt.split(" performing ")[0]
        if age_descriptor.startswith("A "):
            age_descriptor = age_descriptor[2:]
    else:
        age_descriptor = f"{row['age_group']} {row['gender']}"
    return f"{desc}, {age_descriptor}"


def get_grid_samples() -> list[dict]:
    """Get the subset of samples shown in the HTML preview grid.

    The grid shows 2 female + 2 male samples per taxonomy entry (197
    entries = 781 samples). This avoids generating the full 2000 when
    only grid samples are needed.
    """
    with open(TAXONOMY) as f:
        raw = json.load(f)
    taxonomy = {k: v for k, v in raw.items() if not k.startswith("_")}

    with open(MANIFEST) as f:
        samples = list(csv.DictReader(f))

    by_key = defaultdict(lambda: {"male": [], "female": []})
    for s in samples:
        by_key[s["vocal_burst_key"]][s["gender"]].append(s)

    grid_samples = []
    for burst_key in taxonomy:
        if burst_key not in by_key:
            continue
        male = by_key[burst_key]["male"][:2]
        female = by_key[burst_key]["female"][:2]
        if not male and not female:
            continue
        grid_samples.extend(female + male)

    return grid_samples


def generate_worker(gpu_id: int, worker_id: int = 0, num_workers: int = 1,
                    grid_only: bool = False):
    """Generate MOSS samples on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    if grid_only:
        all_samples = get_grid_samples()
    else:
        with open(MANIFEST) as f:
            all_samples = list(csv.DictReader(f))

    my_samples = [s for i, s in enumerate(all_samples) if i % num_workers == worker_id]
    total = len(my_samples)
    print(f"[GPU {gpu_id}] Worker {worker_id}/{num_workers}: {total} samples")

    print(f"[GPU {gpu_id}] Loading MOSS-SoundEffect-v2.0...")
    t0 = time.time()
    pipe = MossSoundEffectPipeline.from_pretrained(
        "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    print(f"[GPU {gpu_id}] Model loaded in {time.time()-t0:.1f}s")

    generated = 0
    skipped = 0
    errors = 0

    for idx, s in enumerate(my_samples, 1):
        outpath = Path(s["output_path"].replace("/output/", "/output_moss/"))
        if outpath.exists():
            skipped += 1
            continue

        outpath.parent.mkdir(parents=True, exist_ok=True)
        prompt = adapt_prompt_for_moss(s)
        duration = float(s["duration_s"])

        try:
            audio = pipe(
                prompt=prompt,
                seconds=duration,
                num_inference_steps=100,
                cfg_scale=4.0,
            )
            pipe.save_audio(audio, str(outpath))
            generated += 1

            if idx % 10 == 0 or idx == total:
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


def launch_multi_gpu(num_gpus: int, grid_only: bool = False):
    available = get_available_gpus()
    num_gpus = min(num_gpus, len(available))
    gpu_ids = available[:num_gpus]

    if grid_only:
        total = len(get_grid_samples())
        print(f"Grid-only mode: {total} samples")
    else:
        with open(MANIFEST) as f:
            total = sum(1 for _ in csv.DictReader(f))

    print(f"Launching {num_gpus} MOSS workers on GPUs: {gpu_ids}")

    script = str(Path(__file__).resolve())
    processes = []
    log_files = []

    for wid, gid in enumerate(gpu_ids):
        log_path = BASE_DIR / f"moss_worker_gpu{gid}.log"
        log_f = open(log_path, "w")
        log_files.append(log_f)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["LD_LIBRARY_PATH"] = ""
        cmd = [sys.executable, script, "--gpu", str(gid),
               "--worker-id", str(wid), "--num-workers", str(num_gpus)]
        if grid_only:
            cmd.append("--grid-only")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        processes.append((wid, gid, proc))

    while True:
        all_done = all(p.poll() is not None for _, _, p in processes)
        count = sum(1 for _ in OUTPUT_DIR.rglob("*.wav")) if OUTPUT_DIR.exists() else 0
        print(f"  Progress: {count}/{total}", end="\r")
        if all_done:
            break
        time.sleep(15)

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
    parser.add_argument("--grid-only", action="store_true",
                        help="Only generate the 781 samples needed for the HTML grid")
    args = parser.parse_args()

    if args.multi_gpu:
        launch_multi_gpu(num_gpus=args.num_gpus, grid_only=args.grid_only)
    else:
        generate_worker(args.gpu, args.worker_id, args.num_workers,
                        grid_only=args.grid_only)


if __name__ == "__main__":
    main()
