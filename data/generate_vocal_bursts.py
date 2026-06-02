#!/usr/bin/env python3
"""
Vocal Bursts Generation Pipeline — DramaBox TTS
=================================================

Generates 2000 synthetic vocal burst audio samples using DramaBox TTS,
spanning 202 vocal burst types across 16 age/gender groups.

Prompt format
-------------
DramaBox is prompted with a single text string:

    "A {age_descriptor} performing {burst_key}, {burst_description}"

Examples:
    "A toddler girl performing Belly Laugh, A deep, uncontrollable laugh
     that involves the whole body, originating from the diaphragm."
    "A young man performing Orgasmic Cry, A loud, uncontrolled, rising
     vocalization at the peak of sexual climax."
    "A elderly woman performing Exasperated Sigh, A heavy, audible exhale
     expressing deep frustration or weariness."

The model receives NO voice reference audio — the age/gender is conveyed
purely through the text prompt. Duration is randomised between 3–12 s.

Taxonomy
--------
Two taxonomy files are used:

- vocal_bursts_taxonomy.json      — 202 entries (full, including NSFW)
- vocal_bursts_taxonomy_sfw.json  — 180 entries (safe for minors)

Minor age groups (toddler, pre-puberty, teenage) draw from the SFW
taxonomy only; adult age groups draw from the full extended taxonomy.

Output
------
2000 WAV files (1000 male + 1000 female) organised as:

    output/{gender}/{age_group}/{id}_{burst_key}.wav

Each age group gets 125 samples (1000 / 8 age groups per gender).
Taxonomy entries are shuffled per age group and cycled to fill all slots.

A CSV manifest (prompts_manifest.csv) is written alongside the audio,
recording the prompt, metadata, and output path for every sample.

DramaBox generation parameters:
    - gen_duration: random 3–12 s per sample
    - watermark: False
    - cfg: 2.5, stg: 1.5, 30 steps (DramaBox defaults)
    - No voice reference audio

Multi-GPU
---------
Supports sharded parallel generation across N GPUs:

    python generate_vocal_bursts.py --multi-gpu           # all GPUs
    python generate_vocal_bursts.py --multi-gpu --num-gpus 4

Each worker loads DramaBox independently and processes every N-th sample
from the manifest.

Prerequisites
-------------
    git clone https://github.com/resemble-ai/DramaBox ../DramaBox
    pip install -r ../DramaBox/requirements.txt

Dataset
-------
Generated samples are available at:
    https://huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw
"""

import json
import random
import os
import csv
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fix cuDNN version conflict from stale LD_LIBRARY_PATH entries.
# This is specific to conda environments that bundle their own cuDNN.
# ---------------------------------------------------------------------------
if "LD_LIBRARY_PATH" in os.environ:
    filtered = ":".join(
        p for p in os.environ["LD_LIBRARY_PATH"].split(":")
        if "ml-general" not in p
    )
    if filtered:
        os.environ["LD_LIBRARY_PATH"] = filtered
    else:
        del os.environ["LD_LIBRARY_PATH"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
EXTENDED_TAXONOMY_PATH = BASE_DIR / "vocal_bursts_taxonomy.json"
SFW_TAXONOMY_PATH = BASE_DIR / "vocal_bursts_taxonomy_sfw.json"
MANIFEST_PATH = BASE_DIR / "prompts_manifest.csv"
OUTPUT_DIR = BASE_DIR / "output"

SAMPLES_PER_GENDER = 1000          # 1000 male + 1000 female = 2000 total
AGE_GROUPS_PER_GENDER = 8          # 8 age brackets per gender
SAMPLES_PER_AGE_GROUP = SAMPLES_PER_GENDER // AGE_GROUPS_PER_GENDER  # = 125

MIN_DURATION_S = 3                 # Shortest sample duration (seconds)
MAX_DURATION_S = 12                # Longest sample duration (seconds)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Age group definitions
#
# Each group has:
#   key         – filesystem-safe identifier
#   descriptor  – human-readable label inserted into the prompt
#   age_range   – approximate real-world age range
#   is_minor    – if True, only SFW taxonomy entries are used
# ---------------------------------------------------------------------------

FEMALE_AGE_GROUPS = [
    {"key": "toddler_girl",       "descriptor": "toddler girl",       "age_range": "2-6",   "is_minor": True},
    {"key": "pre_puberty_girl",   "descriptor": "pre-puberty girl",   "age_range": "7-12",  "is_minor": True},
    {"key": "teenage_girl",       "descriptor": "teenage girl",       "age_range": "13-17", "is_minor": True},
    {"key": "young_woman",        "descriptor": "young woman",        "age_range": "18-30", "is_minor": False},
    {"key": "middle_aged_woman",  "descriptor": "middle-aged woman",  "age_range": "31-50", "is_minor": False},
    {"key": "mature_woman",       "descriptor": "mature woman",       "age_range": "51-65", "is_minor": False},
    {"key": "elderly_woman",      "descriptor": "elderly woman",      "age_range": "66-80", "is_minor": False},
    {"key": "senescent_woman",    "descriptor": "senescent woman",    "age_range": "80+",   "is_minor": False},
]

MALE_AGE_GROUPS = [
    {"key": "toddler_boy",       "descriptor": "toddler boy",       "age_range": "2-6",   "is_minor": True},
    {"key": "pre_puberty_boy",   "descriptor": "pre-puberty boy",   "age_range": "7-12",  "is_minor": True},
    {"key": "teenage_boy",       "descriptor": "teenage boy",       "age_range": "13-17", "is_minor": True},
    {"key": "young_man",         "descriptor": "young man",         "age_range": "18-30", "is_minor": False},
    {"key": "middle_aged_man",   "descriptor": "middle-aged man",   "age_range": "31-50", "is_minor": False},
    {"key": "mature_man",        "descriptor": "mature man",        "age_range": "51-65", "is_minor": False},
    {"key": "elderly_man",       "descriptor": "elderly man",       "age_range": "66-80", "is_minor": False},
    {"key": "senescent_man",     "descriptor": "senescent man",     "age_range": "80+",   "is_minor": False},
]

# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_taxonomy(path: Path) -> dict:
    """Load a taxonomy JSON and return {burst_key: burst_description}.

    Skips internal keys that start with '_' (e.g. _meta, __category_*).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def sanitize_filename(name: str) -> str:
    """Convert a vocal burst key like 'Belly Laugh' into 'belly_laugh'."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:60]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(age_descriptor: str, burst_key: str, burst_description: str) -> str:
    """Build the DramaBox text prompt for one vocal burst sample.

    Format:
        "A {age_descriptor} performing {burst_key}, {burst_description}"

    DramaBox interprets this as a stage direction: it synthesises the
    described vocalisation with vocal qualities matching the age and gender
    implied by the descriptor — no voice reference audio is needed.
    """
    return f"A {age_descriptor} performing {burst_key}, {burst_description}"


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def generate_age_group_samples(
    age_group: dict,
    taxonomy: dict,
    gender: str,
    rng: random.Random,
    start_id: int,
) -> list[dict]:
    """Create 125 manifest rows for one age group.

    The taxonomy entries are shuffled and cycled to ensure coverage even
    when there are more sample slots than taxonomy entries. Each sample
    gets a random duration between MIN_DURATION_S and MAX_DURATION_S.
    """
    entries = list(taxonomy.items())
    rng.shuffle(entries)

    samples = []
    for i in range(SAMPLES_PER_AGE_GROUP):
        burst_key, burst_desc = entries[i % len(entries)]
        duration_s = round(rng.uniform(MIN_DURATION_S, MAX_DURATION_S), 1)
        sample_id = start_id + i
        safe_key = sanitize_filename(burst_key)
        output_path = str(
            OUTPUT_DIR / gender / age_group["key"] / f"{sample_id:04d}_{safe_key}.wav"
        )
        prompt = build_prompt(age_group["descriptor"], burst_key, burst_desc)

        samples.append({
            "id": sample_id,
            "prompt": prompt,
            "output_path": output_path,
            "gender": gender,
            "age_group": age_group["key"],
            "age_descriptor": age_group["descriptor"],
            "age_range": age_group["age_range"],
            "vocal_burst_key": burst_key,
            "vocal_burst_description": burst_desc,
            "duration_s": duration_s,
        })

    return samples


def build_manifest(seed: int = RANDOM_SEED) -> list[dict]:
    """Build the complete 2000-row prompt manifest.

    Iterates over all gender × age-group combinations, using the SFW
    taxonomy for minors and the full extended taxonomy for adults.
    """
    extended_taxonomy = load_taxonomy(EXTENDED_TAXONOMY_PATH)
    sfw_taxonomy = load_taxonomy(SFW_TAXONOMY_PATH)

    rng = random.Random(seed)
    all_samples = []
    current_id = 1

    for gender, age_groups in [("female", FEMALE_AGE_GROUPS), ("male", MALE_AGE_GROUPS)]:
        for age_group in age_groups:
            # Minors get the SFW-only taxonomy (no NSFW vocal bursts)
            taxonomy = sfw_taxonomy if age_group["is_minor"] else extended_taxonomy
            samples = generate_age_group_samples(
                age_group=age_group,
                taxonomy=taxonomy,
                gender=gender,
                rng=rng,
                start_id=current_id,
            )
            all_samples.extend(samples)
            current_id += SAMPLES_PER_AGE_GROUP

    return all_samples


def write_manifest(samples: list[dict]) -> None:
    """Write the manifest to CSV."""
    fieldnames = [
        "id", "prompt", "output_path", "gender", "age_group",
        "vocal_burst_key", "vocal_burst_description", "duration_s",
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    print(f"Manifest written: {MANIFEST_PATH} ({len(samples)} rows)")


# ---------------------------------------------------------------------------
# Audio generation via DramaBox
# ---------------------------------------------------------------------------

def generate_audio(samples: list[dict], gpu_id: int = 0) -> None:
    """Run DramaBox TTS inference for each sample on a single GPU.

    Each sample is generated with:
        server.generate_to_file(
            prompt   = "A {age_descriptor} performing {burst_key}, {description}",
            output   = "/path/to/output.wav",
            gen_duration = {3.0 .. 12.0},
            watermark = False,
        )

    Existing output files are skipped (supports resuming interrupted runs).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # DramaBox lives in a sibling directory
    dramabox_dir = Path(__file__).resolve().parent.parent / "DramaBox"
    if dramabox_dir.exists():
        sys.path.insert(0, str(dramabox_dir))

    try:
        from src.inference_server import TTSServer
    except ImportError:
        print(
            "ERROR: DramaBox not found. Install from:\n"
            "  git clone https://github.com/resemble-ai/DramaBox ../DramaBox\n"
            "  pip install -r ../DramaBox/requirements.txt"
        )
        raise SystemExit(1)

    print(f"[GPU {gpu_id}] Initialising DramaBox TTSServer...")
    server = TTSServer(device="cuda")

    total = len(samples)
    generated = 0
    skipped = 0
    errors = 0

    for idx, sample in enumerate(samples, 1):
        output_path = Path(sample["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip already-generated files (resume support)
        if output_path.exists():
            skipped += 1
            continue

        print(
            f"[GPU {gpu_id}] [{idx}/{total}] "
            f"{sample['vocal_burst_key']} ({sample['gender']}/{sample['age_group']}) "
            f"duration={sample['duration_s']}s"
        )

        try:
            server.generate_to_file(
                prompt=sample["prompt"],
                output=str(output_path),
                gen_duration=sample["duration_s"],
                watermark=False,
            )
            generated += 1
        except Exception as e:
            import traceback
            print(f"  [GPU {gpu_id}] ERROR: {e}")
            traceback.print_exc()
            errors += 1

    print(f"\n[GPU {gpu_id}] Done. generated={generated} skipped={skipped} errors={errors}")


# ---------------------------------------------------------------------------
# Multi-GPU launcher
# ---------------------------------------------------------------------------

def get_available_gpus() -> list[int]:
    """Detect available NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [int(l.strip()) for l in result.stdout.strip().split("\n") if l.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def launch_multi_gpu(num_gpus: int, seed: int) -> None:
    """Spawn one worker subprocess per GPU, each handling its slice.

    Worker i processes samples[i::N] from the manifest. Each worker
    loads DramaBox independently onto its assigned GPU.
    """
    available = get_available_gpus()
    num_gpus = min(num_gpus, len(available))
    gpu_ids = available[:num_gpus]

    print(f"Launching {num_gpus} workers on GPUs: {gpu_ids}")

    script = str(Path(__file__).resolve())
    processes = []
    log_files = []

    for worker_id, gpu_id in enumerate(gpu_ids):
        log_path = BASE_DIR / f"worker_gpu{gpu_id}.log"
        log_f = open(log_path, "w")
        log_files.append(log_f)

        cmd = [
            sys.executable, script,
            "--gpu", str(gpu_id),
            "--worker-id", str(worker_id),
            "--num-workers", str(num_gpus),
            "--seed", str(seed),
        ]
        print(f"  Worker {worker_id} -> GPU {gpu_id} (log: {log_path})")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((worker_id, gpu_id, proc))

    print(f"\nAll {num_gpus} workers launched. Monitoring...\n")
    while True:
        all_done = all(proc.poll() is not None for _, _, proc in processes)
        wav_count = len(list(OUTPUT_DIR.rglob("*.wav"))) if OUTPUT_DIR.exists() else 0
        print(f"  Progress: {wav_count}/2000 wav files generated", end="\r")
        if all_done:
            break
        time.sleep(5)

    for f in log_files:
        f.close()

    wav_count = len(list(OUTPUT_DIR.rglob("*.wav"))) if OUTPUT_DIR.exists() else 0
    print(f"\n\nAll workers finished. Total wav files: {wav_count}/2000")


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def print_manifest_stats(samples: list[dict]) -> None:
    """Print summary statistics and a SFW-filter sanity check."""
    print(f"\n{'='*60}\nMANIFEST STATISTICS\n{'='*60}")
    print(f"Total samples: {len(samples)}")

    # Gender breakdown
    gender_counts = {}
    for s in samples:
        gender_counts[s["gender"]] = gender_counts.get(s["gender"], 0) + 1
    print(f"\nBy gender:")
    for g, c in sorted(gender_counts.items()):
        print(f"  {g}: {c}")

    # Age group breakdown
    print(f"\nBy age group:")
    age_counts = {}
    for s in samples:
        key = f"{s['gender']}/{s['age_group']}"
        age_counts[key] = age_counts.get(key, 0) + 1
    for ag, c in sorted(age_counts.items()):
        print(f"  {ag}: {c}")

    # SFW filter: verify no NSFW entries slipped into minor age groups
    nsfw_keywords = {
        "sensual", "erotic", "orgasmic", "lustful", "seductive", "intimate",
        "passionate kiss", "deep sensual moan", "pleasured whimper",
        "heavy panting (intimate)", "lustful growl", "seductive purr",
        "erotic breath catch", "orgasmic cry", "intimate wet kiss",
        "tender post-climax sigh", "exaggerated smooch",
    }
    minor_groups = {
        ag["key"] for ag in FEMALE_AGE_GROUPS + MALE_AGE_GROUPS if ag["is_minor"]
    }
    violations = [
        s for s in samples
        if s["age_group"] in minor_groups
        and any(kw in s["vocal_burst_key"].lower() or kw in s["vocal_burst_description"].lower()
                for kw in nsfw_keywords)
    ]
    print(f"\nSFW filter check: ", end="")
    if violations:
        print(f"FAIL — {len(violations)} violations!")
    else:
        print("PASS")

    # Duration stats
    durations = [s["duration_s"] for s in samples]
    print(f"\nDuration range: {min(durations):.1f}s – {max(durations):.1f}s")
    print(f"Duration mean:  {sum(durations)/len(durations):.1f}s")

    # Sample prompts
    print(f"\nSample prompts:")
    rng = random.Random(0)
    for s in rng.sample(samples, min(5, len(samples))):
        print(f"  [{s['id']}] {s['prompt'][:120]}...")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate vocal burst audio samples using DramaBox TTS"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate manifest only, skip audio")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Parallel generation across all GPUs")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (0 = all available)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    # Always build the manifest first
    print("Building prompt manifest...")
    samples = build_manifest(seed=args.seed)
    write_manifest(samples)

    if args.dry_run:
        print_manifest_stats(samples)
        print("\n--dry-run: Skipping audio generation.")
        return

    # Create output directory structure
    for sample in samples:
        Path(sample["output_path"]).parent.mkdir(parents=True, exist_ok=True)

    if args.multi_gpu:
        print_manifest_stats(samples)
        num_gpus = args.num_gpus if args.num_gpus > 0 else len(get_available_gpus())
        launch_multi_gpu(num_gpus=num_gpus, seed=args.seed)
        return

    # Single worker mode (possibly one of N parallel workers)
    if args.num_workers > 1:
        worker_samples = [s for i, s in enumerate(samples) if i % args.num_workers == args.worker_id]
        print(f"Worker {args.worker_id}/{args.num_workers}: {len(worker_samples)} samples on GPU {args.gpu}")
    else:
        worker_samples = samples
        print_manifest_stats(samples)

    generate_audio(worker_samples, gpu_id=args.gpu)


if __name__ == "__main__":
    main()
