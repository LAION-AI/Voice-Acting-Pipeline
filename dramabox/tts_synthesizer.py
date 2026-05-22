"""Multi-GPU DramaBox TTS audio synthesis.

Distributes prompts across GPUs, each running a TTSServer instance.
Supports raw generation and optional self-voice-cloning (self-VC) second pass.

Requires the DramaBox repository to be installed or on sys.path:
    pip install -e /path/to/DramaBox
"""
import csv
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def tts_worker(gpu_id: int, prompts: list[tuple[int, str]],
               output_dir: str, config: dict):
    """Run TTS on a single GPU for a list of (sample_idx, prompt) pairs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tts_cfg = config.get("tts", {})
    dramabox_dir = tts_cfg.get("dramabox_dir", "")

    # Add DramaBox to path if specified
    if dramabox_dir:
        sys.path.insert(0, os.path.join(dramabox_dir, "src"))
        sys.path.insert(0, os.path.join(dramabox_dir, "ltx2"))

    from inference_server import TTSServer
    from model_downloader import get_all_paths

    paths = get_all_paths()
    log = logging.getLogger(f"gpu{gpu_id}")
    log.info(f"GPU {gpu_id}: loading TTSServer for {len(prompts)} prompts...")

    server = TTSServer(
        checkpoint=paths["transformer"],
        full_checkpoint=paths["audio_components"],
        gemma_root=paths["gemma_root"],
        device="cuda",
        dtype="bf16",
        compile_model=tts_cfg.get("compile", True),
        bnb_4bit=tts_cfg.get("bnb_4bit", True),
    )
    log.info(f"GPU {gpu_id}: TTSServer loaded, starting generation")

    cfg_scale = tts_cfg.get("cfg_scale", 2.0)
    stg_scale = tts_cfg.get("stg_scale", 1.5)
    dur_mult = tts_cfg.get("duration_multiplier", 1.1)
    seed = tts_cfg.get("seed", 42)
    self_vc = tts_cfg.get("self_vc", True)
    ref_duration = tts_cfg.get("ref_duration", 10.0)
    watermark = tts_cfg.get("watermark", False)

    results = []
    for i, (sample_idx, prompt) in enumerate(prompts):
        tag = f"sample_{sample_idx:06d}"
        raw_path = os.path.join(output_dir, f"{tag}_raw.wav")
        vc_path = os.path.join(output_dir, f"{tag}_selfvc.wav")

        if os.path.exists(raw_path) and (not self_vc or os.path.exists(vc_path)):
            log.info(f"GPU {gpu_id}: [{i+1}/{len(prompts)}] {tag} exists, skipping")
            results.append({"sample_idx": sample_idx, "raw": raw_path,
                            "vc": vc_path, "status": "skipped"})
            continue

        try:
            t0 = time.time()
            server.generate_to_file(
                prompt=prompt, output=raw_path,
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                duration_multiplier=dur_mult, gen_duration=0.0,
                seed=seed, watermark=watermark,
            )
            raw_time = time.time() - t0

            vc_time = 0.0
            if self_vc:
                t1 = time.time()
                server.generate_to_file(
                    prompt=prompt, output=vc_path,
                    voice_ref=raw_path,
                    cfg_scale=cfg_scale, stg_scale=stg_scale,
                    duration_multiplier=dur_mult, gen_duration=0.0,
                    ref_duration=ref_duration,
                    seed=seed, watermark=watermark,
                )
                vc_time = time.time() - t1

            log.info(f"GPU {gpu_id}: [{i+1}/{len(prompts)}] {tag} done "
                     f"(raw={raw_time:.1f}s, vc={vc_time:.1f}s)")
            results.append({"sample_idx": sample_idx, "raw": raw_path,
                            "vc": vc_path, "status": "ok"})

        except Exception as e:
            log.error(f"GPU {gpu_id}: [{i+1}/{len(prompts)}] {tag} FAILED: {e}")
            traceback.print_exc()
            results.append({"sample_idx": sample_idx, "raw": raw_path,
                            "vc": vc_path, "status": f"error: {e}"})

    results_path = os.path.join(output_dir, f"results_gpu{gpu_id}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"GPU {gpu_id}: finished all {len(prompts)} prompts")


def synthesize_from_csv(csv_path: str, config: dict) -> Path:
    """Synthesize audio from a CSV file containing dramabox_prompt column.

    Distributes prompts across GPUs in round-robin fashion.

    Returns the output directory path.
    """
    tts_cfg = config.get("tts", {})
    output_cfg = config.get("output", {})
    gpus = tts_cfg.get("gpus", [0])
    stagger = tts_cfg.get("stagger_start_seconds", 2)

    output_dir = Path(output_cfg.get("output_dir", "./output")) / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Build (index, prompt) pairs
    idx_field = "global_idx" if "global_idx" in rows[0] else "sample_idx"
    prompts = [(int(row[idx_field]), row["dramabox_prompt"]) for row in rows]
    print(f"Loaded {len(prompts)} prompts from {csv_path}")

    # Distribute across GPUs (round-robin)
    gpu_prompts = {g: [] for g in gpus}
    for i, p in enumerate(prompts):
        gpu_prompts[gpus[i % len(gpus)]].append(p)

    for g in gpus:
        print(f"  GPU {g}: {len(gpu_prompts[g])} prompts")

    # Launch workers
    processes = []
    for gpu_id in gpus:
        if not gpu_prompts[gpu_id]:
            continue
        p = mp.Process(
            target=tts_worker,
            args=(gpu_id, gpu_prompts[gpu_id], str(output_dir), config),
        )
        p.start()
        processes.append((gpu_id, p))
        time.sleep(stagger)

    for gpu_id, p in processes:
        p.join()
        print(f"GPU {gpu_id} worker finished (exit code {p.exitcode})")

    print(f"\nDone! Audio files in: {output_dir}")
    return output_dir
