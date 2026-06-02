#!/usr/bin/env python3
"""
RE-USE Speech Enhancement Postprocessing
==========================================

Processes all generated vocal burst audio through NVIDIA RE-USE, a 9.6M
parameter multilingual speech enhancement model based on the SEMamba
architecture (State Space Model).

RE-USE improves audio quality by removing noise and artifacts while
preserving the vocal characteristics. It operates in the STFT domain:

    1. Load WAV at its native sample rate
    2. Compute STFT (scaled to model's expected n_fft/hop/win)
    3. Pass magnitude + phase through SEMamba model
    4. Apply sweep artifact removal (zero out mostly-silent frequency bins)
    5. Reconstruct via ISTFT
    6. Pad/trim to match original length

Directory mapping
-----------------
    nsfw_comparison/original/  -> nsfw_comparison/original_reuse/
    nsfw_comparison/sulfur/    -> nsfw_comparison/sulfur_reuse/
    nsfw_comparison/sa3/       -> nsfw_comparison/sa3_reuse/
    nsfw_comparison/moss/      -> nsfw_comparison/moss_reuse/
    output/                    -> output_reuse/
    output_sa3/                -> output_sa3_reuse/
    output_moss/               -> output_moss_reuse/

Existing output files are skipped (resume support).

Usage
-----
    python postprocess_reuse.py --multi-gpu           # all 8 GPUs
    python postprocess_reuse.py --gpu 0               # single GPU
    python postprocess_reuse.py --gpu 0 --worker-id 0 --num-workers 8

Prerequisites
-------------
    git clone https://huggingface.co/nvidia/RE-USE
    pip install mamba-ssm   # requires CUDA, build from source if needed
    # Create __init__.py in RE-USE/models/ and RE-USE/utils/ directories

Model
-----
    nvidia/RE-USE (9.6M params, SEMamba architecture)
    Config: USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_...yaml

Dataset
-------
    https://huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

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
REUSE_DIR = Path("/home/deployer/laion/REUSE")  # Adjust to your RE-USE clone path

# Source -> destination directory pairs
DIR_MAPPINGS = [
    (BASE_DIR / "nsfw_comparison" / "original", BASE_DIR / "nsfw_comparison" / "original_reuse"),
    (BASE_DIR / "nsfw_comparison" / "sulfur",   BASE_DIR / "nsfw_comparison" / "sulfur_reuse"),
    (BASE_DIR / "nsfw_comparison" / "sa3",      BASE_DIR / "nsfw_comparison" / "sa3_reuse"),
    (BASE_DIR / "nsfw_comparison" / "moss",     BASE_DIR / "nsfw_comparison" / "moss_reuse"),
    (BASE_DIR / "output",                       BASE_DIR / "output_reuse"),
    (BASE_DIR / "output_sa3",                   BASE_DIR / "output_sa3_reuse"),
    (BASE_DIR / "output_moss",                  BASE_DIR / "output_moss_reuse"),
]


def collect_all_files() -> list[tuple[Path, Path]]:
    """Collect all (input_wav, output_wav) path pairs."""
    pairs = []
    for src_dir, dst_dir in DIR_MAPPINGS:
        if not src_dir.exists():
            continue
        for wav_file in sorted(src_dir.rglob("*.wav")):
            rel = wav_file.relative_to(src_dir)
            out_file = dst_dir / rel
            pairs.append((wav_file, out_file))
    return pairs


def enhance_files(pairs: list[tuple[Path, Path]], gpu_id: int = 0) -> None:
    """Run RE-USE enhancement on a list of (input, output) pairs.

    The enhancement pipeline:
        1. Load audio at native sample rate
        2. Scale STFT params to match the native rate vs model's 16kHz
        3. Forward pass through SEMamba (magnitude + phase)
        4. Remove sweep artifacts (zero out mostly-silent freq bins)
        5. ISTFT to reconstruct waveform
        6. Pad/trim to match original length
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    import torchaudio
    import torch.nn as nn

    # RE-USE model imports
    sys.path.insert(0, str(REUSE_DIR))
    from models.stfts import mag_phase_stft, mag_phase_istft
    from models.generator_SEMamba_time_d4 import SEMamba
    from utils.util import load_config, pad_or_trim_to_match

    # Load model config and weights
    cfg = load_config(str(
        REUSE_DIR / "recipes" /
        "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
    ))
    model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=cfg).to("cuda")
    model.eval()

    # STFT parameters from config
    n_fft = cfg["stft_cfg"]["n_fft"]
    hop_size = cfg["stft_cfg"]["hop_size"]
    win_size = cfg["stft_cfg"]["win_size"]
    compress_factor = cfg["model_cfg"]["compress_factor"]
    sampling_rate = cfg["stft_cfg"]["sampling_rate"]  # model's native rate (16kHz)
    RELU = nn.ReLU()

    def make_even(v):
        """Round to nearest even integer (required for STFT)."""
        v = int(round(v))
        return v if v % 2 == 0 else v + 1

    print(f"[GPU {gpu_id}] RE-USE loaded. Processing {len(pairs)} files...")

    generated = 0
    skipped = 0
    errors = 0

    with torch.no_grad():
        for idx, (in_path, out_path) in enumerate(pairs, 1):
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                skipped += 1
                continue

            try:
                wav, sr = torchaudio.load(str(in_path))
                noisy = wav.to("cuda")

                # Scale STFT params from model's 16kHz to the file's native rate
                n_fft_s = make_even(n_fft * sr // sampling_rate)
                hop_s = make_even(hop_size * sr // sampling_rate)
                win_s = make_even(win_size * sr // sampling_rate)

                # Forward: STFT -> SEMamba -> ISTFT
                noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                    noisy, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
                    compress_factor=compress_factor, center=True, addeps=False,
                )
                amp_g, pha_g, _ = model(noisy_mag, noisy_pha)

                # Sweep artifact removal: zero out frequency bins that are
                # mostly silent (>50% of time frames are zero)
                mag = torch.expm1(RELU(amp_g))
                zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
                amp_g[:, :, (zero_portion > 0.5)[0]] = 0

                audio_g = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, compress_factor)
                audio_g = pad_or_trim_to_match(noisy.detach(), audio_g, pad_value=1e-8)

                torchaudio.save(str(out_path), audio_g.cpu(), sr)
                generated += 1

                if idx % 50 == 0:
                    print(f"[GPU {gpu_id}] [{idx}/{len(pairs)}] generated={generated} skipped={skipped}")

            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR on {in_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1

    print(f"\n[GPU {gpu_id}] Done. generated={generated} skipped={skipped} errors={errors}")


# ---------------------------------------------------------------------------
# Multi-GPU launcher
# ---------------------------------------------------------------------------

def get_available_gpus() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [int(l.strip()) for l in result.stdout.strip().split("\n") if l.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def launch_multi_gpu(num_gpus: int) -> None:
    available = get_available_gpus()
    num_gpus = min(num_gpus, len(available))
    gpu_ids = available[:num_gpus]

    all_pairs = collect_all_files()
    print(f"Total files: {len(all_pairs)}")
    print(f"Launching {num_gpus} RE-USE workers on GPUs: {gpu_ids}")

    script = str(Path(__file__).resolve())
    processes = []
    log_files = []

    for wid, gid in enumerate(gpu_ids):
        log_path = BASE_DIR / f"reuse_worker_gpu{gid}.log"
        log_f = open(log_path, "w")
        log_files.append(log_f)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["LD_LIBRARY_PATH"] = ""
        cmd = [sys.executable, script, "--gpu", str(gid),
               "--worker-id", str(wid), "--num-workers", str(num_gpus)]
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        processes.append((wid, gid, proc))

    while True:
        all_done = all(p.poll() is not None for _, _, p in processes)
        total_enhanced = sum(
            len(list(d.rglob("*.wav"))) for _, d in DIR_MAPPINGS if d.exists()
        )
        print(f"  Progress: {total_enhanced}/{len(all_pairs)}", end="\r")
        if all_done:
            break
        time.sleep(5)

    for f in log_files:
        f.close()

    total = sum(len(list(d.rglob("*.wav"))) for _, d in DIR_MAPPINGS if d.exists())
    print(f"\n\nDone. Total enhanced: {total}/{len(all_pairs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    all_pairs = collect_all_files()

    if args.multi_gpu:
        num_gpus = args.num_gpus if args.num_gpus > 0 else len(get_available_gpus())
        launch_multi_gpu(num_gpus=num_gpus)
        return

    if args.num_workers > 1:
        worker_pairs = [p for i, p in enumerate(all_pairs) if i % args.num_workers == args.worker_id]
    else:
        worker_pairs = all_pairs

    enhance_files(worker_pairs, gpu_id=args.gpu)


if __name__ == "__main__":
    main()
