#!/usr/bin/env python3
"""
DramaBox Post-Processing Pipeline
===================================

Downloads tars from laion/dramabox-voice-acting-data, applies:
  1. RE-USE speech enhancement (skip for singing/humming prompts)
  2. LavaSR super-resolution
  3. Whisper turbo ASR with word-level timestamps
  4. CUT TO: scene split detection (longest silence gap)
  5. Audio split into part1 + part2, plus concatenated full audio
  6. VoiceCLAP Large embeddings for part1, part2, and full
  7. MP3 conversion (256kbps mono 48kHz) for all three variants

Uploads processed files to laion/dramabox-voice-acting-data-annotated.

Usage:
    python dramabox_postprocess.py                     # Full run, 8 GPUs
    python dramabox_postprocess.py --num-gpus 4        # 4 GPUs
    python dramabox_postprocess.py --test              # 1 tar, 1 GPU
    python dramabox_postprocess.py --worker --gpu 0 --work-file work_0.json
"""

import io
import os
import re
import sys
import json
import time
import base64
import shutil
import struct
import signal
import tarfile
import logging
import argparse
import tempfile
import traceback
import subprocess
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Fix LD_LIBRARY_PATH (conda cuDNN conflicts)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("postprocess")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
REUSE_DIR = Path("/home/deployer/laion/REUSE")
OUTPUT_DIR = BASE_DIR / "dramabox_postprocess_output"
STATE_FILE = OUTPUT_DIR / "postprocess_state.json"
PROGRESS_DIR = OUTPUT_DIR / "progress"

SRC_REPO = "laion/dramabox-voice-acting-data"
DST_REPO = "laion/dramabox-voice-acting-data-annotated"

MP3_BITRATE = "256k"
MP3_SAMPLE_RATE = 48000
CLAP_BATCH_SIZE = 16  # Files to accumulate before batching VoiceCLAP embeddings

# ---------------------------------------------------------------------------
# Singing/humming/whistling keyword detection
# ---------------------------------------------------------------------------
# Match performed vocal actions, not ambient sound descriptions
SINGING_PATTERN = re.compile(
    r'\b(sing(?:s|ing|er)?|'
    r'hum(?:s|ming|med)|'  # skip bare "hum" (often noun)
    r'whistl(?:e|es|ing|ed)|'
    r'lullaby|chant(?:s|ing|ed)?|serenade[sd]?|'
    r'yodel(?:s|ing|ed)?|croon(?:s|ing|ed)?|'
    r'warbl(?:e|es|ing|ed)?)\b',
    re.IGNORECASE,
)

# Bare "hum" only when preceded by action context
BARE_HUM_PATTERN = re.compile(
    r'\b(?:begins?\s+to|starts?\s+to|a\s+soft|softly|gently|quietly)\s+hum\b',
    re.IGNORECASE,
)


def is_singing_prompt(prompt_text: str) -> bool:
    """Check if prompt describes singing/humming/whistling performance."""
    if SINGING_PATTERN.search(prompt_text):
        return True
    if BARE_HUM_PATTERN.search(prompt_text):
        return True
    return False


# ---------------------------------------------------------------------------
# Extract expected scene text from prompt (text inside double quotes)
# ---------------------------------------------------------------------------
def extract_scene_texts(prompt: str) -> tuple[str, str]:
    """Extract quoted dialogue from before and after CUT TO: in the prompt."""
    parts = re.split(r'\bCUT\s+TO\s*:', prompt, maxsplit=1)
    if len(parts) < 2:
        return "", ""

    def get_quotes(text):
        matches = re.findall(r'"([^"]+)"', text)
        return " ".join(matches)

    return get_quotes(parts[0]), get_quotes(parts[1])


def extract_expected_text(prompt: str) -> str:
    """Extract all quoted dialogue from a DramaBox prompt (full text)."""
    matches = re.findall(r'"([^"]*)"', prompt)
    if not matches:
        matches = re.findall(r'\u201c([^\u201d]*)\u201d', prompt)
    return " ".join(matches).strip()


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate via edit distance at the word level.

    Returns WER as a float (0.0 = perfect, >1.0 = more errors than words).
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n = len(ref_words)
    m = len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[n][m] / n


# ---------------------------------------------------------------------------
# CLAP aesthetics text for cosine similarity scoring
# ---------------------------------------------------------------------------
AESTHETICS_TEXT = ("Realistic, genuine, spotanoues, authentic, sensual, natural voice "
                   "with all imperfections and organic microdistractions a natural "
                   "situation brings with it")


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# CUT TO: split detection via word timestamps
# ---------------------------------------------------------------------------
def find_split_point(word_timestamps: list[dict], total_duration: float) -> tuple[float, str]:
    """
    Find the CUT TO: transition point by locating the longest silence gap
    in the middle 20-80% of the audio.

    Returns (split_time_sec, method).
    """
    if not word_timestamps or len(word_timestamps) < 4:
        return total_duration / 2.0, "midpoint_fallback"

    # Compute gaps between consecutive words
    gaps = []
    for i in range(1, len(word_timestamps)):
        gap_start = word_timestamps[i - 1]["end"]
        gap_end = word_timestamps[i]["start"]
        gap_len = gap_end - gap_start
        gap_mid = (gap_start + gap_end) / 2.0
        gaps.append((gap_len, gap_mid, i))

    # Filter to middle region (20%-80% of total duration)
    lo = total_duration * 0.20
    hi = total_duration * 0.80
    middle_gaps = [(g, mid, i) for g, mid, i in gaps if lo <= mid <= hi]

    if not middle_gaps:
        # Widen to 10%-90%
        lo = total_duration * 0.10
        hi = total_duration * 0.90
        middle_gaps = [(g, mid, i) for g, mid, i in gaps if lo <= mid <= hi]

    if not middle_gaps:
        return total_duration / 2.0, "midpoint_fallback"

    # Find the longest gap in the middle region
    best_gap, best_mid, best_idx = max(middle_gaps, key=lambda x: x[0])

    if best_gap < 0.3:
        # No clear gap found, use midpoint
        return total_duration / 2.0, "midpoint_fallback"

    return best_mid, "silence_gap"


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------
def apply_fade(audio_np: np.ndarray, sr: int, fade_ms: int = 50) -> np.ndarray:
    """Apply fade-in at start and fade-out at end."""
    fade_samples = int(sr * fade_ms / 1000)
    if fade_samples >= len(audio_np):
        return audio_np
    audio = audio_np.copy()
    # Fade in
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    # Fade out
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return audio


def split_audio(audio_np: np.ndarray, sr: int, split_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Split mono audio at split_sec, apply 50ms fade."""
    split_sample = int(split_sec * sr)
    split_sample = max(0, min(split_sample, len(audio_np)))
    part1 = apply_fade(audio_np[:split_sample], sr)
    part2 = apply_fade(audio_np[split_sample:], sr)
    return part1, part2


def to_mono(audio_np: np.ndarray) -> np.ndarray:
    """Convert to mono by averaging channels."""
    if audio_np.ndim == 2:
        return audio_np.mean(axis=0)
    return audio_np


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "256k", sr: int = 48000):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-ac", "1",  # mono
            "-ar", str(sr),
            "-b:a", bitrate,
            "-f", "mp3",
            mp3_path,
        ],
        capture_output=True, check=True,
    )


def float32_to_base64(arr: np.ndarray) -> str:
    """Encode float32 numpy array as base64."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Worker: loads all models and processes files
# ---------------------------------------------------------------------------
def run_worker(gpu_id: int, work_file: str):
    """Process audio files on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    import torchaudio
    import torch.nn as nn
    import warnings
    warnings.filterwarnings("ignore")

    with open(work_file) as f:
        work_items = json.load(f)

    total = len(work_items)
    log.info(f"[GPU {gpu_id}] {total} items to process. Loading models...")
    t0 = time.time()

    # --- Load RE-USE ---
    sys.path.insert(0, str(REUSE_DIR))
    from models.stfts import mag_phase_stft, mag_phase_istft
    from models.generator_SEMamba_time_d4 import SEMamba
    from utils.util import load_config, pad_or_trim_to_match

    reuse_cfg = load_config(str(
        REUSE_DIR / "recipes" /
        "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
    ))
    reuse_model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=reuse_cfg).to("cuda")
    reuse_model.eval()

    reuse_n_fft = reuse_cfg["stft_cfg"]["n_fft"]
    reuse_hop = reuse_cfg["stft_cfg"]["hop_size"]
    reuse_win = reuse_cfg["stft_cfg"]["win_size"]
    reuse_compress = reuse_cfg["model_cfg"]["compress_factor"]
    reuse_sr = reuse_cfg["stft_cfg"]["sampling_rate"]
    RELU = nn.ReLU()
    log.info(f"[GPU {gpu_id}] RE-USE loaded")

    # --- Load LavaSR (with vocos monkey-patch) ---
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
            if f_min is not None:
                mel_kwargs["f_min"] = f_min
            if f_max is not None:
                mel_kwargs["f_max"] = f_max
            if norm is not None:
                mel_kwargs["norm"] = norm
            if mel_scale is not None:
                mel_kwargs["mel_scale"] = mel_scale
            self.mel_spec = torchaudio.transforms.MelSpectrogram(**mel_kwargs)

    _vfe.MelSpectrogramFeatures = _PatchedMSF

    from LavaSR.model import LavaEnhance2
    lavasr_model = LavaEnhance2("YatharthS/LavaSR", device="cuda:0")
    log.info(f"[GPU {gpu_id}] LavaSR loaded")

    # --- Load Whisper turbo ---
    import whisper
    whisper_model = whisper.load_model("turbo", device="cuda")
    log.info(f"[GPU {gpu_id}] Whisper turbo loaded")

    # --- Load VoiceCLAP Large ---
    from sentence_transformers import SentenceTransformer
    import librosa

    clap_model = SentenceTransformer("laion/voiceclap-large", trust_remote_code=True, device="cuda")
    # Pre-compute aesthetics text embedding for CLAP similarity scoring
    aesthetics_text_emb = clap_model.encode([AESTHETICS_TEXT], normalize_embeddings=True)[0]
    log.info(f"[GPU {gpu_id}] VoiceCLAP Large loaded (aesthetics text embedding cached)")

    load_time = time.time() - t0
    log.info(f"[GPU {gpu_id}] All models loaded in {load_time:.1f}s")

    # --- Helper functions ---
    def make_even(v):
        v = int(round(v))
        return v if v % 2 == 0 else v + 1

    def enhance_reuse(wav_tensor, sr):
        """Apply RE-USE enhancement. wav_tensor: (C, T) on cuda."""
        with torch.no_grad():
            n_fft_s = make_even(reuse_n_fft * sr // reuse_sr)
            hop_s = make_even(reuse_hop * sr // reuse_sr)
            win_s = make_even(reuse_win * sr // reuse_sr)

            noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
                wav_tensor, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
                compress_factor=reuse_compress, center=True, addeps=False,
            )
            amp_g, pha_g, _ = reuse_model(noisy_mag, noisy_pha)
            mag = torch.expm1(RELU(amp_g))
            zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
            amp_g[:, :, (zero_portion > 0.5)[0]] = 0

            audio_g = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, reuse_compress)
            audio_g = pad_or_trim_to_match(wav_tensor.detach(), audio_g, pad_value=1e-8)
            return audio_g

    def enhance_lavasr(wav_path_in, wav_path_out):
        """Apply LavaSR super-resolution. LavaSR expects 16kHz input and outputs 48kHz."""
        wav, sr_in = lavasr_model.load_audio(wav_path_in, input_sr=16000)
        output = lavasr_model.enhance(wav, enhance=True, denoise=False)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        torchaudio.save(wav_path_out, output.cpu(), 48000)

    def run_whisper(audio_path):
        """Run Whisper ASR with word-level timestamps. Returns (text, word_timestamps)."""
        result = whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None,  # auto-detect
        )
        text = result.get("text", "").strip()
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })
        return text, words

    def resample_to_16k(audio_np, sr):
        """Resample audio to 16kHz using torchaudio (GPU-accelerated)."""
        if sr == 16000:
            return audio_np
        t = torch.from_numpy(audio_np).unsqueeze(0).to("cuda")
        t_16k = torchaudio.functional.resample(t, sr, 16000)
        result = t_16k.squeeze(0).cpu().numpy()
        del t, t_16k
        return result

    def get_clap_embeddings_batch(audio_list_16k):
        """Get VoiceCLAP Large embeddings for a batch of 16kHz mono arrays."""
        inputs = [{"array": a.astype(np.float32), "sampling_rate": 16000} for a in audio_list_16k]
        embs = clap_model.encode(inputs, normalize_embeddings=True, batch_size=len(inputs))
        return embs

    def _score_and_write(d, embeddings_map):
        """Score a single item and write annotation JSON.

        embeddings_map: dict mapping 'full' (and optionally 'part1','part2') to numpy arrays.
        """
        ann = d["annotation"]

        # Store CLAP embeddings
        ann["voiceclap_embedding_full"] = float32_to_base64(embeddings_map["full"])
        if d["has_cut_to"]:
            ann["voiceclap_embedding_part1"] = float32_to_base64(embeddings_map["part1"])
            ann["voiceclap_embedding_part2"] = float32_to_base64(embeddings_map["part2"])
        else:
            ann["voiceclap_embedding_part1"] = None
            ann["voiceclap_embedding_part2"] = None

        # CLAP cosine similarity to aesthetics text
        sim_full = cosine_sim(embeddings_map["full"], aesthetics_text_emb)
        ann["clap_cosine_similarity_full"] = round(sim_full, 6)
        if d["has_cut_to"]:
            ann["clap_cosine_similarity_part1"] = round(cosine_sim(embeddings_map["part1"], aesthetics_text_emb), 6)
            ann["clap_cosine_similarity_part2"] = round(cosine_sim(embeddings_map["part2"], aesthetics_text_emb), 6)
        else:
            ann["clap_cosine_similarity_part1"] = None
            ann["clap_cosine_similarity_part2"] = None
        ann["clap_aesthetics_text"] = AESTHETICS_TEXT

        # Reward = cosine_similarity * (1 - min(WER, 1.0))
        def _reward(sim, wer):
            if sim is None or wer is None:
                return None
            return round(sim * (1.0 - min(wer, 1.0)), 6)

        ann["reward_full"] = _reward(ann["clap_cosine_similarity_full"], ann.get("wer_full"))
        ann["reward_part1"] = _reward(ann.get("clap_cosine_similarity_part1"), ann.get("wer_part1"))
        ann["reward_part2"] = _reward(ann.get("clap_cosine_similarity_part2"), ann.get("wer_part2"))

        ann_path = os.path.join(d["output_dir"], f"{d['base_name']}.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=1)

    def flush_clap_batch(deferred):
        """Process accumulated CLAP embeddings in batch, score, and write JSONs.

        Handles both two-scene (part1+part2+full) and single-prompt (full only) items.
        Returns number of successfully processed items.
        """
        if not deferred:
            return 0

        # Build flat list of audio arrays for batched encoding
        clap_inputs = []
        index_map = []  # (item_idx, key) tuples to map results back
        for i, d in enumerate(deferred):
            if d["has_cut_to"]:
                clap_inputs.append(d["part1_16k"])
                index_map.append((i, "part1"))
                clap_inputs.append(d["part2_16k"])
                index_map.append((i, "part2"))
            clap_inputs.append(d["full_16k"])
            index_map.append((i, "full"))

        done = 0
        try:
            embeddings = get_clap_embeddings_batch(clap_inputs)
            # Scatter embeddings back to per-item dicts
            emb_dicts = [{} for _ in deferred]
            for emb_idx, (item_idx, key) in enumerate(index_map):
                emb_dicts[item_idx][key] = embeddings[emb_idx]

            for i, d in enumerate(deferred):
                _score_and_write(d, emb_dicts[i])
                done += 1

        except torch.cuda.OutOfMemoryError:
            log.warning(f"[GPU {gpu_id}] CLAP batch OOM ({len(deferred)} items), falling back to singles")
            torch.cuda.empty_cache()
            for d in deferred:
                try:
                    if d["has_cut_to"]:
                        audio_list = [d["part1_16k"], d["part2_16k"], d["full_16k"]]
                        embs = get_clap_embeddings_batch(audio_list)
                        emb_dict = {"part1": embs[0], "part2": embs[1], "full": embs[2]}
                    else:
                        embs = get_clap_embeddings_batch([d["full_16k"]])
                        emb_dict = {"full": embs[0]}
                    _score_and_write(d, emb_dict)
                    done += 1
                except Exception as e2:
                    log.error(f"[GPU {gpu_id}] CLAP failed for {d['base_name']}: {e2}")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        return done

    # --- Process each item (deferred VoiceCLAP batching) ---
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0
    deferred_items = []  # Accumulate for batched VoiceCLAP

    for idx, item in enumerate(work_items):
        wav_path = item["wav_path"]
        json_path = item["json_path"]
        output_dir = item["output_dir"]
        base_name = item["base_name"]  # e.g. "cca_voicenet_42_seed07"

        try:
            # Load sidecar JSON
            with open(json_path) as f:
                sidecar = json.load(f)

            prompt_id = sidecar["prompt_id"]
            seed = sidecar["seed"]
            original_prompt = sidecar.get("original_prompt", sidecar.get("prompt", ""))

            # [1] Check singing keywords
            singing = is_singing_prompt(original_prompt)

            # [2] Load audio
            wav_tensor, sr = torchaudio.load(wav_path)
            wav_cuda = wav_tensor.to("cuda")

            # [3] RE-USE enhancement (skip for singing)
            if not singing:
                wav_cuda = enhance_reuse(wav_cuda, sr)
                reuse_applied = True
            else:
                reuse_applied = False

            # Save to temp WAV for LavaSR input, free GPU memory
            tmp_pre_lavasr = os.path.join(output_dir, f"_tmp_pre_lavasr_{base_name}.wav")
            torchaudio.save(tmp_pre_lavasr, wav_cuda.cpu(), sr)
            del wav_cuda, wav_tensor
            torch.cuda.empty_cache()

            # [4] LavaSR super-resolution
            tmp_post_lavasr = os.path.join(output_dir, f"_tmp_post_lavasr_{base_name}.wav")
            try:
                enhance_lavasr(tmp_pre_lavasr, tmp_post_lavasr)
            except Exception as e:
                log.warning(f"[GPU {gpu_id}] LavaSR failed for {base_name}: {e}, using pre-LavaSR audio")
                shutil.copy2(tmp_pre_lavasr, tmp_post_lavasr)
            torch.cuda.empty_cache()

            # Load LavaSR output
            wav_enhanced, sr_out = torchaudio.load(tmp_post_lavasr)
            mono_np = to_mono(wav_enhanced.numpy())
            total_duration = len(mono_np) / sr_out

            # [5] Whisper ASR
            asr_text, word_ts = run_whisper(tmp_post_lavasr)

            # [5b] Detect prompt type
            has_cut_to = bool(re.search(r'\bCUT\s+TO\s*:', original_prompt))

            # [6] Full audio (always saved)
            full_np = mono_np.copy()
            full_dur = len(full_np) / sr_out
            full_16k = resample_to_16k(full_np, sr_out)

            import soundfile as sf
            full_wav = os.path.join(output_dir, f"_tmp_full_{base_name}.wav")
            sf.write(full_wav, full_np, sr_out)
            full_mp3 = os.path.join(output_dir, f"{base_name}_full.mp3")

            # Extract expected text for WER
            full_expected = extract_expected_text(original_prompt)
            wer_full = compute_wer(asr_text, full_expected) if full_expected else None

            if has_cut_to:
                # ── Two-scene pathway (CUT TO:) ──────────────────────────
                split_sec, split_method = find_split_point(word_ts, total_duration)

                # Split transcript at the split point
                scene1_words = [w for w in word_ts if w["end"] <= split_sec + 0.1]
                scene2_words = [w for w in word_ts if w["start"] >= split_sec - 0.1]
                scene1_transcript = " ".join(w["word"] for w in scene1_words)
                scene2_transcript = " ".join(w["word"] for w in scene2_words)

                scene1_expected, scene2_expected = extract_scene_texts(original_prompt)

                # Split audio into parts
                part1_np, part2_np = split_audio(mono_np, sr_out, split_sec)
                part1_dur = len(part1_np) / sr_out
                part2_dur = len(part2_np) / sr_out

                part1_16k = resample_to_16k(part1_np, sr_out)
                part2_16k = resample_to_16k(part2_np, sr_out)

                # Save part WAVs + MP3
                part1_wav = os.path.join(output_dir, f"_tmp_part1_{base_name}.wav")
                part2_wav = os.path.join(output_dir, f"_tmp_part2_{base_name}.wav")
                sf.write(part1_wav, part1_np, sr_out)
                sf.write(part2_wav, part2_np, sr_out)

                part1_mp3 = os.path.join(output_dir, f"{base_name}_part1.mp3")
                part2_mp3 = os.path.join(output_dir, f"{base_name}_part2.mp3")

                # Run all ffmpeg conversions in parallel
                ffp1 = subprocess.Popen(
                    ["ffmpeg", "-y", "-i", part1_wav, "-ac", "1", "-ar", str(MP3_SAMPLE_RATE),
                     "-b:a", MP3_BITRATE, "-f", "mp3", part1_mp3],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                ffp2 = subprocess.Popen(
                    ["ffmpeg", "-y", "-i", part2_wav, "-ac", "1", "-ar", str(MP3_SAMPLE_RATE),
                     "-b:a", MP3_BITRATE, "-f", "mp3", part2_mp3],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                ffp3 = subprocess.Popen(
                    ["ffmpeg", "-y", "-i", full_wav, "-ac", "1", "-ar", str(MP3_SAMPLE_RATE),
                     "-b:a", MP3_BITRATE, "-f", "mp3", full_mp3],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                ffp1.wait()
                ffp2.wait()
                ffp3.wait()

                # Cleanup temp files
                for tmp in [tmp_pre_lavasr, tmp_post_lavasr, part1_wav, part2_wav, full_wav]:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

                # WER per scene
                wer_part1 = compute_wer(scene1_transcript, scene1_expected) if scene1_expected else None
                wer_part2 = compute_wer(scene2_transcript, scene2_expected) if scene2_expected else None

                annotation = {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "pathway": sidecar.get("pathway", ""),
                    "language": sidecar.get("language", ""),
                    "original_prompt": original_prompt,
                    "singing_flag": singing,
                    "reuse_applied": reuse_applied,
                    "asr_transcript": asr_text,
                    "word_timestamps": word_ts,
                    "split_point_sec": round(split_sec, 3),
                    "split_method": split_method,
                    "scene1_transcript": scene1_transcript,
                    "scene2_transcript": scene2_transcript,
                    "scene1_expected_text": scene1_expected,
                    "scene2_expected_text": scene2_expected,
                    "part1_duration_sec": round(part1_dur, 3),
                    "part2_duration_sec": round(part2_dur, 3),
                    "full_duration_sec": round(full_dur, 3),
                    "wer_full": round(wer_full, 4) if wer_full is not None else None,
                    "wer_part1": round(wer_part1, 4) if wer_part1 is not None else None,
                    "wer_part2": round(wer_part2, 4) if wer_part2 is not None else None,
                    "sample_info": sidecar.get("sample_info", {}),
                    "cfg_scale": sidecar.get("cfg_scale", 2.5),
                    "stg_scale": sidecar.get("stg_scale", 1.5),
                    "moss_refined_prompt_full": None,
                    "moss_refined_prompt_part1": None,
                    "moss_refined_prompt_part2": None,
                }
                deferred_items.append({
                    "base_name": base_name,
                    "output_dir": output_dir,
                    "part1_16k": part1_16k,
                    "part2_16k": part2_16k,
                    "full_16k": full_16k,
                    "has_cut_to": True,
                    "annotation": annotation,
                })
            else:
                # ── Single-prompt pathway (no CUT TO:) ───────────────────
                log.info(f"[GPU {gpu_id}] {base_name}: single-prompt (no CUT TO:)")

                # Convert full to MP3
                ffp = subprocess.Popen(
                    ["ffmpeg", "-y", "-i", full_wav, "-ac", "1", "-ar", str(MP3_SAMPLE_RATE),
                     "-b:a", MP3_BITRATE, "-f", "mp3", full_mp3],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                ffp.wait()

                # Cleanup temp files
                for tmp in [tmp_pre_lavasr, tmp_post_lavasr, full_wav]:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

                annotation = {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "pathway": "single_prompt",
                    "language": sidecar.get("language", ""),
                    "original_prompt": original_prompt,
                    "singing_flag": singing,
                    "reuse_applied": reuse_applied,
                    "asr_transcript": asr_text,
                    "word_timestamps": word_ts,
                    "split_point_sec": None,
                    "split_method": None,
                    "scene1_transcript": None,
                    "scene2_transcript": None,
                    "scene1_expected_text": None,
                    "scene2_expected_text": None,
                    "part1_duration_sec": None,
                    "part2_duration_sec": None,
                    "full_duration_sec": round(full_dur, 3),
                    "wer_full": round(wer_full, 4) if wer_full is not None else None,
                    "wer_part1": None,
                    "wer_part2": None,
                    "sample_info": sidecar.get("sample_info", {}),
                    "cfg_scale": sidecar.get("cfg_scale", 2.5),
                    "stg_scale": sidecar.get("stg_scale", 1.5),
                    "moss_refined_prompt_full": None,
                    "moss_refined_prompt_part1": None,
                    "moss_refined_prompt_part2": None,
                }
                deferred_items.append({
                    "base_name": base_name,
                    "output_dir": output_dir,
                    "part1_16k": None,
                    "part2_16k": None,
                    "full_16k": full_16k,
                    "has_cut_to": False,
                    "annotation": annotation,
                })

        except torch.cuda.OutOfMemoryError:
            log.warning(f"[GPU {gpu_id}] OOM on {base_name}, clearing cache and skipping")
            torch.cuda.empty_cache()
            errors += 1
            for suffix in ["_tmp_pre_lavasr_", "_tmp_post_lavasr_", "_tmp_part1_", "_tmp_part2_", "_tmp_full_"]:
                tmp_f = os.path.join(output_dir, f"{suffix}{base_name}.wav")
                try:
                    os.unlink(tmp_f)
                except OSError:
                    pass

        except Exception as e:
            log.error(f"[GPU {gpu_id}] ERROR on {base_name}: {e}")
            traceback.print_exc()
            errors += 1
            for suffix in ["_tmp_pre_lavasr_", "_tmp_post_lavasr_", "_tmp_part1_", "_tmp_part2_", "_tmp_full_"]:
                tmp_f = os.path.join(output_dir, f"{suffix}{base_name}.wav")
                try:
                    os.unlink(tmp_f)
                except OSError:
                    pass

        # Clear GPU cache after every file
        torch.cuda.empty_cache()

        # Flush CLAP batch when full
        if len(deferred_items) >= CLAP_BATCH_SIZE:
            log.info(f"[GPU {gpu_id}] Flushing CLAP batch ({len(deferred_items)} items)")
            processed += flush_clap_batch(deferred_items)
            deferred_items = []

        # Write progress
        if (idx + 1) % 5 == 0 or idx == total - 1:
            elapsed = time.time() - t0 - load_time
            rate = (processed + len(deferred_items)) / max(elapsed, 1)
            remaining = total - idx - 1
            eta_min = remaining / max(rate, 0.001) / 60

            progress_data = {
                "gpu_id": gpu_id,
                "processed": processed,
                "errors": errors,
                "total": total,
                "current": idx + 1,
                "deferred_clap": len(deferred_items),
                "rate_per_min": round(rate * 60, 2),
                "eta_min": round(eta_min, 1),
                "timestamp": time.time(),
            }
            progress_path = PROGRESS_DIR / f"gpu_{gpu_id}.json"
            tmp_p = progress_path.with_suffix(".tmp")
            with open(tmp_p, "w") as f:
                json.dump(progress_data, f)
            tmp_p.rename(progress_path)

            log.info(
                f"[GPU {gpu_id}] [{idx+1}/{total}] "
                f"ok={processed} err={errors} deferred={len(deferred_items)} "
                f"rate={rate*60:.1f}/min ETA={eta_min:.1f}min"
            )

    # Final CLAP flush
    if deferred_items:
        log.info(f"[GPU {gpu_id}] Final CLAP batch flush ({len(deferred_items)} items)")
        processed += flush_clap_batch(deferred_items)
        deferred_items = []

    # Final progress
    progress_data = {
        "gpu_id": gpu_id,
        "processed": processed,
        "errors": errors,
        "total": total,
        "current": total,
        "done": True,
        "timestamp": time.time(),
    }
    progress_path = PROGRESS_DIR / f"gpu_{gpu_id}.json"
    tmp_p = progress_path.with_suffix(".tmp")
    with open(tmp_p, "w") as f:
        json.dump(progress_data, f)
    tmp_p.rename(progress_path)

    log.info(f"[GPU {gpu_id}] DONE. processed={processed} errors={errors}")


# =========================================================================
# State Management
# =========================================================================

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "processed_tars": [],
        "current_tar": None,
        "tar_index": 0,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


# =========================================================================
# Progress Monitor (HTTP + Cloudflare tunnel)
# =========================================================================

MONITOR_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>DramaBox Post-Process Monitor</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; margin: 0; padding: 20px; }
.container { max-width: 1000px; margin: 0 auto; }
h1 { color: #ff9800; font-size: 1.4em; border-bottom: 1px solid #333; padding-bottom: 10px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 15px 0; }
.stat-card { background: #1a1a2e; border-radius: 8px; padding: 12px; border: 1px solid #333; }
.stat-value { font-size: 1.6em; font-weight: bold; color: #ff9800; }
.stat-label { font-size: 0.8em; color: #888; margin-top: 4px; }
.progress-bar { width: 100%%; height: 28px; background: #1a1a2e; border-radius: 14px; overflow: hidden; margin: 12px 0; }
.progress-fill { height: 100%%; background: linear-gradient(90deg, #e65100, #ff9800); display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85em; min-width: 60px; }
.gpu-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0; }
.gpu-card { background: #1a1a2e; border-radius: 6px; padding: 8px; text-align: center; border: 1px solid #333; }
.gpu-id { color: #ff9800; font-weight: bold; }
.ts { color: #555; font-size: 0.75em; text-align: right; margin-top: 15px; }
</style>
</head><body>
<div class="container">
<h1>DramaBox Post-Processing Pipeline</h1>
<div id="c">Loading...</div>
</div>
<script>
async function r(){try{const d=(await(await fetch('/api/progress')).json());const op=d.overall_pct||0;const p=d.progress_pct;
document.getElementById('c').innerHTML=`
<h3 style="color:#ff9800;margin-top:8px">Overall: ${(d.total_processed||0).toLocaleString()} / ${(d.total_files_all||0).toLocaleString()} files (${op}%%)</h3>
<div class="progress-bar"><div class="progress-fill" style="width:${Math.max(op,1)}%%">${op}%%</div></div>
<div class="stats-grid">
<div class="stat-card"><div class="stat-value" style="color:#ffd54f">${d.eta||'—'}</div><div class="stat-label">ETA</div></div>
<div class="stat-card"><div class="stat-value">${d.tars_done}/${d.tars_total}</div><div class="stat-label">Tars Done</div></div>
<div class="stat-card"><div class="stat-value">${d.rate_per_hour}/h</div><div class="stat-label">Files/Hour</div></div>
<div class="stat-card"><div class="stat-value">${d.elapsed_h}h</div><div class="stat-label">Elapsed</div></div>
<div class="stat-card"><div class="stat-value">${d.errors}</div><div class="stat-label">Errors</div></div>
<div class="stat-card"><div class="stat-value">${d.current_tar||'—'}</div><div class="stat-label">Current Tar</div></div>
<div class="stat-card"><div class="stat-value">${d.processed}/${d.total_files}</div><div class="stat-label">Batch Progress</div></div>
</div>
<h3 style="color:#ff9800">GPUs</h3>
<div class="gpu-grid">${Object.entries(d.gpus).map(([id,g])=>
'<div class="gpu-card"><div class="gpu-id">GPU '+id+'</div><div>'+g.processed+'/'+g.total+'</div><div style="color:#888;font-size:0.8em">'+(g.rate_per_min||0)+'/min</div></div>'
).join('')}</div>
<div class="ts">Updated: ${d.timestamp}</div>`;
}catch(e){console.error(e);}}
r(); setInterval(r,5000);
</script></body></html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/progress":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = self.server.get_progress()
            self.wfile.write(json.dumps(data, indent=2).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(MONITOR_HTML.encode())

    def log_message(self, fmt, *args):
        pass


def start_monitor(get_progress_fn, port=8768):
    server = HTTPServer(("0.0.0.0", port), MonitorHandler)
    server.get_progress = get_progress_fn
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info(f"Monitor: http://0.0.0.0:{port}")

    # Cloudflare tunnel
    try:
        tunnel_proc = subprocess.Popen(
            ["/tmp/cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        def read_url():
            for line in iter(tunnel_proc.stdout.readline, ""):
                if "trycloudflare.com" in line:
                    m = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if m:
                        url = m.group(0)
                        log.info(f"MONITOR URL: {url}")
                        (OUTPUT_DIR / "monitor_url.txt").write_text(url + "\n")
                        return

        threading.Thread(target=read_url, daemon=True).start()
    except FileNotFoundError:
        log.warning("cloudflared not found, skipping tunnel")
        tunnel_proc = None

    return server, tunnel_proc


# =========================================================================
# Coordinator
# =========================================================================

def get_available_gpus():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [int(l.strip()) for l in r.stdout.strip().split("\n") if l.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def list_source_tars() -> list[str]:
    """List all tar files in the source HF repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(SRC_REPO, repo_type="dataset")
    tars = sorted([f for f in files if f.endswith(".tar")])
    return tars


def download_tar(tar_name: str, local_dir: Path) -> Path:
    """Download a tar from the source HF repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=SRC_REPO,
        filename=tar_name,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path)


def extract_tar(tar_path: Path, extract_dir: Path) -> list[tuple[str, str]]:
    """Extract tar, return list of (wav_path, json_path) pairs."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir, filter="data")

    # Find all WAV files and pair with JSON sidecars
    pairs = []
    for wav_file in sorted(extract_dir.glob("*.wav")):
        json_file = wav_file.with_suffix(".json")
        if json_file.exists():
            pairs.append((str(wav_file), str(json_file)))
        else:
            log.warning(f"No JSON sidecar for {wav_file.name}")
    return pairs


def create_output_tar(staging_dir: Path, tar_name: str, tars_dir: Path) -> Path:
    """Create a tar of processed MP3s and annotation JSONs."""
    tar_path = tars_dir / tar_name
    with tarfile.open(tar_path, "w") as tf:
        for fpath in sorted(staging_dir.iterdir()):
            if fpath.name.startswith("_tmp_"):
                continue
            if fpath.suffix in (".mp3", ".json"):
                tf.add(str(fpath), arcname=fpath.name)
    size_mb = tar_path.stat().st_size / 1e6
    log.info(f"Created output tar {tar_name} ({size_mb:.1f} MB)")
    return tar_path


def upload_tar(tar_path: Path) -> bool:
    """Upload processed tar to destination HF repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=f"data/{tar_path.name}",
            repo_id=DST_REPO,
            repo_type="dataset",
        )
        log.info(f"Uploaded {tar_path.name}")
        return True
    except Exception as e:
        log.error(f"Upload failed {tar_path.name}: {e}")
        return False


def coordinator_main(num_gpus: int, test_mode: bool = False):
    """Main coordinator: download tars, distribute work, collect results, upload."""
    gpu_ids = get_available_gpus()[:num_gpus]
    num_workers = len(gpu_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tars").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    state = load_state()
    start_time = time.time()

    # List source tars
    log.info("Listing source tars...")
    all_tars = list_source_tars()
    log.info(f"Found {len(all_tars)} tars in source repo")

    processed_tars = set(state.get("processed_tars", []))
    pending_tars = [t for t in all_tars if t not in processed_tars]
    log.info(f"Already processed: {len(processed_tars)}, pending: {len(pending_tars)}")

    if test_mode:
        pending_tars = pending_tars[:1]
        gpu_ids = gpu_ids[:1]
        num_workers = 1
        log.info("TEST MODE: 1 tar, 1 GPU")

    total_tars = len(all_tars)
    tars_done = len(processed_tars)
    files_per_tar = 250  # each tar has ~250 wav/json pairs

    # Progress function for monitor
    current_tar_name = [None]
    current_total_files = [0]
    cumulative_processed = [tars_done * files_per_tar]  # total across all completed tars

    def get_progress():
        workers = {}
        batch_proc = 0
        total_errs = 0
        if PROGRESS_DIR.exists():
            for p in PROGRESS_DIR.glob("gpu_*.json"):
                try:
                    with open(p) as f:
                        wd = json.load(f)
                    gid = str(wd.get("gpu_id", p.stem))
                    workers[gid] = wd
                    batch_proc += wd.get("processed", 0)
                    total_errs += wd.get("errors", 0)
                except (json.JSONDecodeError, IOError):
                    pass

        elapsed = time.time() - start_time
        total_proc = cumulative_processed[0] + batch_proc
        total_files_all = total_tars * files_per_tar
        overall_pct = round(total_proc / max(total_files_all, 1) * 100, 1)
        rate = total_proc / max(elapsed, 1)

        # ETA based on overall progress
        remaining = total_files_all - total_proc
        if rate > 0 and elapsed > 60:
            eta_h = remaining / rate / 3600
            eta_str = f"{eta_h:.1f}h"
        else:
            eta_str = "calculating..."

        return {
            "processed": batch_proc,
            "total_processed": total_proc,
            "total_files": current_total_files[0],
            "total_files_all": total_files_all,
            "errors": total_errs,
            "progress_pct": round(batch_proc / max(current_total_files[0], 1) * 100, 1),
            "overall_pct": overall_pct,
            "tars_done": tars_done,
            "tars_total": total_tars,
            "current_tar": current_tar_name[0],
            "rate_per_hour": round(rate * 3600),
            "eta": eta_str,
            "elapsed_h": round(elapsed / 3600, 2),
            "gpus": workers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Start monitor
    monitor, tunnel = start_monitor(get_progress, port=8768)

    # Process each pending tar
    for tar_idx, tar_name in enumerate(pending_tars):
        log.info(f"\n{'='*60}")
        log.info(f"Processing tar {tar_idx+1}/{len(pending_tars)}: {tar_name}")
        log.info(f"{'='*60}")

        current_tar_name[0] = tar_name
        state["current_tar"] = tar_name
        save_state(state)

        # Clear progress files
        for p in PROGRESS_DIR.glob("gpu_*.json"):
            p.unlink()

        # Download tar
        download_dir = OUTPUT_DIR / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Downloading {tar_name}...")
        tar_path = download_tar(tar_name, download_dir)

        # Extract
        extract_dir = OUTPUT_DIR / "extracted" / Path(tar_name).stem
        pairs = extract_tar(tar_path, extract_dir)
        log.info(f"Extracted {len(pairs)} wav/json pairs")

        # Delete downloaded tar after extraction to free disk space
        try:
            tar_path.unlink()
            log.info(f"  Freed disk: deleted source download {tar_name}")
        except OSError:
            pass

        if not pairs:
            log.warning(f"No pairs in {tar_name}, skipping")
            state["processed_tars"].append(tar_name)
            save_state(state)
            tars_done += 1
            continue

        current_total_files[0] = len(pairs)

        # Create staging dir for outputs
        staging_dir = OUTPUT_DIR / "staging" / Path(tar_name).stem
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Create work items and distribute across GPUs
        work_items = []
        for wav_path, json_path in pairs:
            base_name = Path(wav_path).stem  # e.g. "cca_voicenet_42_seed07"
            work_items.append({
                "wav_path": wav_path,
                "json_path": json_path,
                "output_dir": str(staging_dir),
                "base_name": base_name,
            })

        # Distribute round-robin
        shards = [[] for _ in range(num_workers)]
        for i, item in enumerate(work_items):
            shards[i % num_workers].append(item)

        # Write work files
        work_dir = OUTPUT_DIR / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        work_files = []
        for wid, shard in enumerate(shards):
            wf = work_dir / f"work_{gpu_ids[wid]}.json"
            with open(wf, "w") as f:
                json.dump(shard, f)
            work_files.append(wf)
            log.info(f"  Worker GPU {gpu_ids[wid]}: {len(shard)} items")

        # Launch worker subprocesses
        script = str(Path(__file__).resolve())
        processes = []
        log_handles = []

        for wid, gid in enumerate(gpu_ids):
            if not shards[wid]:
                continue

            log_path = OUTPUT_DIR / "logs" / f"worker_gpu{gid}.log"
            log_f = open(log_path, "w")
            log_handles.append(log_f)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gid)
            env["LD_LIBRARY_PATH"] = ""
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            cmd = [
                sys.executable, "-u", script,
                "--worker",
                "--gpu", str(gid),
                "--work-file", str(work_files[wid]),
            ]
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            processes.append((wid, gid, proc))
            log.info(f"  Launched worker GPU {gid} (PID {proc.pid})")

        # Wait for all workers to finish
        while True:
            all_done = all(p.poll() is not None for _, _, p in processes)
            if all_done:
                break

            # Log progress
            prog = get_progress()
            log.info(
                f"  [{tar_name}] {prog['processed']}/{prog['total_files']} "
                f"({prog['progress_pct']}%) errors={prog['errors']} "
                f"rate={prog['rate_per_hour']}/h"
            )
            time.sleep(30)

        for f in log_handles:
            f.close()

        # Check exit codes
        for wid, gid, proc in processes:
            rc = proc.wait()
            if rc != 0:
                log.warning(f"  Worker GPU {gid} exit code {rc}")

        # Create output tar
        output_tar_name = Path(tar_name).name  # e.g. batch_000000.tar
        output_tar_path = create_output_tar(
            staging_dir, output_tar_name, OUTPUT_DIR / "tars"
        )

        # Upload
        if upload_tar(output_tar_path):
            tars_done += 1
            cumulative_processed[0] += current_total_files[0]
            state["processed_tars"].append(tar_name)
            save_state(state)

            # Cleanup
            shutil.rmtree(extract_dir, ignore_errors=True)
            shutil.rmtree(staging_dir, ignore_errors=True)
            output_tar_path.unlink(missing_ok=True)
            log.info(f"  Cleaned up {tar_name}")
        else:
            log.error(f"  Upload failed for {tar_name}, keeping files for retry")

        # Cleanup work files
        for wf in work_files:
            wf.unlink(missing_ok=True)

    log.info(f"\nALL DONE. {tars_done}/{total_tars} tars processed.")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DramaBox Post-Processing Pipeline")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--test", action="store_true", help="Process 1 tar with 1 GPU")

    # Worker mode
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--work-file", type=str)

    args = parser.parse_args()

    if args.worker:
        run_worker(args.gpu, args.work_file)
        return

    coordinator_main(args.num_gpus, test_mode=args.test)


if __name__ == "__main__":
    main()
