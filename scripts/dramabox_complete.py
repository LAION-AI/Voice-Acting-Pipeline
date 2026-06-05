#!/usr/bin/env python3
"""
DramaBox Annotation Completion Pipeline.

Completes annotation for all 363 batches of dramabox voice acting data:
  - Batches 0-172:   "enrich" mode — add 16 missing fields to existing annotations
  - Batches 173-362: "full" mode   — full annotation pipeline from raw WAVs

Two HuggingFace datasets:
  - laion/dramabox-voice-acting-data           (363 raw TARs)
  - laion/dramabox-voice-acting-data-annotated  (173 partially annotated TARs)

Usage:
    # Full run with config
    python scripts/dramabox_complete.py --config configs/complete.yaml

    # Test mode (1 batch, 1 GPU)
    python scripts/dramabox_complete.py --config configs/complete.yaml --test

    # Process specific batch range
    python scripts/dramabox_complete.py --config configs/complete.yaml --batches 0-50

    # Force a specific mode
    python scripts/dramabox_complete.py --config configs/complete.yaml --mode enrich

    # Worker mode (launched by coordinator)
    python scripts/dramabox_complete.py --config configs/complete.yaml --worker --gpu 0 --work-file work.json
"""

# =============================================================================
# Section 1: Imports and LD_LIBRARY_PATH fix
# =============================================================================
import os
import sys

# Filter out conda ml-general paths that break native libraries
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _ld:
    _filtered = [p for p in _ld.split(":") if "ml-general" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_filtered)

import argparse
import base64
import io
import json
import logging
import re
import shutil
import signal
import struct
import subprocess
import tarfile
import tempfile
import time
import traceback
from pathlib import Path


# =============================================================================
# Section 2: Config Loading
# =============================================================================

def load_yaml_config(config_path: str) -> dict:
    """Parse YAML config file and resolve relative paths."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)

    # Resolve paths relative to Voice-Acting-Pipeline root
    vap_root = config_path.parent.parent
    config["_vap_root"] = str(vap_root)

    # Resolve output_dir
    output_dir = config.get("output", {}).get("output_dir", "./complete_output")
    if not os.path.isabs(output_dir):
        output_dir = str(vap_root / output_dir)
    config.setdefault("output", {})["output_dir"] = output_dir

    return config


def validate_config(config: dict) -> list:
    """Validate config and return list of warnings."""
    warnings = []

    storage_mode = config.get("storage", {}).get("mode", "local")
    if storage_mode != "local":
        if not _get_hf_token(config):
            warnings.append("HuggingFace token not set (config, HF_TOKEN env var, or huggingface-cli login)")

    reuse_dir = config.get("postprocess", {}).get("reuse_dir", "")
    if reuse_dir and not Path(reuse_dir).exists():
        warnings.append(f"RE-USE directory not found: {reuse_dir}")

    return warnings


# =============================================================================
# Section 3: HuggingFace I/O
# =============================================================================

def _get_hf_token(config: dict) -> str:
    """Get HuggingFace token from config, environment, or cached login."""
    token = config.get("huggingface", {}).get("token", "")
    if not token or token == "YOUR_HF_TOKEN_HERE":
        token = os.environ.get("HF_TOKEN", "")
    if not token:
        # Check cached token from `huggingface-cli login`
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    return token


def download_tar(repo_id: str, batch_idx: int, dest_dir: str, config: dict) -> Path:
    """Download and extract a TAR from HuggingFace Hub.

    Returns the directory containing extracted files.
    """
    from huggingface_hub import hf_hub_download

    token = _get_hf_token(config)
    tar_name = f"batch_{batch_idx:06d}.tar"
    tar_path_in_repo = f"data/{tar_name}"

    logging.info("Downloading %s from %s ...", tar_name, repo_id)
    local_tar = hf_hub_download(
        repo_id=repo_id,
        filename=tar_path_in_repo,
        repo_type="dataset",
        token=token if token else None,
        local_dir=dest_dir,
    )

    # Extract
    extract_dir = Path(dest_dir) / f"batch_{batch_idx:06d}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(local_tar, "r") as tar:
        tar.extractall(path=str(extract_dir))

    logging.info("Extracted %s -> %s (%d files)", tar_name, extract_dir,
                 len(list(extract_dir.rglob("*"))))
    return extract_dir


def upload_tar(tar_path: str, repo_id: str, config: dict) -> bool:
    """Upload a TAR file to HuggingFace Hub."""
    token = _get_hf_token(config)
    if not token:
        logging.error("No HuggingFace token available for upload")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        tar_name = Path(tar_path).name
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=f"data/{tar_name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logging.info("Uploaded %s to %s", tar_name, repo_id)
        return True
    except Exception as e:
        logging.error("Upload failed for %s: %s", Path(tar_path).name, e)
        return False


def list_annotated_batches(repo_id: str, config: dict) -> set:
    """Query which batch indices exist in the annotated repo.

    Returns set of integer batch indices.
    """
    token = _get_hf_token(config)
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token if token else None)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        indices = set()
        for f in files:
            # Match data/batch_NNNNNN.tar
            m = re.match(r'data/batch_(\d{6})\.tar$', f)
            if m:
                indices.add(int(m.group(1)))
        return indices
    except Exception as e:
        logging.warning("Could not list annotated batches from %s: %s", repo_id, e)
        return set()


def list_raw_batches(repo_id: str, config: dict) -> set:
    """Query which batch indices exist in the raw repo.

    Returns set of integer batch indices.
    """
    token = _get_hf_token(config)
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token if token else None)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        indices = set()
        for f in files:
            m = re.match(r'data/batch_(\d{6})\.tar$', f)
            if m:
                indices.add(int(m.group(1)))
        return indices
    except Exception as e:
        logging.warning("Could not list raw batches from %s: %s", repo_id, e)
        return set()


# =============================================================================
# Section 4: Batch Discovery
# =============================================================================

def discover_batches(config: dict) -> list:
    """Discover batches to process and their modes.

    Returns list of (batch_idx, mode) tuples where mode is "enrich" or "full".
    """
    raw_repo = config.get("raw_repo", "laion/dramabox-voice-acting-data")
    annotated_repo = config.get("annotated_repo", "laion/dramabox-voice-acting-data-annotated")

    batch_start = config.get("batch_range", {}).get("start", 0)
    batch_end = config.get("batch_range", {}).get("end", 363)

    logging.info("Discovering batches %d-%d ...", batch_start, batch_end - 1)
    logging.info("  Raw repo:       %s", raw_repo)
    logging.info("  Annotated repo: %s", annotated_repo)

    # Query existing annotated batches
    annotated_indices = list_annotated_batches(annotated_repo, config)
    logging.info("  Found %d existing annotated batches", len(annotated_indices))

    # Query raw batches
    raw_indices = list_raw_batches(raw_repo, config)
    logging.info("  Found %d raw batches", len(raw_indices))

    batches = []
    for idx in range(batch_start, batch_end):
        if idx not in raw_indices:
            logging.debug("Batch %d not in raw repo, skipping", idx)
            continue

        if idx in annotated_indices:
            mode = "enrich"
        else:
            mode = "full"

        batches.append((idx, mode))

    enrich_count = sum(1 for _, m in batches if m == "enrich")
    full_count = sum(1 for _, m in batches if m == "full")
    logging.info("Discovered %d batches: %d enrich, %d full", len(batches), enrich_count, full_count)

    return batches


# =============================================================================
# Section 5: GPU Detection
# =============================================================================

def detect_gpus() -> list:
    """Detect available GPU indices via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        pass
    return [0]


def get_gpu_vram_mb(gpu_id: int) -> int:
    """Get total VRAM in MB for a specific GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
             f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


# =============================================================================
# Section 6: Model Manager
# =============================================================================

class ModelManager:
    """Manages GPU model lifecycle for the completion pipeline.

    Stages:
      - "postprocess": RE-USE + LavaSR + VoiceCLAP (needed for both modes)
      - "whisper": Whisper ASR (only needed for "full" mode)
    """

    def __init__(self, config: dict, loading_mode: str = "auto"):
        self.config = config
        self.loading_mode = loading_mode
        self._models = {}  # stage -> dict of models
        self._loaded_stages = set()

    def ensure_stage(self, stage: str) -> dict:
        """Load models for the given stage. Returns dict of model objects."""
        if stage in self._loaded_stages:
            return self._models[stage]

        if self.loading_mode == "sequential":
            for s in list(self._loaded_stages):
                if s != stage:
                    self.unload_stage(s)

        if stage == "postprocess":
            self._models[stage] = self._load_postprocess()
        elif stage == "whisper":
            self._models[stage] = self._load_whisper()
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self._loaded_stages.add(stage)
        return self._models[stage]

    def unload_stage(self, stage: str):
        """Free models for the given stage."""
        if stage not in self._loaded_stages:
            return

        self._models.pop(stage, {})
        self._loaded_stages.discard(stage)

        try:
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass

        logging.info("Unloaded stage '%s'", stage)

    def unload_all(self):
        """Unload all stages."""
        for s in list(self._loaded_stages):
            self.unload_stage(s)

    def _load_postprocess(self) -> dict:
        """Load RE-USE, LavaSR, and VoiceCLAP."""
        import torch
        import torchaudio
        import torch.nn as nn
        import warnings
        warnings.filterwarnings("ignore")

        models = {}
        t0 = time.time()

        # --- RE-USE ---
        reuse_dir = self.config.get("postprocess", {}).get("reuse_dir", "/home/deployer/laion/REUSE")
        if reuse_dir not in sys.path:
            sys.path.insert(0, str(reuse_dir))

        from models.stfts import mag_phase_stft, mag_phase_istft
        from models.generator_SEMamba_time_d4 import SEMamba
        from utils.util import load_config, pad_or_trim_to_match

        reuse_cfg_path = os.path.join(
            reuse_dir, "recipes",
            "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
        )
        reuse_cfg = load_config(reuse_cfg_path)
        reuse_model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=reuse_cfg).to("cuda")
        reuse_model.eval()

        models["reuse_model"] = reuse_model
        models["reuse_cfg"] = reuse_cfg
        models["reuse_stft_fns"] = (mag_phase_stft, mag_phase_istft)
        models["reuse_utils"] = pad_or_trim_to_match
        models["reuse_relu"] = nn.ReLU()
        logging.info("RE-USE loaded")

        # --- LavaSR (with vocos monkey-patch) ---
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
        models["lavasr_model"] = lavasr_model
        logging.info("LavaSR loaded")

        # --- VoiceCLAP Large ---
        from sentence_transformers import SentenceTransformer
        clap_model = SentenceTransformer("laion/voiceclap-large", trust_remote_code=True, device="cuda")

        aesthetics_text = self.config.get("quality_prefix", "").strip()
        if not aesthetics_text:
            aesthetics_text = ("Realistic, genuine, spontaneous, authentic, sensual, natural voice "
                               "with all imperfections and organic microdistractions a natural "
                               "situation brings with it")
        aesthetics_emb = clap_model.encode([aesthetics_text], normalize_embeddings=True)[0]

        models["clap_model"] = clap_model
        models["aesthetics_text"] = aesthetics_text
        models["aesthetics_emb"] = aesthetics_emb
        logging.info("VoiceCLAP Large loaded")

        logging.info("All postprocess models loaded in %.1fs", time.time() - t0)
        return models

    def _load_whisper(self) -> dict:
        """Load Whisper ASR model."""
        import whisper
        model_name = self.config.get("postprocess", {}).get("whisper_model", "turbo")
        logging.info("Loading Whisper %s ...", model_name)
        t0 = time.time()
        whisper_model = whisper.load_model(model_name, device="cuda")
        logging.info("Whisper %s loaded in %.1fs", model_name, time.time() - t0)
        return {"whisper_model": whisper_model}


# =============================================================================
# Section 7: Postprocessing Functions
# =============================================================================

# --- Singing/humming keyword detection ---
SINGING_PATTERN = re.compile(
    r'\b(sing(?:s|ing|er)?|'
    r'hum(?:s|ming|med)|'
    r'whistl(?:e|es|ing|ed)|'
    r'lullaby|chant(?:s|ing|ed)?|serenade[sd]?|'
    r'yodel(?:s|ing|ed)?|croon(?:s|ing|ed)?|'
    r'warbl(?:e|es|ing|ed)?)\b',
    re.IGNORECASE,
)
BARE_HUM_PATTERN = re.compile(
    r'\b(?:begins?\s+to|starts?\s+to|a\s+soft|softly|gently|quietly)\s+hum\b',
    re.IGNORECASE,
)


def is_singing_prompt(prompt_text: str) -> bool:
    """Check if prompt describes singing/humming/whistling performance."""
    return bool(SINGING_PATTERN.search(prompt_text)) or bool(BARE_HUM_PATTERN.search(prompt_text))


def extract_expected_text(prompt: str) -> str:
    """Extract all quoted dialogue from a DramaBox prompt."""
    matches = re.findall(r'"([^"]*)"', prompt)
    if not matches:
        matches = re.findall(r'\u201c([^\u201d]*)\u201d', prompt)
    return " ".join(matches).strip()


def extract_scene_texts(prompt: str) -> tuple:
    """Extract quoted dialogue from before and after CUT TO: in the prompt."""
    parts = re.split(r'\bCUT\s+TO\s*:', prompt, maxsplit=1)
    if len(parts) < 2:
        return "", ""

    def get_quotes(text):
        matches = re.findall(r'"([^"]+)"', text)
        return " ".join(matches)

    return get_quotes(parts[0]), get_quotes(parts[1])


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate via edit distance."""
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


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def float32_to_base64(arr) -> str:
    """Encode float32 numpy array as base64."""
    import numpy as np
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def base64_to_float32(s: str):
    """Decode base64 string back to float32 numpy array."""
    import numpy as np
    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype=np.float32).copy()


def find_split_point(word_timestamps: list, total_duration: float) -> tuple:
    """Find CUT TO: transition point via longest silence gap in middle 20-80%."""
    if not word_timestamps or len(word_timestamps) < 4:
        return total_duration / 2.0, "midpoint_fallback"

    gaps = []
    for i in range(1, len(word_timestamps)):
        gap_start = word_timestamps[i - 1]["end"]
        gap_end = word_timestamps[i]["start"]
        gap_len = gap_end - gap_start
        gap_mid = (gap_start + gap_end) / 2.0
        gaps.append((gap_len, gap_mid, i))

    lo = total_duration * 0.20
    hi = total_duration * 0.80
    middle_gaps = [(g, mid, i) for g, mid, i in gaps if lo <= mid <= hi]
    if not middle_gaps:
        lo = total_duration * 0.10
        hi = total_duration * 0.90
        middle_gaps = [(g, mid, i) for g, mid, i in gaps if lo <= mid <= hi]
    if not middle_gaps:
        return total_duration / 2.0, "midpoint_fallback"

    best_gap, best_mid, _ = max(middle_gaps, key=lambda x: x[0])
    if best_gap < 0.3:
        return total_duration / 2.0, "midpoint_fallback"
    return best_mid, "silence_gap"


def apply_fade(audio_np, sr: int, fade_ms: int = 50):
    """Apply fade-in at start and fade-out at end."""
    import numpy as np
    fade_samples = int(sr * fade_ms / 1000)
    if fade_samples >= len(audio_np):
        return audio_np
    audio = audio_np.copy()
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return audio


def split_audio(audio_np, sr: int, split_sec: float):
    """Split mono audio at split_sec, apply 50ms fade."""
    split_sample = int(split_sec * sr)
    split_sample = max(0, min(split_sample, len(audio_np)))
    part1 = apply_fade(audio_np[:split_sample], sr)
    part2 = apply_fade(audio_np[split_sample:], sr)
    return part1, part2


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "256k", sr: int = 48000):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path,
         "-ac", "1", "-ar", str(sr), "-b:a", bitrate, "-f", "mp3", mp3_path],
        capture_output=True, check=True,
    )


def run_reuse(wav_tensor, sr: int, pp_models: dict):
    """Apply RE-USE speech enhancement. wav_tensor: (C, T) on cuda."""
    import torch
    import torch.nn as nn

    reuse_model = pp_models["reuse_model"]
    reuse_cfg = pp_models["reuse_cfg"]
    mag_phase_stft, mag_phase_istft = pp_models["reuse_stft_fns"]
    pad_or_trim = pp_models["reuse_utils"]
    RELU = pp_models["reuse_relu"]

    n_fft = reuse_cfg["stft_cfg"]["n_fft"]
    hop = reuse_cfg["stft_cfg"]["hop_size"]
    win = reuse_cfg["stft_cfg"]["win_size"]
    compress = reuse_cfg["model_cfg"]["compress_factor"]
    reuse_sr = reuse_cfg["stft_cfg"]["sampling_rate"]

    def make_even(v):
        v = int(round(v))
        return v if v % 2 == 0 else v + 1

    with torch.no_grad():
        n_fft_s = make_even(n_fft * sr // reuse_sr)
        hop_s = make_even(hop * sr // reuse_sr)
        win_s = make_even(win * sr // reuse_sr)

        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
            wav_tensor, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
            compress_factor=compress, center=True, addeps=False,
        )
        amp_g, pha_g, _ = reuse_model(noisy_mag, noisy_pha)
        mag = torch.expm1(RELU(amp_g))
        zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0

        audio_g = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, compress)
        audio_g = pad_or_trim(wav_tensor.detach(), audio_g, pad_value=1e-8)
        return audio_g


def run_lavasr(wav_path_in: str, wav_path_out: str, pp_models: dict):
    """Apply LavaSR super-resolution (16kHz -> 48kHz)."""
    import torchaudio
    lavasr = pp_models["lavasr_model"]
    wav, sr_in = lavasr.load_audio(wav_path_in, input_sr=16000)
    output = lavasr.enhance(wav, enhance=True, denoise=False)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    torchaudio.save(wav_path_out, output.cpu(), 48000)


def run_whisper_asr(audio_path: str, whisper_models: dict) -> tuple:
    """Run Whisper ASR with word-level timestamps. Returns (text, word_timestamps)."""
    whisper_model = whisper_models["whisper_model"]
    result = whisper_model.transcribe(audio_path, word_timestamps=True, language=None)
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


def get_clap_embeddings_batch(audio_list_16k: list, pp_models: dict):
    """Get VoiceCLAP Large embeddings for a batch of 16kHz mono arrays."""
    import numpy as np
    clap_model = pp_models["clap_model"]
    inputs = [{"array": a.astype(np.float32), "sampling_rate": 16000} for a in audio_list_16k]
    return clap_model.encode(inputs, normalize_embeddings=True, batch_size=len(inputs))


def resample_to_16k(audio_np, current_sr: int):
    """Resample audio numpy array to 16kHz."""
    import torch
    import torchaudio
    if current_sr == 16000:
        return audio_np
    t = torch.from_numpy(audio_np).unsqueeze(0).to("cuda")
    t_16k = torchaudio.functional.resample(t, current_sr, 16000)
    return t_16k.squeeze(0).cpu().numpy()


# =============================================================================
# Section 8: Enrich Worker (Mode A) — Batches 0-172
# =============================================================================

def _find_raw_wav(raw_dir: Path, prompt_id: str, seed: int) -> Path:
    """Find the raw WAV file in extracted raw TAR for a given prompt_id and seed.

    Raw TARs contain files like: {prompt_id}_seed{NN}.wav and {prompt_id}_seed{NN}.json
    """
    # Try direct pattern
    wav_name = f"{prompt_id}_seed{seed:02d}.wav"
    wav_path = raw_dir / wav_name
    if wav_path.exists():
        return wav_path

    # Search recursively (in case of nested directories)
    for p in raw_dir.rglob(wav_name):
        return p

    return None


def _find_raw_sidecar(raw_dir: Path, prompt_id: str, seed: int) -> dict:
    """Load the raw sidecar JSON for a given prompt_id and seed."""
    json_name = f"{prompt_id}_seed{seed:02d}.json"
    json_path = raw_dir / json_name
    if not json_path.exists():
        for p in raw_dir.rglob(json_name):
            json_path = p
            break
        else:
            return {}

    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _find_annotated_file(annotated_dir: Path, filename: str) -> Path:
    """Find a file in the annotated TAR extract directory."""
    fp = annotated_dir / filename
    if fp.exists():
        return fp
    for p in annotated_dir.rglob(filename):
        return p
    return None


def _collect_samples_from_annotated(annotated_dir: Path) -> list:
    """Collect all annotation JSONs from an extracted annotated TAR.

    Returns list of dicts with keys: prompt_id, seed, json_path, annotation.
    """
    samples = []
    json_files = sorted(annotated_dir.rglob("*.json"))

    for jp in json_files:
        try:
            with open(jp, encoding="utf-8") as f:
                ann = json.load(f)
        except Exception:
            continue

        prompt_id = ann.get("prompt_id", "")
        seed = ann.get("seed", 0)
        if not prompt_id:
            # Try to parse from filename: {prompt_id}_seed{NN}.json
            m = re.match(r'^(.+)_seed(\d+)\.json$', jp.name)
            if m:
                prompt_id = m.group(1)
                seed = int(m.group(2))
            else:
                continue

        samples.append({
            "prompt_id": prompt_id,
            "seed": seed,
            "json_path": jp,
            "annotation": ann,
        })

    return samples


def _collect_samples_from_raw(raw_dir: Path) -> list:
    """Collect all samples from an extracted raw TAR.

    Returns list of dicts with keys: prompt_id, seed, wav_path, sidecar.
    """
    samples = []
    wav_files = sorted(raw_dir.rglob("*.wav"))

    for wp in wav_files:
        m = re.match(r'^(.+)_seed(\d+)\.wav$', wp.name)
        if not m:
            continue

        prompt_id = m.group(1)
        seed = int(m.group(2))

        # Load sidecar JSON
        json_name = f"{prompt_id}_seed{seed:02d}.json"
        json_path = wp.parent / json_name
        sidecar = {}
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as f:
                    sidecar = json.load(f)
            except Exception:
                pass

        samples.append({
            "prompt_id": prompt_id,
            "seed": seed,
            "wav_path": wp,
            "sidecar": sidecar,
        })

    return samples


def process_batch_enrich(batch_idx: int, raw_dir: Path, annotated_dir: Path,
                         output_dir: Path, pp_models: dict, config: dict) -> dict:
    """Enrich an already-annotated batch with 16 missing fields.

    For each sample:
    1. Load raw sidecar JSON for metadata fields
    2. Load existing annotation JSON for existing 22 fields
    3. Copy existing part1.mp3 and part2.mp3
    4. RE-USE raw WAV -> enhanced WAV
    5. LavaSR enhanced -> upsampled WAV
    6. full_duration_sec from upsampled WAV
    7. wav_to_mp3 -> full.mp3
    8. VoiceCLAP embedding for full audio
    9. Decode existing part1/part2 embeddings
    10. Compute CLAP cosine similarities
    11. WER computation
    12. Reward computation
    13. Merge into complete annotation
    """
    import torch
    import torchaudio
    import numpy as np

    pp_cfg = config.get("postprocess", {})
    mp3_bitrate = pp_cfg.get("mp3_bitrate", "256k")
    mp3_sr = pp_cfg.get("mp3_sample_rate", 48000)
    clap_batch_size = pp_cfg.get("clap_batch_size", 16)

    aesthetics_emb = pp_models["aesthetics_emb"]
    aesthetics_text = pp_models["aesthetics_text"]

    # Collect existing annotated samples
    annotated_samples = _collect_samples_from_annotated(annotated_dir)
    logging.info("Batch %d enrich: found %d annotated samples", batch_idx, len(annotated_samples))

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"processed": 0, "errors": 0, "skipped": 0}

    # CLAP batch accumulator
    clap_batch = []

    for sample in annotated_samples:
        prompt_id = sample["prompt_id"]
        seed = sample["seed"]
        ann = sample["annotation"]
        base_name = f"{prompt_id}_seed{seed:02d}"

        try:
            # 1. Load raw sidecar for missing metadata
            raw_sidecar = _find_raw_sidecar(raw_dir, prompt_id, seed)
            modified_prompt = raw_sidecar.get("modified_prompt", ann.get("modified_prompt", ""))
            duration_multiplier = raw_sidecar.get("duration_multiplier",
                                                  raw_sidecar.get("dur_mult", 1.1))

            # 2. Copy existing part1.mp3 and part2.mp3 to output
            for part_suffix in ["_part1.mp3", "_part2.mp3"]:
                src = _find_annotated_file(annotated_dir, base_name + part_suffix)
                if src:
                    dst = output_dir / (base_name + part_suffix)
                    shutil.copy2(str(src), str(dst))

            # 3. Find raw WAV
            raw_wav_path = _find_raw_wav(raw_dir, prompt_id, seed)
            if raw_wav_path is None:
                logging.warning("Enrich %s: raw WAV not found, skipping full audio processing", base_name)
                stats["skipped"] += 1
                # Still write annotation with what we have
                _write_partial_enrich(ann, raw_sidecar, output_dir, base_name,
                                      aesthetics_text, aesthetics_emb, pp_models)
                continue

            # 4. Load raw WAV and apply RE-USE (skip if singing)
            wav_tensor, sr = torchaudio.load(str(raw_wav_path))
            wav_tensor = wav_tensor.to("cuda")
            if wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

            prompt_text = ann.get("original_prompt", "")
            singing = ann.get("singing_flag", is_singing_prompt(prompt_text))
            reuse_applied = False

            if not singing:
                wav_tensor = run_reuse(wav_tensor, sr, pp_models)
                reuse_applied = True

            # 5. Save enhanced WAV for LavaSR
            tmp_dir = output_dir / "_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            enhanced_path = tmp_dir / f"{base_name}_enhanced.wav"
            torchaudio.save(str(enhanced_path), wav_tensor.cpu(), sr)

            # 6. LavaSR super-resolution
            lavasr_path = tmp_dir / f"{base_name}_lavasr.wav"
            run_lavasr(str(enhanced_path), str(lavasr_path), pp_models)

            # 7. Load LavaSR output and compute full_duration
            lava_wav, lava_sr = torchaudio.load(str(lavasr_path))
            mono_np = lava_wav.squeeze(0).numpy()
            full_duration_sec = round(len(mono_np) / lava_sr, 3)

            # 8. wav_to_mp3 -> full.mp3
            full_mp3 = output_dir / f"{base_name}_full.mp3"
            wav_to_mp3(str(lavasr_path), str(full_mp3), mp3_bitrate, mp3_sr)

            # 9. Resample full audio to 16kHz for CLAP
            full_16k = resample_to_16k(mono_np, lava_sr)

            # 10. Decode existing part1/part2 embeddings
            part1_emb = None
            part2_emb = None
            if ann.get("voiceclap_embedding_part1"):
                try:
                    part1_emb = base64_to_float32(ann["voiceclap_embedding_part1"])
                except Exception:
                    pass
            if ann.get("voiceclap_embedding_part2"):
                try:
                    part2_emb = base64_to_float32(ann["voiceclap_embedding_part2"])
                except Exception:
                    pass

            # 11. Compute WER
            asr_text = ann.get("asr_transcript", "")
            full_ref = extract_expected_text(prompt_text)
            wer_full = round(compute_wer(asr_text, full_ref), 4) if full_ref else None

            # Part WER from existing annotation data
            scene1_text = ann.get("scene1_transcript", "")
            scene2_text = ann.get("scene2_transcript", "")
            scene1_expected = ann.get("scene1_expected_text", "")
            scene2_expected = ann.get("scene2_expected_text", "")

            # If expected texts not in annotation, extract from prompt
            if not scene1_expected and not scene2_expected:
                scene1_expected, scene2_expected = extract_scene_texts(prompt_text)

            wer_part1 = round(compute_wer(scene1_text, scene1_expected), 4) if scene1_expected else None
            wer_part2 = round(compute_wer(scene2_text, scene2_expected), 4) if scene2_expected else None

            # Queue for CLAP batch processing
            clap_item = {
                "base_name": base_name,
                "output_dir": str(output_dir),
                "full_16k": full_16k,
                "part1_emb": part1_emb,
                "part2_emb": part2_emb,
                "annotation": ann,
                "raw_sidecar": raw_sidecar,
                "full_duration_sec": full_duration_sec,
                "reuse_applied": reuse_applied,
                "wer_full": wer_full,
                "wer_part1": wer_part1,
                "wer_part2": wer_part2,
                "modified_prompt": modified_prompt,
                "duration_multiplier": duration_multiplier,
                "aesthetics_text": aesthetics_text,
                "scene1_expected": scene1_expected,
                "scene2_expected": scene2_expected,
            }
            clap_batch.append(clap_item)

            # Flush CLAP batch if full
            if len(clap_batch) >= clap_batch_size:
                _flush_enrich_clap_batch(clap_batch, pp_models, output_dir)
                clap_batch.clear()

            # Cleanup temp files
            for tmp in [enhanced_path, lavasr_path]:
                if tmp.exists():
                    tmp.unlink()

            stats["processed"] += 1

        except Exception as e:
            logging.error("Enrich failed %s: %s\n%s", base_name, e, traceback.format_exc())
            stats["errors"] += 1

    # Flush remaining CLAP batch
    if clap_batch:
        _flush_enrich_clap_batch(clap_batch, pp_models, output_dir)
        clap_batch.clear()

    # Cleanup tmp dir
    tmp_dir = output_dir / "_tmp"
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    logging.info("Batch %d enrich complete: %d processed, %d errors, %d skipped",
                 batch_idx, stats["processed"], stats["errors"], stats["skipped"])
    return stats


def _write_partial_enrich(ann: dict, raw_sidecar: dict, output_dir: Path,
                          base_name: str, aesthetics_text: str, aesthetics_emb,
                          pp_models: dict):
    """Write enriched annotation when raw WAV is unavailable (no full audio processing).

    Fills in what fields it can from existing annotation and raw sidecar.
    """
    import numpy as np

    prompt_text = ann.get("original_prompt", "")

    # Compute WER from existing transcripts
    full_ref = extract_expected_text(prompt_text)
    asr_text = ann.get("asr_transcript", "")
    wer_full = round(compute_wer(asr_text, full_ref), 4) if full_ref else None

    scene1_expected = ann.get("scene1_expected_text", "")
    scene2_expected = ann.get("scene2_expected_text", "")
    if not scene1_expected and not scene2_expected:
        scene1_expected, scene2_expected = extract_scene_texts(prompt_text)

    scene1_text = ann.get("scene1_transcript", "")
    scene2_text = ann.get("scene2_transcript", "")
    wer_part1 = round(compute_wer(scene1_text, scene1_expected), 4) if scene1_expected else None
    wer_part2 = round(compute_wer(scene2_text, scene2_expected), 4) if scene2_expected else None

    # Decode existing embeddings for CLAP similarity
    part1_sim = None
    part2_sim = None
    if ann.get("voiceclap_embedding_part1"):
        try:
            part1_emb = base64_to_float32(ann["voiceclap_embedding_part1"])
            part1_sim = round(cosine_sim(part1_emb, aesthetics_emb), 6)
        except Exception:
            pass
    if ann.get("voiceclap_embedding_part2"):
        try:
            part2_emb = base64_to_float32(ann["voiceclap_embedding_part2"])
            part2_sim = round(cosine_sim(part2_emb, aesthetics_emb), 6)
        except Exception:
            pass

    def _reward(sim, wer):
        if sim is None or wer is None:
            return None
        return round(sim * (1.0 - min(wer, 1.0)), 6)

    # Merge complete annotation
    complete = dict(ann)
    complete.update({
        "format": ann.get("format", "cut_to"),
        "modified_prompt": raw_sidecar.get("modified_prompt", ann.get("modified_prompt", "")),
        "has_cut_to": ann.get("has_cut_to", "CUT TO:" in prompt_text),
        "duration_multiplier": raw_sidecar.get("duration_multiplier",
                                                raw_sidecar.get("dur_mult", 1.1)),
        "full_duration_sec": None,  # Cannot compute without WAV
        "wer_full": wer_full,
        "wer_part1": wer_part1,
        "wer_part2": wer_part2,
        "voiceclap_embedding_full": None,
        "clap_cosine_similarity_full": None,
        "clap_cosine_similarity_part1": part1_sim,
        "clap_cosine_similarity_part2": part2_sim,
        "clap_aesthetics_text": aesthetics_text,
        "reward_full": None,
        "reward_part1": _reward(part1_sim, wer_part1),
        "reward_part2": _reward(part2_sim, wer_part2),
        "scene1_expected_text": scene1_expected,
        "scene2_expected_text": scene2_expected,
    })

    ann_path = output_dir / f"{base_name}.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(complete, f, ensure_ascii=False, indent=1)


def _flush_enrich_clap_batch(batch: list, pp_models: dict, output_dir: Path):
    """Process CLAP embeddings for enrich batch and write complete annotations."""
    import numpy as np

    if not batch:
        return

    # Collect all full audio arrays for batched CLAP
    clap_inputs = [d["full_16k"] for d in batch]

    try:
        embeddings = get_clap_embeddings_batch(clap_inputs, pp_models)
        aesthetics_emb = pp_models["aesthetics_emb"]

        for i, d in enumerate(batch):
            full_emb = embeddings[i]
            ann = d["annotation"]
            raw_sidecar = d["raw_sidecar"]
            base_name = d["base_name"]
            prompt_text = ann.get("original_prompt", "")

            # CLAP cosine similarities
            sim_full = round(cosine_sim(full_emb, aesthetics_emb), 6)
            sim_part1 = None
            sim_part2 = None

            if d["part1_emb"] is not None:
                sim_part1 = round(cosine_sim(d["part1_emb"], aesthetics_emb), 6)
            if d["part2_emb"] is not None:
                sim_part2 = round(cosine_sim(d["part2_emb"], aesthetics_emb), 6)

            def _reward(sim, wer):
                if sim is None or wer is None:
                    return None
                return round(sim * (1.0 - min(wer, 1.0)), 6)

            # Scene expected texts
            scene1_expected = d.get("scene1_expected", "")
            scene2_expected = d.get("scene2_expected", "")

            # Build complete annotation (merge old + new)
            complete = dict(ann)
            complete.update({
                "format": ann.get("format", "cut_to"),
                "modified_prompt": d["modified_prompt"],
                "has_cut_to": ann.get("has_cut_to", "CUT TO:" in prompt_text),
                "duration_multiplier": d["duration_multiplier"],
                "full_duration_sec": d["full_duration_sec"],
                "reuse_applied": d.get("reuse_applied", ann.get("reuse_applied", False)),
                "wer_full": d["wer_full"],
                "wer_part1": d["wer_part1"],
                "wer_part2": d["wer_part2"],
                "voiceclap_embedding_full": float32_to_base64(full_emb),
                "clap_cosine_similarity_full": sim_full,
                "clap_cosine_similarity_part1": sim_part1,
                "clap_cosine_similarity_part2": sim_part2,
                "clap_aesthetics_text": d["aesthetics_text"],
                "reward_full": _reward(sim_full, d["wer_full"]),
                "reward_part1": _reward(sim_part1, d["wer_part1"]),
                "reward_part2": _reward(sim_part2, d["wer_part2"]),
                "scene1_expected_text": scene1_expected,
                "scene2_expected_text": scene2_expected,
            })

            # Write annotation JSON
            ann_path = output_dir / f"{base_name}.json"
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(complete, f, ensure_ascii=False, indent=1)

    except Exception as e:
        logging.error("Enrich CLAP batch failed: %s\n%s", e, traceback.format_exc())


# =============================================================================
# Section 9: Full Worker (Mode B) — Batches 173-362
# =============================================================================

def process_batch_full(batch_idx: int, raw_dir: Path, output_dir: Path,
                       pp_models: dict, whisper_models: dict, config: dict) -> dict:
    """Full annotation pipeline for unannotated batches.

    Per sample: RE-USE -> LavaSR -> Whisper -> split -> MP3 -> VoiceCLAP -> WER -> rewards.
    """
    import torch
    import torchaudio
    import numpy as np

    pp_cfg = config.get("postprocess", {})
    mp3_bitrate = pp_cfg.get("mp3_bitrate", "256k")
    mp3_sr = pp_cfg.get("mp3_sample_rate", 48000)
    clap_batch_size = pp_cfg.get("clap_batch_size", 16)

    aesthetics_emb = pp_models["aesthetics_emb"]
    aesthetics_text = pp_models["aesthetics_text"]

    # Collect raw samples
    raw_samples = _collect_samples_from_raw(raw_dir)
    logging.info("Batch %d full: found %d raw samples", batch_idx, len(raw_samples))

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"processed": 0, "errors": 0, "skipped": 0}

    clap_batch = []

    for sample in raw_samples:
        prompt_id = sample["prompt_id"]
        seed = sample["seed"]
        wav_path = sample["wav_path"]
        sidecar = sample["sidecar"]
        base_name = f"{prompt_id}_seed{seed:02d}"

        try:
            # Extract metadata from sidecar
            original_prompt = sidecar.get("original_prompt", sidecar.get("prompt", ""))
            modified_prompt = sidecar.get("modified_prompt", "")
            pathway = sidecar.get("pathway", "unknown")
            language = sidecar.get("language", "English")
            fmt = sidecar.get("format", "cut_to")
            has_cut = sidecar.get("has_cut_to", "CUT TO:" in original_prompt)
            singing = sidecar.get("singing_flag", is_singing_prompt(original_prompt))
            sample_info = sidecar.get("sample_info", {})
            cfg_scale = sidecar.get("cfg_scale", 2.5)
            stg_scale = sidecar.get("stg_scale", 1.5)
            duration_multiplier = sidecar.get("duration_multiplier",
                                              sidecar.get("dur_mult", 1.1))

            # 1. Load raw WAV
            wav_tensor, sr = torchaudio.load(str(wav_path))
            wav_tensor = wav_tensor.to("cuda")
            if wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

            # 2. RE-USE (skip if singing)
            reuse_applied = False
            if not singing:
                wav_tensor = run_reuse(wav_tensor, sr, pp_models)
                reuse_applied = True

            # 3. Save enhanced WAV for LavaSR
            tmp_dir = output_dir / "_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            enhanced_path = tmp_dir / f"{base_name}_enhanced.wav"
            torchaudio.save(str(enhanced_path), wav_tensor.cpu(), sr)

            # 4. LavaSR super-resolution
            lavasr_path = tmp_dir / f"{base_name}_lavasr.wav"
            run_lavasr(str(enhanced_path), str(lavasr_path), pp_models)

            # 5. Load LavaSR output (48kHz)
            lava_wav, lava_sr = torchaudio.load(str(lavasr_path))
            mono_np = lava_wav.squeeze(0).numpy()
            full_duration_sec = round(len(mono_np) / lava_sr, 3)

            # 6. Whisper ASR
            asr_text, word_timestamps = run_whisper_asr(str(lavasr_path), whisper_models)

            # 7. Extract expected text
            full_ref = extract_expected_text(original_prompt)

            # 8. Split detection and audio split (if CUT TO:)
            split_sec, split_method = None, None
            part1_np, part2_np = None, None
            scene1_text, scene2_text = "", ""
            scene1_expected, scene2_expected = "", ""
            wer_part1, wer_part2 = None, None
            part1_duration, part2_duration = None, None

            if has_cut:
                split_sec, split_method = find_split_point(word_timestamps, full_duration_sec)
                part1_np, part2_np = split_audio(mono_np, lava_sr, split_sec)
                scene1_expected, scene2_expected = extract_scene_texts(original_prompt)

                # Transcript for parts
                scene1_text = " ".join(w["word"] for w in word_timestamps if w["end"] <= split_sec)
                scene2_text = " ".join(w["word"] for w in word_timestamps if w["start"] >= split_sec)

                if scene1_expected:
                    wer_part1 = round(compute_wer(scene1_text, scene1_expected), 4)
                if scene2_expected:
                    wer_part2 = round(compute_wer(scene2_text, scene2_expected), 4)

                part1_duration = round(split_sec, 3)
                part2_duration = round(full_duration_sec - split_sec, 3)

            # 9. Full WER
            wer_full = round(compute_wer(asr_text, full_ref), 4) if full_ref else None

            # 10. MP3 conversion
            full_mp3 = output_dir / f"{base_name}_full.mp3"
            wav_to_mp3(str(lavasr_path), str(full_mp3), mp3_bitrate, mp3_sr)

            if has_cut and part1_np is not None and part2_np is not None:
                p1_wav = tmp_dir / f"{base_name}_part1.wav"
                p2_wav = tmp_dir / f"{base_name}_part2.wav"
                torchaudio.save(str(p1_wav), torch.from_numpy(part1_np).unsqueeze(0), lava_sr)
                torchaudio.save(str(p2_wav), torch.from_numpy(part2_np).unsqueeze(0), lava_sr)

                p1_mp3 = output_dir / f"{base_name}_part1.mp3"
                p2_mp3 = output_dir / f"{base_name}_part2.mp3"
                wav_to_mp3(str(p1_wav), str(p1_mp3), mp3_bitrate, mp3_sr)
                wav_to_mp3(str(p2_wav), str(p2_mp3), mp3_bitrate, mp3_sr)

                # Cleanup temp part WAVs
                for tmp in [p1_wav, p2_wav]:
                    if tmp.exists():
                        tmp.unlink()

            # 11. Resample to 16kHz for CLAP
            full_16k = resample_to_16k(mono_np, lava_sr)

            # 12. Build annotation (without CLAP fields — added in batch flush)
            annotation = {
                "prompt_id": prompt_id,
                "seed": seed,
                "pathway": pathway,
                "language": language,
                "format": fmt,
                "original_prompt": original_prompt,
                "modified_prompt": modified_prompt,
                "singing_flag": singing,
                "reuse_applied": reuse_applied,
                "has_cut_to": has_cut,
                "asr_transcript": asr_text,
                "word_timestamps": word_timestamps,
                "split_point_sec": round(split_sec, 3) if split_sec else None,
                "split_method": split_method,
                "scene1_transcript": scene1_text,
                "scene2_transcript": scene2_text,
                "scene1_expected_text": scene1_expected,
                "scene2_expected_text": scene2_expected,
                "full_duration_sec": full_duration_sec,
                "part1_duration_sec": part1_duration,
                "part2_duration_sec": part2_duration,
                "wer_full": wer_full,
                "wer_part1": wer_part1,
                "wer_part2": wer_part2,
                "sample_info": sample_info,
                "cfg_scale": cfg_scale,
                "stg_scale": stg_scale,
                "duration_multiplier": duration_multiplier,
            }

            # Queue for CLAP batch
            clap_item = {
                "base_name": base_name,
                "output_dir": str(output_dir),
                "has_cut_to": has_cut,
                "full_16k": full_16k,
                "annotation": annotation,
            }
            if has_cut and part1_np is not None:
                clap_item["part1_16k"] = resample_to_16k(part1_np, lava_sr)
                clap_item["part2_16k"] = resample_to_16k(part2_np, lava_sr)

            clap_batch.append(clap_item)

            if len(clap_batch) >= clap_batch_size:
                _flush_full_clap_batch(clap_batch, pp_models)
                clap_batch.clear()

            # Cleanup temp files
            for tmp in [enhanced_path, lavasr_path]:
                if tmp.exists():
                    tmp.unlink()

            stats["processed"] += 1

        except Exception as e:
            logging.error("Full process failed %s: %s\n%s", base_name, e, traceback.format_exc())
            stats["errors"] += 1

    # Flush remaining CLAP batch
    if clap_batch:
        _flush_full_clap_batch(clap_batch, pp_models)
        clap_batch.clear()

    # Cleanup tmp dir
    tmp_dir = output_dir / "_tmp"
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    logging.info("Batch %d full complete: %d processed, %d errors, %d skipped",
                 batch_idx, stats["processed"], stats["errors"], stats["skipped"])
    return stats


def _flush_full_clap_batch(batch: list, pp_models: dict):
    """Process CLAP embeddings for full batch and write annotation JSONs."""
    import numpy as np

    if not batch:
        return

    clap_inputs = []
    index_map = []
    for i, d in enumerate(batch):
        if d["has_cut_to"] and "part1_16k" in d:
            clap_inputs.append(d["part1_16k"])
            index_map.append((i, "part1"))
            clap_inputs.append(d["part2_16k"])
            index_map.append((i, "part2"))
        clap_inputs.append(d["full_16k"])
        index_map.append((i, "full"))

    try:
        embeddings = get_clap_embeddings_batch(clap_inputs, pp_models)
        emb_dicts = [{} for _ in batch]
        for idx, (item_idx, key) in enumerate(index_map):
            emb_dicts[item_idx][key] = embeddings[idx]

        aesthetics_emb = pp_models["aesthetics_emb"]
        aesthetics_text = pp_models["aesthetics_text"]

        for i, d in enumerate(batch):
            ann = d["annotation"]
            embs = emb_dicts[i]

            ann["voiceclap_embedding_full"] = float32_to_base64(embs["full"])
            sim_full = cosine_sim(embs["full"], aesthetics_emb)
            ann["clap_cosine_similarity_full"] = round(sim_full, 6)

            if d["has_cut_to"] and "part1" in embs:
                ann["voiceclap_embedding_part1"] = float32_to_base64(embs["part1"])
                ann["voiceclap_embedding_part2"] = float32_to_base64(embs["part2"])
                ann["clap_cosine_similarity_part1"] = round(cosine_sim(embs["part1"], aesthetics_emb), 6)
                ann["clap_cosine_similarity_part2"] = round(cosine_sim(embs["part2"], aesthetics_emb), 6)
            else:
                ann["voiceclap_embedding_part1"] = None
                ann["voiceclap_embedding_part2"] = None
                ann["clap_cosine_similarity_part1"] = None
                ann["clap_cosine_similarity_part2"] = None

            ann["clap_aesthetics_text"] = aesthetics_text

            def _reward(sim, wer):
                if sim is None or wer is None:
                    return None
                return round(sim * (1.0 - min(wer, 1.0)), 6)

            ann["reward_full"] = _reward(ann["clap_cosine_similarity_full"], ann.get("wer_full"))
            ann["reward_part1"] = _reward(ann.get("clap_cosine_similarity_part1"), ann.get("wer_part1"))
            ann["reward_part2"] = _reward(ann.get("clap_cosine_similarity_part2"), ann.get("wer_part2"))

            # Write annotation JSON
            ann_path = os.path.join(d["output_dir"], f"{d['base_name']}.json")
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=1)

    except Exception as e:
        logging.error("Full CLAP batch failed: %s\n%s", e, traceback.format_exc())


# =============================================================================
# Section 10: Worker Entry Point
# =============================================================================

def run_worker(gpu_id: int, work_file: str, config_path: str, overrides: dict = None):
    """Worker process for one GPU. Processes assigned batches sequentially."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config = load_yaml_config(config_path)

    # Apply overrides
    if overrides:
        if "mode_filter" in overrides:
            config["_mode_filter"] = overrides["mode_filter"]

    output_dir = Path(config["output"]["output_dir"])
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"gpu_{gpu_id}.json"

    # Setup logging
    log_file = output_dir / f"worker_gpu{gpu_id}.log"
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
        format=f"%(asctime)s [GPU {gpu_id}] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger(f"worker.gpu{gpu_id}")

    with open(work_file) as f:
        work_items = json.load(f)

    total = len(work_items)
    log.info("Worker starting: %d work items on GPU %d", total, gpu_id)

    # Determine loading mode
    loading_mode = config.get("gpu", {}).get("loading_mode", "auto")
    if loading_mode == "auto":
        vram = get_gpu_vram_mb(gpu_id)
        loading_mode = "simultaneous" if vram >= 40000 else "sequential"
        log.info("Auto-detected VRAM: %d MB -> mode: %s", vram, loading_mode)

    manager = ModelManager(config, loading_mode)

    # Check if we need whisper (any "full" mode batches)
    needs_whisper = any(item["mode"] == "full" for item in work_items)

    # Progress tracking
    progress = {
        "gpu_id": gpu_id,
        "total": total,
        "completed": 0,
        "errors": 0,
        "current_batch": None,
        "current_mode": None,
        "current_stage": "starting",
        "start_time": time.time(),
        "completed_batches": [],
    }

    def save_progress():
        try:
            tmp = str(progress_file) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(progress, f, indent=1)
            os.replace(tmp, str(progress_file))
        except Exception:
            pass

    save_progress()

    # Process each batch
    for work_idx, item in enumerate(work_items):
        batch_idx = item["batch_idx"]
        mode = item["mode"]

        progress["current_batch"] = batch_idx
        progress["current_mode"] = mode
        progress["current_stage"] = f"downloading batch {batch_idx}"
        save_progress()

        log.info("Processing batch %d (%d/%d) mode=%s", batch_idx, work_idx + 1, total, mode)

        raw_repo = config.get("raw_repo", "laion/dramabox-voice-acting-data")
        annotated_repo = config.get("annotated_repo", "laion/dramabox-voice-acting-data-annotated")
        output_repo = config.get("output_repo", annotated_repo)

        batch_work_dir = output_dir / "working" / f"batch_{batch_idx:06d}"
        batch_work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download raw TAR
            raw_download_dir = batch_work_dir / "raw"
            raw_download_dir.mkdir(parents=True, exist_ok=True)
            raw_dir = download_tar(raw_repo, batch_idx, str(raw_download_dir), config)

            annotated_dir = None
            if mode == "enrich":
                # Download annotated TAR
                ann_download_dir = batch_work_dir / "annotated"
                ann_download_dir.mkdir(parents=True, exist_ok=True)
                annotated_dir = download_tar(annotated_repo, batch_idx,
                                             str(ann_download_dir), config)

            # Ensure models are loaded
            progress["current_stage"] = f"loading models for batch {batch_idx}"
            save_progress()

            pp_models = manager.ensure_stage("postprocess")
            whisper_models = None
            if mode == "full":
                whisper_models = manager.ensure_stage("whisper")

            # Process batch
            batch_output_dir = batch_work_dir / "output"
            batch_output_dir.mkdir(parents=True, exist_ok=True)

            progress["current_stage"] = f"processing batch {batch_idx} ({mode})"
            save_progress()

            if mode == "enrich":
                stats = process_batch_enrich(
                    batch_idx, raw_dir, annotated_dir,
                    batch_output_dir, pp_models, config,
                )
            else:
                stats = process_batch_full(
                    batch_idx, raw_dir, batch_output_dir,
                    pp_models, whisper_models, config,
                )

            # Package output TAR
            progress["current_stage"] = f"packaging batch {batch_idx}"
            save_progress()

            tar_path = create_output_tar(batch_idx, batch_output_dir, config)

            if tar_path:
                # Upload if configured
                storage_mode = config.get("storage", {}).get("mode", "local")
                if storage_mode in ("upload_and_delete", "upload_and_keep"):
                    progress["current_stage"] = f"uploading batch {batch_idx}"
                    save_progress()

                    success = upload_tar(str(tar_path), output_repo, config)
                    if success and storage_mode == "upload_and_delete":
                        if tar_path.exists():
                            tar_path.unlink()
                        log.info("Uploaded and deleted TAR for batch %d", batch_idx)

            # Update progress
            progress["completed"] += 1
            progress["completed_batches"].append(batch_idx)
            progress["errors"] += stats.get("errors", 0)
            save_progress()

            # Cleanup working directory
            if config.get("storage", {}).get("mode", "local") == "upload_and_delete":
                shutil.rmtree(str(batch_work_dir), ignore_errors=True)

            log.info("Batch %d complete: %s", batch_idx, stats)

        except Exception as e:
            log.error("Batch %d failed: %s\n%s", batch_idx, e, traceback.format_exc())
            progress["errors"] += 1
            save_progress()

    manager.unload_all()

    progress["current_stage"] = "done"
    progress["end_time"] = time.time()
    save_progress()
    log.info("Worker GPU %d finished: %d completed, %d errors",
             gpu_id, progress["completed"], progress["errors"])


# =============================================================================
# Section 11: TAR Packaging
# =============================================================================

def create_output_tar(batch_idx: int, output_dir: Path, config: dict) -> Path:
    """Package output files into a WebDataset TAR.

    Per sample:
    - {prompt_id}_seed{NN}_full.mp3
    - {prompt_id}_seed{NN}_part1.mp3 (if CUT TO)
    - {prompt_id}_seed{NN}_part2.mp3 (if CUT TO)
    - {prompt_id}_seed{NN}.json
    """
    tar_dir = Path(config["output"]["output_dir"]) / "tars"
    tar_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tar_dir / f"batch_{batch_idx:06d}.tar"

    # Collect all output files
    mp3_files = sorted(output_dir.glob("*.mp3"))
    json_files = sorted(output_dir.glob("*.json"))

    if not mp3_files and not json_files:
        logging.warning("No output files for batch %d", batch_idx)
        return None

    with tarfile.open(str(tar_path), "w") as tar:
        for f in mp3_files:
            tar.add(str(f), arcname=f.name)
        for f in json_files:
            tar.add(str(f), arcname=f.name)

    file_count = len(mp3_files) + len(json_files)
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    logging.info("Created batch_%06d.tar (%d files, %.1f MB)", batch_idx, file_count, size_mb)
    return tar_path


# =============================================================================
# Section 12: State Management
# =============================================================================

def load_state(config: dict) -> dict:
    """Load pipeline state from JSON file, or return fresh state."""
    output_dir = Path(config["output"]["output_dir"])
    state_file = output_dir / "complete_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            logging.info("Loaded state: %d completed batches, %d uploaded",
                         len(state.get("completed_batches", [])),
                         len(state.get("uploaded_batches", [])))
            return state
        except Exception as e:
            logging.warning("Failed to load state: %s", e)

    return {
        "completed_batches": [],
        "uploaded_batches": [],
        "total_batches": 0,
        "errors": {},
        "start_time": time.time(),
    }


def save_state(state: dict, config: dict):
    """Atomic save of pipeline state."""
    output_dir = Path(config["output"]["output_dir"])
    state_file = output_dir / "complete_state.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp = str(state_file) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, str(state_file))


# =============================================================================
# Section 13: Coordinator
# =============================================================================

def coordinator_main(config: dict):
    """Main coordinator: discover batches, distribute to GPUs, launch workers, monitor."""
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect GPUs
    gpu_cfg = config.get("gpu", {})
    devices = gpu_cfg.get("devices", "auto")
    if devices == "auto":
        gpus = detect_gpus()
    elif isinstance(devices, list):
        gpus = devices
    else:
        gpus = [int(x) for x in str(devices).split(",")]

    logging.info("Using GPUs: %s", gpus)

    # Load state
    state = load_state(config)
    completed_set = set(state.get("completed_batches", []))

    # Discover batches
    mode_filter = config.get("_mode_filter")
    batches = discover_batches(config)

    # Filter mode if specified
    if mode_filter and mode_filter != "auto":
        batches = [(idx, mode_filter) for idx, _ in batches]

    # Filter out completed batches (resume support)
    remaining = [(idx, mode) for idx, mode in batches if idx not in completed_set]
    logging.info("After resume filter: %d batches remaining (of %d total)",
                 len(remaining), len(batches))

    if not remaining:
        logging.info("No batches to process — pipeline may already be complete")
        return

    state["total_batches"] = len(batches)

    # Distribute batches round-robin across GPUs
    gpu_work = {g: [] for g in gpus}
    for i, (batch_idx, mode) in enumerate(remaining):
        gpu_work[gpus[i % len(gpus)]].append({
            "batch_idx": batch_idx,
            "mode": mode,
        })

    for g in gpus:
        enrich_count = sum(1 for w in gpu_work[g] if w["mode"] == "enrich")
        full_count = sum(1 for w in gpu_work[g] if w["mode"] == "full")
        logging.info("GPU %d: %d batches (%d enrich, %d full)",
                     g, len(gpu_work[g]), enrich_count, full_count)

    # Write per-GPU work files
    work_dir = output_dir / "work_files"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_files = {}
    for g in gpus:
        wf = work_dir / f"work_gpu_{g}.json"
        with open(wf, "w") as f:
            json.dump(gpu_work[g], f, indent=1)
        work_files[g] = str(wf)

    # Launch worker subprocesses
    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    config_path = config["_config_path"]

    # Build CLI overrides
    cli_overrides = []
    if mode_filter and mode_filter != "auto":
        cli_overrides += ["--mode", mode_filter]

    start_time = time.time()
    processes = []
    for i, g in enumerate(gpus):
        cmd = [
            python_exe, "-u", script_path,
            "--config", config_path,
            "--worker",
            "--gpu", str(g),
            "--work-file", work_files[g],
        ] + cli_overrides
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)

        # Ensure HF_TOKEN is propagated to workers
        hf_token = _get_hf_token(config)
        if hf_token:
            env["HF_TOKEN"] = hf_token

        logging.info("Launching worker GPU %d (stagger %ds)", g, i * 2)
        if i > 0:
            time.sleep(2)

        proc = subprocess.Popen(cmd, env=env)
        processes.append((g, proc))

    logging.info("All %d workers launched", len(processes))

    # Monitor loop
    try:
        while True:
            all_done = all(proc.poll() is not None for _, proc in processes)
            if all_done:
                break

            # Read progress from workers
            total_completed = 0
            total_errors = 0
            for g in gpus:
                pf = output_dir / "progress" / f"gpu_{g}.json"
                try:
                    if pf.exists():
                        with open(pf) as f:
                            gp = json.load(f)
                        total_completed += gp.get("completed", 0)
                        total_errors += gp.get("errors", 0)
                except Exception:
                    pass

            pct = (total_completed / max(len(remaining), 1)) * 100
            elapsed = time.time() - start_time
            rate = total_completed / elapsed * 3600 if elapsed > 0 and total_completed > 0 else 0

            print(f"\r  Progress: {total_completed}/{len(remaining)} batches "
                  f"({pct:.1f}%) | Errors: {total_errors} | "
                  f"Rate: {rate:.0f}/hr | Elapsed: {elapsed/60:.0f}m",
                  end="", flush=True)

            # Update state
            for g in gpus:
                pf = output_dir / "progress" / f"gpu_{g}.json"
                try:
                    if pf.exists():
                        with open(pf) as f:
                            gp = json.load(f)
                        for bidx in gp.get("completed_batches", []):
                            if bidx not in completed_set:
                                completed_set.add(bidx)
                                state["completed_batches"] = sorted(completed_set)
                except Exception:
                    pass

            save_state(state, config)
            time.sleep(30)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt — sending SIGTERM to workers")
        for g, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for g, proc in processes:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    # Wait for all workers
    for g, proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
        rc = proc.returncode
        if rc and rc != 0:
            logging.warning("Worker GPU %d exited with code %d", g, rc)

    print()  # Newline after progress line

    # Final state update from worker progress
    for g in gpus:
        pf = output_dir / "progress" / f"gpu_{g}.json"
        try:
            if pf.exists():
                with open(pf) as f:
                    gp = json.load(f)
                for bidx in gp.get("completed_batches", []):
                    completed_set.add(bidx)
        except Exception:
            pass

    state["completed_batches"] = sorted(completed_set)
    save_state(state, config)

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  DramaBox Annotation Completion Pipeline — Complete")
    print("=" * 70)
    print(f"  Batches: {len(completed_set)}/{len(batches)} completed")
    enrich_done = sum(1 for idx in completed_set
                      if any(b == idx and m == "enrich" for b, m in batches))
    full_done = sum(1 for idx in completed_set
                    if any(b == idx and m == "full" for b, m in batches))
    print(f"  Enrich:  {enrich_done}")
    print(f"  Full:    {full_done}")
    print(f"  Time:    {elapsed/60:.1f} min")
    if completed_set:
        rate = len(completed_set) / elapsed * 3600
        print(f"  Rate:    {rate:.1f} batches/hr")
    print(f"  Output:  {output_dir}")
    print("=" * 70)


# =============================================================================
# Section 14: CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DramaBox Annotation Completion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full run
  python scripts/dramabox_complete.py --config configs/complete.yaml

  # Test mode (1 batch, 1 GPU)
  python scripts/dramabox_complete.py --config configs/complete.yaml --test

  # Process specific batch range
  python scripts/dramabox_complete.py --config configs/complete.yaml --batches 0-50

  # Force enrich mode only
  python scripts/dramabox_complete.py --config configs/complete.yaml --mode enrich
""",
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process 1 batch, 1 GPU")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Override GPU list: '0,1,2,3'")
    parser.add_argument("--batches", type=str, default=None,
                        help="Override batch range: 'START-END' (e.g., '0-50')")
    parser.add_argument("--mode", choices=["enrich", "full", "auto"], default=None,
                        help="Override processing mode (default: auto)")

    # Internal worker mode (launched by coordinator)
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--work-file", type=str, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Worker mode
    if args.worker:
        if args.gpu is None or args.work_file is None:
            parser.error("--worker requires --gpu and --work-file")
        overrides = {}
        if args.mode:
            overrides["mode_filter"] = args.mode
        run_worker(args.gpu, args.work_file, args.config, overrides=overrides)
        return

    # Coordinator mode
    config = load_yaml_config(args.config)

    # Setup logging
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_cfg = config.get("logging", {})
    log_file = output_dir / log_cfg.get("file", "complete.log")
    log_level = getattr(logging, log_cfg.get("level", "INFO"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # Apply CLI overrides
    if args.gpus:
        config.setdefault("gpu", {})["devices"] = [int(x) for x in args.gpus.split(",")]

    if args.batches:
        parts = args.batches.split("-")
        if len(parts) == 2:
            config.setdefault("batch_range", {})["start"] = int(parts[0])
            config["batch_range"]["end"] = int(parts[1])
        else:
            parser.error("--batches must be START-END (e.g., '0-50')")

    if args.mode:
        config["_mode_filter"] = args.mode

    # Test mode
    if args.test:
        logging.info("TEST MODE: 1 batch, 1 GPU")
        config.setdefault("gpu", {})["devices"] = [detect_gpus()[0]]
        # Find first unprocessed batch
        state = load_state(config)
        completed = set(state.get("completed_batches", []))
        # Start from batch 0 if nothing specified, find first non-completed
        batch_start = config.get("batch_range", {}).get("start", 0)
        test_batch = batch_start
        for i in range(batch_start, config.get("batch_range", {}).get("end", 363)):
            if i not in completed:
                test_batch = i
                break
        config.setdefault("batch_range", {})["start"] = test_batch
        config["batch_range"]["end"] = test_batch + 1
        logging.info("Test batch: %d", test_batch)

    # Validate
    warnings = validate_config(config)
    for w in warnings:
        logging.warning("Config: %s", w)

    # Print banner
    batch_start = config.get("batch_range", {}).get("start", 0)
    batch_end = config.get("batch_range", {}).get("end", 363)
    gpu_devs = config.get("gpu", {}).get("devices", "auto")
    if gpu_devs == "auto":
        gpu_devs = detect_gpus()

    print("=" * 70)
    print("  DramaBox Annotation Completion Pipeline")
    print("=" * 70)
    print(f"  Config:    {config['_config_path']}")
    print(f"  Batches:   {batch_start}-{batch_end - 1} ({batch_end - batch_start} total)")
    print(f"  GPUs:      {gpu_devs}")
    print(f"  Storage:   {config.get('storage', {}).get('mode', 'local')}")
    print(f"  Output:    {config['output']['output_dir']}")
    if args.mode:
        print(f"  Mode:      {args.mode}")
    print("=" * 70)
    print()

    coordinator_main(config)


if __name__ == "__main__":
    main()
