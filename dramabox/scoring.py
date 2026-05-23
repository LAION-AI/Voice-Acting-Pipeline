"""Best-of-N ranking with composite reward scoring.

Scoring components:
1. WER (Word Error Rate) via Parakeet v3 ASR (nvidia/parakeet-tdt-0.6b-v3)
   - Transcribes audio, computes WER against expected text
   - Expected text extracted from double-quoted segments in DramaBox prompt

2. Content Enjoyment via Empathic Insight Plus (laion/Empathic-Insight-Voice-Plus)
   - Uses BUD-E-Whisper encoder for audio embeddings
   - MLP expert predicts content enjoyment score

Composite reward:
    reward = (1 - min(WER, 1.0)) * content_enjoyment

Best-of-N:
    Generate N samples, score each, select the one with highest reward.
"""
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

# Per-process model caches
_asr_cache: dict = {}
_enjoyment_cache: dict = {}


# ─── Expected text extraction ───────────────────────────────────────────────

def extract_expected_text(dramabox_prompt: str) -> str:
    """Extract expected spoken text from a DramaBox prompt.

    Finds all text inside double quotes (the dialogue portions)
    and concatenates them.

    Args:
        dramabox_prompt: Full DramaBox prompt string.

    Returns:
        Concatenated dialogue text (what should be spoken).
    """
    # Match text inside double quotes, handling escaped quotes
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, dramabox_prompt)
    if not matches:
        # Try Unicode quotes
        pattern = r'\u201c([^\u201d]*)\u201d'
        matches = re.findall(pattern, dramabox_prompt)
    return " ".join(matches).strip()


# ─── ASR / WER ──────────────────────────────────────────────────────────────

def load_asr_model(device: str = "cuda"):
    """Load Parakeet v3 ASR model.

    Returns the ASR model instance.
    """
    if device in _asr_cache:
        return _asr_cache[device]

    import nemo.collections.asr as nemo_asr

    log.info("Loading Parakeet v3 ASR on %s...", device)
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3"
    )
    model = model.to(device)
    model.eval()

    _asr_cache[device] = model
    log.info("Parakeet v3 ASR loaded on %s", device)
    return model


def _ensure_mono_wav(audio_path: str | Path) -> str:
    """Ensure audio is mono WAV at 16kHz for ASR. Returns path to mono file.

    Parakeet ASR expects (batch, time) — stereo audio causes shape mismatch.
    Converts in-place to a temp file if needed.
    """
    import tempfile

    waveform, sr = torchaudio.load(str(audio_path))
    needs_convert = waveform.shape[0] > 1 or sr != 16000

    if not needs_convert:
        return str(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, 16000)
    return tmp.name


def transcribe_audio(audio_path: str | Path, device: str = "cuda") -> str:
    """Transcribe an audio file using Parakeet v3 ASR.

    Args:
        audio_path: Path to audio file.
        device: CUDA device.

    Returns:
        Transcribed text string.
    """
    mono_path = _ensure_mono_wav(audio_path)
    try:
        model = load_asr_model(device)
        results = model.transcribe([mono_path])
        # Results is a list of transcription strings or hypothesis objects
        if isinstance(results[0], str):
            return results[0]
        return results[0].text if hasattr(results[0], 'text') else str(results[0])
    finally:
        if mono_path != str(audio_path):
            import os
            os.unlink(mono_path)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate between hypothesis and reference.

    Uses simple edit distance at the word level.

    Args:
        hypothesis: ASR transcription.
        reference: Expected text.

    Returns:
        WER as a float (0.0 = perfect, >1.0 = more errors than words).
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Dynamic programming edit distance
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


def score_wer(audio_path: str | Path, expected_text: str,
              device: str = "cuda") -> dict:
    """Transcribe audio and compute WER against expected text.

    Args:
        audio_path: Path to audio file.
        expected_text: Reference text (from DramaBox prompt quotes).
        device: CUDA device.

    Returns:
        Dict with "transcription", "expected", "wer" keys.
    """
    transcription = transcribe_audio(audio_path, device=device)
    wer = compute_wer(transcription, expected_text)
    return {
        "transcription": transcription,
        "expected": expected_text,
        "wer": wer,
    }


# ─── Content Enjoyment (Empathic Insight Plus) ──────────────────────────────

def load_enjoyment_model(device: str = "cuda"):
    """Load the Empathic Insight Plus content enjoyment scorer.

    Uses BUD-E-Whisper encoder for embeddings + MLP expert for scoring.

    Returns (encoder, scorer, processor) tuple.
    """
    if device in _enjoyment_cache:
        return _enjoyment_cache[device]

    from transformers import WhisperModel, WhisperFeatureExtractor
    import torch.nn as nn

    log.info("Loading Empathic Insight Plus on %s...", device)

    # Load BUD-E-Whisper encoder (fine-tuned Whisper Small, 768-dim embeddings)
    encoder_id = "laion/BUD-E-Whisper"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(encoder_id)
    encoder = WhisperModel.from_pretrained(
        encoder_id, torch_dtype=torch.float32
    ).to(device)
    encoder.eval()

    # Load MLP expert for content enjoyment scoring
    # Pooled features: mean + min + max + std = 4 * 768 = 3072 input dim
    scorer_path = _find_enjoyment_scorer()
    if scorer_path:
        scorer = _load_pooled_mlp_scorer(scorer_path, input_dim=3072, device=device)
    else:
        log.warning("Enjoyment scorer weights not found, using random baseline")
        scorer = None

    _enjoyment_cache[device] = (encoder, scorer, feature_extractor)
    log.info("Empathic Insight Plus loaded on %s", device)
    return encoder, scorer, feature_extractor


def _find_enjoyment_scorer() -> str | None:
    """Find the Empathic Insight Plus content enjoyment scorer weights."""
    candidates = [
        Path.home() / ".cache/huggingface/hub/models--laion--Empathic-Insight-Voice-Plus",
        Path("/tmp/empathic_insight_plus"),
    ]
    for base in candidates:
        for p in base.rglob("model_score_content_enjoyment_best.pth"):
            return str(p)
    # Try HF download
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "laion/Empathic-Insight-Voice-Plus",
            "model_score_content_enjoyment_best.pth",
        )
        return path
    except Exception:
        return None


class _PooledEmbeddingMLP(torch.nn.Module):
    """PooledEmbeddingMLP matching Empathic Insight Plus quality experts.

    Architecture: proj(3072->64) -> ReLU/Dropout -> 64->64 -> 64->32 -> 32->16 -> 16->1
    State dict keys: proj.weight, proj.bias, mlp.{2,5,8,11}.weight/bias
    """

    def __init__(self, input_dim: int = 3072):
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


def _load_pooled_mlp_scorer(weights_path: str, input_dim: int = 3072,
                            device: str = "cuda"):
    """Load a PooledEmbeddingMLP scorer from a .pth file."""
    model = _PooledEmbeddingMLP(input_dim)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _pool_features(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pool encoder hidden states: mean + min + max + std -> (4*hidden_dim,)."""
    # hidden_states: (1, seq_len, hidden_dim)
    h = hidden_states.squeeze(0)  # (seq_len, hidden_dim)
    mean = h.mean(dim=0)
    min_val = h.min(dim=0).values
    max_val = h.max(dim=0).values
    std = h.std(dim=0)
    return torch.cat([mean, min_val, max_val, std], dim=0).unsqueeze(0)


def score_content_enjoyment(audio_path: str | Path,
                            device: str = "cuda") -> float:
    """Score content enjoyment of an audio file.

    Args:
        audio_path: Path to audio file.
        device: CUDA device.

    Returns:
        Content enjoyment score (higher is better, typically 0-5 range).
    """
    encoder, scorer, feature_extractor = load_enjoyment_model(device)

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = feature_extractor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        encoder_output = encoder.encoder(input_features)
        hidden_states = encoder_output.last_hidden_state

        if scorer is not None:
            pooled = _pool_features(hidden_states)
            score = scorer(pooled).item()
        else:
            # Fallback: use mean activation magnitude as proxy
            score = hidden_states.mean().item() * 10

    return score


# ─── Composite Reward ────────────────────────────────────────────────────────

def compute_reward(wer: float, content_enjoyment: float) -> float:
    """Compute composite reward from WER and content enjoyment.

    reward = (1 - min(WER, 1.0)) * content_enjoyment

    Args:
        wer: Word Error Rate (0.0 = perfect).
        content_enjoyment: Content enjoyment score (higher is better).

    Returns:
        Composite reward score.
    """
    return (1.0 - min(wer, 1.0)) * max(content_enjoyment, 0.0)


def score_audio(
    audio_path: str | Path,
    dramabox_prompt: str,
    device: str = "cuda",
) -> dict:
    """Full scoring pipeline for a single audio file.

    Args:
        audio_path: Path to audio file to score.
        dramabox_prompt: The DramaBox prompt used to generate this audio.
        device: CUDA device.

    Returns:
        Dict with wer_info, content_enjoyment, and reward.
    """
    expected_text = extract_expected_text(dramabox_prompt)
    wer_info = score_wer(audio_path, expected_text, device=device)
    enjoyment = score_content_enjoyment(audio_path, device=device)
    reward = compute_reward(wer_info["wer"], enjoyment)

    return {
        "audio_path": str(audio_path),
        "transcription": wer_info["transcription"],
        "expected_text": expected_text,
        "wer": wer_info["wer"],
        "content_enjoyment": enjoyment,
        "reward": reward,
    }


def best_of_n(
    audio_paths: list[str | Path],
    dramabox_prompt: str,
    device: str = "cuda",
) -> dict:
    """Select the best audio from N candidates.

    Scores each candidate and returns the one with the highest composite reward.

    Args:
        audio_paths: List of candidate audio file paths.
        dramabox_prompt: The DramaBox prompt used to generate these audios.
        device: CUDA device.

    Returns:
        Dict with "best_path", "best_score", "all_scores" keys.
    """
    all_scores = []
    for path in audio_paths:
        try:
            score = score_audio(path, dramabox_prompt, device=device)
            all_scores.append(score)
        except Exception as e:
            log.error("Failed to score %s: %s", path, e)
            all_scores.append({
                "audio_path": str(path),
                "wer": 1.0,
                "content_enjoyment": 0.0,
                "reward": 0.0,
                "error": str(e),
            })

    best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i].get("reward", 0.0))

    return {
        "best_path": str(audio_paths[best_idx]),
        "best_idx": best_idx,
        "best_score": all_scores[best_idx],
        "all_scores": all_scores,
    }


def unload_all():
    """Free all scoring models from GPU memory."""
    global _asr_cache, _enjoyment_cache
    _asr_cache.clear()
    _enjoyment_cache.clear()
    torch.cuda.empty_cache()
