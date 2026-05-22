"""On-the-fly timbre caption generation using laion/timbre-whisper.

If a reference audio file lacks a precomputed timbre caption, this module
generates one using the Timbre Whisper model (fine-tuned Whisper for
voice timbre description).

Model: laion/timbre-whisper
Input: 16kHz mono audio
Output: Natural language timbre description (age, gender, vocal qualities)
"""
import logging
from pathlib import Path

import torch
import torchaudio

log = logging.getLogger(__name__)

# Singleton model cache (per-process)
_model_cache: dict = {}


def load_timbre_whisper(device: str = "cuda") -> tuple:
    """Load the Timbre Whisper model and processor.

    Returns (processor, model) tuple. Caches per device.
    """
    cache_key = device
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    model_id = "laion/timbre-whisper"
    # The timbre-whisper repo is missing preprocessor_config.json,
    # so we load the processor from the base whisper-small model
    # and the fine-tuned weights from laion/timbre-whisper.
    processor_id = "openai/whisper-small"
    log.info("Loading Timbre Whisper from %s on %s...", model_id, device)

    processor = WhisperProcessor.from_pretrained(processor_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    _model_cache[cache_key] = (processor, model)
    log.info("Timbre Whisper loaded on %s", device)
    return processor, model


def generate_timbre_caption(
    audio_path: str | Path,
    device: str = "cuda",
    max_new_tokens: int = 440,
) -> str:
    """Generate a timbre caption for an audio file.

    Args:
        audio_path: Path to audio file (any format torchaudio supports).
        device: CUDA device string.
        max_new_tokens: Max tokens for caption generation.

    Returns:
        Natural language timbre caption string.
    """
    processor, model = load_timbre_whisper(device)

    # Load and resample to 16kHz mono
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # Prepare inputs
    inputs = processor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device, dtype=torch.float16)

    # Generate caption
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=max_new_tokens,
        )

    caption = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return caption.strip()


def ensure_timbre_caption(
    metadata: dict,
    audio_path: str | Path,
    device: str = "cuda",
    caption_key: str = "timbre_caption",
) -> str:
    """Get timbre caption from metadata, or generate on-the-fly if missing.

    Args:
        metadata: Reference audio metadata dict (may contain timbre_caption).
        audio_path: Path to the reference audio file.
        device: CUDA device for Timbre Whisper model.
        caption_key: Key to look up in metadata.

    Returns:
        Timbre caption string (from metadata or freshly generated).
    """
    # Check metadata for existing caption
    caption = metadata.get(caption_key, "")
    if not caption:
        for alt_key in ["timbre_full_prediction", "timbre_description", "caption"]:
            caption = metadata.get(alt_key, "")
            if caption:
                break

    if caption:
        return caption

    # Generate on-the-fly
    log.info("No timbre caption in metadata, generating with Timbre Whisper for %s", audio_path)
    return generate_timbre_caption(audio_path, device=device)


def unload_timbre_whisper():
    """Free Timbre Whisper model from GPU memory."""
    global _model_cache
    _model_cache.clear()
    torch.cuda.empty_cache()
