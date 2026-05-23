"""Path D: MOSS Audio Thinking pipeline.

Uses MOSS-Audio-8B-Thinking to listen to reference audio and extract
speaker attributes via chain-of-thought reasoning. The model directly
analyzes the audio to produce a DramaBox-compatible speaker description,
bypassing the need for separate timbre whisper annotation.

Model: OpenMOSS-Team/MOSS-Audio-8B-Thinking (multi-modal audio understanding with reasoning)
Input: Reference audio + text instruction
Output: Reasoning trace + final speaker description / DramaBox prompt

API pattern (from MOSS-Audio/infer.py):
    model = MossAudioModel.from_pretrained(path, dtype="auto", device_map=device)
    processor = MossAudioProcessor.from_pretrained(path, enable_time_marker=True)
    raw_audio = load_audio(audio_path, sample_rate=processor.config.mel_sr)
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    text = processor.decode(generated_ids[0, input_len:], skip_special_tokens=True)
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch

log = logging.getLogger(__name__)

# Per-process model cache
_moss_cache: dict = {}

# Default model path — 8B Thinking model
DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"

# MOSS Audio prompts for DramaBox speaker analysis
MOSS_ANALYSIS_PROMPT = """\
Listen carefully to this audio recording of a speaker. Analyze the speaker's voice \
and provide a detailed description of their vocal characteristics.

Describe the following:
1. Perceived gender and approximate age range
2. Pitch range (high, medium, low)
3. Voice quality (breathy, clear, rough, smooth, nasal, resonant, etc.)
4. Vocal energy and tempo
5. Emotional state and delivery style
6. Any distinctive vocal features (accent, vocal fry, vibrato, etc.)

Think step by step about what you hear, then provide a final concise speaker \
description suitable for a voice acting prompt (2-3 sentences)."""

MOSS_DRAMABOX_PROMPT = """\
Listen to this audio recording. Your ONLY task regarding the audio is to analyze \
the SPEAKER'S VOICE CHARACTERISTICS: age, gender, timbre, pitch, accent, and \
vocal quality. Completely IGNORE the words and content spoken in the audio — \
they are irrelevant.

Then, create a DramaBox-format voice prompt with ENTIRELY ORIGINAL dialogue \
that you invent from scratch. Do NOT transcribe, quote, paraphrase, or \
reproduce ANY of the words spoken in the recording. The dialogue content must \
be completely new and unrelated to what is said in the audio.

DramaBox format:
- Start with a speaker description (age, gender, timbre, voice quality) in English, \
matching the voice you heard
- Include that this is a pristine, high-quality studio recording with no background noise
- Then alternate between stage directions (English, outside quotes) and \
spoken dialogue (in {language}, inside double quotes "...")
- Write approximately {word_count} words of ORIGINAL dialogue (minimum 10 words)
- Make the scene dramatically interesting — the words should tell a compelling \
story fragment, reveal character, or capture a vivid emotional moment
- The emotions to convey: {emotions}
- Everything is ONE speaker only. No dialogue partners.

Think step by step: first describe the speaker's voice characteristics only \
(ignore what they say), then write the complete DramaBox prompt with brand \
new creative dialogue that fits the emotions."""


def _check_transformers_version():
    """Check if transformers version is compatible with MOSS-Audio.

    MOSS-Audio-8B-Thinking requires transformers ~4.57.x. With 5.x, the
    Qwen3 backbone produces degenerate output (repeated <think> tokens).
    """
    import transformers
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        log.warning(
            "transformers %s detected. MOSS-Audio requires transformers ~4.57.x. "
            "With transformers 5.x, output may be degenerate. "
            "Use a venv with transformers==4.57.1 for MOSS inference.",
            transformers.__version__,
        )
        return False
    return True


def load_moss_audio(device: str = "cuda", moss_dir: str = "",
                    model_path: str = ""):
    """Load the MOSS-Audio-8B-Thinking model using native API.

    Args:
        device: CUDA device string (e.g. "cuda:0").
        moss_dir: Path to MOSS-Audio source directory (for imports).
        model_path: Model path or HuggingFace ID. Defaults to 8B-Thinking.

    Returns:
        (model, processor) tuple.

    Note:
        Requires transformers ~4.57.x. With transformers 5.x, the Qwen3
        backbone produces degenerate output. Use the venv at /tmp/moss_venv
        or install transformers==4.57.1 in a separate environment.
    """
    if device in _moss_cache:
        return _moss_cache[device]

    _check_transformers_version()

    # Add MOSS-Audio src/ to sys.path for imports
    if moss_dir:
        src_dir = os.path.join(moss_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
    else:
        # Try default location
        default_src = "/home/deployer/laion/MOSS-Audio/src"
        if os.path.isdir(default_src) and default_src not in sys.path:
            sys.path.insert(0, default_src)

    from modeling_moss_audio import MossAudioModel
    from processing_moss_audio import MossAudioProcessor
    from audio_io import load_audio  # noqa: F401 — stash in cache for later

    model_path = model_path or DEFAULT_MODEL_PATH

    # Check for local weights first
    local_weights = os.path.join(
        moss_dir or "/home/deployer/laion/MOSS-Audio",
        "weights", "MOSS-Audio-8B-Thinking",
    )
    if os.path.isdir(local_weights):
        model_path = local_weights
        log.info("Using local MOSS-Audio-8B-Thinking weights at %s", model_path)

    log.info("Loading MOSS-Audio-8B-Thinking from %s on %s...", model_path, device)

    model = MossAudioModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map=device,
    )
    model.eval()

    processor = MossAudioProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    _moss_cache[device] = (model, processor)
    log.info("MOSS-Audio-8B-Thinking loaded on %s", device)
    return model, processor


def _load_audio_for_moss(audio_path: str | Path, moss_dir: str = ""):
    """Load audio using MOSS-Audio's native load_audio function."""
    if moss_dir:
        src_dir = os.path.join(moss_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
    else:
        default_src = "/home/deployer/laion/MOSS-Audio/src"
        if os.path.isdir(default_src) and default_src not in sys.path:
            sys.path.insert(0, default_src)

    from audio_io import load_audio
    from processing_moss_audio import MelConfig

    config = MelConfig()
    return load_audio(str(audio_path), sample_rate=config.mel_sr)


def _run_moss_generation(
    model, processor, prompt: str, audio_path: str | Path,
    moss_dir: str = "", max_new_tokens: int = 1024,
    temperature: float = 0.7, top_p: float = 0.9,
) -> tuple[str, float, int]:
    """Run MOSS-Audio generation with proper input preparation.

    Returns (output_text, elapsed_seconds, n_tokens).
    """
    raw_audio = _load_audio_for_moss(audio_path, moss_dir)

    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs = inputs.to(model.device)

    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)

    # Set audio input mask (required by MOSS-Audio)
    audio_input_mask = inputs["input_ids"] == processor.audio_token_id
    inputs["audio_input_mask"] = audio_input_mask

    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            use_cache=True,
        )
    elapsed = time.time() - t0

    new_tokens = generated_ids[0, input_len:]
    output_text = processor.decode(new_tokens, skip_special_tokens=True)
    n_tokens = len(new_tokens)

    return output_text, elapsed, n_tokens


def analyze_speaker(
    audio_path: str | Path,
    device: str = "cuda",
    moss_dir: str = "",
    model_path: str = "",
    max_new_tokens: int = 1024,
) -> dict:
    """Analyze a speaker's voice using MOSS Audio 8B thinking.

    Args:
        audio_path: Path to reference audio file.
        device: CUDA device.
        moss_dir: Path to MOSS-Audio source directory.
        model_path: Model path override.
        max_new_tokens: Maximum tokens for generation.

    Returns:
        Dict with "reasoning_trace" and "speaker_description" keys.
    """
    model, processor = load_moss_audio(device, moss_dir, model_path)

    output_text, elapsed, n_tokens = _run_moss_generation(
        model, processor, MOSS_ANALYSIS_PROMPT, audio_path,
        moss_dir=moss_dir, max_new_tokens=max_new_tokens,
    )

    reasoning, description = _split_reasoning(output_text)

    log.info("MOSS analysis of %s completed in %.1fs (%d tokens)",
             audio_path, elapsed, n_tokens)

    return {
        "audio_path": str(audio_path),
        "reasoning_trace": reasoning,
        "speaker_description": description,
        "full_output": output_text,
        "generation_time": elapsed,
        "tokens_generated": n_tokens,
    }


def generate_dramabox_prompt(
    audio_path: str | Path,
    language: str = "English",
    word_count: int = 30,
    emotions: str = "Contemplation (clearly present)",
    device: str = "cuda",
    moss_dir: str = "",
    model_path: str = "",
    max_new_tokens: int = 2048,
) -> dict:
    """Generate a full DramaBox prompt from reference audio using MOSS Audio 8B.

    This is Path D: the model listens to the audio, reasons about the voice,
    and generates a complete DramaBox prompt.

    Args:
        audio_path: Path to reference audio file.
        language: Target language for dialogue.
        word_count: Target word count for dialogue.
        emotions: Emotions to convey in the performance.
        device: CUDA device.
        moss_dir: Path to MOSS-Audio source directory.
        model_path: Model path override.
        max_new_tokens: Maximum tokens for generation.

    Returns:
        Dict with reasoning_trace, dramabox_prompt, and metadata.
    """
    model, processor = load_moss_audio(device, moss_dir, model_path)

    prompt = MOSS_DRAMABOX_PROMPT.format(
        language=language,
        word_count=word_count,
        emotions=emotions,
    )

    output_text, elapsed, n_tokens = _run_moss_generation(
        model, processor, prompt, audio_path,
        moss_dir=moss_dir, max_new_tokens=max_new_tokens,
        temperature=0.8, top_p=0.95,
    )

    reasoning, dramabox_prompt = _split_reasoning(output_text)

    log.info("MOSS DramaBox generation for %s completed in %.1fs",
             audio_path, elapsed)

    return {
        "audio_path": str(audio_path),
        "language": language,
        "word_count": word_count,
        "emotions": emotions,
        "reasoning_trace": reasoning,
        "dramabox_prompt": dramabox_prompt,
        "full_output": output_text,
        "generation_time": elapsed,
        "tokens_generated": n_tokens,
    }


def _split_reasoning(text: str) -> tuple[str, str]:
    """Split MOSS output into reasoning trace and final content.

    The model typically uses <think>...</think> tags to separate
    reasoning from the final output.
    """
    # Try <think>...</think> pattern
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        after_think = text[think_match.end():].strip()
        return reasoning, after_think

    # Try common separator patterns
    for marker in ["Final answer:", "Final description:", "DramaBox prompt:",
                    "Speaker description:", "Here is the prompt:",
                    "Here's the DramaBox", "Here is the DramaBox"]:
        if marker.lower() in text.lower():
            idx = text.lower().index(marker.lower())
            reasoning = text[:idx].strip()
            final = text[idx + len(marker):].strip()
            return reasoning, final

    # If no clear separation, treat last paragraph as final
    paragraphs = text.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1])
        final = paragraphs[-1]
        return reasoning, final

    return "", text


def save_moss_result(result: dict, output_path: str | Path):
    """Save MOSS analysis/generation result to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("Saved MOSS result to %s", output_path)


def unload_moss():
    """Free MOSS Audio model from GPU memory."""
    global _moss_cache
    _moss_cache.clear()
    torch.cuda.empty_cache()
