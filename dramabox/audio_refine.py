"""Audio refinement pipeline: Resemble Enhance + Chatterbox Voice Conversion.

Two use cases:
1. Clean reference audio: denoise/enhance → optionally self-VC for consistency
2. Polish generated output: enhance → self-VC to match reference voice

Resemble Enhance:
  - denoise(): Removes background noise, outputs at original sample rate
  - enhance(): Full enhancement (denoise + super-resolution), outputs at 48kHz

Chatterbox VC:
  - Voice conversion that clones a target voice from a reference audio
  - Output at 24kHz
"""
import logging
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

log = logging.getLogger(__name__)

# Per-process model caches
_enhance_cache: dict = {}
_vc_cache: dict = {}


def load_resemble_enhance(device: str = "cuda"):
    """Load Resemble Enhance functions.

    Returns (denoise_fn, enhance_fn) tuple.
    """
    if device in _enhance_cache:
        return _enhance_cache[device]

    from resemble_enhance.enhancer.inference import denoise, enhance

    _enhance_cache[device] = (denoise, enhance)
    log.info("Resemble Enhance loaded")
    return denoise, enhance


def load_chatterbox_vc(device: str = "cuda"):
    """Load Chatterbox voice conversion model.

    Returns ChatterboxVC instance.
    """
    if device in _vc_cache:
        return _vc_cache[device]

    from chatterbox.vc import ChatterboxVC

    log.info("Loading ChatterboxVC on %s...", device)
    vc = ChatterboxVC.from_pretrained(device=device)
    _vc_cache[device] = vc
    log.info("ChatterboxVC loaded on %s", device)
    return vc


def enhance_audio(
    audio_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    mode: str = "enhance",
) -> Path:
    """Enhance an audio file using Resemble Enhance.

    Args:
        audio_path: Input audio file path.
        output_path: Where to write the enhanced audio.
        device: CUDA device.
        mode: "denoise" for denoising only, "enhance" for full enhancement.

    Returns:
        Path to the enhanced audio file.
    """
    denoise_fn, enhance_fn = load_resemble_enhance(device)

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    wav_mono = waveform.squeeze(0)

    if mode == "denoise":
        result, out_sr = denoise_fn(wav_mono, sr, device=device)
    else:
        result, out_sr = enhance_fn(wav_mono, sr, device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), result.cpu().numpy(), out_sr)
    log.info("Enhanced %s -> %s (mode=%s, sr=%d)", audio_path, output_path, mode, out_sr)
    return output_path


def voice_convert(
    source_audio: str | Path,
    reference_audio: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    skip_watermark: bool = True,
) -> Path:
    """Convert voice in source audio to match reference audio using Chatterbox VC.

    Args:
        source_audio: Audio to transform (keeps content, changes voice).
        reference_audio: Target voice to clone.
        output_path: Where to write the converted audio.
        device: CUDA device.
        skip_watermark: Skip Perth watermarker (saves ~4.7s per file).

    Returns:
        Path to the voice-converted audio file.
    """
    vc = load_chatterbox_vc(device)

    if skip_watermark:
        result = _vc_generate_fast(vc, str(source_audio), str(reference_audio))
    else:
        result = vc.generate(
            audio=str(source_audio),
            target_voice_path=str(reference_audio),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ChatterboxVC outputs tensor at 24kHz
    out_sr = 24000
    if isinstance(result, torch.Tensor):
        audio_np = result.cpu().squeeze().numpy()
    else:
        audio_np = result
    sf.write(str(output_path), audio_np, out_sr)
    log.info("Voice converted %s -> %s (ref=%s)", source_audio, output_path, reference_audio)
    return output_path


# Track last reference to avoid redundant set_target_voice calls
_last_ref_path: str = ""


def _vc_generate_fast(vc, source_audio: str, reference_audio: str):
    """Fast VC generation: skip watermark, cache reference, use torchaudio."""
    import librosa
    global _last_ref_path

    # Only re-embed reference voice if it changed
    if reference_audio != _last_ref_path:
        vc.set_target_voice(reference_audio)
        _last_ref_path = reference_audio

    with torch.inference_mode():
        audio_16, _ = librosa.load(source_audio, sr=16000)
        audio_16 = torch.from_numpy(audio_16).float().to(vc.device)[None,]

        s3_tokens, _ = vc.s3gen.tokenizer(audio_16)
        wav, _ = vc.s3gen.inference(
            speech_tokens=s3_tokens,
            ref_dict=vc.ref_dict,
        )
        wav = wav.squeeze(0).detach().cpu()
        # Skip watermark — saves ~4.7s per file
    return wav


def self_voice_convert(
    audio_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> Path:
    """Self-voice-clone: use the audio as both source and reference.

    This helps regularize the output by passing it through the VC model
    with itself as reference, improving voice consistency.

    Args:
        audio_path: Audio file to self-clone.
        output_path: Where to write the result.
        device: CUDA device.

    Returns:
        Path to the self-cloned audio file.
    """
    return voice_convert(audio_path, audio_path, output_path, device=device)


def refine_reference_audio(
    audio_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    enhance_mode: str = "enhance",
    do_self_vc: bool = True,
) -> dict:
    """Full refinement pipeline for reference audio.

    Steps:
    1. Resemble Enhance (denoise or full enhance)
    2. Optionally self-VC with Chatterbox

    Args:
        audio_path: Input reference audio.
        output_dir: Directory for refined outputs.
        device: CUDA device.
        enhance_mode: "denoise" or "enhance".
        do_self_vc: Whether to apply self-VC after enhancement.

    Returns:
        Dict with paths: {"enhanced": path, "self_vc": path_or_None, "final": path}
    """
    output_dir = Path(output_dir)
    stem = Path(audio_path).stem

    enhanced_path = output_dir / f"{stem}_enhanced.wav"
    enhance_audio(audio_path, enhanced_path, device=device, mode=enhance_mode)

    result = {"enhanced": enhanced_path, "self_vc": None, "final": enhanced_path}

    if do_self_vc:
        vc_path = output_dir / f"{stem}_enhanced_selfvc.wav"
        self_voice_convert(enhanced_path, vc_path, device=device)
        result["self_vc"] = vc_path
        result["final"] = vc_path

    return result


def refine_generated_audio(
    generated_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    enhance_mode: str = "enhance",
    do_vc_to_ref: bool = True,
) -> dict:
    """Full refinement pipeline for generated TTS output.

    Steps:
    1. Resemble Enhance (denoise or full enhance)
    2. Optionally VC to match reference voice

    Args:
        generated_path: TTS output audio.
        reference_path: Reference audio for voice matching.
        output_dir: Directory for refined outputs.
        device: CUDA device.
        enhance_mode: "denoise" or "enhance".
        do_vc_to_ref: Whether to VC the enhanced output to match reference.

    Returns:
        Dict with paths: {"enhanced": path, "vc": path_or_None, "final": path}
    """
    output_dir = Path(output_dir)
    stem = Path(generated_path).stem

    enhanced_path = output_dir / f"{stem}_enhanced.wav"
    enhance_audio(generated_path, enhanced_path, device=device, mode=enhance_mode)

    result = {"enhanced": enhanced_path, "vc": None, "final": enhanced_path}

    if do_vc_to_ref:
        vc_path = output_dir / f"{stem}_vc.wav"
        voice_convert(enhanced_path, reference_path, vc_path, device=device)
        result["vc"] = vc_path
        result["final"] = vc_path

    return result


def voice_convert_batch(
    source_paths: list[str | Path],
    reference_audio: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
    skip_watermark: bool = True,
) -> list[Path]:
    """Convert voice in multiple source audio files to match a single reference.

    Pre-loads the VC model once and caches the reference voice embedding.

    Args:
        source_paths: List of audio files to transform.
        reference_audio: Target voice to clone for all conversions.
        output_dir: Directory for converted outputs.
        device: CUDA device.
        skip_watermark: Skip Perth watermarker (saves ~4.7s per file).

    Returns:
        List of paths to voice-converted audio files.
    """
    vc = load_chatterbox_vc(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-set reference voice once for all files
    vc.set_target_voice(str(reference_audio))
    global _last_ref_path
    _last_ref_path = str(reference_audio)

    results = []
    for src in source_paths:
        src = Path(src)
        out_path = output_dir / f"{src.stem}_vc.wav"

        try:
            if skip_watermark:
                result = _vc_generate_fast(vc, str(src), str(reference_audio))
            else:
                result = vc.generate(audio=str(src))

            out_sr = 24000
            if isinstance(result, torch.Tensor):
                audio_np = result.cpu().squeeze().numpy()
            else:
                audio_np = result
            sf.write(str(out_path), audio_np, out_sr)
            log.info("Batch VC: %s -> %s", src.name, out_path.name)
            results.append(out_path)
        except Exception as e:
            log.error("Batch VC failed for %s: %s", src.name, e)
            results.append(None)

    return results


def unload_all():
    """Free all refinement models from GPU memory."""
    global _enhance_cache, _vc_cache
    _enhance_cache.clear()
    _vc_cache.clear()
    torch.cuda.empty_cache()
