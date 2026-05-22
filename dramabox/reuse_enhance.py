"""NVIDIA RE-USE: Universal Speech Enhancement.

Wraps the nvidia/RE-USE model (9.6M param SEMamba architecture) for
speech enhancement. Removes noise, reverberation, clipping, codec
artifacts, and bandwidth limitations while preserving speaker identity,
emotion, and linguistic content.

Model: nvidia/RE-USE
Architecture: SEMamba (bi-directional Mamba, 30 layers, 9.6M params)
Input/Output: mono WAV, 8-48 kHz
"""
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

log = logging.getLogger(__name__)

_reuse_cache: dict = {}

RELU = nn.ReLU()


def _ensure_reuse_modules():
    """Ensure RE-USE model code is importable.

    Downloads from HuggingFace Hub if needed, adds to sys.path.
    """
    # Check if already importable
    try:
        from models.generator_SEMamba_time_d4 import SEMamba
        from models.stfts import mag_phase_stft, mag_phase_istft
        from utils.util import load_config, pad_or_trim_to_match
        return
    except ImportError:
        pass

    # Download model files from HuggingFace
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = "nvidia/RE-USE"
    cache_dir = Path.home() / ".cache" / "reuse_model"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download the full repo to get model code
    local_dir = snapshot_download(
        repo_id,
        local_dir=str(cache_dir / "RE-USE"),
        ignore_patterns=["*.md", ".gitattributes", "noisy_audio/*", "long_noisy_audio/*"],
    )

    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)

    log.info("RE-USE model code loaded from %s", local_dir)


def load_reuse(device: str = "cuda"):
    """Load the RE-USE speech enhancement model.

    Returns (model, config) tuple.
    """
    if device in _reuse_cache:
        return _reuse_cache[device]

    _ensure_reuse_modules()

    from huggingface_hub import hf_hub_download
    from models.generator_SEMamba_time_d4 import SEMamba
    from utils.util import load_config

    config_path = hf_hub_download(repo_id="nvidia/RE-USE", filename="config.json")
    cfg = load_config(config_path)

    log.info("Loading RE-USE model on %s...", device)
    model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=cfg).to(device)
    model.eval()

    _reuse_cache[device] = (model, cfg)
    log.info("RE-USE loaded on %s (9.6M params)", device)
    return model, cfg


def _make_even(value):
    """Round to nearest even integer."""
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def enhance_audio(
    input_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    bandwidth_extend_to: int | None = None,
) -> Path:
    """Enhance a single audio file using RE-USE.

    Args:
        input_path: Path to noisy/degraded audio file.
        output_path: Where to write the enhanced audio.
        device: CUDA device.
        bandwidth_extend_to: Optional target sample rate for bandwidth extension.
            If set, input is first resampled to this rate before enhancement.

    Returns:
        Path to the enhanced audio file.
    """
    _ensure_reuse_modules()
    from models.stfts import mag_phase_stft, mag_phase_istft
    from utils.util import pad_or_trim_to_match

    model, cfg = load_reuse(device)

    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    # Load audio
    noisy_wav, noisy_sr = torchaudio.load(str(input_path))

    # Bandwidth extension: resample to target rate
    if bandwidth_extend_to is not None:
        try:
            import librosa
            noisy_wav = torch.FloatTensor(
                librosa.resample(
                    noisy_wav.cpu().numpy(),
                    orig_sr=noisy_sr,
                    target_sr=bandwidth_extend_to,
                    res_type="kaiser_best",
                )
            )
            noisy_sr = bandwidth_extend_to
        except ImportError:
            noisy_wav = torchaudio.functional.resample(noisy_wav, noisy_sr, bandwidth_extend_to)
            noisy_sr = bandwidth_extend_to

    noisy_wav = noisy_wav.to(device)

    # Scale STFT params to match input sample rate
    n_fft_scaled = _make_even(n_fft * noisy_sr // sampling_rate)
    hop_size_scaled = _make_even(hop_size * noisy_sr // sampling_rate)
    win_size_scaled = _make_even(win_size * noisy_sr // sampling_rate)

    with torch.no_grad():
        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
            noisy_wav,
            n_fft=n_fft_scaled,
            hop_size=hop_size_scaled,
            win_size=win_size_scaled,
            compress_factor=compress_factor,
            center=True,
            addeps=False,
        )

        amp_g, pha_g, _ = model(noisy_mag, noisy_pha)

        # Remove "strange sweep artifact" (from official inference.py)
        mag = torch.expm1(RELU(amp_g))
        zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0

        audio_g = mag_phase_istft(
            amp_g, pha_g,
            n_fft_scaled, hop_size_scaled, win_size_scaled,
            compress_factor,
        )
        audio_g = pad_or_trim_to_match(noisy_wav.detach(), audio_g, pad_value=1e-8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio_g.cpu(), noisy_sr)

    log.info("RE-USE enhanced %s -> %s (sr=%d)", input_path, output_path, noisy_sr)
    return output_path


def enhance_batch(
    input_paths: list[str | Path],
    output_dir: str | Path,
    device: str = "cuda",
    bandwidth_extend_to: int | None = None,
) -> list[dict]:
    """Enhance multiple audio files.

    Args:
        input_paths: List of input audio file paths.
        output_dir: Directory for enhanced outputs.
        device: CUDA device.
        bandwidth_extend_to: Optional bandwidth extension target rate.

    Returns:
        List of dicts with "input", "output", "status" keys.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for path in input_paths:
        path = Path(path)
        out_path = output_dir / f"{path.stem}_reuse.wav"
        try:
            enhance_audio(path, out_path, device=device,
                          bandwidth_extend_to=bandwidth_extend_to)
            results.append({"input": str(path), "output": str(out_path), "status": "ok"})
        except Exception as e:
            log.error("RE-USE failed on %s: %s", path, e)
            results.append({"input": str(path), "output": "", "status": f"error: {e}"})

    return results


def unload_reuse():
    """Free RE-USE model from GPU memory."""
    global _reuse_cache
    _reuse_cache.clear()
    torch.cuda.empty_cache()
