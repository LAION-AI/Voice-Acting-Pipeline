"""Sidon + ChatterboxVC augmentation for DramaBox post-processing.

Replaces the RE-USE + LavaSR enhancement chain. For each raw TTS sample,
runs two parallel paths and picks the best by DNS-MOS OVR score:

  Path A: Sidon only       (16 kHz -> 48 kHz)
  Path B: ChatterboxVC     (any SR -> 24 kHz) then Sidon (-> 48 kHz)

Models:
  - Sidon (sarulab-speech/sidon_raw_weight): w2v-BERT LoRA encoder + DAC decoder
  - Chatterbox VC (chatterbox.vc.ChatterboxVC): S3Gen flow-matching VC
  - DNS-MOS (PyTorch native): SIG/BAK/OVR scorer at 16 kHz

For two-part CUT TO: audio, a self-VC of the full audio provides the VC
target to ensure speaker consistency across both scenes.
"""
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNS-MOS scorer (loaded from sidon repo)
# ---------------------------------------------------------------------------
DNSMOS_SCRIPT = Path("/home/deployer/laion/sidon/dnsmos_pytorch.py")

_dnsmos_cache: dict = {}


def _load_dnsmos(device: str = "cuda"):
    """Load the PyTorch DNS-MOS model, cached per device."""
    if device in _dnsmos_cache:
        return _dnsmos_cache[device]

    # Import from the sidon repo
    import importlib.util
    spec = importlib.util.spec_from_file_location("dnsmos_pytorch", str(DNSMOS_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.DNSMOSPyTorch.from_onnx(device=device)
    _dnsmos_cache[device] = model
    log.info("DNS-MOS loaded on %s", device)
    return model


def score_dnsmos(waveform_16k: np.ndarray, device: str = "cuda") -> float:
    """Score a 16 kHz mono waveform and return the OVR score (1-5 scale).

    The waveform is chunked into 9-second windows (the model's expected
    input length) and scores are averaged.
    """
    model = _load_dnsmos(device)
    CHUNK = 144160  # 9.01s at 16 kHz

    wav = torch.from_numpy(waveform_16k).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Chunk long audio
    total = wav.shape[-1]
    if total <= CHUNK:
        chunks = [wav]
    else:
        chunks = []
        for start in range(0, total, CHUNK):
            end = min(start + CHUNK, total)
            if end - start < 16000:  # skip very short trailing chunks
                continue
            chunks.append(wav[..., start:end])

    ovr_scores = []
    with torch.no_grad():
        for c in chunks:
            c = c.to(device)
            scores = model(c)  # [B, 3] = [SIG, BAK, OVR]
            ovr_scores.append(scores[0, 2].item())

    return float(np.mean(ovr_scores)) if ovr_scores else 1.0


# ---------------------------------------------------------------------------
# Sidon model loader
# ---------------------------------------------------------------------------
_sidon_cache: dict = {}


def _load_sidon(device: str = "cuda"):
    """Load Sidon encoder (w2v-BERT 2.0 + LoRA) + decoder (DAC).

    The encoder is a w2v-BERT 2.0 model with a PEFT LoRA adapter.
    The decoder is a DAC Decoder with a pretrained state dict.

    Returns (encoder, decoder, processor) cached per device.
    """
    if device in _sidon_cache:
        return _sidon_cache[device]

    import transformers
    from peft import PeftModel
    from huggingface_hub import hf_hub_download
    import dac

    log.info("Loading Sidon models on %s...", device)

    # --- Encoder: w2v-BERT 2.0 + LoRA adapter ---
    base_model = transformers.Wav2Vec2BertModel.from_pretrained(
        "facebook/w2v-bert-2.0",
        num_hidden_layers=8,
        layerdrop=0.0,
    )

    # Download adapter files
    adapter_dir = str(Path(hf_hub_download(
        "sarulab-speech/sidon_raw_weight",
        "adapter_config.json",
    )).parent)
    hf_hub_download("sarulab-speech/sidon_raw_weight", "adapter_model.safetensors")

    encoder = PeftModel.from_pretrained(base_model, adapter_dir)
    encoder = encoder.merge_and_unload()
    encoder = encoder.to(device).eval()

    # --- Processor: for converting raw audio to input features ---
    processor = transformers.AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    # --- Decoder: DAC with pretrained state dict ---
    decoder = dac.model.dac.Decoder(
        input_channel=1024,
        channels=1536,
        rates=[8, 5, 4, 3, 2],
    )
    decoder_path = hf_hub_download(
        "sarulab-speech/sidon_raw_weight",
        "decoder_state_dict.pt",
    )
    state_dict = torch.load(decoder_path, map_location="cpu", weights_only=True)
    decoder.load_state_dict(state_dict, strict=True)
    decoder = decoder.to(device).eval()

    _sidon_cache[device] = (encoder, decoder, processor)
    log.info("Sidon loaded on %s", device)
    return encoder, decoder, processor


# ---------------------------------------------------------------------------
# ChatterboxVC loader
# ---------------------------------------------------------------------------
_vc_cache: dict = {}


def _load_chatterbox_vc(device: str = "cuda"):
    """Load Chatterbox VC model, cached per device."""
    if device in _vc_cache:
        return _vc_cache[device]

    from chatterbox.vc import ChatterboxVC

    log.info("Loading ChatterboxVC on %s...", device)
    vc = ChatterboxVC.from_pretrained(device=device)
    _vc_cache[device] = vc
    log.info("ChatterboxVC loaded on %s", device)
    return vc


# ---------------------------------------------------------------------------
# Core enhancement functions
# ---------------------------------------------------------------------------
def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample numpy waveform."""
    if orig_sr == target_sr:
        return wav
    t = torch.from_numpy(wav).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    t = torchaudio.functional.resample(t, orig_sr, target_sr)
    return t.squeeze(0).numpy()


def enhance_sidon(wav_path: str | Path, device: str = "cuda") -> tuple[np.ndarray, float]:
    """Apply Sidon enhancement only.

    Input: any audio file.
    Output: (enhanced_48k_numpy, dns_mos_ovr_score).

    Pipeline: load audio -> resample to 16 kHz -> w2v-BERT feature extraction
    -> DAC decoder -> 48 kHz output -> DNS-MOS scoring.
    """
    encoder, decoder, processor = _load_sidon(device)

    # Load and resample to 16 kHz
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # Normalize to ±0.9
    wav_mono = wav.squeeze(0)
    max_val = wav_mono.abs().max().clamp_min(1e-6)
    wav_mono = 0.9 * wav_mono / max_val

    # Extract input features via processor
    inputs = processor(
        wav_mono.numpy(), sampling_rate=16000,
        return_tensors="pt", padding=True,
    )
    input_features = inputs["input_features"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        # Extract SSL features via encoder [B, T, 1024]
        features = encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        ).last_hidden_state

        # Transpose for decoder [B, 1024, T]
        features_t = features.transpose(1, 2)

        # Decode to 48 kHz [B, 1, T_48k]
        enhanced = decoder(features_t)

    enhanced_np = enhanced.squeeze().float().cpu().numpy()

    # Normalize output
    out_max = np.abs(enhanced_np).max()
    if out_max > 0:
        enhanced_np = enhanced_np * (0.9 / out_max)

    # Score with DNS-MOS (needs 16 kHz)
    enhanced_16k = _resample(enhanced_np, 48000, 16000)
    ovr = score_dnsmos(enhanced_16k, device=device)

    return enhanced_np, ovr


def enhance_vc_sidon(
    wav_path: str | Path,
    ref_path: str | Path | None = None,
    device: str = "cuda",
) -> tuple[np.ndarray, float]:
    """Apply Chatterbox VC then Sidon.

    If ref_path is provided, voice-converts to match the reference speaker.
    If ref_path is None, performs self-VC (voice-converts to itself for
    artifact removal / consistency).

    Output: (enhanced_48k_numpy, dns_mos_ovr_score).
    """
    vc = _load_chatterbox_vc(device)

    # Determine VC target
    target = str(ref_path) if ref_path else str(wav_path)

    # Run Chatterbox VC (outputs 24 kHz)
    vc_wav = vc.generate(
        audio=str(wav_path),
        target_voice_path=target,
    )
    # vc_wav is a torch tensor at 24 kHz
    if isinstance(vc_wav, torch.Tensor):
        vc_np = vc_wav.squeeze().cpu().numpy()
    else:
        vc_np = np.asarray(vc_wav).squeeze()

    # Save VC output to temp file for Sidon input
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        import soundfile as sf
        sf.write(tmp_path, vc_np, 24000)

        # Apply Sidon to VC output -> 48 kHz
        enhanced_np, _ = enhance_sidon(tmp_path, device=device)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Score with DNS-MOS
    enhanced_16k = _resample(enhanced_np, 48000, 16000)
    ovr = score_dnsmos(enhanced_16k, device=device)

    return enhanced_np, ovr


# ---------------------------------------------------------------------------
# High-level augmentation API
# ---------------------------------------------------------------------------
class SidonEnhancer:
    """Augmentation engine: runs both Sidon-only and VC+Sidon paths,
    picks the best by DNS-MOS OVR score.

    Usage::

        enhancer = SidonEnhancer(device="cuda:0")

        # Single sample
        best_wav, method = enhancer.augment_sample(wav_path)

        # Two-part CUT TO:
        results = enhancer.augment_two_part(full_wav, part1_wav, part2_wav)
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        # Pre-load all models
        _load_dnsmos(device)
        _load_sidon(device)
        _load_chatterbox_vc(device)
        log.info("SidonEnhancer initialized on %s", device)

    def augment_sample(
        self,
        wav_path: str | Path,
        ref_path: str | Path | None = None,
    ) -> tuple[np.ndarray, str]:
        """Run both enhancement paths, pick best by DNS-MOS OVR.

        Args:
            wav_path: Input audio file.
            ref_path: Optional reference audio for voice conversion target.
                If None and VC path is run, self-VC is used.

        Returns:
            (best_waveform_48k, method_used) where method is "sidon" or "vc+sidon".
        """
        sidon_wav, sidon_ovr = enhance_sidon(wav_path, device=self.device)
        log.info("  Sidon-only: DNS-MOS OVR=%.3f", sidon_ovr)

        try:
            vc_sidon_wav, vc_sidon_ovr = enhance_vc_sidon(
                wav_path, ref_path=ref_path, device=self.device
            )
            log.info("  VC+Sidon: DNS-MOS OVR=%.3f", vc_sidon_ovr)
        except Exception as e:
            log.warning("  VC+Sidon failed (%s), using Sidon-only", e)
            return sidon_wav, "sidon"

        if vc_sidon_ovr > sidon_ovr:
            return vc_sidon_wav, "vc+sidon"
        return sidon_wav, "sidon"

    def augment_two_part(
        self,
        full_wav_path: str | Path,
        part1_wav_path: str | Path,
        part2_wav_path: str | Path,
    ) -> dict:
        """Handle CUT TO: two-part audio with speaker consistency.

        Strategy:
        1. Self-VC the full audio -> Sidon -> full_enhanced (VC target)
        2. For each part: compare Sidon-only vs VC-to-full_enhanced + Sidon
        3. Pick best per part by DNS-MOS OVR

        Returns:
            {
                "full_enhanced": np.ndarray (48 kHz),
                "full_method": str,
                "full_ovr": float,
                "part1_enhanced": np.ndarray (48 kHz),
                "part1_method": str,
                "part1_ovr": float,
                "part2_enhanced": np.ndarray (48 kHz),
                "part2_method": str,
                "part2_ovr": float,
            }
        """
        # Step 1: Create full_enhanced as VC target
        # Self-VC full audio then Sidon
        full_vc_wav, full_vc_ovr = enhance_vc_sidon(
            full_wav_path, ref_path=None, device=self.device
        )
        full_sidon_wav, full_sidon_ovr = enhance_sidon(
            full_wav_path, device=self.device
        )

        if full_vc_ovr > full_sidon_ovr:
            full_enhanced = full_vc_wav
            full_method = "vc+sidon"
            full_ovr = full_vc_ovr
        else:
            full_enhanced = full_sidon_wav
            full_method = "sidon"
            full_ovr = full_sidon_ovr

        log.info("  Full: %s (OVR=%.3f)", full_method, full_ovr)

        # Save full_enhanced as VC reference for parts
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            full_ref_path = tmp.name
        import soundfile as sf
        sf.write(full_ref_path, full_enhanced, 48000)

        results = {
            "full_enhanced": full_enhanced,
            "full_method": full_method,
            "full_ovr": full_ovr,
        }

        try:
            # Step 2: Process each part
            for part_name, part_path in [("part1", part1_wav_path), ("part2", part2_wav_path)]:
                # Path A: Sidon only
                sidon_wav, sidon_ovr = enhance_sidon(part_path, device=self.device)

                # Path B: VC to full_enhanced + Sidon
                try:
                    vc_wav, vc_ovr = enhance_vc_sidon(
                        part_path, ref_path=full_ref_path, device=self.device
                    )
                except Exception as e:
                    log.warning("  %s VC+Sidon failed (%s), using Sidon-only", part_name, e)
                    vc_wav, vc_ovr = None, -1.0

                if vc_ovr > sidon_ovr:
                    results[f"{part_name}_enhanced"] = vc_wav
                    results[f"{part_name}_method"] = "vc+sidon"
                    results[f"{part_name}_ovr"] = vc_ovr
                else:
                    results[f"{part_name}_enhanced"] = sidon_wav
                    results[f"{part_name}_method"] = "sidon"
                    results[f"{part_name}_ovr"] = sidon_ovr

                log.info("  %s: %s (OVR=%.3f)",
                         part_name, results[f"{part_name}_method"], results[f"{part_name}_ovr"])
        finally:
            try:
                os.unlink(full_ref_path)
            except OSError:
                pass

        return results

    def cleanup(self):
        """Free all cached models from GPU memory."""
        global _sidon_cache, _vc_cache, _dnsmos_cache
        _sidon_cache.clear()
        _vc_cache.clear()
        _dnsmos_cache.clear()
        torch.cuda.empty_cache()
        log.info("SidonEnhancer cleaned up")
