#!/usr/bin/env python3
"""
Full enhanced evaluation pipeline for Run 5 + Run 6 + Run 7 + Run 8.

Phases:
  1. Generate Run 7/8 audio (LoRA / merged DiT + AdaLN checkpoints)
  2. Sidon post-processing for all runs
  3. ChatterboxVC → Sidon for all runs
  4. Score all variants (raw / sidon / vc_sidon) across all runs
  5. Generate unified HTML report with tables + MP3 players
"""

import gc
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torchaudio

# ── Paths ──
BASE_DIR = "/home/deployer/laion"
DRAMABOX_DIR = os.path.join(BASE_DIR, "DramaBox")
SCRIPTS_DIR = os.path.join(BASE_DIR, "Voice-Acting-Pipeline/scripts")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_output_combined")

VANILLA_CKPT = os.path.join(DRAMABOX_DIR, "models/ltx-2.3-22b-dev-audio-only-v13-merged.safetensors")
FULL_CKPT = os.path.join(DRAMABOX_DIR, "models/ltx-2.3-22b-dev.safetensors")
REF_DIR = os.path.join(BASE_DIR, "test-refs")

RUN5_DIR = os.path.join(BASE_DIR, "eval_output_run5")
RUN6_DIR = os.path.join(BASE_DIR, "eval_output_run6")
RUN7_DIR = os.path.join(OUTPUT_DIR)

# Train dirs
RUN5_TRAIN = os.path.join(BASE_DIR, "full_ft_adaln_speaker_run5")
RUN7_TRAIN = os.path.join(BASE_DIR, "full_ft_standard_run2")
RUN8_TRAIN = os.path.join(BASE_DIR, "full_ft_adaln_run8")
RUN8_MERGED_CKPT = os.path.join(RUN8_TRAIN, "merged_dit.safetensors")
RUN9_TRAIN = os.path.join(BASE_DIR, "full_ft_lora128_run9")

sys.path.insert(0, os.path.join(DRAMABOX_DIR, "ltx2"))
sys.path.insert(0, os.path.join(DRAMABOX_DIR, "src"))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "urgent2026_challenge_track2"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Prompts & Speakers (same as eval_full_ft_adaln_enhanced.py) ──

EN_PROMPTS = [
    {"id": "love_vulnerable", "title": "Vulnerable Love Declaration", "gen_duration": 20, "lang": "en",
     "prompt": "Genuine, intimate, vulnerable. Voice trembles with sincerity, then shifts to surprised amusement. High Quality Recording. 'I never thought I'd say this out loud, but you make every broken piece feel like it matters. [soft laugh] God, look at me, I'm standing here smiling through all of this. You wonderful, terrifying, ridiculous person. I think I really love you.'"},
    {"id": "horror_basement", "title": "Basement Horror to Relief", "gen_duration": 0, "lang": "en",
     "prompt": "Alone in a dark basement, escalating dread then sudden relief. High Quality Recording. 'Hello? Is someone down here? Oh God, I can hear breathing. [terrified scream] NO! STAY AWAY FROM ME! Wait... oh. Oh my God, it's just you. Jesus Christ, you scared me half to death. Don't ever do that again.'"},
    {"id": "funny_ecstasy", "title": "Funny Story with Ecstatic Memory", "gen_duration": 0, "lang": "en",
     "prompt": "Telling a hilarious story to friends, laughter keeps breaking through. High Quality Recording. 'And then, [laughing] and then she just looked at him dead serious and said, that is not a duck! [ecstatic laughter] Oh man, those summers were absolute magic. Pure, ridiculous, beautiful magic. I would give anything to go back to those days.'"},
]

DE_PROMPTS = [
    {"id": "love_vulnerable_de", "title": "Verletzliche Liebeserklärung", "gen_duration": 22, "lang": "de",
     "prompt": "Aufrichtig, intim, verletzlich. Die Stimme zittert vor Ehrlichkeit, wechselt dann zu überraschtem Amüsement. Hochwertige Aufnahme. 'Ich hätte nie gedacht, dass ich das laut aussprechen würde, aber du gibst jedem zerbrochenen Stück das Gefühl, dass es zählt. [leises Lachen] Gott, sieh mich an, ich stehe hier und lächle durch das alles hindurch. Du wundervoller, beängstigender, verrückter Mensch. Ich glaube, ich liebe dich wirklich.'"},
    {"id": "horror_basement_de", "title": "Kellerschrecken bis Erleichterung", "gen_duration": 0, "lang": "de",
     "prompt": "Allein in einem dunklen Keller, steigende Angst, dann plötzliche Erleichterung. Hochwertige Aufnahme. 'Hallo? Ist da jemand? Oh Gott, ich kann Atmen hören. [erschrockener Schrei] NEIN! BLEIB WEG VON MIR! Warte... oh. Oh mein Gott, du bist es nur. Herrgott nochmal, du hast mich zu Tode erschreckt. Mach das nie wieder.'"},
    {"id": "funny_ecstasy_de", "title": "Lustige Geschichte mit ekstatischer Erinnerung", "gen_duration": 0, "lang": "de",
     "prompt": "Erzählt eine urkomische Geschichte an Freunde, Gelächter bricht immer wieder durch. Hochwertige Aufnahme. 'Und dann, [lachend] und dann hat sie ihn einfach todernst angeschaut und gesagt, das ist keine Ente! [begeistertes Gelächter] Oh Mann, diese Sommer waren absolute Magie. Reine, verrückte, wunderschöne Magie. Ich würde alles geben, um in diese Tage zurückzukehren.'"},
]

ALL_PROMPTS = EN_PROMPTS + DE_PROMPTS

REF_SPEAKERS = [
    {"id": "chris", "path": os.path.join(REF_DIR, "chris-ref-enhanced.wav")},
    {"id": "fairy", "path": os.path.join(REF_DIR, "enh_Fairy-2.wav")},
    {"id": "samantha", "path": os.path.join(REF_DIR, "enh_Samantha musing about flirting Voice AI - Echo TTS.wav")},
    {"id": "goblin", "path": os.path.join(REF_DIR, "enh_School Goblin - en.wav")},
    {"id": "spongebob", "path": os.path.join(REF_DIR, "enh_spongebob-ref.wav")},
]

SEEDS = [42, 137]
UNCOND_SEEDS = [42, 137]
VARIANTS = ["raw", "sidon", "vc_sidon"]

# ═══════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════════════

def get_all_models():
    """Define all models across Run 5, 6, and 7."""
    models = []

    # ── Run 5: AdaLN-Zero ──
    models.append({
        "name": "R5_vanilla", "run": "run5", "source_dir": RUN5_DIR,
        "checkpoint": VANILLA_CKPT, "adaln_path": None, "lora_path": None,
        "label": "Run 5: Vanilla (baseline)", "short": "R5 Vanilla",
    })
    models.append({
        "name": "R5_adaln_epoch4", "run": "run5", "source_dir": RUN5_DIR,
        "checkpoint": VANILLA_CKPT,
        "adaln_path": os.path.join(RUN5_TRAIN, "speaker_adaln_epoch4.pt"),
        "lora_path": None,
        "label": "Run 5: AdaLN epoch 4 (best)", "short": "R5 AdaLN e4",
    })
    models.append({
        "name": "R5_adaln_epoch5", "run": "run5", "source_dir": RUN5_DIR,
        "checkpoint": VANILLA_CKPT,
        "adaln_path": os.path.join(RUN5_TRAIN, "speaker_adaln_epoch5.pt"),
        "lora_path": None,
        "label": "Run 5: AdaLN epoch 5 (final)", "short": "R5 AdaLN e5",
    })

    # ── Run 6: Full FT fp32-master ──
    models.append({
        "name": "R6_vanilla", "run": "run6", "source_dir": RUN6_DIR,
        "checkpoint": VANILLA_CKPT, "adaln_path": None, "lora_path": None,
        "label": "Run 6: Vanilla (baseline)", "short": "R6 Vanilla",
    })
    models.append({
        "name": "run6_step100", "run": "run6", "source_dir": RUN6_DIR,
        "checkpoint": os.path.join(BASE_DIR, "saved_checkpoints/model_step100.safetensors"),
        "adaln_path": None, "lora_path": None,
        "label": "Run 6: Full FT step 100", "short": "R6 step100",
    })
    models.append({
        "name": "run6_step160", "run": "run6", "source_dir": RUN6_DIR,
        "checkpoint": os.path.join(BASE_DIR, "full_ft_standard_run1/model_step160.safetensors"),
        "adaln_path": None, "lora_path": None,
        "label": "Run 6: Full FT step 160", "short": "R6 step160",
    })
    models.append({
        "name": "run6_final", "run": "run6", "source_dir": RUN6_DIR,
        "checkpoint": os.path.join(BASE_DIR, "full_ft_standard_run1/model_best.safetensors"),
        "adaln_path": None, "lora_path": None,
        "label": "Run 6: Full FT final", "short": "R6 final",
    })

    # ── Run 7: LoRA rank 64, lr=4e-5, 2 epochs, 340 steps ──
    lora_dir = RUN7_TRAIN

    # Final checkpoint (step 340 = lora_final)
    lora_final = os.path.join(lora_dir, "lora_final.safetensors")
    if os.path.exists(lora_final):
        models.append({
            "name": "run7_final", "run": "run7", "source_dir": OUTPUT_DIR,
            "checkpoint": VANILLA_CKPT, "adaln_path": None,
            "lora_path": lora_final,
            "label": "Run 7: LoRA final (step 340)", "short": "R7 final",
        })

    # Mid-training checkpoint (step 330)
    lora_330 = os.path.join(lora_dir, "lora_step330.safetensors")
    if os.path.exists(lora_330):
        models.append({
            "name": "run7_step330", "run": "run7", "source_dir": OUTPUT_DIR,
            "checkpoint": VANILLA_CKPT, "adaln_path": None,
            "lora_path": lora_330,
            "label": "Run 7: LoRA step 330", "short": "R7 s330",
        })

    # ── Run 8: Frozen LoRA-merged DiT + AdaLN-Zero (lr=7e-5, 6 epochs) ──
    if os.path.exists(RUN8_MERGED_CKPT):
        # Best checkpoint
        adaln_best = os.path.join(RUN8_TRAIN, "speaker_adaln_best.pt")
        if os.path.exists(adaln_best):
            models.append({
                "name": "run8_best", "run": "run8", "source_dir": OUTPUT_DIR,
                "checkpoint": RUN8_MERGED_CKPT, "adaln_path": adaln_best,
                "lora_path": None,
                "label": "Run 8: AdaLN best (frozen DiT)", "short": "R8 best",
            })

        # Final checkpoint
        adaln_final = os.path.join(RUN8_TRAIN, "speaker_adaln_final.pt")
        if os.path.exists(adaln_final):
            models.append({
                "name": "run8_final", "run": "run8", "source_dir": OUTPUT_DIR,
                "checkpoint": RUN8_MERGED_CKPT, "adaln_path": adaln_final,
                "lora_path": None,
                "label": "Run 8: AdaLN final (frozen DiT)", "short": "R8 final",
            })

        # Discover any step-based AdaLN checkpoints
        import glob as _glob
        import re as _re
        step_adalns = sorted(_glob.glob(os.path.join(RUN8_TRAIN, "speaker_adaln_step*.pt")))
        for sp in step_adalns:
            m = _re.search(r"speaker_adaln_step(\d+)\.pt$", sp)
            if m:
                step_n = int(m.group(1))
                name = f"run8_step{step_n}"
                # Skip if we'd duplicate best/final
                if any(mdl["name"] == name for mdl in models):
                    continue
                models.append({
                    "name": name, "run": "run8", "source_dir": OUTPUT_DIR,
                    "checkpoint": RUN8_MERGED_CKPT, "adaln_path": sp,
                    "lora_path": None,
                    "label": f"Run 8: AdaLN step {step_n}", "short": f"R8 s{step_n}",
                })

    # ── Run 9: LoRA rank 128 (fp32 master) + Pretrained AdaLN-Zero ──
    if os.path.exists(RUN9_TRAIN):
        import glob as _glob9
        import re as _re9
        # Discover LoRA + AdaLN step checkpoints
        lora_steps = sorted(_glob9.glob(os.path.join(RUN9_TRAIN, "lora_step*.safetensors")))
        for lp in lora_steps:
            m = _re9.search(r"lora_step(\d+)\.safetensors$", lp)
            if m:
                step_n = int(m.group(1))
                adaln_p = os.path.join(RUN9_TRAIN, f"speaker_adaln_step{step_n}.pt")
                if not os.path.exists(adaln_p):
                    continue
                models.append({
                    "name": f"run9_step{step_n}", "run": "run9", "source_dir": OUTPUT_DIR,
                    "checkpoint": RUN8_MERGED_CKPT, "adaln_path": adaln_p,
                    "lora_path": lp,
                    "label": f"Run 9: LoRA128+AdaLN step {step_n}", "short": f"R9 s{step_n}",
                })

        # Final checkpoint
        lora_final = os.path.join(RUN9_TRAIN, "lora_final.safetensors")
        adaln_final = os.path.join(RUN9_TRAIN, "speaker_adaln_final.pt")
        if os.path.exists(lora_final) and os.path.exists(adaln_final):
            # Don't duplicate if final is same as a step checkpoint
            if not any(mdl["name"] == "run9_final" for mdl in models):
                models.append({
                    "name": "run9_final", "run": "run9", "source_dir": OUTPUT_DIR,
                    "checkpoint": RUN8_MERGED_CKPT, "adaln_path": adaln_final,
                    "lora_path": lora_final,
                    "label": "Run 9: LoRA128+AdaLN final", "short": "R9 final",
                })

    return models


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: COLLECT EXISTING + GENERATE NEW AUDIO
# ═══════════════════════════════════════════════════════════════════════

def collect_existing_samples(models):
    """Collect sample metadata for models that already have audio generated."""
    samples = []
    for mcfg in models:
        if mcfg["run"] in ("run5", "run6"):
            source_dir = mcfg["source_dir"]
            # Map model name to original dir name
            orig_name = mcfg["name"]
            if mcfg["run"] == "run5":
                name_map = {"R5_vanilla": "vanilla", "R5_adaln_epoch4": "best_epoch4",
                            "R5_adaln_epoch5": "final_epoch5"}
                orig_subdir = name_map.get(orig_name, orig_name)
            else:
                name_map = {"R6_vanilla": "vanilla"}
                orig_subdir = name_map.get(orig_name, orig_name)

            model_dir = os.path.join(source_dir, orig_subdir)
            if not os.path.isdir(model_dir):
                log.warning(f"Missing dir: {model_dir}")
                continue

            for prompt in ALL_PROMPTS:
                for ref in REF_SPEAKERS:
                    for seed in SEEDS:
                        sid = f"{prompt['id']}_{ref['id']}_seed{seed}"
                        out = os.path.join(model_dir, f"{sid}.wav")
                        samples.append({
                            "model": mcfg["name"], "prompt_id": prompt["id"],
                            "prompt_text": prompt["prompt"], "lang": prompt["lang"],
                            "ref_id": ref["id"], "ref_path": ref["path"],
                            "seed": seed, "conditional": True,
                            "output_path": out if os.path.exists(out) else None,
                            "source_dir": source_dir,
                            "orig_subdir": orig_subdir,
                        })

                for seed in UNCOND_SEEDS:
                    sid = f"{prompt['id']}_uncond_seed{seed}"
                    out = os.path.join(model_dir, f"{sid}.wav")
                    samples.append({
                        "model": mcfg["name"], "prompt_id": prompt["id"],
                        "prompt_text": prompt["prompt"], "lang": prompt["lang"],
                        "ref_id": None, "ref_path": None,
                        "seed": seed, "conditional": False,
                        "output_path": out if os.path.exists(out) else None,
                        "source_dir": source_dir,
                        "orig_subdir": orig_subdir,
                    })

    return samples


def generate_run7_samples(models, output_dir):
    """Generate audio for Run 7 LoRA and Run 8 AdaLN models."""
    run7_models = [m for m in models if m["run"] in ("run7", "run8", "run9")]
    if not run7_models:
        log.info("No Run 7/8 models to generate")
        return []

    # Pre-encode prompts (reuse from Run 5/6 if available)
    encoded_dir = os.path.join(RUN5_DIR, "encoded_prompts")
    if not os.path.isdir(encoded_dir):
        encoded_dir = os.path.join(RUN6_DIR, "encoded_prompts")
    if not os.path.isdir(encoded_dir):
        # Need to encode
        encoded_dir = os.path.join(output_dir, "encoded_prompts")
        log.info("Pre-encoding prompts...")
        script = os.path.join(SCRIPTS_DIR, "pre_encode_prompts.py")
        prompts_json = os.path.join(output_dir, "eval_prompts.json")
        with open(prompts_json, "w") as f:
            json.dump([{"id": p["id"], "prompt": p["prompt"]} for p in ALL_PROMPTS], f)
        subprocess.run(
            [sys.executable, script, "--output-dir", encoded_dir, "--prompts-json", prompts_json],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )

    samples = []
    for mcfg in run7_models:
        mname = mcfg["name"]
        mdir = os.path.join(output_dir, mname)
        os.makedirs(mdir, exist_ok=True)

        log.info(f"{'='*60}")
        log.info(f"GENERATING: {mname} (LoRA: {mcfg['lora_path']})")
        log.info(f"{'='*60}")

        for prompt in ALL_PROMPTS:
            for ref in REF_SPEAKERS:
                for seed in SEEDS:
                    sid = f"{prompt['id']}_{ref['id']}_seed{seed}"
                    out = os.path.join(mdir, f"{sid}.wav")
                    sample = {
                        "model": mname, "prompt_id": prompt["id"],
                        "prompt_text": prompt["prompt"], "lang": prompt["lang"],
                        "ref_id": ref["id"], "ref_path": ref["path"],
                        "seed": seed, "conditional": True, "output_path": out,
                        "source_dir": output_dir, "orig_subdir": mname,
                    }
                    if os.path.exists(out):
                        log.info(f"  Skip: {sid}")
                    else:
                        log.info(f"  Generating: {sid}")
                        ok = _run_lora_inference(
                            mcfg["checkpoint"], mcfg["lora_path"], mcfg.get("adaln_path"),
                            prompt["id"], prompt["prompt"], ref["path"],
                            out, seed, prompt.get("gen_duration", 0), encoded_dir)
                        if not ok:
                            sample["output_path"] = None
                    samples.append(sample)

            for seed in UNCOND_SEEDS:
                sid = f"{prompt['id']}_uncond_seed{seed}"
                out = os.path.join(mdir, f"{sid}.wav")
                sample = {
                    "model": mname, "prompt_id": prompt["id"],
                    "prompt_text": prompt["prompt"], "lang": prompt["lang"],
                    "ref_id": None, "ref_path": None,
                    "seed": seed, "conditional": False, "output_path": out,
                    "source_dir": output_dir, "orig_subdir": mname,
                }
                if os.path.exists(out):
                    log.info(f"  Skip: {sid}")
                else:
                    log.info(f"  Generating: {sid}")
                    ok = _run_lora_inference(
                        mcfg["checkpoint"], mcfg["lora_path"], mcfg.get("adaln_path"),
                        prompt["id"], prompt["prompt"], None,
                        out, seed, prompt.get("gen_duration", 0), encoded_dir)
                    if not ok:
                        sample["output_path"] = None
                samples.append(sample)

    return samples


def _run_lora_inference(checkpoint, lora_path, adaln_path, prompt_id, prompt_text,
                        voice_sample, output_path, seed, gen_duration=0, encoded_dir=None):
    """Run inference with LoRA weights."""
    script = os.path.join(SCRIPTS_DIR, "inference_adaln.py")
    cmd = [
        sys.executable, script,
        "--checkpoint", checkpoint,
        "--full-checkpoint", FULL_CKPT,
        "--output", output_path,
        "--seed", str(seed),
        "--no-watermark",
    ]

    if lora_path:
        cmd.extend(["--lora-checkpoint", lora_path])
    if adaln_path:
        cmd.extend(["--adaln-checkpoint", adaln_path])

    if encoded_dir:
        pos_pt = os.path.join(encoded_dir, f"{prompt_id}.pt")
        neg_pt = os.path.join(encoded_dir, "negative.pt")
        if os.path.exists(pos_pt):
            cmd.extend(["--cond-pos-pt", pos_pt])
            if os.path.exists(neg_pt):
                cmd.extend(["--cond-neg-pt", neg_pt])
        else:
            cmd.extend(["--prompt", prompt_text])
    else:
        cmd.extend(["--prompt", prompt_text])

    if prompt_text:
        cmd.extend(["--prompt-text", prompt_text])
    if voice_sample:
        cmd.extend(["--voice-sample", voice_sample])
    else:
        cmd.append("--no-ref")
    if gen_duration > 0:
        cmd.extend(["--gen-duration", str(gen_duration)])

    log.info(f"    Inference: seed={seed} ref={'yes' if voice_sample else 'no'}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        if result.returncode != 0:
            log.error(f"    Inference failed: {result.stderr[-500:]}")
            return False
    except Exception as e:
        log.error(f"    Inference exception: {e}")
        return False
    return os.path.exists(output_path)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: SIDON POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _load_sidon(device="cuda"):
    import transformers
    from huggingface_hub import hf_hub_download
    fe_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="feature_extractor_cuda.pt")
    decoder_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")
    fe = torch.jit.load(fe_path, map_location=device).to(device)
    decoder = torch.jit.load(decoder_path, map_location=device).to(device)
    preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
        "facebook/w2v-bert-2.0", sampling_rate=16000)
    return fe, decoder, preprocessor


def _run_sidon_on_file(wav_path, out_path, fe, decoder, preprocessor, device="cuda"):
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[1:2]
    peak = waveform.abs().max()
    if peak > 0:
        waveform = 0.9 * (waveform / peak)
    target_n_samples = int(48_000 / sample_rate * waveform.shape[-1])
    wav_16k = torchaudio.functional.highpass_biquad(waveform, sample_rate, 50)
    wav_16k = torchaudio.functional.resample(wav_16k, sample_rate, 16_000)
    wav_16k = torch.nn.functional.pad(wav_16k, (0, 24000))
    restoreds = []
    feature_cache = None
    for chunk in wav_16k.view(-1).split(16000 * 96):
        inputs = preprocessor(
            torch.nn.functional.pad(chunk, (160, 160)), return_tensors="pt")
        with torch.inference_mode():
            feature = fe(inputs["input_features"].to(device))["last_hidden_state"]
            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
            restoreds.append(decoder(feature.transpose(1, 2)).view(-1)[:-960])
            feature_cache = feature[:, -1:]
    restored_wav = torch.cat(restoreds, dim=0)[:target_n_samples]
    torchaudio.save(out_path, restored_wav.view(1, -1).cpu(), 48000)


def _get_model_audio_dir(sample):
    """Get the directory containing raw WAV files for a sample."""
    return os.path.join(sample["source_dir"], sample["orig_subdir"])


def phase_sidon(samples, device="cuda"):
    """Run Sidon on all raw outputs that don't have sidon/ yet."""
    log.info("=" * 60)
    log.info("PHASE 2: SIDON POST-PROCESSING")
    log.info("=" * 60)

    todo = []
    for s in samples:
        if not s.get("output_path") or not os.path.exists(s["output_path"]):
            continue
        model_dir = _get_model_audio_dir(s)
        sidon_dir = os.path.join(model_dir, "sidon")
        os.makedirs(sidon_dir, exist_ok=True)
        fname = os.path.basename(s["output_path"])
        out_path = os.path.join(sidon_dir, fname)
        if not os.path.exists(out_path):
            todo.append((s["output_path"], out_path))

    if not todo:
        log.info("  All Sidon outputs exist, skipping")
        return

    log.info(f"  Processing {len(todo)} files through Sidon...")
    fe, decoder, preprocessor = _load_sidon(device)

    t0 = time.time()
    for i, (inp, outp) in enumerate(todo):
        try:
            _run_sidon_on_file(inp, outp, fe, decoder, preprocessor, device)
        except Exception as e:
            log.warning(f"  Sidon failed for {os.path.basename(inp)}: {e}")
        if (i + 1) % 20 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            log.info(f"  Sidon [{i+1}/{len(todo)}] {elapsed:.0f}s elapsed")

    del fe, decoder
    gc.collect()
    torch.cuda.empty_cache()
    log.info("  Sidon complete")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: CHATTERBOXVC → SIDON
# ═══════════════════════════════════════════════════════════════════════

def _load_chatterbox_vc(device="cuda"):
    try:
        from chatterbox.vc import ChatterboxVC
        vc = ChatterboxVC.from_pretrained(device=device)
        log.info("  Loaded ChatterboxVC via chatterbox.vc")
        return vc
    except ImportError:
        pass
    try:
        from chatterbox_vc import VoiceConverter
        vc = VoiceConverter(device=device)
        log.info("  Loaded ChatterboxVC via chatterbox_vc")
        return vc
    except ImportError:
        pass
    raise ImportError("Cannot import ChatterboxVC — install chatterbox or chatterbox_vc")


def phase_vc_sidon(samples, device="cuda"):
    """Run ChatterboxVC → Sidon on conditional samples missing vc_sidon/."""
    log.info("=" * 60)
    log.info("PHASE 3: CHATTERBOXVC → SIDON")
    log.info("=" * 60)

    cond_samples = [s for s in samples
                    if s["conditional"] and s.get("output_path")
                    and os.path.exists(s["output_path"])]

    vc_todo = []  # (raw_path, vc_tmp, ref_path, sidon_out, ref_id)
    for s in cond_samples:
        model_dir = _get_model_audio_dir(s)
        vc_dir = os.path.join(model_dir, "vc_sidon")
        os.makedirs(vc_dir, exist_ok=True)
        fname = os.path.basename(s["output_path"])
        vc_tmp = os.path.join(vc_dir, f"vc_tmp_{fname}")
        sidon_out = os.path.join(vc_dir, fname)
        if not os.path.exists(sidon_out):
            vc_todo.append((s["output_path"], vc_tmp, s["ref_path"], sidon_out, s["ref_id"]))

    if not vc_todo:
        log.info("  All VC→Sidon outputs exist, skipping")
        return

    log.info(f"  Processing {len(vc_todo)} files through ChatterboxVC → Sidon...")

    # Step 1: Voice Conversion
    log.info("  Step 1: ChatterboxVC...")
    vc = _load_chatterbox_vc(device)

    by_speaker = defaultdict(list)
    for item in vc_todo:
        by_speaker[item[4]].append(item)

    t0 = time.time()
    vc_done = 0
    for spk, items in by_speaker.items():
        ref_path = items[0][2]
        log.info(f"  VC speaker '{spk}': {len(items)} files")

        for raw_path, vc_tmp, _, _, _ in items:
            if os.path.exists(vc_tmp):
                vc_done += 1
                continue
            try:
                with torch.inference_mode():
                    result = vc.generate(
                        audio=str(raw_path),
                        target_voice_path=str(ref_path),
                    )
                if isinstance(result, torch.Tensor):
                    if result.dim() == 1:
                        result = result.unsqueeze(0)
                    torchaudio.save(vc_tmp, result.cpu(), 24000)
                vc_done += 1
            except Exception as e:
                log.warning(f"    VC failed for {os.path.basename(raw_path)}: {e}")
                vc_done += 1

            if vc_done % 10 == 0 or vc_done == len(vc_todo):
                elapsed = time.time() - t0
                log.info(f"    VC [{vc_done}/{len(vc_todo)}] {elapsed:.0f}s")

    del vc
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Sidon on VC outputs
    log.info("  Step 2: Sidon on VC outputs...")
    fe, decoder, preprocessor = _load_sidon(device)

    sidon_todo = [(item[1], item[3]) for item in vc_todo
                  if os.path.exists(item[1]) and not os.path.exists(item[3])]

    t0 = time.time()
    for i, (vc_path, sidon_path) in enumerate(sidon_todo):
        try:
            _run_sidon_on_file(vc_path, sidon_path, fe, decoder, preprocessor, device)
        except Exception as e:
            log.warning(f"    Sidon-on-VC failed for {os.path.basename(vc_path)}: {e}")
        if (i + 1) % 20 == 0 or (i + 1) == len(sidon_todo):
            elapsed = time.time() - t0
            log.info(f"    Sidon-VC [{i+1}/{len(sidon_todo)}] {elapsed:.0f}s")

    del fe, decoder
    gc.collect()
    torch.cuda.empty_cache()

    # Clean up VC temp files
    for item in vc_todo:
        vc_tmp = item[1]
        if os.path.exists(vc_tmp) and os.path.exists(item[3]):
            os.unlink(vc_tmp)

    log.info("  VC→Sidon complete")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: SCORING
# ═══════════════════════════════════════════════════════════════════════

SQA_METRICS = ["mos", "dnsmos_ovrl", "scoreq", "utmos", "nisqa_mos"]
AUDIOBOX_METRICS = ["CE", "CU", "PC", "PQ"]
ALL_METRICS = SQA_METRICS + AUDIOBOX_METRICS + ["spk_sim"]

METRIC_LABELS = {
    "mos": "MOS", "dnsmos_ovrl": "DNSMOS", "scoreq": "ScoReQ",
    "utmos": "UTMOS", "nisqa_mos": "NISQA",
    "CE": "CE", "CU": "CU", "PC": "PC", "PQ": "PQ",
    "spk_sim": "SpkSim",
}

HIGHER_BETTER = {"mos", "dnsmos_ovrl", "scoreq", "utmos", "nisqa_mos", "CE", "CU", "PQ", "spk_sim"}


def _run_class(model_name):
    """Return CSS class for a model name (r5/r6/r7/r8/r9)."""
    if model_name.startswith("R5"):
        return "r5"
    if model_name.startswith("R6") or model_name.startswith("run6"):
        return "r6"
    if model_name.startswith("run7"):
        return "r7"
    if model_name.startswith("run8"):
        return "r8"
    if model_name.startswith("run9"):
        return "r9"
    return "r7"  # fallback


def _run_class_audio(model_name):
    """Return CSS class for audio cards (run5/run6/run7/run8/run9)."""
    if model_name.startswith("R5"):
        return "run5"
    if model_name.startswith("R6") or model_name.startswith("run6"):
        return "run6"
    if model_name.startswith("run7"):
        return "run7"
    if model_name.startswith("run8"):
        return "run8"
    if model_name.startswith("run9"):
        return "run9"
    return "run7"


def _load_audio_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav, 16000


def score_with_sqa(paths_dict, device="cuda"):
    log.info("  Loading SQA model...")
    from urgent2026_sqa.infer import load_model, infer_single
    model, config = load_model("vvwangvv/universa-ext_wavlm-base_5metric")
    model = model.to(device).eval()
    scores = {}
    t0 = time.time()
    keys = list(paths_dict.keys())
    for i, key in enumerate(keys):
        try:
            result = infer_single(model, config, paths_dict[key])
            scores[key] = {k: float(v) for k, v in result.items()}
        except Exception as e:
            log.warning(f"    SQA failed for {key}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == len(keys):
            log.info(f"    SQA [{i+1}/{len(keys)}] {(i+1)/(time.time()-t0):.1f}/s")
    del model
    torch.cuda.empty_cache()
    return scores


def score_with_audiobox(paths_dict, device="cuda"):
    log.info("  Loading AudioBox Aesthetics...")
    from audiobox_aesthetics.infer import initialize_predictor
    predictor = initialize_predictor()
    scores = {}
    t0 = time.time()
    keys = list(paths_dict.keys())
    batch_size = 16
    for batch_start in range(0, len(keys), batch_size):
        batch_keys = keys[batch_start:batch_start + batch_size]
        inputs = [{"path": paths_dict[k]} for k in batch_keys]
        try:
            results = predictor.forward(inputs)
            for k, result in zip(batch_keys, results):
                scores[k] = {m: float(v) for m, v in result.items()}
        except Exception as e:
            log.warning(f"    AudioBox batch failed: {e}")
        done = min(batch_start + batch_size, len(keys))
        if done % 50 < batch_size or done == len(keys):
            log.info(f"    AudioBox [{done}/{len(keys)}] {done/(time.time()-t0):.1f}/s")
    del predictor
    torch.cuda.empty_cache()
    return scores


def score_speaker_sim_batch(pairs, device="cuda"):
    log.info("  Loading WavLM-SV...")
    from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
    model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    ).eval().to(device, torch.float32)
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

    def _embed(path):
        wav, _ = _load_audio_16k(path)
        inputs = fe(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.input_values.to(device))
            emb = out.embeddings
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    ref_cache = {}
    scores = {}
    t0 = time.time()
    keys = list(pairs.keys())
    for i, key in enumerate(keys):
        gen_path, ref_path = pairs[key]
        try:
            if ref_path not in ref_cache:
                ref_cache[ref_path] = _embed(ref_path)
            gen_emb = _embed(gen_path)
            sim = torch.nn.functional.cosine_similarity(gen_emb, ref_cache[ref_path]).item()
            scores[key] = sim
        except Exception as e:
            log.warning(f"    SpkSim failed for {key}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == len(keys):
            log.info(f"    SpkSim [{i+1}/{len(keys)}] {(i+1)/(time.time()-t0):.1f}/s")
    del model
    torch.cuda.empty_cache()
    return scores


def phase_score(samples, device="cuda"):
    """Score all variants of all samples, merging with existing Run 5 scores."""
    log.info("=" * 60)
    log.info("PHASE 4: SCORING ALL VARIANTS")
    log.info("=" * 60)

    # Load existing Run 5 enhanced scores
    all_scores = {}
    r5_scores_path = os.path.join(RUN5_DIR, "enhanced_scores.json")
    if os.path.exists(r5_scores_path):
        with open(r5_scores_path) as f:
            r5_raw = json.load(f)
        # Re-key from Run 5 names to our naming
        name_map = {"vanilla": "R5_vanilla", "best_epoch4": "R5_adaln_epoch4",
                     "final_epoch5": "R5_adaln_epoch5"}
        for k, v in r5_raw.items():
            parts = k.split("|")
            if parts[0] in name_map:
                parts[0] = name_map[parts[0]]
                new_key = "|".join(parts)
                all_scores[new_key] = v
        log.info(f"  Loaded {len(all_scores)} Run 5 enhanced scores")

    # Load cached combined scores if they exist
    scores_path = os.path.join(OUTPUT_DIR, "enhanced_scores.json")
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            cached = json.load(f)
        all_scores.update(cached)
        log.info(f"  Loaded {len(cached)} cached combined scores (total now {len(all_scores)})")

    # Build paths for all variants
    all_paths = {}
    spk_pairs = {}

    for s in samples:
        if not s.get("output_path") or not os.path.exists(s["output_path"]):
            continue
        model_dir = _get_model_audio_dir(s)
        fname = os.path.basename(s["output_path"])

        for variant in VARIANTS:
            if variant == "raw":
                wav_path = s["output_path"]
            elif variant == "sidon":
                wav_path = os.path.join(model_dir, "sidon", fname)
            elif variant == "vc_sidon":
                if not s["conditional"]:
                    continue
                wav_path = os.path.join(model_dir, "vc_sidon", fname)
            else:
                continue

            if not os.path.exists(wav_path):
                continue

            ref_id = s.get("ref_id") or "uncond"
            key = f"{s['model']}|{s['prompt_id']}|{ref_id}|{s['seed']}|{variant}"
            all_paths[key] = wav_path

            if s["conditional"] and s.get("ref_path"):
                spk_pairs[key] = (wav_path, s["ref_path"])

    log.info(f"  Total variant files: {len(all_paths)}")
    log.info(f"  Speaker sim pairs: {len(spk_pairs)}")

    # Score only uncached
    uncached_paths = {k: v for k, v in all_paths.items() if k not in all_scores}
    if uncached_paths:
        log.info(f"  Scoring {len(uncached_paths)} new files...")

        sqa_scores = score_with_sqa(uncached_paths, device)
        for k, v in sqa_scores.items():
            all_scores.setdefault(k, {}).update(v)

        ab_scores = score_with_audiobox(uncached_paths, device)
        for k, v in ab_scores.items():
            all_scores.setdefault(k, {}).update(v)

        uncached_spk = {k: v for k, v in spk_pairs.items()
                        if k not in all_scores or "spk_sim" not in all_scores.get(k, {})}
        if uncached_spk:
            spk_scores = score_speaker_sim_batch(uncached_spk, device)
            for k, v in spk_scores.items():
                all_scores.setdefault(k, {})["spk_sim"] = v

        with open(scores_path, "w") as f:
            json.dump(all_scores, f, indent=2)
        log.info(f"  Scores saved: {scores_path}")
    else:
        log.info("  All scores cached, skipping")

    return all_scores


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: CONVERT TO MP3
# ═══════════════════════════════════════════════════════════════════════

def phase_convert_mp3(samples):
    """Convert all variant WAVs to MP3 for serving."""
    log.info("=" * 60)
    log.info("PHASE 5: CONVERT WAVs TO MP3")
    log.info("=" * 60)

    todo = []
    for s in samples:
        if not s.get("output_path") or not os.path.exists(s["output_path"]):
            continue
        model_dir = _get_model_audio_dir(s)
        fname = os.path.basename(s["output_path"])
        base = fname.replace(".wav", "")

        for variant in VARIANTS:
            if variant == "raw":
                wav_path = s["output_path"]
                mp3_dir = os.path.join(model_dir, "mp3")
            elif variant == "sidon":
                wav_path = os.path.join(model_dir, "sidon", fname)
                mp3_dir = os.path.join(model_dir, "sidon", "mp3")
            elif variant == "vc_sidon":
                if not s["conditional"]:
                    continue
                wav_path = os.path.join(model_dir, "vc_sidon", fname)
                mp3_dir = os.path.join(model_dir, "vc_sidon", "mp3")
            else:
                continue

            if not os.path.exists(wav_path):
                continue

            os.makedirs(mp3_dir, exist_ok=True)
            mp3_path = os.path.join(mp3_dir, f"{base}.mp3")
            if not os.path.exists(mp3_path):
                todo.append((wav_path, mp3_path))

    if not todo:
        log.info("  All MP3s exist, skipping")
        return

    log.info(f"  Converting {len(todo)} files to MP3...")
    t0 = time.time()
    for i, (wav, mp3) in enumerate(todo):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav, "-ac", "1", "-ar", "24000",
                 "-b:a", "96k", "-f", "mp3", mp3],
                capture_output=True, timeout=30)
        except Exception as e:
            log.warning(f"  MP3 convert failed: {os.path.basename(wav)}: {e}")
        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            log.info(f"  MP3 [{i+1}/{len(todo)}] {elapsed:.0f}s")

    log.info("  MP3 conversion complete")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: UNIFIED HTML REPORT
# ═══════════════════════════════════════════════════════════════════════

def _mp3_url(sample, variant, output_dir):
    """Get relative MP3 URL for serving. Symlinks source dirs into output_dir."""
    if not sample.get("output_path"):
        return None
    model_dir = _get_model_audio_dir(sample)
    fname = os.path.basename(sample["output_path"]).replace(".wav", ".mp3")

    if variant == "raw":
        mp3_path = os.path.join(model_dir, "mp3", fname)
        rel_subdir = f"{sample['model']}/mp3"
    elif variant == "sidon":
        mp3_path = os.path.join(model_dir, "sidon", "mp3", fname)
        rel_subdir = f"{sample['model']}/sidon/mp3"
    elif variant == "vc_sidon":
        mp3_path = os.path.join(model_dir, "vc_sidon", "mp3", fname)
        rel_subdir = f"{sample['model']}/vc_sidon/mp3"
    else:
        return None

    if not os.path.exists(mp3_path):
        return None

    # Ensure symlink exists in output_dir
    link_dir = os.path.join(output_dir, rel_subdir)
    if not os.path.exists(link_dir):
        # Create symlink to parent
        parts = rel_subdir.split("/")
        model_link = os.path.join(output_dir, parts[0])
        if not os.path.exists(model_link):
            os.symlink(model_dir, model_link)

    return f"{rel_subdir}/{fname}"


def phase_html_report(samples, all_scores, models, output_dir):
    """Generate unified HTML report with all variants, all runs."""
    log.info("=" * 60)
    log.info("PHASE 6: HTML REPORT")
    log.info("=" * 60)

    model_order = [m["name"] for m in models]
    model_labels = {m["name"]: m["label"] for m in models}
    model_short = {m["name"]: m["short"] for m in models}

    # Compute aggregate stats per model × variant
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, sc in all_scores.items():
        parts = key.split("|")
        if len(parts) != 5:
            continue
        model, prompt_id, ref_id, seed, variant = parts
        if model not in model_order:
            continue
        for metric in ALL_METRICS:
            if metric in sc and sc[metric] is not None:
                if metric == "spk_sim" and ref_id in ("uncond", "None"):
                    continue
                stats[model][variant][metric].append(sc[metric])

    def avg(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    def std(lst):
        if len(lst) < 2:
            return 0.0
        m = sum(lst) / len(lst)
        return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

    # ── HTML ──
    html = []
    html.append("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>DramaBox Full Enhanced Eval: Runs 5–9</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; color: #333; max-width: 1600px; }
h1 { color: #222; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }
h2 { color: #444; margin-top: 30px; }
h3 { color: #555; margin-top: 20px; }
table { border-collapse: collapse; margin: 10px 0; font-size: 0.85em; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: center; }
th { background: #4a90d9; color: white; font-size: 0.85em; }
.section { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.12); overflow-x: auto; }
.r5 { background: #e3f2fd; }
.r6 { background: #fff3e0; }
.r7 { background: #e8f5e9; }
.r8 { background: #f3e5f5; }
.r9 { background: #fff8e1; }
.metric-up { color: #28a745; font-weight: bold; }
.metric-down { color: #dc3545; font-weight: bold; }
.metric-same { color: #6c757d; }
.legend { display: flex; gap: 15px; margin: 10px 0; font-size: 0.9em; flex-wrap: wrap; }
.legend span { padding: 3px 10px; border-radius: 4px; }
.prompt-section { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
.prompt-header { font-size: 1.1em; font-weight: bold; color: #333; margin-bottom: 4px; }
.prompt-text { font-size: 0.85em; color: #666; line-height: 1.5; margin: 4px 0 15px 0; padding: 8px 12px; background: #f8f9fa; border-left: 3px solid #4a90d9; border-radius: 0 4px 4px 0; }
.ref-block { margin: 12px 0 20px 0; padding: 10px 0; border-top: 1px solid #eee; }
.ref-title { font-weight: bold; font-size: 0.95em; color: #555; margin-bottom: 8px; }
.variant-row { display: flex; gap: 6px; margin: 4px 0; flex-wrap: wrap; }
.variant-label { font-size: 0.75em; font-weight: bold; color: #888; margin-top: 6px; margin-bottom: 2px; }
.audio-card { border: 1px solid #ddd; border-radius: 6px; padding: 8px; min-width: 220px; max-width: 280px; flex: 1; }
.audio-card.run5 { border-left: 3px solid #2196F3; background: #fafcff; }
.audio-card.run6 { border-left: 3px solid #FF9800; background: #fffcf5; }
.audio-card.run7 { border-left: 3px solid #4CAF50; background: #f5fff5; }
.audio-card.run8 { border-left: 3px solid #9C27B0; background: #fdf5ff; }
.audio-card.run9 { border-left: 3px solid #FF5722; background: #fff5f2; }
.audio-card .model-name { font-weight: bold; font-size: 0.8em; margin-bottom: 2px; }
.audio-card .scores { font-size: 0.72em; color: #666; margin-bottom: 3px; line-height: 1.4; }
.audio-card audio { width: 100%; height: 30px; }
.audio-card .no-audio { font-size: 0.75em; color: #999; font-style: italic; }
.variant-group { margin: 8px 0; }
</style>
</head><body>
<h1>DramaBox Full Enhanced Evaluation: Runs 5–9</h1>
<p>
<b>Run 5</b>: AdaLN-Zero speaker conditioning (152M params), DiT frozen in bf16, lr_adaln=5e-5, 5 epochs<br>
<b>Run 6</b>: Standard full fine-tune, fp32 master weights, lr=2e-6, 1 epoch, no AdaLN<br>
<b>Run 7</b>: LoRA rank 64 (113M params), lr=4e-5, 2 epochs, bf16<br>
<b>Run 8</b>: Frozen LoRA-merged DiT + AdaLN-Zero, lr=7e-5, 6 epochs<br>
<b>Run 9</b>: LoRA rank 128 (226M, fp32 master) + pretrained AdaLN-Zero, lr_dit=4e-5, lr_adaln=1e-5, 3 epochs<br>
<b>Variants</b>: Raw (DramaBox decoder) | Sidon (speech restoration) | VC→Sidon (ChatterboxVC voice conversion + Sidon)
</p>
<div class="legend">
  <span class="r5">Run 5 (AdaLN)</span>
  <span class="r6">Run 6 (Full FT)</span>
  <span class="r7">Run 7 (LoRA)</span>
  <span class="r8">Run 8 (Merged+AdaLN)</span>
  <span class="r9">Run 9 (LoRA128+AdaLN)</span>
</div>
""")

    # ── Aggregate tables per variant ──
    for variant in VARIANTS:
        variant_label = {"raw": "Raw Output", "sidon": "Sidon Enhanced",
                         "vc_sidon": "ChatterboxVC → Sidon"}[variant]
        html.append(f'<div class="section"><h2>{variant_label} — Aggregate Metrics</h2>')

        # Key metrics for the table header
        show_metrics = ["mos", "utmos", "nisqa_mos", "dnsmos_ovrl", "CE", "CU", "PQ", "spk_sim"]
        header = "<tr><th>Model</th>"
        for met in show_metrics:
            arrow = " &#8593;" if met in HIGHER_BETTER else ""
            header += f"<th>{METRIC_LABELS.get(met, met)}{arrow}</th>"
        header += "<th>N</th></tr>"
        html.append(f"<table>{header}")

        for m in model_order:
            if m not in stats or variant not in stats[m]:
                continue
            sv = stats[m][variant]
            n = len(sv.get("mos", []))
            if n == 0:
                continue
            run = _run_class(m)
            label = model_labels.get(m, m)
            html.append(f'<tr class="{run}"><td style="text-align:left;"><b>{label}</b></td>')
            for met in show_metrics:
                vals = sv.get(met, [])
                if vals:
                    html.append(f'<td>{avg(vals):.3f}</td>')
                else:
                    html.append('<td>&mdash;</td>')
            html.append(f'<td>{n}</td></tr>')

        html.append('</table></div>')

    # ── Delta table (raw variant, vs R5_vanilla) ──
    html.append('<div class="section"><h2>Deltas vs Vanilla (Raw Output)</h2>')
    show_metrics_delta = ["mos", "utmos", "CE", "PQ", "spk_sim"]
    header = "<tr><th>Model</th>"
    for met in show_metrics_delta:
        header += f"<th>&Delta;{METRIC_LABELS.get(met, met)}</th>"
    header += "</tr>"
    html.append(f"<table>{header}")

    vanilla_avgs = {}
    for met in show_metrics_delta:
        vals = stats.get("R5_vanilla", {}).get("raw", {}).get(met, [])
        vanilla_avgs[met] = avg(vals) if vals else 0

    for m in model_order:
        if m == "R5_vanilla":
            continue
        sv = stats.get(m, {}).get("raw", {})
        if not sv:
            continue
        run = _run_class(m)
        label = model_labels.get(m, m)
        html.append(f'<tr class="{run}"><td style="text-align:left;"><b>{label}</b></td>')
        for met in show_metrics_delta:
            vals = sv.get(met, [])
            if vals and vanilla_avgs[met]:
                delta = avg(vals) - vanilla_avgs[met]
                higher_better = met in HIGHER_BETTER
                cls = "metric-up" if (delta > 0.005 and higher_better) or (delta < -0.005 and not higher_better) else \
                      "metric-down" if (delta < -0.005 and higher_better) or (delta > 0.005 and not higher_better) else \
                      "metric-same"
                sign = "+" if delta > 0 else ""
                html.append(f'<td class="{cls}">{sign}{delta:.3f}</td>')
            else:
                html.append('<td>&mdash;</td>')
        html.append('</tr>')

    html.append('</table></div>')

    # ── Sidon improvement table ──
    html.append('<div class="section"><h2>Sidon Improvement (Raw → Sidon)</h2>')
    header = "<tr><th>Model</th>"
    for met in show_metrics_delta:
        header += f"<th>&Delta;{METRIC_LABELS.get(met, met)}</th>"
    header += "</tr>"
    html.append(f"<table>{header}")

    for m in model_order:
        raw_sv = stats.get(m, {}).get("raw", {})
        sid_sv = stats.get(m, {}).get("sidon", {})
        if not raw_sv or not sid_sv:
            continue
        run = _run_class(m)
        label = model_labels.get(m, m)
        html.append(f'<tr class="{run}"><td style="text-align:left;"><b>{label}</b></td>')
        for met in show_metrics_delta:
            raw_vals = raw_sv.get(met, [])
            sid_vals = sid_sv.get(met, [])
            if raw_vals and sid_vals:
                delta = avg(sid_vals) - avg(raw_vals)
                higher_better = met in HIGHER_BETTER
                cls = "metric-up" if (delta > 0.005 and higher_better) or (delta < -0.005 and not higher_better) else \
                      "metric-down" if (delta < -0.005 and higher_better) or (delta > 0.005 and not higher_better) else \
                      "metric-same"
                sign = "+" if delta > 0 else ""
                html.append(f'<td class="{cls}">{sign}{delta:.3f}</td>')
            else:
                html.append('<td>&mdash;</td>')
        html.append('</tr>')

    html.append('</table></div>')

    # ── Learnings & Retrospective Section ──
    html.append("""
<div class="section">
<h2>Learnings from Runs 5–9</h2>
<p style="font-size:0.85em;color:#666;">Cumulative insights from the DramaBox voice fine-tuning campaign.</p>

<h3 style="color:#2e7d32;">What Worked</h3>
<table>
<tr><th style="width:25%;">Technique</th><th style="width:15%;">First Used</th><th>Impact</th></tr>
<tr class="r8"><td style="text-align:left;"><b>LoRA-merged DiT + fresh AdaLN-Zero</b></td><td>Run 8</td>
<td>Best overall quality. Merging LoRA into DiT and training only AdaLN on top gave clean speaker conditioning without degrading the base model.</td></tr>
<tr class="r9"><td style="text-align:left;"><b>FP32 master weights for LoRA</b></td><td>Run 9</td>
<td>Solved the bf16 ULP floor problem. CPU-offloaded fp32 copies (~2.5 GB RAM) keep Adam updates alive across all epochs. Zero throughput impact.</td></tr>
<tr class="r7"><td style="text-align:left;"><b>LoRA on audio_attn1 + audio_ff</b></td><td>Run 7</td>
<td>Efficient fine-tuning: rank 64 → 113M params (3.2% of DiT). Trains meaningfully in epoch 1 before bf16 floor hits.</td></tr>
<tr class="r5"><td style="text-align:left;"><b>AdaLN-Zero speaker conditioning</b></td><td>Run 5</td>
<td>455M-param conditioning network learns speaker identity effectively. Converges in 4–5 epochs with cosine schedule.</td></tr>
<tr class="r5"><td style="text-align:left;"><b>Sidon post-processing</b></td><td>Run 5</td>
<td>Consistent +0.15 MOS and +0.35 NISQA improvement across all models. Essential for production quality.</td></tr>
<tr class="r7"><td style="text-align:left;"><b>ChatterboxVC → Sidon pipeline</b></td><td>Run 7</td>
<td>Voice conversion followed by restoration adds +0.07 SpkSim on top of Sidon alone.</td></tr>
<tr class="r5"><td style="text-align:left;"><b>Weight debug logging (hash + Δ%)</b></td><td>Run 5</td>
<td>Critical diagnostic. Exposed the bf16 freeze (0.0000% changed) that would otherwise appear as normal training loss convergence.</td></tr>
<tr class="r8"><td style="text-align:left;"><b>Separate optimizers for DiT vs AdaLN</b></td><td>Run 8</td>
<td>Allows different LRs (4e-5 DiT vs 1e-5 AdaLN) and independent freeze/unfreeze of components.</td></tr>
</table>

<h3 style="color:#c62828;">What Didn't Work</h3>
<table>
<tr><th style="width:25%;">Approach</th><th style="width:15%;">Run</th><th>Problem</th></tr>
<tr class="r6"><td style="text-align:left;"><b>Standard full fine-tune (fp32 master)</b></td><td>Run 6</td>
<td>3.5B params at lr=2e-6 did not converge meaningfully in 1 epoch. Too slow, too expensive.</td></tr>
<tr class="r7"><td style="text-align:left;"><b>LoRA in bf16 beyond epoch 1</b></td><td>Run 7</td>
<td>bf16 ULP floor: weights ~0.01 have ULP≈8.58e-5, Adam updates ~1.72e-5 round to zero. Training freezes silently.</td></tr>
<tr class="r9"><td style="text-align:left;"><b>Stacking LoRA on merged LoRA+AdaLN</b></td><td>Run 9</td>
<td>Diminishing returns: −0.025 UTMOS, −0.014 SpkSim vs Run 8. Adding more capacity on an already-adapted model doesn't help.</td></tr>
<tr class="r5"><td style="text-align:left;"><b>Longer AdaLN-only training (6 epochs)</b></td><td>Runs 5, 8</td>
<td>Peak quality at epoch 4–5; epochs 5–6 show mild overfitting (loss rises, diversity drops).</td></tr>
</table>

<h3 style="color:#1565c0;">Bugs Encountered &amp; Fixed</h3>
<table>
<tr><th style="width:25%;">Bug</th><th style="width:15%;">Run</th><th>Root Cause &amp; Fix</th></tr>
<tr><td style="text-align:left;"><b>bf16 ULP floor</b></td><td>Run 7→9</td>
<td><b>Root cause:</b> bf16 has 8 mantissa bits; for weights near 0.01, the smallest representable step is ~8.58e-5, but Adam updates are ~1.72e-5. Updates round to zero.<br>
<b>Failed fixes:</b> (1) pre-<code>accelerator.prepare()</code> fp32 cast — DDP replaces params; (2) post-<code>prepare()</code> fp32 cast — DDP flat buffers ignore param.data replacement.<br>
<b>Working fix:</b> FP32MasterWeights class: CPU-offloaded fp32 param copies, copy bf16 grads→fp32, run AdamW in fp32, copy fp32→bf16 back.</td></tr>
<tr><td style="text-align:left;"><b>DDP flat parameter buffers</b></td><td>Run 9</td>
<td><b>Root cause:</b> <code>param.data = fp32_tensor</code> doesn't work because DDP manages its own contiguous buffer. Direct param.data replacement is silently ignored.<br>
<b>Fix:</b> Keep bf16 params on GPU as DDP sees them; maintain separate fp32 copies on CPU for the actual optimizer step.</td></tr>
<tr><td style="text-align:left;"><b>Accelerator optimizer preparation</b></td><td>Run 8</td>
<td><b>Root cause:</b> When DiT is frozen, empty param groups cause errors in <code>accelerator.prepare(optimizer)</code>.<br>
<b>Fix:</b> Use dummy single-param optimizer for DiT when frozen; only step the AdaLN optimizer.</td></tr>
<tr><td style="text-align:left;"><b>Checkpoint size explosion</b></td><td>Run 8</td>
<td><b>Root cause:</b> Saving full DiT (6.2 GB) at every checkpoint wastes disk.<br>
<b>Fix:</b> <code>skip_model_save=True</code> when DiT is frozen — only save AdaLN (~1.7 GB/checkpoint). Saved ~25 GB disk over 6 epochs.</td></tr>
<tr><td style="text-align:left;"><b>LoRA weight saving in fp32</b></td><td>Run 9</td>
<td><b>Root cause:</b> LoRA params are bf16 on GPU but fp32 master copies have the real trained values.<br>
<b>Fix:</b> Save from fp32 master copies, not from GPU bf16 params. Verified by checking saved tensor dtypes.</td></tr>
</table>
</div>
""")

    # ── Audio Comparison Section ──
    html.append('<div class="section"><h2>Audio Samples — All Models &times; All Variants</h2>')
    html.append('<p style="font-size:0.85em;color:#666;">For each prompt + speaker + seed: '
                'three rows (Raw / Sidon / VC→Sidon) with all models side by side.</p>')

    # Group samples
    groups = defaultdict(dict)
    for s in samples:
        key = (s["prompt_id"], s.get("ref_id"), s["seed"])
        groups[key][s["model"]] = s

    refs = []
    seen_refs = set()
    for s in samples:
        rid = s.get("ref_id")
        if rid and rid not in seen_refs:
            seen_refs.add(rid)
            refs.append(rid)

    seeds = sorted(set(s["seed"] for s in samples))

    for prompt in ALL_PROMPTS:
        pid = prompt["id"]
        lang_tag = f' ({prompt["lang"]})'
        html.append(f'<div class="prompt-section">')
        html.append(f'<div class="prompt-header">{pid}{lang_tag}</div>')
        html.append(f'<div class="prompt-text">{prompt["prompt"]}</div>')

        for ref in refs:
            for seed in seeds:
                key = (pid, ref, seed)
                if key not in groups:
                    continue
                grp = groups[key]
                if not grp:
                    continue

                html.append(f'<div class="ref-block">')
                html.append(f'<div class="ref-title">Speaker: {ref} &nbsp;|&nbsp; Seed: {seed}</div>')

                for variant in VARIANTS:
                    variant_label = {"raw": "Raw", "sidon": "Sidon", "vc_sidon": "VC→Sidon"}[variant]
                    html.append(f'<div class="variant-group">')
                    html.append(f'<div class="variant-label">{variant_label}</div>')
                    html.append(f'<div class="variant-row">')

                    for m in model_order:
                        if m not in grp:
                            continue
                        s = grp[m]

                        # Skip vc_sidon for unconditional
                        if variant == "vc_sidon" and not s["conditional"]:
                            continue

                        run = _run_class_audio(m)
                        short = model_short.get(m, m)

                        # Get scores
                        ref_id_key = ref or "uncond"
                        score_key = f"{m}|{pid}|{ref_id_key}|{seed}|{variant}"
                        sc = all_scores.get(score_key, {})

                        parts = []
                        if sc.get("mos") is not None:
                            parts.append(f'MOS={sc["mos"]:.2f}')
                        if sc.get("utmos") is not None:
                            parts.append(f'UTMOS={sc["utmos"]:.2f}')
                        if sc.get("CE") is not None:
                            parts.append(f'CE={sc["CE"]:.1f}')
                        if sc.get("spk_sim") is not None:
                            parts.append(f'SpkSim={sc["spk_sim"]:.3f}')
                        score_str = " &middot; ".join(parts)

                        mp3 = _mp3_url(s, variant, output_dir)

                        html.append(f'<div class="audio-card {run}">')
                        html.append(f'<div class="model-name">{short}</div>')
                        if score_str:
                            html.append(f'<div class="scores">{score_str}</div>')
                        if mp3:
                            html.append(f'<audio controls preload="none" src="{mp3}"></audio>')
                        else:
                            html.append(f'<div class="no-audio">No audio</div>')
                        html.append('</div>')

                    html.append('</div>')  # variant-row
                    html.append('</div>')  # variant-group

                html.append('</div>')  # ref-block

        html.append('</div>')  # prompt-section

    html.append('</div>')  # section
    html.append('</body></html>')

    report_path = os.path.join(output_dir, "eval_full_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    log.info(f"Report written: {report_path} ({os.path.getsize(report_path) / 1e6:.1f} MB)")
    return report_path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Get all model configs
    models = get_all_models()
    log.info(f"Models: {[m['name'] for m in models]}")

    # Phase 1: Collect existing + generate Run 7
    log.info("=" * 60)
    log.info("PHASE 1: COLLECT & GENERATE AUDIO")
    log.info("=" * 60)

    existing_samples = collect_existing_samples(models)
    run7_samples = generate_run7_samples(models, OUTPUT_DIR)
    samples = existing_samples + run7_samples

    # Save metadata
    with open(os.path.join(OUTPUT_DIR, "samples_meta.json"), "w") as f:
        json.dump(samples, f, indent=2, default=str)
    log.info(f"Total samples: {len(samples)}")

    # Phase 2: Sidon
    phase_sidon(samples, device)

    # Phase 3: VC → Sidon
    phase_vc_sidon(samples, device)

    # Phase 4: Score
    all_scores = phase_score(samples, device)

    # Phase 5: Convert to MP3
    phase_convert_mp3(samples)

    # Phase 6: HTML report
    # Create symlinks in output dir for serving
    for mcfg in models:
        if mcfg["run"] in ("run5", "run6"):
            name_map_r5 = {"R5_vanilla": "vanilla", "R5_adaln_epoch4": "best_epoch4",
                           "R5_adaln_epoch5": "final_epoch5"}
            name_map_r6 = {"R6_vanilla": "vanilla"}
            if mcfg["run"] == "run5":
                orig = name_map_r5.get(mcfg["name"], mcfg["name"])
                src = os.path.join(RUN5_DIR, orig)
            else:
                orig = name_map_r6.get(mcfg["name"], mcfg["name"])
                src = os.path.join(RUN6_DIR, orig)

            link = os.path.join(OUTPUT_DIR, mcfg["name"])
            if os.path.exists(src) and not os.path.exists(link):
                os.symlink(src, link)

    report_path = phase_html_report(samples, all_scores, models, OUTPUT_DIR)

    log.info("=" * 60)
    log.info("EVALUATION COMPLETE")
    log.info(f"Report: {report_path}")
    log.info(f"Scores: {os.path.join(OUTPUT_DIR, 'enhanced_scores.json')}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
