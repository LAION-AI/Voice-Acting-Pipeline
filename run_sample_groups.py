#!/usr/bin/env python3
"""
Generate 20 sample groups (25 candidates each), process through the full pipeline,
and produce an HTML report with embedded MP3s showing top-2 candidates.

Groups:
  1-10:  CC2-C Archetype (character consistency, two-scene CUT TO:)
  11-20: Extreme Physical
  For reference audio groups: round-robin assignment across 5 reference files

Pipeline per candidate:
  1. DramaBox TTS (CFG=2.5, STG=1.5)
  2. Sidon + ChatterboxVC augmentation (best-of-2 by DNS-MOS OVR)
  3. Whisper turbo ASR + CUT TO: split
  4. VoiceCLAP + WER scoring
  5. MOSS re-annotation (Gemma 4 E4B-it prompt rewriting)
  6. HTML report generation

Uses all 8 GPUs via multiprocessing.
"""

import gc
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sample_groups")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DRAMABOX_DIR = Path("/home/deployer/laion/DramaBox")
VOICE_PIPELINE_DIR = Path("/home/deployer/laion/Voice-Acting-Pipeline")
SIDON_DIR = Path("/home/deployer/laion/sidon")
OUTPUT_DIR = Path("/home/deployer/laion/Voice-Acting-Pipeline/sample_groups_output")

CC2C_PROMPTS_FILE = VOICE_PIPELINE_DIR / "data" / "dramabox_cc2c_archetype.json"
EXTREME_PROMPTS_FILE = VOICE_PIPELINE_DIR / "data" / "dramabox_extreme_physical.json"

REF_DIR = Path("/home/deployer/laion/test-refs")
REF_FILES = [
    REF_DIR / "chris-ref.mp3",
    REF_DIR / "Fairy-2_trim.wav",
    REF_DIR / "Samantha musing about flirting Voice AI - Echo TTS_trim.wav",
    REF_DIR / "School Goblin - en_trim.wav",
    REF_DIR / "spongebob-ref.mp3",
]

NUM_CANDIDATES = 25
CFG_SCALE = 2.5
STG_SCALE = 1.5
NUM_GPUS = 8
NUM_ARCHETYPE_GROUPS = 10
NUM_EXTREME_GROUPS = 10
TOTAL_GROUPS = NUM_ARCHETYPE_GROUPS + NUM_EXTREME_GROUPS

MP3_BITRATE = "256k"
MP3_SAMPLE_RATE = 48000

# Timing tracker
TIMINGS = {}


def record_time(stage, elapsed):
    TIMINGS.setdefault(stage, []).append(elapsed)


# ---------------------------------------------------------------------------
# Step 1: Select prompts
# ---------------------------------------------------------------------------
def select_prompts():
    """Select 10 CC2-C archetype + 10 extreme physical prompts."""
    import random
    random.seed(42)

    with open(CC2C_PROMPTS_FILE) as f:
        cc2c_all = json.load(f)
    with open(EXTREME_PROMPTS_FILE) as f:
        extreme_all = json.load(f)

    cc2c_selected = random.sample(cc2c_all, NUM_ARCHETYPE_GROUPS)
    extreme_selected = random.sample(extreme_all, NUM_EXTREME_GROUPS)

    groups = []
    ref_idx = 0

    for i, prompt_data in enumerate(cc2c_selected):
        ref_file = str(REF_FILES[ref_idx % len(REF_FILES)])
        ref_idx += 1
        groups.append({
            "group_id": i,
            "pathway": prompt_data["pathway"],
            "prompt": prompt_data["dramabox_prompt"],
            "sample_info": prompt_data.get("sample_info", {}),
            "ref_audio": ref_file,
            "source": "cc2c_archetype",
        })

    for i, prompt_data in enumerate(extreme_selected):
        ref_file = str(REF_FILES[ref_idx % len(REF_FILES)])
        ref_idx += 1
        groups.append({
            "group_id": NUM_ARCHETYPE_GROUPS + i,
            "pathway": prompt_data["pathway"],
            "prompt": prompt_data["dramabox_prompt"],
            "sample_info": prompt_data.get("sample_info", {}),
            "ref_audio": ref_file,
            "source": "extreme_physical",
        })

    return groups


# ---------------------------------------------------------------------------
# Step 2: TTS synthesis (multi-GPU)
# ---------------------------------------------------------------------------
def tts_worker(gpu_id, work_items, output_dir):
    """Generate TTS for assigned items on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    sys.path.insert(0, str(DRAMABOX_DIR / "src"))
    sys.path.insert(0, str(DRAMABOX_DIR / "ltx2"))

    from inference_server import TTSServer
    from model_downloader import get_all_paths

    paths = get_all_paths()
    server = TTSServer(
        checkpoint=paths["transformer"],
        full_checkpoint=paths["audio_components"],
        gemma_root=paths["gemma_root"],
        device="cuda",
        dtype="bf16",
        compile_model=True,
        bnb_4bit=True,
    )

    results = []
    for item in work_items:
        gid = item["group_id"]
        cid = item["candidate_id"]
        prompt = item["prompt"]
        seed = item["seed"]
        tag = f"g{gid:02d}_c{cid:02d}"
        out_path = os.path.join(output_dir, f"{tag}_raw.wav")

        if os.path.exists(out_path):
            results.append({"tag": tag, "path": out_path, "status": "skipped"})
            continue

        try:
            t0 = time.time()
            server.generate_to_file(
                prompt=prompt,
                output=out_path,
                cfg_scale=CFG_SCALE,
                stg_scale=STG_SCALE,
                duration_multiplier=1.1,
                seed=seed,
                watermark=False,
            )
            elapsed = time.time() - t0
            results.append({"tag": tag, "path": out_path, "status": "ok",
                            "elapsed": elapsed})
            print(f"  [GPU {gpu_id}] {tag} done ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [GPU {gpu_id}] {tag} FAILED: {e}", flush=True)
            results.append({"tag": tag, "path": "", "status": f"error: {e}"})

    # Save results
    res_path = os.path.join(output_dir, f"tts_results_gpu{gpu_id}.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def run_tts_synthesis(groups, output_dir):
    """Generate 25 candidates per group across 8 GPUs."""
    log.info("=== STEP 1: TTS Synthesis ===")
    t0 = time.time()

    tts_dir = output_dir / "tts_raw"
    tts_dir.mkdir(parents=True, exist_ok=True)

    # Build work items: 20 groups x 25 candidates = 500 total
    work_items = []
    for g in groups:
        for c in range(NUM_CANDIDATES):
            work_items.append({
                "group_id": g["group_id"],
                "candidate_id": c,
                "prompt": g["prompt"],
                "seed": 1000 * g["group_id"] + c,
            })

    # Distribute round-robin across GPUs
    shards = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work_items):
        shards[i % NUM_GPUS].append(item)

    log.info(f"  {len(work_items)} items across {NUM_GPUS} GPUs "
             f"({len(shards[0])} per GPU)")

    # Launch workers
    processes = []
    for gpu_id in range(NUM_GPUS):
        if not shards[gpu_id]:
            continue
        p = mp.Process(target=tts_worker,
                       args=(gpu_id, shards[gpu_id], str(tts_dir)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t0
    record_time("tts_synthesis", elapsed)
    log.info(f"  TTS done in {elapsed:.1f}s ({len(work_items)/elapsed:.1f} samples/s)")
    return tts_dir


# ---------------------------------------------------------------------------
# Step 3: Enhancement + ASR + Splitting + Scoring (multi-GPU)
# ---------------------------------------------------------------------------
def enhance_worker(gpu_id, work_items, tts_dir, enhanced_dir):
    """Run enhancement + ASR + splitting + scoring on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    import torchaudio
    import soundfile as sf
    import whisper

    # Load Sidon enhancer
    sys.path.insert(0, str(VOICE_PIPELINE_DIR))
    from dramabox.sidon_enhance import SidonEnhancer, score_dnsmos

    enhancer = SidonEnhancer(device="cuda:0")
    print(f"  [GPU {gpu_id}] Sidon enhancer loaded", flush=True)

    # Load Whisper turbo
    whisper_model = whisper.load_model("turbo", device="cuda")
    print(f"  [GPU {gpu_id}] Whisper turbo loaded", flush=True)

    # Load VoiceCLAP (optional — requires transformers>=5)
    clap_model = None
    neg_emb = None
    try:
        from sentence_transformers import SentenceTransformer
        clap_model = SentenceTransformer(
            "laion/voiceclap-large", trust_remote_code=True, device="cuda"
        )
        neg_text = "robotic, distorted, uncanny, distortion"
        neg_emb = clap_model.encode([neg_text], normalize_embeddings=True)[0]
        print(f"  [GPU {gpu_id}] VoiceCLAP loaded", flush=True)
    except Exception as e:
        print(f"  [GPU {gpu_id}] VoiceCLAP unavailable ({e}), using WER+DNS-MOS scoring", flush=True)

    results = []
    for item in work_items:
        gid = item["group_id"]
        cid = item["candidate_id"]
        prompt = item["prompt"]
        ref_audio = item.get("ref_audio")
        tag = f"g{gid:02d}_c{cid:02d}"

        result_path = os.path.join(enhanced_dir, f"{tag}_result.json")
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        raw_path = os.path.join(tts_dir, f"{tag}_raw.wav")
        if not os.path.exists(raw_path):
            continue

        result = {
            "group_id": gid, "candidate_id": cid, "tag": tag,
            "prompt": prompt, "ref_audio": ref_audio,
        }

        try:
            t0 = time.time()

            # --- Enhancement ---
            t_enh = time.time()
            enhanced_np, method = enhancer.augment_sample(raw_path, ref_path=ref_audio)
            result["enhance_method"] = method
            result["enhance_time"] = round(time.time() - t_enh, 2)

            # DNS-MOS score
            from dramabox.sidon_enhance import _resample
            enhanced_16k = _resample(enhanced_np, 48000, 16000)
            result["dnsmos_ovr"] = round(score_dnsmos(enhanced_16k, "cuda:0"), 4)

            # Save enhanced WAV
            enh_wav = os.path.join(enhanced_dir, f"{tag}_enhanced.wav")
            sf.write(enh_wav, enhanced_np, 48000)
            result["enhanced_path"] = enh_wav

            # --- Whisper ASR ---
            t_asr = time.time()
            asr_result = whisper_model.transcribe(enh_wav, word_timestamps=True)
            asr_text = asr_result.get("text", "").strip()
            word_ts = []
            for seg in asr_result.get("segments", []):
                for w in seg.get("words", []):
                    word_ts.append({
                        "word": w["word"].strip(),
                        "start": round(w["start"], 3),
                        "end": round(w["end"], 3),
                    })
            result["asr_text"] = asr_text
            result["word_timestamps"] = word_ts
            result["asr_time"] = round(time.time() - t_asr, 2)

            # --- WER ---
            # Extract expected text from prompt (quoted dialogue)
            matches = re.findall(r'"([^"]*)"', prompt)
            expected_text = " ".join(matches).strip()
            result["expected_text"] = expected_text

            if expected_text:
                hyp = asr_text.lower().split()
                ref = expected_text.lower().split()
                if ref:
                    n, m = len(ref), len(hyp)
                    d = [[0]*(m+1) for _ in range(n+1)]
                    for i in range(n+1): d[i][0] = i
                    for j in range(m+1): d[0][j] = j
                    for i in range(1, n+1):
                        for j in range(1, m+1):
                            d[i][j] = (d[i-1][j-1] if ref[i-1]==hyp[j-1]
                                       else 1+min(d[i-1][j], d[i][j-1], d[i-1][j-1]))
                    result["wer"] = round(d[n][m] / n, 4)
                else:
                    result["wer"] = 0.0
            else:
                result["wer"] = None

            # --- CUT TO: split ---
            has_cut_to = bool(re.search(r'\bCUT\s+TO\s*:', prompt))
            result["has_cut_to"] = has_cut_to

            mono_np = enhanced_np
            sr_out = 48000
            total_duration = len(mono_np) / sr_out

            if has_cut_to and word_ts:
                # Find longest silence gap in middle 20-80%
                lo, hi = total_duration * 0.20, total_duration * 0.80
                gaps = []
                for i in range(1, len(word_ts)):
                    gap_s = word_ts[i-1]["end"]
                    gap_e = word_ts[i]["start"]
                    gap_mid = (gap_s + gap_e) / 2
                    if lo <= gap_mid <= hi:
                        gaps.append((gap_e - gap_s, gap_mid))
                if gaps:
                    _, split_sec = max(gaps)
                else:
                    split_sec = total_duration / 2
                result["split_sec"] = round(split_sec, 3)

                split_sample = int(split_sec * sr_out)
                split_sample = max(0, min(split_sample, len(mono_np)))
                part1_np = mono_np[:split_sample]
                part2_np = mono_np[split_sample:]

                # Apply fades
                fade = int(0.05 * sr_out)
                if fade < len(part1_np):
                    part1_np = part1_np.copy()
                    part1_np[-fade:] *= np.linspace(1, 0, fade)
                if fade < len(part2_np):
                    part2_np = part2_np.copy()
                    part2_np[:fade] *= np.linspace(0, 1, fade)

                p1_wav = os.path.join(enhanced_dir, f"{tag}_part1.wav")
                p2_wav = os.path.join(enhanced_dir, f"{tag}_part2.wav")
                sf.write(p1_wav, part1_np, sr_out)
                sf.write(p2_wav, part2_np, sr_out)
                result["part1_path"] = p1_wav
                result["part2_path"] = p2_wav
                result["part1_dur"] = round(len(part1_np) / sr_out, 3)
                result["part2_dur"] = round(len(part2_np) / sr_out, 3)
            else:
                result["split_sec"] = None

            result["full_dur"] = round(total_duration, 3)

            # --- VoiceCLAP scoring (if available) ---
            if clap_model is not None:
                t_clap = time.time()
                audio_16k = _resample(mono_np, 48000, 16000)
                audio_input = [{"array": audio_16k.astype(np.float32), "sampling_rate": 16000}]

                san_prompt = re.sub(r'"[^"]*"', '', prompt).strip()
                san_prompt = re.sub(r'\s+', ' ', san_prompt)[:500]

                emb_audio = clap_model.encode(audio_input, normalize_embeddings=True)[0]
                emb_text = clap_model.encode([san_prompt], normalize_embeddings=True)[0]

                result["clap_sim"] = round(float(np.dot(emb_audio, emb_text)), 4)
                result["clap_neg"] = round(float(np.dot(emb_audio, neg_emb)), 4)
                result["clap_time"] = round(time.time() - t_clap, 2)
            else:
                result["clap_sim"] = 0.0
                result["clap_neg"] = 0.0
                result["clap_time"] = 0.0

            # Composite score: WER + CLAP (if available) + DNS-MOS
            wer_val = result.get("wer", 1.0) or 1.0
            dnsmos_val = result.get("dnsmos_ovr", 1.0)
            if clap_model is not None:
                result["reward"] = round(
                    (1 - min(wer_val, 1.0)) * (result["clap_sim"] - result["clap_neg"] + 2),
                    4
                )
            else:
                # Fallback: WER accuracy * DNS-MOS (normalized to 0-1)
                result["reward"] = round(
                    (1 - min(wer_val, 1.0)) * (dnsmos_val / 5.0),
                    4
                )

            # --- MP3 conversion ---
            for suffix in ["enhanced", "part1", "part2"]:
                wav_p = result.get(f"{suffix}_path") or result.get(f"{'enhanced' if suffix == 'enhanced' else suffix}_path")
                if wav_p and os.path.exists(wav_p):
                    mp3_p = wav_p.replace(".wav", ".mp3")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", wav_p, "-ac", "1", "-ar",
                         str(MP3_SAMPLE_RATE), "-b:a", MP3_BITRATE, "-f", "mp3", mp3_p],
                        capture_output=True, check=False,
                    )
                    result[f"{suffix}_mp3"] = mp3_p

            result["total_time"] = round(time.time() - t0, 2)
            result["status"] = "ok"

        except Exception as e:
            print(f"  [GPU {gpu_id}] {tag} FAILED: {e}", flush=True)
            traceback.print_exc()
            result["status"] = f"error: {e}"
            result["total_time"] = 0

        torch.cuda.empty_cache()

        # Save result
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        results.append(result)

        if len(results) % 5 == 0:
            print(f"  [GPU {gpu_id}] {len(results)}/{len(work_items)} done", flush=True)

    return results


def run_enhancement(groups, tts_dir, output_dir):
    """Run enhancement + ASR + scoring across 8 GPUs."""
    log.info("=== STEP 2: Enhancement + ASR + Scoring ===")
    t0 = time.time()

    enhanced_dir = output_dir / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # Build work items
    work_items = []
    for g in groups:
        for c in range(NUM_CANDIDATES):
            work_items.append({
                "group_id": g["group_id"],
                "candidate_id": c,
                "prompt": g["prompt"],
                "ref_audio": g.get("ref_audio"),
            })

    # Distribute
    shards = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work_items):
        shards[i % NUM_GPUS].append(item)

    log.info(f"  {len(work_items)} items across {NUM_GPUS} GPUs")

    processes = []
    for gpu_id in range(NUM_GPUS):
        if not shards[gpu_id]:
            continue
        p = mp.Process(target=enhance_worker,
                       args=(gpu_id, shards[gpu_id], str(tts_dir), str(enhanced_dir)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t0
    record_time("enhancement", elapsed)
    log.info(f"  Enhancement done in {elapsed:.1f}s")
    return enhanced_dir


# ---------------------------------------------------------------------------
# Step 4: MOSS-style re-annotation with Gemma 4 E4B-it
# ---------------------------------------------------------------------------
def reannotate_worker(gpu_id, work_items, enhanced_dir):
    """Run Gemma 4 E4B-it prompt re-annotation on assigned items."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "google/gemma-4-E4B-it"
    print(f"  [GPU {gpu_id}] Loading Gemma 4 E4B-it...", flush=True)

    # Patch tokenizer config: transformers 4.x expects extra_special_tokens as dict,
    # but Gemma 4 ships it as a list. Copy tokenizer files to a temp dir and fix.
    from huggingface_hub import snapshot_download
    tok_src = snapshot_download(model_id, allow_patterns=["tokenizer*", "special_tokens*", "*.model"])
    tok_dir = tempfile.mkdtemp(prefix="gemma_tok_")
    for fname in os.listdir(tok_src):
        src_f = os.path.join(tok_src, fname)
        if os.path.isfile(src_f):
            shutil.copy2(src_f, os.path.join(tok_dir, fname))
    tc_path = os.path.join(tok_dir, "tokenizer_config.json")
    with open(tc_path, encoding="utf-8") as f:
        tc = json.load(f)
    if isinstance(tc.get("extra_special_tokens"), list):
        tc["extra_special_tokens"] = {t: t for t in tc["extra_special_tokens"]}
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f)

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"  [GPU {gpu_id}] Gemma loaded", flush=True)

    system_prompt = """You rewrite DramaBox prompts to match what was actually performed in the audio.
Given the original DramaBox prompt and the ASR transcript of the generated audio,
rewrite the prompt so it accurately describes what was ACTUALLY said and how it was performed.

Rules:
1. Keep the speaker description paragraph (age, gender, timbre, recording quality).
2. Use the ASR transcript as the new dialogue in double quotes.
3. Adjust stage directions to match the actual delivery implied by the transcript.
4. Keep the same format: speaker description, then alternating stage directions + dialogue.
5. Output ONLY the rewritten DramaBox prompt, no explanation."""

    for item in work_items:
        tag = item["tag"]
        result_path = os.path.join(enhanced_dir, f"{tag}_result.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)

        if result.get("status") != "ok":
            continue
        if result.get("refined_prompt_full"):
            continue  # already done

        asr_text = result.get("asr_text", "")
        original_prompt = result.get("prompt", "")
        if not asr_text or not original_prompt:
            continue

        try:
            # Full prompt re-annotation
            user_msg = (
                f"Original DramaBox prompt:\n{original_prompt}\n\n"
                f"ASR transcript of generated audio:\n{asr_text}\n\n"
                f"Rewrite the DramaBox prompt to match the actual performance."
            )
            messages = [{"role": "user", "content": system_prompt + "\n\n" + user_msg}]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=1024,
                                         temperature=0.7, top_p=0.9, do_sample=True)
            new_tokens = outputs[0][inputs.shape[-1]:]
            refined = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            result["refined_prompt_full"] = refined

            # Part-level re-annotation if CUT TO:
            if result.get("has_cut_to"):
                split_sec = result.get("split_sec", 0)
                words = result.get("word_timestamps", [])
                p1_words = " ".join(w["word"] for w in words if w["end"] <= split_sec + 0.1)
                p2_words = " ".join(w["word"] for w in words if w["start"] >= split_sec - 0.1)

                for part_name, part_transcript in [("part1", p1_words), ("part2", p2_words)]:
                    if not part_transcript.strip():
                        result[f"refined_prompt_{part_name}"] = ""
                        continue
                    user_msg_p = (
                        f"Original full DramaBox prompt:\n{original_prompt}\n\n"
                        f"ASR transcript of {part_name} audio only:\n{part_transcript}\n\n"
                        f"Write a standalone single-scene DramaBox prompt for JUST this part."
                    )
                    msgs_p = [{"role": "user", "content": system_prompt + "\n\n" + user_msg_p}]
                    inp_p = tokenizer.apply_chat_template(
                        msgs_p, return_tensors="pt", add_generation_prompt=True
                    ).to("cuda")
                    with torch.no_grad():
                        out_p = model.generate(inp_p, max_new_tokens=768,
                                               temperature=0.7, top_p=0.9, do_sample=True)
                    new_p = out_p[0][inp_p.shape[-1]:]
                    result[f"refined_prompt_{part_name}"] = tokenizer.decode(
                        new_p, skip_special_tokens=True
                    ).strip()

            # Save updated result
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"  [GPU {gpu_id}] {tag} re-annotated", flush=True)

        except Exception as e:
            print(f"  [GPU {gpu_id}] {tag} re-annotation FAILED: {e}", flush=True)

        torch.cuda.empty_cache()


def run_reannotation(groups, enhanced_dir):
    """Run Gemma re-annotation across GPUs (only for top-2 candidates per group)."""
    log.info("=== STEP 3: Prompt Re-annotation ===")
    t0 = time.time()

    # Load all results and find top-2 per group
    all_results = {}
    for f in sorted(enhanced_dir.glob("g*_result.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("status") == "ok":
            gid = r["group_id"]
            all_results.setdefault(gid, []).append(r)

    top_items = []
    for gid in sorted(all_results.keys()):
        candidates = all_results[gid]
        candidates.sort(key=lambda x: x.get("reward", 0), reverse=True)
        for r in candidates[:2]:
            top_items.append({"tag": r["tag"]})

    log.info(f"  Re-annotating {len(top_items)} top candidates (top-2 per group)")

    # Distribute across GPUs (fewer items, use fewer GPUs)
    n_gpus = min(NUM_GPUS, max(1, len(top_items) // 2))
    shards = [[] for _ in range(n_gpus)]
    for i, item in enumerate(top_items):
        shards[i % n_gpus].append(item)

    processes = []
    for gpu_id in range(n_gpus):
        if not shards[gpu_id]:
            continue
        p = mp.Process(target=reannotate_worker,
                       args=(gpu_id, shards[gpu_id], str(enhanced_dir)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t0
    record_time("reannotation", elapsed)
    log.info(f"  Re-annotation done in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Step 5: Build HTML report
# ---------------------------------------------------------------------------
def build_html_report(groups, enhanced_dir, output_dir):
    """Build HTML report with embedded MP3s showing top-2 candidates per group."""
    import base64

    log.info("=== STEP 4: Building HTML Report ===")
    t0 = time.time()

    # Load all results
    all_results = {}
    for f in sorted(enhanced_dir.glob("g*_result.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("status") == "ok":
            gid = r["group_id"]
            all_results.setdefault(gid, []).append(r)

    def embed_mp3(path):
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f'<audio controls preload="none"><source src="data:audio/mpeg;base64,{data}"></audio>'

    def escape_html(text):
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html_parts = ["""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>DramaBox Sample Groups — Top-2 Candidates</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; margin: 0; padding: 20px; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { color: #ff9800; border-bottom: 2px solid #333; padding-bottom: 12px; }
h2 { color: #ffd54f; margin-top: 40px; border-top: 1px solid #333; padding-top: 20px; }
h3 { color: #4fc3f7; }
.group { background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 20px 0; border: 1px solid #333; }
.candidate { background: #0d1117; border-radius: 8px; padding: 16px; margin: 12px 0; border: 1px solid #2a2a3e; }
.scores { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin: 10px 0; }
.score-card { background: #1a1a2e; border-radius: 6px; padding: 8px; text-align: center; border: 1px solid #333; }
.score-val { font-size: 1.3em; font-weight: bold; color: #ff9800; }
.score-label { font-size: 0.75em; color: #888; }
.prompt-box { background: #0d1117; border-radius: 6px; padding: 12px; margin: 8px 0; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; border-left: 3px solid #4fc3f7; max-height: 300px; overflow-y: auto; }
.meta { color: #888; font-size: 0.85em; margin: 4px 0; }
.rank-1 { border-left: 4px solid #4caf50; }
.rank-2 { border-left: 4px solid #ff9800; }
audio { width: 100%; margin: 4px 0; }
.audio-row { display: grid; grid-template-columns: 80px 1fr; gap: 8px; align-items: center; margin: 4px 0; }
.audio-label { color: #888; font-size: 0.85em; font-weight: bold; }
.tag { display: inline-block; background: #2a2a3e; border-radius: 4px; padding: 2px 8px; font-size: 0.8em; color: #4fc3f7; margin: 2px 4px; }
details { margin: 8px 0; }
summary { cursor: pointer; color: #4fc3f7; font-weight: bold; }
.throughput { background: #1a1a2e; border-radius: 8px; padding: 16px; margin: 20px 0; border: 1px solid #333; }
.throughput table { width: 100%; border-collapse: collapse; }
.throughput td, .throughput th { padding: 8px 12px; border-bottom: 1px solid #333; text-align: left; }
.throughput th { color: #ff9800; }
</style>
</head><body>
<div class="container">
<h1>DramaBox Sample Groups — Top-2 Candidates</h1>
<p class="meta">Generated: """ + time.strftime("%Y-%m-%d %H:%M:%S") + f"""
 | {TOTAL_GROUPS} groups x {NUM_CANDIDATES} candidates | CFG={CFG_SCALE}, STG={STG_SCALE}</p>
"""]

    for g in groups:
        gid = g["group_id"]
        candidates = all_results.get(gid, [])
        if not candidates:
            continue

        candidates.sort(key=lambda x: x.get("reward", 0), reverse=True)
        top2 = candidates[:2]

        html_parts.append(f"""
<div class="group">
<h2>Group {gid} — {escape_html(g['source'])} ({escape_html(g['pathway'])})</h2>
<div class="meta">
<strong>Sample Info:</strong> {escape_html(json.dumps(g.get('sample_info', {})))}
</div>
<div class="meta"><strong>Reference Audio:</strong> {escape_html(os.path.basename(g.get('ref_audio', 'None')))}</div>
<details>
<summary>Original Prompt</summary>
<div class="prompt-box">{escape_html(g['prompt'])}</div>
</details>
""")

        for rank, r in enumerate(top2, 1):
            rank_class = f"rank-{rank}"
            html_parts.append(f"""
<div class="candidate {rank_class}">
<h3>#{rank} — Candidate {r['candidate_id']} <span class="tag">{r['enhance_method']}</span>
<span class="tag">seed={r.get('candidate_id', 0) + 1000 * gid}</span></h3>

<div class="scores">
<div class="score-card"><div class="score-val">{r.get('reward', 0):.3f}</div><div class="score-label">Reward</div></div>
<div class="score-card"><div class="score-val">{r.get('wer', 0):.1%}</div><div class="score-label">WER</div></div>
<div class="score-card"><div class="score-val">{r.get('clap_sim', 0):.3f}</div><div class="score-label">CLAP Sim</div></div>
<div class="score-card"><div class="score-val">{r.get('clap_neg', 0):.3f}</div><div class="score-label">CLAP Neg</div></div>
<div class="score-card"><div class="score-val">{r.get('dnsmos_ovr', 0):.2f}</div><div class="score-label">DNS-MOS</div></div>
<div class="score-card"><div class="score-val">{r.get('full_dur', 0):.1f}s</div><div class="score-label">Duration</div></div>
</div>

<div class="audio-row"><div class="audio-label">Full:</div>{embed_mp3(r.get('enhanced_mp3', r.get('enhanced_path', '').replace('.wav', '.mp3')))}</div>
""")
            if r.get("has_cut_to"):
                html_parts.append(f"""
<div class="audio-row"><div class="audio-label">Part 1:</div>{embed_mp3(r.get('part1_mp3', r.get('part1_path', '').replace('.wav', '.mp3')))}</div>
<div class="audio-row"><div class="audio-label">Part 2:</div>{embed_mp3(r.get('part2_mp3', r.get('part2_path', '').replace('.wav', '.mp3')))}</div>
<div class="meta">Split at {r.get('split_sec', 0):.2f}s | Part 1: {r.get('part1_dur', 0):.1f}s | Part 2: {r.get('part2_dur', 0):.1f}s</div>
""")

            html_parts.append(f"""
<div class="meta"><strong>ASR Transcript:</strong> {escape_html(r.get('asr_text', ''))}</div>

<details>
<summary>Refined Prompt (Full)</summary>
<div class="prompt-box">{escape_html(r.get('refined_prompt_full', 'Not available'))}</div>
</details>
""")
            if r.get("has_cut_to"):
                html_parts.append(f"""
<details>
<summary>Refined Prompt (Part 1)</summary>
<div class="prompt-box">{escape_html(r.get('refined_prompt_part1', 'Not available'))}</div>
</details>
<details>
<summary>Refined Prompt (Part 2)</summary>
<div class="prompt-box">{escape_html(r.get('refined_prompt_part2', 'Not available'))}</div>
</details>
""")

            html_parts.append(f"""
<div class="meta">Enhance: {r.get('enhance_time', 0):.1f}s | ASR: {r.get('asr_time', 0):.1f}s | CLAP: {r.get('clap_time', 0):.1f}s | Total: {r.get('total_time', 0):.1f}s</div>
</div>""")

        html_parts.append("</div>")  # close group

    # Throughput report
    html_parts.append("""
<div class="throughput">
<h2>Throughput Report (8x A100-80GB)</h2>
<table>
<tr><th>Stage</th><th>Total Time</th><th>Items</th><th>Throughput</th><th>Per-Item</th></tr>
""")
    total_items = TOTAL_GROUPS * NUM_CANDIDATES
    for stage, times in TIMINGS.items():
        total_t = sum(times)
        per_item = total_t / total_items if total_items > 0 else 0
        throughput = total_items / total_t if total_t > 0 else 0
        html_parts.append(
            f"<tr><td>{stage}</td><td>{total_t:.1f}s</td>"
            f"<td>{total_items}</td>"
            f"<td>{throughput:.2f}/s ({throughput*3600:.0f}/h)</td>"
            f"<td>{per_item:.2f}s</td></tr>"
        )
    overall = sum(sum(t) for t in TIMINGS.values())
    html_parts.append(
        f"<tr style='font-weight:bold;color:#ff9800'>"
        f"<td>TOTAL</td><td>{overall:.1f}s ({overall/60:.1f}min)</td>"
        f"<td>{total_items}</td>"
        f"<td>{total_items/overall:.2f}/s</td>"
        f"<td>{overall/total_items:.2f}s</td></tr>"
    )
    html_parts.append("</table></div>")

    html_parts.append("</div></body></html>")

    html_path = output_dir / "sample_groups_report.html"
    html_path.write_text("".join(html_parts), encoding="utf-8")
    log.info(f"  HTML report: {html_path}")

    elapsed = time.time() - t0
    record_time("html_report", elapsed)
    return html_path


# ---------------------------------------------------------------------------
# Step 6: Write protocol
# ---------------------------------------------------------------------------
def write_protocol(groups, output_dir, html_path):
    """Write protocol.md with full pipeline documentation."""
    log.info("=== Writing protocol.md ===")

    # Load results for stats
    enhanced_dir = output_dir / "enhanced"
    all_results = {}
    for f in sorted(enhanced_dir.glob("g*_result.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("status") == "ok":
            gid = r["group_id"]
            all_results.setdefault(gid, []).append(r)

    total_items = TOTAL_GROUPS * NUM_CANDIDATES
    total_ok = sum(len(v) for v in all_results.values())

    lines = [
        "# Sample Groups Generation Protocol",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Hardware:** 8x NVIDIA A100-SXM4-80GB",
        "",
        "## Configuration",
        "",
        f"- **Groups:** {TOTAL_GROUPS} ({NUM_ARCHETYPE_GROUPS} CC2-C Archetype + {NUM_EXTREME_GROUPS} Extreme Physical)",
        f"- **Candidates per group:** {NUM_CANDIDATES}",
        f"- **Total candidates:** {total_items}",
        f"- **Successfully processed:** {total_ok}",
        f"- **TTS settings:** CFG={CFG_SCALE}, STG={STG_SCALE}, duration_multiplier=1.1",
        f"- **Enhancement:** Sidon + ChatterboxVC augmentation (best-of-2 by DNS-MOS OVR)",
        f"- **ASR:** Whisper turbo (word-level timestamps)",
        f"- **Scoring:** WER + VoiceCLAP Large (sanitized + negative)",
        f"- **Re-annotation:** Gemma 4 E4B-it (top-2 candidates per group)",
        "",
        "## Reference Audio Files",
        "",
    ]
    for ref in REF_FILES:
        lines.append(f"- `{ref.name}`")

    lines += [
        "",
        "## Pipeline Stages",
        "",
        "### Stage 1: DramaBox TTS Synthesis",
        "- Model: ResembleAI/DramaBox (22B DiT transformer)",
        "- 8 GPU workers, each loading its own TTSServer",
        "- ~24GB VRAM per GPU with 4-bit Gemma quantization",
        "- Output: raw WAV at 24kHz",
        "",
        "### Stage 2: Sidon + ChatterboxVC Enhancement",
        "- **Path A:** Sidon only (w2v-BERT LoRA encoder + DAC decoder, 16kHz -> 48kHz)",
        "- **Path B:** ChatterboxVC (S3Gen flow-matching VC, any -> 24kHz) then Sidon (-> 48kHz)",
        "- DNS-MOS OVR score selects the better path per candidate",
        "- Reference audio from test-refs/ used as VC target (round-robin assignment)",
        "",
        "### Stage 3: Whisper Turbo ASR",
        "- Word-level timestamps for CUT TO: scene splitting",
        "- WER computed against expected dialogue (quoted text from prompt)",
        "",
        "### Stage 4: VoiceCLAP Large Scoring",
        "- Sanitized prompt (stage directions only) similarity",
        "- Negative text (\"robotic, distorted, uncanny\") similarity",
        "- Reward = (1 - WER) * (CLAP_sim - CLAP_neg + 2)",
        "",
        "### Stage 5: CUT TO: Split Detection",
        "- Longest silence gap in middle 20-80% of audio",
        "- 50ms fade at split boundary",
        "- Separate MP3s for full, part1, part2",
        "",
        "### Stage 6: Gemma 4 E4B-it Re-annotation",
        "- Rewrites DramaBox prompts to match actual audio performance",
        "- 3 passes per sample: full, part1, part2",
        "- Only applied to top-2 candidates per group",
        "",
        "## Throughput Report (8x A100-80GB)",
        "",
        "| Stage | Total Time | Items | Throughput | Per-Item |",
        "|-------|-----------|-------|------------|---------|",
    ]

    for stage, times in TIMINGS.items():
        total_t = sum(times)
        per_item = total_t / total_items if total_items > 0 else 0
        throughput = total_items / total_t if total_t > 0 else 0
        lines.append(
            f"| {stage} | {total_t:.1f}s | {total_items} | "
            f"{throughput:.2f}/s ({throughput*3600:.0f}/h) | {per_item:.2f}s |"
        )

    overall = sum(sum(t) for t in TIMINGS.values())
    lines.append(
        f"| **TOTAL** | **{overall:.1f}s ({overall/60:.1f}min)** | "
        f"**{total_items}** | **{total_items/overall:.2f}/s** | **{overall/total_items:.2f}s** |"
    )

    lines += [
        "",
        "## Group Details",
        "",
    ]

    for g in groups:
        gid = g["group_id"]
        candidates = all_results.get(gid, [])
        candidates.sort(key=lambda x: x.get("reward", 0), reverse=True)
        winner = candidates[0] if candidates else {}

        lines.append(f"### Group {gid}: {g['source']} — {g['pathway']}")
        lines.append(f"- **Sample Info:** {json.dumps(g.get('sample_info', {}))}")
        lines.append(f"- **Reference:** {os.path.basename(g.get('ref_audio', 'None'))}")
        lines.append(f"- **Candidates OK:** {len(candidates)}/{NUM_CANDIDATES}")
        if winner:
            lines.append(f"- **Winner:** candidate {winner.get('candidate_id')} "
                         f"(reward={winner.get('reward', 0):.3f}, "
                         f"WER={winner.get('wer', 0):.1%}, "
                         f"DNS-MOS={winner.get('dnsmos_ovr', 0):.2f}, "
                         f"method={winner.get('enhance_method', '')})")
        lines.append("")

    lines += [
        f"## Output",
        f"",
        f"- HTML Report: `{html_path}`",
        f"- Enhanced audio: `{enhanced_dir}/`",
        f"- Raw TTS audio: `{output_dir / 'tts_raw'}/`",
    ]

    protocol_path = Path("/home/deployer/laion/sidon/protocol.md")
    protocol_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Protocol written to {protocol_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mp.set_start_method("spawn", force=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("SAMPLE GROUP GENERATION PIPELINE")
    log.info(f"  {TOTAL_GROUPS} groups x {NUM_CANDIDATES} candidates = {TOTAL_GROUPS * NUM_CANDIDATES} total")
    log.info(f"  GPUs: {NUM_GPUS}x A100-80GB")
    log.info("=" * 60)

    overall_t0 = time.time()

    # Select prompts
    groups = select_prompts()
    groups_path = OUTPUT_DIR / "groups.json"
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    log.info(f"Selected {len(groups)} groups -> {groups_path}")

    # Step 1: TTS synthesis
    tts_dir = run_tts_synthesis(groups, OUTPUT_DIR)

    # Step 2: Enhancement + ASR + Scoring
    enhanced_dir = run_enhancement(groups, tts_dir, OUTPUT_DIR)

    # Step 3: Re-annotation (top-2 only)
    run_reannotation(groups, enhanced_dir)

    # Step 4: HTML report
    html_path = build_html_report(groups, enhanced_dir, OUTPUT_DIR)

    # Step 5: Protocol
    write_protocol(groups, OUTPUT_DIR, html_path)

    overall_elapsed = time.time() - overall_t0
    log.info("=" * 60)
    log.info(f"ALL DONE in {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    log.info(f"HTML: {html_path}")
    log.info(f"Protocol: /home/deployer/laion/sidon/protocol.md")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
