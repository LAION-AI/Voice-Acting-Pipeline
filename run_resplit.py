#!/usr/bin/env python3
"""Re-run ONLY the LLM-guided CUT TO: splitting, re-annotation, and HTML report.

Assumes enhancement + ASR are already done (result JSONs with asr_text/word_timestamps
exist in enhanced/).  Clears and recomputes: split_sec, fade_strategy, part WAVs/MP3s,
refined prompts.
"""

import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("resplit")

OUTPUT_DIR = Path("/home/deployer/laion/Voice-Acting-Pipeline/sample_groups_output")
NUM_GPUS = 8
NUM_CANDIDATES = 25
TOTAL_GROUPS = 20
MP3_BITRATE = "256k"
MP3_SAMPLE_RATE = 48000

# Import LLM splitting helpers from the main script
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Splitting + re-annotation worker (one per GPU)
# ---------------------------------------------------------------------------
def resplit_worker(gpu_id, work_items, enhanced_dir):
    """Re-split CUT TO: audio with LLM guidance, then re-annotate."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    import soundfile as sf
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Import helpers
    from run_sample_groups import (
        find_llm_split_point, find_quiet_spot, get_fade_strategy,
    )

    # Load Gemma 4 E4B-it
    model_id = "google/gemma-4-E4B-it"
    print(f"  [GPU {gpu_id}] Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        do_reannotate = item.get("reannotate", False)
        result_path = os.path.join(enhanced_dir, f"{tag}_result.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)

        if result.get("status") != "ok":
            continue

        prompt = result.get("prompt", "")
        word_ts = result.get("word_timestamps", [])
        asr_text = result.get("asr_text", "")

        # --- Re-split ---
        has_cut_to = bool(re.search(r'\bCUT\s+TO\s*:', prompt))
        enh_wav = result.get("enhanced_path", "")

        if has_cut_to and word_ts and enh_wav and os.path.exists(enh_wav):
            try:
                mono_np, sr_file = sf.read(enh_wav, dtype="float32")
                if mono_np.ndim > 1:
                    mono_np = mono_np.mean(axis=1)
                sr_out = sr_file
                total_duration = len(mono_np) / sr_out

                # Phase 1: LLM split point
                llm_split_sec, split_method = find_llm_split_point(
                    prompt, word_ts, total_duration, tokenizer, model)

                # Phase 2: Quiet spot
                quiet_sec, quiet_found = find_quiet_spot(
                    mono_np, sr_out, llm_split_sec, word_ts)

                split_sec = max(0.5, min(quiet_sec, total_duration - 0.5))

                # Phase 3: Fade strategy
                fade_strategy = get_fade_strategy(
                    prompt, split_sec, quiet_found, word_ts, tokenizer, model)

                result["split_sec"] = round(split_sec, 3)
                result["llm_split_sec"] = round(llm_split_sec, 3)
                result["split_method"] = split_method
                result["quiet_spot_found"] = bool(quiet_found)
                result["fade_strategy"] = fade_strategy

                # Split audio
                split_sample = int(split_sec * sr_out)
                split_sample = max(0, min(split_sample, len(mono_np)))
                part1_np = mono_np[:split_sample].copy()
                part2_np = mono_np[split_sample:].copy()

                # Apply fades
                fo = int(fade_strategy["fade_out_ms"] * sr_out / 1000)
                fi = int(fade_strategy["fade_in_ms"] * sr_out / 1000)
                if fo > 0 and fo < len(part1_np):
                    part1_np[-fo:] *= np.linspace(1, 0, fo)
                if fi > 0 and fi < len(part2_np):
                    part2_np[:fi] *= np.linspace(0, 1, fi)

                # Silence gap
                gap_ms = fade_strategy.get("silence_gap_ms", 0)
                if gap_ms > 0:
                    gap_samp = int(gap_ms * sr_out / 1000)
                    part1_np = np.concatenate([part1_np,
                                               np.zeros(gap_samp // 2, dtype=part1_np.dtype)])
                    part2_np = np.concatenate([np.zeros(gap_samp - gap_samp // 2,
                                                        dtype=part2_np.dtype), part2_np])

                p1_wav = os.path.join(enhanced_dir, f"{tag}_part1.wav")
                p2_wav = os.path.join(enhanced_dir, f"{tag}_part2.wav")
                sf.write(p1_wav, part1_np, sr_out)
                sf.write(p2_wav, part2_np, sr_out)
                result["part1_path"] = p1_wav
                result["part2_path"] = p2_wav
                result["part1_dur"] = round(len(part1_np) / sr_out, 3)
                result["part2_dur"] = round(len(part2_np) / sr_out, 3)

                # MP3 conversion
                for suffix, wav_p in [("part1", p1_wav), ("part2", p2_wav)]:
                    mp3_p = wav_p.replace(".wav", ".mp3")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", wav_p, "-ac", "1", "-ar",
                         str(MP3_SAMPLE_RATE), "-b:a", MP3_BITRATE, "-f", "mp3", mp3_p],
                        capture_output=True, check=False,
                    )
                    result[f"{suffix}_mp3"] = mp3_p

                print(f"  [GPU {gpu_id}] {tag} re-split at {split_sec:.2f}s ({split_method})",
                      flush=True)

            except Exception as e:
                print(f"  [GPU {gpu_id}] {tag} re-split FAILED: {e}", flush=True)
                traceback.print_exc()

        # --- Re-annotate (only for top candidates) ---
        if do_reannotate and asr_text and prompt:
            try:
                # Clear old refined prompts
                for key in ["refined_prompt_full", "refined_prompt_part1",
                            "refined_prompt_part2"]:
                    result.pop(key, None)

                # Full re-annotation
                user_msg = (
                    f"Original DramaBox prompt:\n{prompt}\n\n"
                    f"ASR transcript of generated audio:\n{asr_text}\n\n"
                    f"Rewrite the DramaBox prompt to match the actual performance."
                )
                messages = [{"role": "user", "content": system_prompt + "\n\n" + user_msg}]
                encoded = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )
                input_ids = encoded["input_ids"].to("cuda")
                with torch.no_grad():
                    outputs = model.generate(input_ids=input_ids, max_new_tokens=1024,
                                             temperature=0.7, top_p=0.9, do_sample=True)
                new_tokens = outputs[0][input_ids.shape[-1]:]
                refined = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                result["refined_prompt_full"] = refined

                # Part-level re-annotation
                if has_cut_to and result.get("split_sec"):
                    split_sec = result["split_sec"]
                    p1_words = " ".join(w["word"] for w in word_ts
                                        if w["end"] <= split_sec + 0.1)
                    p2_words = " ".join(w["word"] for w in word_ts
                                        if w["start"] >= split_sec - 0.1)

                    for part_name, part_transcript in [("part1", p1_words),
                                                       ("part2", p2_words)]:
                        if not part_transcript.strip():
                            result[f"refined_prompt_{part_name}"] = ""
                            continue
                        user_msg_p = (
                            f"Original full DramaBox prompt:\n{prompt}\n\n"
                            f"ASR transcript of {part_name} audio only:\n{part_transcript}\n\n"
                            f"Write a standalone single-scene DramaBox prompt for JUST this part."
                        )
                        msgs_p = [{"role": "user",
                                   "content": system_prompt + "\n\n" + user_msg_p}]
                        enc_p = tokenizer.apply_chat_template(
                            msgs_p, return_tensors="pt", add_generation_prompt=True
                        )
                        ids_p = enc_p["input_ids"].to("cuda")
                        with torch.no_grad():
                            out_p = model.generate(input_ids=ids_p, max_new_tokens=768,
                                                   temperature=0.7, top_p=0.9,
                                                   do_sample=True)
                        new_p = out_p[0][ids_p.shape[-1]:]
                        result[f"refined_prompt_{part_name}"] = tokenizer.decode(
                            new_p, skip_special_tokens=True
                        ).strip()

                print(f"  [GPU {gpu_id}] {tag} re-annotated", flush=True)

            except Exception as e:
                print(f"  [GPU {gpu_id}] {tag} re-annotation FAILED: {e}", flush=True)
                traceback.print_exc()

        # Save
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        torch.cuda.empty_cache()


def main():
    mp.set_start_method("spawn", force=True)

    enhanced_dir = OUTPUT_DIR / "enhanced"
    groups_path = OUTPUT_DIR / "groups.json"
    with open(groups_path, encoding="utf-8") as f:
        groups = json.load(f)

    # Load all results, find top-2 per group for re-annotation
    all_results = {}
    for fp in sorted(enhanced_dir.glob("g*_result.json")):
        with open(fp, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("status") == "ok":
            gid = r["group_id"]
            all_results.setdefault(gid, []).append(r)

    top_tags = set()
    for gid in sorted(all_results.keys()):
        candidates = all_results[gid]
        candidates.sort(key=lambda x: x.get("reward", 0), reverse=True)
        for r in candidates[:2]:
            top_tags.add(r["tag"])

    # Build work items: ALL candidates get re-split, top-2 also get re-annotated
    work_items = []
    for gid in sorted(all_results.keys()):
        for r in all_results[gid]:
            work_items.append({
                "tag": r["tag"],
                "reannotate": r["tag"] in top_tags,
            })

    log.info(f"Re-splitting {len(work_items)} candidates "
             f"({len(top_tags)} will also be re-annotated)")

    # Distribute across GPUs
    shards = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work_items):
        shards[i % NUM_GPUS].append(item)

    t0 = time.time()
    processes = []
    for gpu_id in range(NUM_GPUS):
        if not shards[gpu_id]:
            continue
        p = mp.Process(target=resplit_worker,
                       args=(gpu_id, shards[gpu_id], str(enhanced_dir)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t0
    log.info(f"Re-split + re-annotate done in {elapsed:.1f}s")

    # Stats
    n_split = 0
    n_llm = 0
    n_refined = 0
    for fp in sorted(enhanced_dir.glob("g*_result.json")):
        with open(fp, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("split_sec") is not None:
            n_split += 1
        if r.get("split_method", "").startswith("llm_guided"):
            n_llm += 1
        if r.get("refined_prompt_full"):
            n_refined += 1
    log.info(f"  Splits: {n_split} total, {n_llm} LLM-guided")
    log.info(f"  Refined prompts: {n_refined}")

    # Rebuild HTML report
    from run_reannotate import build_html_report
    html_path = build_html_report(groups, enhanced_dir, OUTPUT_DIR)
    log.info(f"Report rebuilt: {html_path}")


if __name__ == "__main__":
    main()
