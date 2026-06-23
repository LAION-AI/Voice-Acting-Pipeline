#!/usr/bin/env python3
"""Re-run ONLY the re-annotation step (Gemma 4 E4B-it prompt rewriting) for top-2
candidates per group, then rebuild the HTML report.

Fixes the tokenizer incompatibility with transformers 4.x by patching
extra_special_tokens from list to dict before loading.
"""

import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reannotate")

OUTPUT_DIR = Path("/home/deployer/laion/Voice-Acting-Pipeline/sample_groups_output")
NUM_GPUS = 8
NUM_CANDIDATES = 25
TOTAL_GROUPS = 20
MP3_SAMPLE_RATE = 48000


def reannotate_worker(gpu_id, work_items, enhanced_dir):
    """Run Gemma 4 E4B-it prompt re-annotation on assigned items."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download

    # Gemma 4 E4B-it requires transformers>=5.0 (gemma4 architecture).
    # Fall back to Gemma 2 9B-it which works with transformers 4.x.
    model_id = "google/gemma-2-9b-it"
    print(f"  [GPU {gpu_id}] Loading {model_id}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"  [GPU {gpu_id}] Gemma loaded ({model_id})", flush=True)

    system_prompt = """You rewrite DramaBox prompts to match what was actually performed in the audio.
Given the original DramaBox prompt and the ASR transcript of the generated audio,
rewrite the prompt so it accurately describes what was ACTUALLY said and how it was performed.

Rules:
1. Keep the speaker description paragraph (age, gender, timbre, recording quality).
2. Use the ASR transcript as the new dialogue in double quotes.
3. Adjust stage directions to match the actual delivery implied by the transcript.
4. Keep the same format: speaker description, then alternating stage directions + dialogue.
5. Output ONLY the rewritten DramaBox prompt, no explanation."""

    import re
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
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()


def build_html_report(groups, enhanced_dir, output_dir):
    """Rebuild HTML report with embedded MP3s showing top-2 candidates per group."""
    import base64
    import re

    log.info("=== Building HTML Report ===")

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

    CFG_SCALE = 2.5
    STG_SCALE = 1.5

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
<h3>#{rank} — Candidate {r['candidate_id']} <span class="tag">{r.get('enhance_method','')}</span>
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

        html_parts.append("</div>")

    html_parts.append("</div></body></html>")

    html_path = output_dir / "sample_groups_report.html"
    html_path.write_text("".join(html_parts), encoding="utf-8")
    log.info(f"  HTML report: {html_path}")
    return html_path


def main():
    mp.set_start_method("spawn", force=True)

    enhanced_dir = OUTPUT_DIR / "enhanced"
    groups_path = OUTPUT_DIR / "groups.json"
    with open(groups_path, encoding="utf-8") as f:
        groups = json.load(f)

    # Find top-2 per group
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

    log.info(f"Re-annotating {len(top_items)} top candidates (top-2 x {len(all_results)} groups)")

    # Distribute across GPUs
    n_gpus = min(NUM_GPUS, max(1, len(top_items) // 2))
    shards = [[] for _ in range(n_gpus)]
    for i, item in enumerate(top_items):
        shards[i % n_gpus].append(item)

    t0 = time.time()
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
    log.info(f"Re-annotation done in {elapsed:.1f}s")

    # Check results
    n_refined = 0
    for f in sorted(enhanced_dir.glob("g*_result.json")):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if r.get("refined_prompt_full"):
            n_refined += 1
    log.info(f"Results with refined prompts: {n_refined}")

    # Rebuild HTML
    html_path = build_html_report(groups, enhanced_dir, OUTPUT_DIR)
    log.info(f"Report rebuilt: {html_path}")


if __name__ == "__main__":
    main()
