#!/usr/bin/env python3
"""ACCC LavaSR Pipeline: RE-USE + LavaSR BWE, 6-method VoiceCLAP ranking.

50 groups × 25 candidates = 1,250 clips
Pipeline: Concatenate reuse parts → LavaSR BWE (no denoising) → Score
          (WER + CLAP Small + CLAP Large) → Merge → 10-page HTML grid
          with 6 ranking methods.
"""

import base64
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/deployer/laion/dramabox-pipeline")
sys.path.insert(0, "/tmp/dramabox_demo")

OUTDIR = Path("/tmp/dramabox_demo")
AUDIO_DIR = OUTDIR / "audio_accc"
LAVASR_DIR = OUTDIR / "audio_accc_lavasr"
HTML_OUTPUT = Path("/home/deployer/laion/voice-acting-pipeline/docs/demo/accc_lavasr.html")

GROUPS_FILE = OUTDIR / "accc_groups.json"
SCORES_FILE = OUTDIR / "lavasr_scores.json"
CLAP_SMALL_FILE = OUTDIR / "lavasr_clap_small.json"
CLAP_LARGE_FILE = OUTDIR / "lavasr_clap_large.json"
EMONET_FILE = OUTDIR / "lavasr_emonet.json"
MERGED_FILE = OUTDIR / "lavasr_merged.json"

N_CANDIDATES = 25
GPUS = list(range(8))
MP3_BITRATE = "48k"
GROUPS_PER_PAGE = 5

NEG_SAN_TEXT = "robotic, distorted, uncanny, distorted, distortion"

EMONET_EMOTIONS = [
    "Affection", "Amusement", "Anger", "Astonishment_Surprise", "Awe",
    "Bitterness", "Concentration", "Confusion", "Contemplation", "Contempt",
    "Contentment", "Disappointment", "Disgust", "Distress", "Doubt",
    "Elation", "Embarrassment", "Emotional_Numbness", "Fatigue_Exhaustion",
    "Fear", "Helplessness", "Hope_Enthusiasm_Optimism",
    "Impatience_and_Irritability", "Infatuation", "Interest",
    "Intoxication_Altered_States_of_Consciousness", "Jealousy_&_Envy",
    "Longing", "Malevolence_Malice", "Pain", "Pleasure_Ecstasy", "Pride",
    "Relief", "Sadness", "Sexual_Lust", "Shame", "Sourness", "Teasing",
    "Thankfulness_Gratitude", "Triumph",
]

RANKING_METHODS = [
    ("v_snr_L", "v_snr_L (DEFAULT): (1−WER) × (san_L − neg_san_L + 2)"),
    ("v_snr_S", "v_snr_S: (1−WER) × (san_S − neg_san_S + 2)"),
    ("v_san_L", "v_san_L: (1−WER) × (san_L + 1)"),
    ("v_san_S", "v_san_S: (1−WER) × (san_S + 1)"),
    ("content_enjoyment_rank", "Content Enjoyment"),
    ("standard", "Standard: (1−WER) × Content Enjoyment"),
] + [
    (f"emo_{e}", f"EmoNet: {e.replace('_', ' ')} ↓ (WER<10% only)")
    for e in EMONET_EMOTIONS
]


def sanitize_prompt(prompt):
    """Remove all double-quoted dialogue, keeping only stage directions."""
    return re.sub(r'"[^"]*"', '', prompt).strip()


def _run_subprocess_workers(worker_script, gpu_batches, extra_args=None,
                            cwd="/home/deployer/laion/dramabox-pipeline",
                            prefix="lavasr", stagger=2):
    """Launch parallel GPU worker subprocesses and wait for completion."""
    processes = []
    for gpu_id, items in gpu_batches.items():
        if not items:
            continue
        work_file = str(OUTDIR / f"_{prefix}_gpu{gpu_id}.json")
        with open(work_file, "w") as f:
            json.dump(items, f)
        cmd = [sys.executable, str(worker_script), str(gpu_id), work_file]
        if extra_args:
            cmd.extend(extra_args)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=cwd,
        )
        processes.append((gpu_id, proc, work_file))
        time.sleep(stagger)

    def _read(gpu_id, proc):
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
        return proc.wait()

    with ThreadPoolExecutor(max_workers=len(processes)) as pool:
        futures = {pool.submit(_read, g, p): g for g, p, _ in processes}
        for f in as_completed(futures):
            gpu_id = futures[f]
            print(f"  GPU {gpu_id} done (exit {f.result()})", flush=True)

    return processes


def _collect_results(processes) -> list[dict]:
    """Collect JSON results from completed worker subprocesses."""
    all_results = []
    for gpu_id, _, work_file in processes:
        results_file = work_file.replace(".json", "_results.json")
        if Path(results_file).exists():
            with open(results_file) as f:
                all_results.extend(json.load(f))
    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Concatenate RE-USE partA + partB
# ═══════════════════════════════════════════════════════════════════════

def phase1_concat(groups):
    print(f"\n{'='*72}")
    print(f"  PHASE 1: CONCATENATE RE-USE PARTS")
    print(f"{'='*72}\n", flush=True)
    t0 = time.time()

    LAVASR_DIR.mkdir(parents=True, exist_ok=True)

    work = []
    for g in groups:
        gidx = g["gidx"]
        for cand in range(N_CANDIDATES):
            out = LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_reuse_full.wav"
            if out.exists() and out.stat().st_size > 1000:
                continue
            partA = AUDIO_DIR / f"accc_{gidx:03d}_n{cand:02d}_reuse_partA.wav"
            partB = AUDIO_DIR / f"accc_{gidx:03d}_n{cand:02d}_reuse_partB.wav"
            work.append((str(partA), str(partB), str(out)))

    if not work:
        print("  All concatenations already exist, skipping.", flush=True)
        return

    print(f"  {len(work)} files to concatenate...", flush=True)

    def do_concat(args):
        partA, partB, out_path = args
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", partA, "-i", partB,
                 "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
                 "-map", "[out]", out_path],
                capture_output=True, timeout=30,
            )
            return Path(out_path).exists()
        except Exception:
            return False

    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(do_concat, w): w for w in work}
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(work)} concatenated", flush=True)

    print(f"  Concatenation done in {time.time()-t0:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: LavaSR BWE
# ═══════════════════════════════════════════════════════════════════════

def phase2_lavasr(groups):
    print(f"\n{'='*72}")
    print(f"  PHASE 2: LAVASR BWE (enhance=True, denoise=False)")
    print(f"{'='*72}\n", flush=True)
    t0 = time.time()

    work = []
    for g in groups:
        gidx = g["gidx"]
        for cand in range(N_CANDIDATES):
            out = LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_lavasr.wav"
            if out.exists() and out.stat().st_size > 1000:
                continue
            inp = LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_reuse_full.wav"
            work.append({"input_path": str(inp), "output_path": str(out)})

    if not work:
        print("  All LavaSR files already exist, skipping.", flush=True)
        return

    print(f"  {len(work)} files to process through LavaSR...", flush=True)

    gpu_batches = {g: [] for g in GPUS}
    for i, item in enumerate(work):
        gpu_batches[i % len(GPUS)].append(item)

    _run_subprocess_workers(
        OUTDIR / "_lavasr_worker.py", gpu_batches,
        cwd=str(OUTDIR), prefix="lavasr_bwe", stagger=3,
    )

    print(f"  LavaSR done in {time.time()-t0:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3: Scoring (WER + CLAP Small + CLAP Large)
# ═══════════════════════════════════════════════════════════════════════

def phase3_scoring(groups):
    print(f"\n{'='*72}")
    print(f"  PHASE 3: SCORING")
    print(f"{'='*72}\n", flush=True)

    # ── 3a: WER + content_enjoyment ──────────────────────────────────
    cached_scores = {}
    if SCORES_FILE.exists():
        with open(SCORES_FILE) as f:
            cached_scores = json.load(f)

    score_work = []
    for g in groups:
        gidx = g["gidx"]
        prompt = g["dramabox_prompt"]
        for cand in range(N_CANDIDATES):
            key = f"{gidx}_{cand}"
            if key in cached_scores:
                continue
            audio_path = str(LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_lavasr.wav")
            score_work.append({
                "group_key": "LAVASR",
                "item_idx": gidx,
                "n": cand,
                "audio_path": audio_path,
                "dramabox_prompt": prompt,
            })

    if score_work:
        print(f"\n  3a: WER + content_enjoyment ({len(score_work)} items)...", flush=True)
        t0 = time.time()
        gpu_batches = {g: [] for g in GPUS}
        for i, item in enumerate(score_work):
            gpu_batches[i % len(GPUS)].append(item)
        procs = _run_subprocess_workers(
            OUTDIR / "_score_worker.py", gpu_batches,
            cwd="/home/deployer/laion/voice-acting-pipeline",
            prefix="lavasr_score", stagger=3,
        )
        results = _collect_results(procs)
        for r in results:
            key = f"{r['item_idx']}_{r['n']}"
            cached_scores[key] = r
        with open(SCORES_FILE, "w") as f:
            json.dump(cached_scores, f, indent=2)
        print(f"  3a done in {time.time()-t0:.1f}s ({len(results)} scored)", flush=True)
    else:
        print("  3a: WER scores cached, skipping.", flush=True)

    # ── 3b: VoiceCLAP Small ─────────────────────────────────────────
    cached_clap_small = {}
    if CLAP_SMALL_FILE.exists():
        with open(CLAP_SMALL_FILE) as f:
            cached_clap_small = json.load(f)

    clap_work = []
    for g in groups:
        gidx = g["gidx"]
        san_text = sanitize_prompt(g["dramabox_prompt"])
        for cand in range(N_CANDIDATES):
            key = f"{gidx}_{cand}"
            if key in cached_clap_small:
                continue
            audio_path = str(LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_lavasr.wav")
            clap_work.append({
                "group_key": "LAVASR",
                "item_idx": gidx,
                "n": cand,
                "audio_path": audio_path,
                "sanitized_text": san_text,
            })

    if clap_work:
        print(f"\n  3b: VoiceCLAP Small ({len(clap_work)} items)...", flush=True)
        t0 = time.time()
        gpu_batches = {g: [] for g in GPUS}
        for i, item in enumerate(clap_work):
            gpu_batches[i % len(GPUS)].append(item)
        procs = _run_subprocess_workers(
            OUTDIR / "_lavasr_clap_worker.py", gpu_batches,
            extra_args=["small"],
            cwd=str(OUTDIR), prefix="lavasr_clap_s", stagger=3,
        )
        results = _collect_results(procs)
        for r in results:
            key = f"{r['item_idx']}_{r['n']}"
            cached_clap_small[key] = r
        with open(CLAP_SMALL_FILE, "w") as f:
            json.dump(cached_clap_small, f, indent=2)
        print(f"  3b done in {time.time()-t0:.1f}s ({len(results)} scored)", flush=True)
    else:
        print("  3b: CLAP Small scores cached, skipping.", flush=True)

    # ── 3c: VoiceCLAP Large ─────────────────────────────────────────
    cached_clap_large = {}
    if CLAP_LARGE_FILE.exists():
        with open(CLAP_LARGE_FILE) as f:
            cached_clap_large = json.load(f)

    clap_work = []
    for g in groups:
        gidx = g["gidx"]
        san_text = sanitize_prompt(g["dramabox_prompt"])
        for cand in range(N_CANDIDATES):
            key = f"{gidx}_{cand}"
            if key in cached_clap_large:
                continue
            audio_path = str(LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_lavasr.wav")
            clap_work.append({
                "group_key": "LAVASR",
                "item_idx": gidx,
                "n": cand,
                "audio_path": audio_path,
                "sanitized_text": san_text,
            })

    if clap_work:
        print(f"\n  3c: VoiceCLAP Large ({len(clap_work)} items)...", flush=True)
        t0 = time.time()
        gpu_batches = {g: [] for g in GPUS}
        for i, item in enumerate(clap_work):
            gpu_batches[i % len(GPUS)].append(item)
        procs = _run_subprocess_workers(
            OUTDIR / "_lavasr_clap_worker.py", gpu_batches,
            extra_args=["large"],
            cwd=str(OUTDIR), prefix="lavasr_clap_l", stagger=3,
        )
        results = _collect_results(procs)
        for r in results:
            key = f"{r['item_idx']}_{r['n']}"
            cached_clap_large[key] = r
        with open(CLAP_LARGE_FILE, "w") as f:
            json.dump(cached_clap_large, f, indent=2)
        print(f"  3c done in {time.time()-t0:.1f}s ({len(results)} scored)", flush=True)
    else:
        print("  3c: CLAP Large scores cached, skipping.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3d: EmoNet Emotion Scoring (40 emotions)
# ═══════════════════════════════════════════════════════════════════════

def phase3d_emonet(groups):
    print(f"\n{'='*72}")
    print(f"  PHASE 3d: EMONET EMOTION SCORING (40 emotions)")
    print(f"{'='*72}\n", flush=True)

    cached_emonet = {}
    if EMONET_FILE.exists():
        with open(EMONET_FILE) as f:
            cached_emonet = json.load(f)

    emo_work = []
    for g in groups:
        gidx = g["gidx"]
        for cand in range(N_CANDIDATES):
            key = f"{gidx}_{cand}"
            if key in cached_emonet:
                continue
            audio_path = str(LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_lavasr.wav")
            emo_work.append({
                "group_key": "LAVASR",
                "item_idx": gidx,
                "n": cand,
                "audio_path": audio_path,
            })

    if emo_work:
        print(f"\n  3d: EmoNet ({len(emo_work)} items)...", flush=True)
        t0 = time.time()
        gpu_batches = {g: [] for g in GPUS}
        for i, item in enumerate(emo_work):
            gpu_batches[i % len(GPUS)].append(item)
        procs = _run_subprocess_workers(
            OUTDIR / "_emonet_worker.py", gpu_batches,
            cwd=str(OUTDIR), prefix="lavasr_emonet", stagger=3,
        )
        results = _collect_results(procs)
        for r in results:
            key = f"{r['item_idx']}_{r['n']}"
            cached_emonet[key] = r
        with open(EMONET_FILE, "w") as f:
            json.dump(cached_emonet, f, indent=2)
        print(f"  3d done in {time.time()-t0:.1f}s ({len(results)} scored)", flush=True)
    else:
        print("  3d: EmoNet scores cached, skipping.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 4: Merge Scores → Ranking Methods
# ═══════════════════════════════════════════════════════════════════════

def phase4_merge(groups):
    print(f"\n{'='*72}")
    print(f"  PHASE 4: MERGE SCORES")
    print(f"{'='*72}\n", flush=True)

    with open(SCORES_FILE) as f:
        cached_scores = json.load(f)
    with open(CLAP_SMALL_FILE) as f:
        cached_clap_small = json.load(f)
    with open(CLAP_LARGE_FILE) as f:
        cached_clap_large = json.load(f)

    cached_emonet = {}
    if EMONET_FILE.exists():
        with open(EMONET_FILE) as f:
            cached_emonet = json.load(f)

    merged = {}
    for g in groups:
        gidx = g["gidx"]
        for cand in range(N_CANDIDATES):
            key = f"{gidx}_{cand}"
            sc = cached_scores.get(key, {})
            cl_s = cached_clap_small.get(key, {})
            cl_l = cached_clap_large.get(key, {})
            emo = cached_emonet.get(key, {}).get("emotions", {})

            wer = sc.get("wer", 1.0)
            enjoy = sc.get("content_enjoyment", 0)
            inv_wer = 1.0 - min(wer, 1.0)

            san_S = cl_s.get("clap_sanitized", 0.0)
            neg_san_S = cl_s.get("neg_san", 0.0)
            san_L = cl_l.get("clap_sanitized", 0.0)
            neg_san_L = cl_l.get("neg_san", 0.0)

            entry = {
                "gidx": gidx,
                "cand": cand,
                "wer": wer,
                "content_enjoyment": enjoy,
                "v_snr_L": inv_wer * (san_L - neg_san_L + 2),
                "v_snr_S": inv_wer * (san_S - neg_san_S + 2),
                "v_san_L": inv_wer * (san_L + 1),
                "v_san_S": inv_wer * (san_S + 1),
                "content_enjoyment_rank": enjoy,
                "standard": inv_wer * enjoy,
                "san_S": san_S,
                "neg_san_S": neg_san_S,
                "san_L": san_L,
                "neg_san_L": neg_san_L,
                "transcription": sc.get("transcription", ""),
                "status": sc.get("status", "ok"),
            }
            # Add 40 emotion scores (raw values, no WER factor)
            for e in EMONET_EMOTIONS:
                entry[f"emo_{e}"] = emo.get(e, 0.0)

            merged[key] = entry

    with open(MERGED_FILE, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"  Merged {len(merged)} entries → {MERGED_FILE}", flush=True)

    return merged


# ═══════════════════════════════════════════════════════════════════════
#  Phase 5: HTML Grid (10 pages × 5 groups)
# ═══════════════════════════════════════════════════════════════════════

def wav_to_mp3_b64(wav_path):
    if not Path(wav_path).exists():
        return ""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-ac", "1", "-ar", "24000",
             "-b:a", MP3_BITRATE, "-f", "mp3", "pipe:1"],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and len(result.stdout) > 100:
            return base64.b64encode(result.stdout).decode()
    except Exception:
        pass
    return ""


def phase5_html(groups, merged):
    print(f"\n{'='*72}")
    print(f"  PHASE 5: HTML GRID GENERATION")
    print(f"{'='*72}\n", flush=True)
    t0 = time.time()

    n_groups = len(groups)
    n_pages = math.ceil(n_groups / GROUPS_PER_PAGE)

    # Pre-convert all LavaSR audio to MP3 base64
    print("  Converting audio to MP3...", flush=True)
    audio_cache = {}
    count = 0
    for g in groups:
        gidx = g["gidx"]
        for cand in range(N_CANDIDATES):
            wav = LAVASR_DIR / f"accc_{gidx:03d}_n{cand:02d}_lavasr.wav"
            b64 = wav_to_mp3_b64(str(wav))
            if b64:
                audio_cache[(gidx, cand)] = b64
                count += 1
        if (gidx + 1) % 10 == 0:
            print(f"    {gidx + 1}/{n_groups} groups", flush=True)
    print(f"    {count} audio clips converted", flush=True)

    # Method options for dropdown
    method_options = "\n".join(
        f'<option value="{m[0]}"{" selected" if i == 0 else ""}>{m[1]}</option>'
        for i, m in enumerate(RANKING_METHODS)
    )

    # Escape helper
    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root { --bg: #0e1117; --card: #1a1d23; --border: #2a2d35; --text: #e0e0e0;
            --dim: #888; --accent: #6c9bff; --green: #4caf50; --red: #f44;
            --orange: #ff9800; --purple: #b388ff; }
    body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
           padding: 20px; max-width: 1200px; margin: 0 auto; }
    h1 { font-size: 1.5em; margin-bottom: 4px; }
    h1 span { color: var(--accent); }
    .subtitle { color: var(--dim); margin-bottom: 16px; font-size: 0.9em; }
    .methodology { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
                   padding: 16px; margin-bottom: 20px; font-size: 0.85em; line-height: 1.6; }
    .methodology h3 { color: var(--accent); margin-bottom: 8px; cursor: pointer; }
    .methodology h3:hover { text-decoration: underline; }
    .methodology h4 { color: var(--orange); margin: 10px 0 4px; font-size: 0.95em; }
    .methodology code { background: var(--bg); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
    .methodology pre { background: var(--bg); padding: 10px; border-radius: 6px; margin: 6px 0;
                       overflow-x: auto; font-size: 0.85em; white-space: pre-wrap; }
    .methodology .formula { color: var(--green); font-weight: 600; }
    .methodology .neg { color: var(--red); }
    .methodology .pos { color: var(--green); }
    .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
                margin-bottom: 20px; padding: 12px; background: var(--card);
                border-radius: 8px; border: 1px solid var(--border); }
    select { background: var(--bg); color: var(--text); border: 1px solid var(--border);
             padding: 6px 10px; border-radius: 4px; font-size: 0.9em; }
    .nav { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .nav a { color: var(--accent); text-decoration: none; padding: 4px 10px;
             background: var(--card); border-radius: 4px; border: 1px solid var(--border);
             font-size: 0.85em; }
    .page-nav { display: flex; gap: 6px; justify-content: center; margin: 16px 0; flex-wrap: wrap; }
    .page-nav a { color: var(--accent); text-decoration: none; padding: 4px 12px;
                  background: var(--card); border-radius: 4px; border: 1px solid var(--border); }
    .page-nav a.current { background: var(--accent); color: var(--bg); font-weight: bold; }
    .group-section { margin-bottom: 30px; }
    .group-header { background: var(--card); padding: 16px; border-radius: 8px 8px 0 0;
                    border: 1px solid var(--border); border-bottom: none; }
    .group-header h2 { font-size: 1.05em; color: var(--accent); margin-bottom: 6px; }
    .group-meta { font-size: 0.82em; color: var(--dim); margin-bottom: 6px; line-height: 1.5; }
    .group-meta strong { color: var(--text); }
    .prompt-block { margin-top: 8px; }
    .prompt-toggle { font-size: 0.82em; color: var(--accent); cursor: pointer; margin-bottom: 4px; }
    .prompt-toggle:hover { text-decoration: underline; }
    .prompt-content { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                     padding: 10px; font-size: 0.8em; white-space: pre-wrap; max-height: 300px;
                     overflow-y: auto; line-height: 1.5; }
    .prompt-content.collapsed { display: none; }
    .clap-texts { margin-top: 8px; }
    .clap-label { font-size: 0.78em; font-weight: 600; margin-bottom: 2px; }
    .clap-label.pos { color: var(--green); }
    .clap-label.neg { color: var(--red); }
    .clap-text-box { background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
                     padding: 6px 10px; font-size: 0.78em; font-family: monospace; margin-bottom: 6px;
                     white-space: pre-wrap; }
    .gemma-params { font-size: 0.78em; color: var(--dim); margin-top: 6px; padding: 6px;
                    background: var(--bg); border-radius: 4px; }
    .candidates { border: 1px solid var(--border); border-radius: 0 0 8px 8px; }
    .candidate { padding: 8px 12px; border-bottom: 1px solid var(--border); display: none; }
    .candidate.visible { display: flex; gap: 10px; align-items: center; }
    .cand-rank { font-weight: bold; font-size: 1.2em; min-width: 36px; text-align: center; }
    .rank-1 { color: #ffd700; }
    .rank-2 { color: #c0c0c0; }
    .rank-3 { color: #cd7f32; }
    .cand-info { flex: 1; min-width: 0; }
    .cand-meta { display: flex; gap: 10px; align-items: center; margin-bottom: 3px; flex-wrap: wrap; }
    .score-val { font-size: 0.85em; color: var(--green); font-weight: 600; }
    .wer-val { font-size: 0.75em; color: var(--dim); }
    .clap-detail { font-size: 0.72em; color: var(--dim); }
    audio { width: 100%; height: 30px; }
    .show-more { text-align: center; padding: 8px; cursor: pointer; color: var(--accent);
                 background: var(--card); border-radius: 0 0 8px 8px; font-size: 0.85em; }
    """

    js_template = """
const PAGE_DATA = %s;

function rankAll(methodKey) {
  const isEmotion = methodKey.startsWith('emo_');

  document.querySelectorAll('.group-section').forEach(section => {
    const gidx = parseInt(section.dataset.gidx);
    const data = PAGE_DATA[gidx];
    if (!data) return;

    let candidates = data.candidates.slice();

    if (isEmotion) {
      let qualifying = candidates.filter(c => c.wer < 0.1);
      let rejected = candidates.filter(c => c.wer >= 0.1);
      qualifying.sort((a, b) => (b[methodKey] || -999) - (a[methodKey] || -999));
      candidates = qualifying.concat(rejected);
    } else {
      candidates.sort((a, b) => (b[methodKey] || -999) - (a[methodKey] || -999));
    }

    const container = section.querySelector('.candidates');
    const cards = Array.from(container.querySelectorAll('.candidate'));

    let qualCount = 0;
    candidates.forEach((c, rank) => {
      const card = cards[rank];
      if (!card) return;
      const isRejected = isEmotion && c.wer >= 0.1;

      const re = card.querySelector('.cand-rank');
      if (isRejected) {
        re.textContent = '—';
        re.className = 'cand-rank';
      } else {
        qualCount++;
        re.textContent = '#' + qualCount;
        re.className = 'cand-rank' + (qualCount <= 3 ? ' rank-' + qualCount : '');
      }

      card.querySelector('.score-val').textContent = (c[methodKey] || 0).toFixed(4);
      card.querySelector('.wer-val').textContent = 'WER: ' + (c.wer * 100).toFixed(1) + '%%';
      card.querySelector('.n-val').textContent = 'n=' + c.cand;

      if (isEmotion) {
        const eName = methodKey.replace('emo_', '').replace(/_/g, ' ');
        card.querySelector('.clap-detail').textContent =
          eName + '=' + (c[methodKey] || 0).toFixed(3) +
          ' | WER=' + (c.wer * 100).toFixed(1) + '%%';
      } else {
        card.querySelector('.clap-detail').textContent =
          'san_L=' + c.san_L.toFixed(3) + ' neg_L=' + c.neg_san_L.toFixed(3) +
          ' | san_S=' + c.san_S.toFixed(3) + ' neg_S=' + c.neg_san_S.toFixed(3);
      }

      const audio = card.querySelector('audio');
      audio.src = c.b64 ? 'data:audio/mpeg;base64,' + c.b64 : '';

      if (isRejected) {
        card.classList.remove('visible');
        card.dataset.rejected = 'true';
      } else {
        card.classList.toggle('visible', qualCount <= 10);
        card.dataset.rejected = 'false';
      }
      card.dataset.sortRank = rank;
    });

    const btn = section.querySelector('.show-more');
    if (btn) {
      const totalQual = isEmotion ? candidates.filter(c => c.wer < 0.1).length : candidates.length;
      btn.textContent = 'Show all ' + totalQual + '...';
      btn.style.display = totalQual > 10 ? 'block' : 'none';
      btn.dataset.expanded = 'false';
    }
  });
}

function toggleMore(btn) {
  const section = btn.closest('.group-section');
  const cards = section.querySelectorAll('.candidate');
  const expanded = btn.dataset.expanded === 'true';
  cards.forEach(card => {
    if (card.dataset.rejected === 'true') return;
    const rank = parseInt(card.dataset.sortRank || 0);
    if (!card.classList.contains('visible') || expanded) {
      card.classList.toggle('visible', !expanded || parseInt(card.style.order || rank) < 10);
    }
  });
  if (!expanded) {
    cards.forEach(card => {
      if (card.dataset.rejected === 'true') return;
      card.classList.add('visible');
    });
    btn.dataset.expanded = 'true';
    btn.textContent = 'Show top 10';
  } else {
    let shown = 0;
    cards.forEach(card => {
      if (card.dataset.rejected === 'true') return;
      shown++;
      card.classList.toggle('visible', shown <= 10);
    });
    btn.dataset.expanded = 'false';
    const totalQual = Array.from(cards).filter(c => c.dataset.rejected !== 'true').length;
    btn.textContent = 'Show all ' + totalQual + '...';
  }
}

function togglePrompt(el) {
  const content = el.nextElementSibling;
  if (content) content.classList.toggle('collapsed');
  el.textContent = content.classList.contains('collapsed')
    ? el.textContent.replace('Hide', 'Show')
    : el.textContent.replace('Show', 'Hide');
}

document.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('method-select');
  sel.addEventListener('change', () => rankAll(sel.value));
  rankAll(sel.value);
});
"""

    nav = """<div class="nav">
  <a href="accc_pooled.html">ACCC Pooled Grid</a>
  <a href="accc_best_of_n.html">ACCC Standard</a>
  <a href="best_of_n_300.html">Best-of-300</a>
  <a href="pitch_analysis.html">Pitch Analysis</a>
</div>"""

    methodology_html = f"""<div class="methodology" id="methodology">
<h3 onclick="document.getElementById('meth-body').classList.toggle('collapsed')">
Methodology &amp; Scoring Strategy (click to collapse)</h3>
<div id="meth-body">

<h4>Audio Pipeline</h4>
<p>Each audio sample passes through this processing chain:</p>
<pre>1. Acting Challenge + Voice Profile sampled from taxonomy
2. Gemma 4 (google/gemma-4-E4B-it) generates a two-scene DramaBox prompt
   Parameters: temperature=1.0, top_p=0.95, max_new_tokens=1536
3. DramaBox TTS renders the prompt as two audio scenes (partA + partB)
4. RE-USE speech enhancement post-processes each scene
5. Scenes concatenated into a single file
6. LavaSR BWE (bandwidth extension via Vocos, <strong>no denoising</strong>) upsamples to 48kHz
7. VoiceCLAP scoring against positive and negative text prompts</pre>

<h4>VoiceCLAP Scoring</h4>
<p>VoiceCLAP computes cosine similarity between audio embeddings and text embeddings.
Two model sizes are used: <strong>Large</strong> (3584-dim) and <strong>Small</strong> (768-dim).</p>

<h4 class="pos">Positive Prompt ("Sanitized")</h4>
<p>The DramaBox prompt with all <code>"double-quoted dialogue"</code> removed, keeping only
stage directions. This measures how well the voice performance matches the emotional and
directional context (tone, delivery, character), independent of whether specific words were
correctly spoken.</p>
<p><strong>Example:</strong> If the full prompt is:<br>
<code>You whisper softly "I can't believe you're here" then pause, overcome with emotion.</code><br>
The sanitized version becomes:<br>
<code>You whisper softly  then pause, overcome with emotion.</code></p>

<h4 class="neg">Negative Prompt ("neg_san")</h4>
<p>Fixed text: <code>{esc(NEG_SAN_TEXT)}</code></p>
<p>Higher similarity to this text means the audio sounds more robotic or distorted (undesirable).
This score is <strong>subtracted</strong> from the positive score to penalize synthetic-sounding output.</p>

<h4>Ranking Formulas</h4>
<pre class="formula">v_snr_L (DEFAULT) = (1 - WER) &times; (san_L - neg_san_L + 2)
v_snr_S           = (1 - WER) &times; (san_S - neg_san_S + 2)
v_san_L           = (1 - WER) &times; (san_L + 1)
v_san_S           = (1 - WER) &times; (san_S + 1)
Content Enjoyment = raw content enjoyment score (no WER factor)
Standard          = (1 - WER) &times; Content Enjoyment</pre>
<p><code>WER</code> = Word Error Rate from ASR transcription vs expected dialogue.<br>
The <code>+ 2</code> / <code>+ 1</code> offsets shift cosine similarities (which can be negative) into positive range.</p>
<p>The <code>_L</code> suffix = VoiceCLAP Large model, <code>_S</code> = VoiceCLAP Small model.</p>

<h4>EmoNet Emotion Scoring</h4>
<p>Each audio sample is scored across <strong>40 EmoNet emotion dimensions</strong>
using the <strong>Empathic Insight Plus</strong> model
(<code>laion/Empathic-Insight-Voice-Plus</code>). The encoder (BUD-E-Whisper) produces
768-dim audio embeddings, which are pooled (mean+min+max+std &rarr; 3072-dim) and fed
through 40 specialized MLP expert heads &mdash; one per emotion.</p>

<h4>Emotion Ranking Rules</h4>
<p>For emotion-based rankings, a <strong>WER cutoff of 10%</strong> is applied: only
samples with WER &lt; 0.1 are shown and ranked. Samples above this threshold are
excluded entirely (they said the wrong words). Within qualifying samples, candidates
are ranked by <strong>descending emotion score</strong>.</p>
<p>This differs from quality/CLAP rankings, which use (1&minus;WER) as a multiplicative
factor and show all 25 candidates.</p>

</div>
</div>"""

    def _page_nav(current, total):
        stem = "accc_lavasr"
        links = []
        for p in range(1, total + 1):
            cls = ' class="current"' if p == current else ''
            links.append(f'<a href="{stem}_p{p}.html"{cls}>Page {p}</a>')
        return '<div class="page-nav">' + "\n".join(links) + "</div>"

    for page_num in range(1, n_pages + 1):
        start = (page_num - 1) * GROUPS_PER_PAGE
        end = min(start + GROUPS_PER_PAGE, n_groups)
        page_groups = groups[start:end]

        # Build page data for JS
        page_data = {}
        for g in page_groups:
            gidx = g["gidx"]
            candidates = []
            for cand in range(N_CANDIDATES):
                key = f"{gidx}_{cand}"
                m = merged.get(key, {})
                b64 = audio_cache.get((gidx, cand), "")
                cand_data = {
                    "cand": cand,
                    "v_snr_L": m.get("v_snr_L", 0),
                    "v_snr_S": m.get("v_snr_S", 0),
                    "v_san_L": m.get("v_san_L", 0),
                    "v_san_S": m.get("v_san_S", 0),
                    "content_enjoyment_rank": m.get("content_enjoyment_rank", 0),
                    "standard": m.get("standard", 0),
                    "wer": m.get("wer", 1.0),
                    "san_L": m.get("san_L", 0),
                    "neg_san_L": m.get("neg_san_L", 0),
                    "san_S": m.get("san_S", 0),
                    "neg_san_S": m.get("neg_san_S", 0),
                    "b64": b64,
                }
                for e in EMONET_EMOTIONS:
                    cand_data[f"emo_{e}"] = m.get(f"emo_{e}", 0.0)
                candidates.append(cand_data)
            page_data[gidx] = {"candidates": candidates}

        # Build sections HTML
        sections_html = ""
        for gi, g in enumerate(page_groups):
            gidx = g["gidx"]
            title = g.get("challenge_title", f"Group {gidx}")
            lang = g.get("language", "English")
            challenge_instr = g.get("challenge_instruction", "")
            gender_desc = g.get("gender_desc", "")
            age_desc = g.get("age_desc", "")
            word_count = g.get("word_count", "?")
            prompt = g.get("dramabox_prompt", "")
            san_text = sanitize_prompt(prompt)

            # First group on first page: prompt expanded; rest collapsed
            prompt_collapsed = "" if (page_num == 1 and gi == 0) else " collapsed"
            toggle_verb = "Hide" if (page_num == 1 and gi == 0) else "Show"

            sections_html += f"""
<div class="group-section" data-gidx="{gidx}">
<div class="group-header">
  <h2>Group {gidx} &mdash; {esc(title)} ({lang})</h2>
  <div class="group-meta">
    <strong>Challenge:</strong> {esc(challenge_instr[:500])}
  </div>
  <div class="group-meta">
    <strong>Gender:</strong> {esc(gender_desc[:200])}<br>
    <strong>Age:</strong> {esc(age_desc[:200])}<br>
    <strong>Word count:</strong> ~{word_count}
  </div>
  <div class="prompt-block">
    <div class="prompt-toggle" onclick="togglePrompt(this)">{toggle_verb} full DramaBox prompt</div>
    <div class="prompt-content{prompt_collapsed}">{esc(prompt)}</div>
  </div>
  <div class="clap-texts">
    <div class="clap-label pos">Positive CLAP text (sanitized prompt):</div>
    <div class="clap-text-box">{esc(san_text[:800])}</div>
    <div class="clap-label neg">Negative CLAP text (subtracted):</div>
    <div class="clap-text-box">{esc(NEG_SAN_TEXT)}</div>
  </div>
  <div class="gemma-params">
    <strong>Gemma 4 params:</strong> model=google/gemma-4-E4B-it, temperature=1.0,
    top_p=0.95, max_new_tokens=1536, do_sample=True
  </div>
</div>
<div class="candidates">
"""
            for cand in range(N_CANDIDATES):
                sections_html += f"""
  <div class="candidate" data-cand="{cand}">
    <div class="cand-rank">#</div>
    <div class="cand-info">
      <div class="cand-meta">
        <span class="score-val"></span>
        <span class="wer-val"></span>
        <span class="n-val" style="font-size:0.75em;color:var(--dim)"></span>
      </div>
      <div class="clap-detail"></div>
      <audio controls preload="none"></audio>
    </div>
  </div>
"""
            sections_html += f"""
</div>
<div class="show-more" onclick="toggleMore(this)">Show all 25...</div>
</div>
"""

        page_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ACCC LavaSR Grid — Page {page_num}</title>
<style>{css}</style>
</head><body>
<h1>ACCC <span>LavaSR BWE</span> Ranking</h1>
<p class="subtitle">RE-USE + LavaSR BWE (no denoising) &rarr; VoiceCLAP scoring | 25 candidates per group |
Page {page_num}/{n_pages} (groups {start}-{end-1})</p>
{nav}
{_page_nav(page_num, n_pages)}
{methodology_html}
<div class="controls">
  <label>Ranking method:</label>
  <select id="method-select">{method_options}</select>
</div>
{sections_html}
{_page_nav(page_num, n_pages)}
<script>{js_template % json.dumps(page_data)}</script>
</body></html>"""

        page_path = HTML_OUTPUT.parent / f"accc_lavasr_p{page_num}.html"
        with open(page_path, "w") as f:
            f.write(page_html)
        size_mb = os.path.getsize(page_path) / 1e6
        print(f"  Page {page_num}: {page_path.name} ({size_mb:.1f} MB) — groups {start}-{end-1}", flush=True)

    # Index page (redirect to page 1)
    with open(HTML_OUTPUT, "w") as f:
        f.write(f'<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0;url=accc_lavasr_p1.html"></head></html>')
    print(f"  Index: {HTML_OUTPUT.name}", flush=True)
    print(f"\n  HTML generation done in {time.time()-t0:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  ACCC LAVASR PIPELINE")
    print("=" * 72, flush=True)

    with open(GROUPS_FILE) as f:
        groups = json.load(f)
    print(f"  {len(groups)} groups × {N_CANDIDATES} candidates = {len(groups)*N_CANDIDATES} clips", flush=True)

    phase1_concat(groups)
    phase2_lavasr(groups)
    phase3_scoring(groups)
    phase3d_emonet(groups)
    merged = phase4_merge(groups)
    phase5_html(groups, merged)

    print("\n  DONE!")
