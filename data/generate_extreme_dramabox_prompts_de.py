#!/usr/bin/env python3
"""
Generate GERMAN DramaBox two-scene (CUT TO:) prompts for Extreme Physical challenges.

600 extreme physical acting challenges → DramaBox prompts (German, no umlauts).
Uses DeepSeek V4 Flash via Hyprlab API.

Output: dramabox_extreme_physical_de.json
"""

import json
import sys
import random
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

# ─── Configuration ───────────────────────────────────────────────────────────
API_KEY = "hypr-lab-dhItb5DFQctQvafMzqgKT3BlbkFJfot58G96B2VMaS4u0015"
API_URL = "https://api.hyprlab.io/v1/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_THREADS = 20
MAX_RETRIES = 5
RETRY_DELAY_BASE = 3

DATA_DIR = Path(__file__).resolve().parent
TMP_DIR = DATA_DIR / "tmp_extreme_dramabox_de"
TMP_DIR.mkdir(exist_ok=True)

OUT_FILE = DATA_DIR / "dramabox_extreme_physical_de.json"

# Progress
progress_lock = threading.Lock()
completed_count = 0
failed_count = 0
total_count = 0

GERMAN_INSTRUCTION = """\
LANGUAGE: German.
ALL spoken dialogue (inside "double quotes") MUST be in German.
CRITICAL: Do NOT use German umlauts (ö, ä, ü, Ö, Ä, Ü). Replace them: ö→oe, ä→ae, ü→ue, Ö→Oe, Ä→Ae, Ü→Ue. For example: "schoene" not "schöne", "ueber" not "über", "Maedchen" not "Mädchen".
Directions (in parentheses) and the speaker description MUST remain in English."""

SYSTEM_PROMPT = """\
You write character-consistent two-scene voice performance prompts in DramaBox format for a single speaker.

CRITICAL RULES:
- It is ALWAYS one single person speaking the entire prompt. The same voice, the same actor, from start to finish. Never introduce a second speaker. Explicitly anchor identity: "the same voice", "the same speaker".
- NO markdown. No bold, no stars, no headers, no labels. Just plain text.
- Directions go in (parentheses). Spoken words go in "double quotes". Alternate between them roughly equally. Keep directions SHORT, 5-12 words each.
- The delivery must sound natural, realistic, genuine, spontaneous — like a real human in a real moment, not a stage performance.
- Scene 1 starts from a place of calm sensuality, heightened perceptiveness — the actor is present, soft, aware of their body and surroundings.
- Scene 2 (after CUT TO:) ERUPTS with the extreme physical sensation — the voice transforms, distorts, breaks under the physical assault. Words dissolve, crack, become raw vocalizations.
- The actor performs with all the little micro-distractions someone in real life in a real situation would have — natural variance, organic reactions.
- TOTAL spoken dialogue (inside "double quotes") must be approximately 50 words — roughly 25 words before CUT TO: and 25 words after.
- Do NOT exceed 60 words of dialogue total. Do NOT go below 40 words.
- NO sound effects. Only the actor's voice, breath, and vocalizations.
- ALL spoken dialogue MUST be in German. But NEVER use umlauts (ö, ä, ü). Always substitute: oe for ö, ae for ä, ue for ü. This is mandatory.
- Directions (parentheses) and speaker descriptions remain in English.

STRUCTURE (write exactly like this, no labels, no headers):

A [age] [gender] with a [timbre/vocal quality], delivering this high-quality studio voice recording with no background noise.

The same voice is [1 sentence: calm, sensual, perceptive state — the actor is present and aware].

(short direction in English) "German spoken words without umlauts." (short direction) "More German spoken words."

CUT TO:

The same voice now [1 sentence: the extreme physical sensation has struck — voice transforms].

(short direction in English) "German words breaking under physical intensity." (short direction) "More distorted German vocalizations."

The performance across both moments should feel [1 sentence about the brutal contrast].

Output ONLY the raw prompt. Nothing else."""


def replace_umlauts(text: str) -> str:
    replacements = {'ö': 'oe', 'ä': 'ae', 'ü': 'ue', 'Ö': 'Oe', 'Ä': 'Ae', 'Ü': 'Ue'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def call_api(user_prompt: str) -> str | None:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.9,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", RETRY_DELAY_BASE * (2 ** attempt)))
                time.sleep(wait)
                continue
            if resp.status_code == 529:
                time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                continue
            if resp.status_code != 200:
                print(f"  [API {resp.status_code}] {resp.text[:150]}", flush=True)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                    continue
                return None
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            if not content or not content.strip():
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return None
            return replace_umlauts(content)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE)
                continue
            return None
    return None


def make_id(index: int) -> str:
    return f"extreme_de_{index:06d}"


def get_tmp_path(item_id: str) -> Path:
    return TMP_DIR / f"{item_id}.json"


def is_done(item_id: str) -> bool:
    p = get_tmp_path(item_id)
    if not p.exists() or p.stat().st_size < 10:
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return bool(d.get("dramabox_prompt", "").strip())
    except:
        return False


def process_item(item: dict) -> dict | None:
    global completed_count, failed_count

    item_id = item["id"]

    if is_done(item_id):
        with progress_lock:
            completed_count += 1
            if completed_count % 100 == 0:
                print(f"  Progress: {completed_count}/{total_count} ({failed_count} failed)", flush=True)
        try:
            with open(get_tmp_path(item_id)) as f:
                return json.load(f)
        except:
            pass

    output = call_api(item["user_prompt"])

    if output is None or not output.strip():
        with progress_lock:
            failed_count += 1
            if failed_count % 20 == 0:
                print(f"  WARNING: {failed_count} failures so far", flush=True)
        return None

    result = {
        "id": item_id,
        "pathway": "Extreme Physical",
        "language": "German",
        "sample_info": item["sample_info"],
        "dramabox_prompt": output,
    }

    with open(get_tmp_path(item_id), "w") as f:
        json.dump(result, f, ensure_ascii=False)

    with progress_lock:
        completed_count += 1
        if completed_count % 100 == 0:
            print(f"  Progress: {completed_count}/{total_count} ({failed_count} failed)", flush=True)

    return result


def main():
    global total_count, completed_count, failed_count

    print("=" * 70)
    print("Extreme Physical DramaBox — GERMAN (no umlauts)")
    print("=" * 70)

    print("\n[1/4] Loading extreme challenges...")
    with open(DATA_DIR / "acting_challenges_extreme_physical.json") as f:
        challenges = json.load(f)
    print(f"  Loaded {len(challenges)} extreme physical challenges")

    print("\n[2/4] Building work items...")
    items = []

    for i, ch in enumerate(challenges):
        item_id = make_id(i)
        cat_name = ch.get("category_name", "")
        subcat = ch.get("subcategory", "")

        user_prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition based on this extreme physical acting challenge:

ACTING CHALLENGE: {ch['title']}
INSTRUCTION: {ch['instruction']}

Category: {cat_name} — {subcat}

Scene 1 MUST begin from a place of calm sensuality and heightened perceptiveness — the actor is soft, present, aware.
Scene 2 (after CUT TO:) MUST erupt with the extreme physical sensation — the voice transforms, distorts, breaks.

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
{GERMAN_INSTRUCTION}
NO sound effects — only the actor's voice, breath, and vocalizations.
SINGLE SPEAKER throughout. Same voice, same person, two moments of brutal contrast.
Choose an appropriate speaker (age, gender, timbre) that fits the challenge.
Output ONLY the raw DramaBox prompt."""

        items.append({
            "id": item_id,
            "sample_info": {
                "challenge_title": ch.get("title", ""),
                "challenge_id": ch.get("id", i),
                "category_code": ch.get("category_code", ""),
                "category_name": cat_name,
                "subcategory": subcat,
            },
            "user_prompt": user_prompt,
        })

    total_count = len(items)
    already_done = sum(1 for it in items if is_done(it["id"]))
    print(f"  Total items: {total_count}")
    print(f"  Already completed (resuming): {already_done}")
    print(f"  Remaining: {total_count - already_done}")

    if already_done == total_count:
        print("\n  All items already done! Skipping to collection.")
    else:
        print(f"\n[3/4] Generating with {MAX_THREADS} threads...")
        completed_count = 0
        failed_count = 0

        try:
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = {executor.submit(process_item, item): item for item in items}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        traceback.print_exc()
        except KeyboardInterrupt:
            print("\nInterrupted! Progress saved.")

        print(f"\n  Completed: {completed_count}, Failed: {failed_count}")

    # Collect results
    print("\n[4/4] Collecting and saving results...")
    results = []
    for tmp_file in sorted(TMP_DIR.glob("*.json")):
        try:
            with open(tmp_file) as f:
                result = json.load(f)
            if result.get("dramabox_prompt", "").strip():
                results.append(result)
        except:
            pass

    output = []
    for i, r in enumerate(results, 1):
        output.append({
            "id": i,
            "pathway": "Extreme Physical",
            "language": "German",
            "sample_info": r.get("sample_info", {}),
            "dramabox_prompt": r.get("dramabox_prompt", ""),
        })

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Check umlauts
    umlaut_count = sum(p["dramabox_prompt"].count(c) for p in output for c in "öäüÖÄÜ")
    print(f"  Saved {len(output)} prompts to {OUT_FILE.name}")
    print(f"  Remaining umlauts: {umlaut_count}")
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
