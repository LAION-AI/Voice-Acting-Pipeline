#!/usr/bin/env python3
"""
Generate DramaBox two-scene (CUT TO:) prompts for Situation-Inspired challenges.

5,749 situation-inspired acting challenges → DramaBox prompts.
Language distribution: English, French, Spanish, German (round-robin).
Uses DeepSeek V4 Flash via Hyprlab API.

Output: dramabox_sit_situation.json
"""

import json
import os
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
TMP_DIR = DATA_DIR / "tmp_situation_dramabox"
TMP_DIR.mkdir(exist_ok=True)

OUT_FILE = DATA_DIR / "dramabox_sit_situation.json"

# Language rotation
LANGUAGES = ["English", "French", "Spanish", "German"]

# Progress
progress_lock = threading.Lock()
completed_count = 0
failed_count = 0
total_count = 0

# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You write character-consistent two-scene voice performance prompts in DramaBox format for a single speaker.

CRITICAL RULES:
- It is ALWAYS one single person speaking the entire prompt. The same voice, the same actor, from start to finish. Never introduce a second speaker. Explicitly anchor identity: "the same voice", "the same speaker".
- NO markdown. No bold, no stars, no headers, no labels. Just plain text.
- Directions go in (parentheses). Spoken words go in "double quotes". Alternate between them roughly equally. Keep directions SHORT, 5-12 words each.
- The delivery must sound natural, realistic, genuine, spontaneous — like a real human in a real moment, not a stage performance.
- The actor performs with all the little micro-distractions someone in real life in a real situation would have — a natural, authentic sensuality and variance in tone, organically reacting to all the micro-distractions around them. Shifting weight, noticing a sound, losing a thought and finding it again. The performance breathes.
- The speaker is PHYSICALLY IN the situation — their body posture, environment, health, or social context naturally affects how they speak, breathe, and move.
- TOTAL spoken dialogue (inside "double quotes") must be approximately 50 words — roughly 25 words before CUT TO: and 25 words after.
- Do NOT exceed 60 words of dialogue total. Do NOT go below 40 words.

STRUCTURE (write exactly like this, no labels, no headers):

A [age] [gender] with a [timbre/vocal quality], delivering this high-quality studio voice recording with no background noise.

The same voice is [1 sentence: emotional state for the first moment, grounded in the physical situation].

(short direction) "Spoken words." (short direction) "More spoken words."

CUT TO:

The same voice now [1 sentence: how the emotion has shifted dramatically, same physical situation].

(short direction) "Spoken words." (short direction) "More spoken words."

The performance across both moments should feel [1 sentence].

EMOTION CONTRAST: Maximize emotional distance between scenes. Polarity flips (joy to grief), arousal shifts (screaming to whisper), control shifts (composure to breakdown).

Output ONLY the raw prompt. Nothing else."""


# ─── Build user prompt ───────────────────────────────────────────────────────

def build_user_prompt(challenge: dict, language: str) -> str:
    """Build user prompt for a situation-inspired DramaBox scene."""
    sit_name = challenge.get("situation_name", "")
    sit_dim = challenge.get("situation_dim", "")
    emotions = challenge.get("emotions_sampled", "")

    lang_instruction = ""
    if language != "English":
        lang_instruction = f"ALL spoken dialogue (inside double quotes) MUST be in {language}. Directions (in parentheses) and the speaker description can remain in English."

    prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition based on this acting challenge:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}

SITUATION: {sit_name} (Dimension: {sit_dim})
The speaker is physically IN this situation — it naturally affects their voice, breathing, posture, and delivery.

EMOTIONS sampled for this challenge: {emotions}
(Scene 1 should lean into these emotions. Scene 2 should shift to a contrasting emotional state — you decide what fits.)

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
LANGUAGE: {language}. {lang_instruction}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
The actor performs genuinely with micro-distractions, natural variance, and organic authenticity.
Choose an appropriate speaker (age, gender, timbre) that fits the challenge.
Output ONLY the raw DramaBox prompt."""

    return prompt


# ─── API call ────────────────────────────────────────────────────────────────

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
            return content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE)
                continue
            return None
    return None


# ─── Item processing ─────────────────────────────────────────────────────────

def make_id(index: int) -> str:
    return f"sit_db_{index:06d}"


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
            if completed_count % 500 == 0:
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
            if failed_count % 50 == 0:
                print(f"  WARNING: {failed_count} failures so far", flush=True)
        return None

    result = {
        "id": item_id,
        "pathway": "SIT (Situation)",
        "language": item["language"],
        "sample_info": item["sample_info"],
        "dramabox_prompt": output,
    }

    with open(get_tmp_path(item_id), "w") as f:
        json.dump(result, f, ensure_ascii=False)

    with progress_lock:
        completed_count += 1
        if completed_count % 500 == 0:
            print(f"  Progress: {completed_count}/{total_count} ({failed_count} failed)", flush=True)

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global total_count, completed_count, failed_count

    print("=" * 70)
    print("Situation DramaBox Two-Scene Prompt Generator")
    print("=" * 70)

    print("\n[1/4] Loading situation challenges...")
    with open(DATA_DIR / "acting_challenges_situation_inspired.json") as f:
        challenges = json.load(f)
    print(f"  Loaded {len(challenges)} situation-inspired challenges")
    print(f"  Languages: {', '.join(LANGUAGES)} (round-robin)")

    print("\n[2/4] Building work items...")
    items = []
    random.seed(42)

    # Shuffle challenges for language distribution diversity
    indices = list(range(len(challenges)))
    random.shuffle(indices)

    for seq, idx in enumerate(indices):
        ch = challenges[idx]
        language = LANGUAGES[seq % len(LANGUAGES)]
        item_id = make_id(seq)
        user_prompt = build_user_prompt(ch, language)

        items.append({
            "id": item_id,
            "language": language,
            "sample_info": {
                "challenge_title": ch.get("title", ""),
                "challenge_id": ch.get("id", seq),
                "situation_name": ch.get("situation_name", ""),
                "situation_dim": ch.get("situation_dim", ""),
                "emotions_sampled": ch.get("emotions_sampled", ""),
                "original_item_id": ch.get("item_id", ""),
            },
            "user_prompt": user_prompt,
        })

    total_count = len(items)
    already_done = sum(1 for it in items if is_done(it["id"]))

    # Count per language
    lang_counts = {}
    for it in items:
        lang = it["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"\n  Total items: {total_count}")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
    print(f"  Already completed (resuming): {already_done}")
    print(f"  Remaining: {total_count - already_done}")

    if already_done == total_count:
        print("\n  All items already done! Skipping to collection.")
    else:
        # Process
        print(f"\n[3/4] Generating with {MAX_THREADS} threads...")
        completed_count = 0
        failed_count = 0

        try:
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = {executor.submit(process_item, item): item for item in items}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
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

    # Count per language in results
    lang_result_counts = {}
    for r in results:
        lang = r.get("language", "English")
        lang_result_counts[lang] = lang_result_counts.get(lang, 0) + 1

    # Assign sequential IDs
    output = []
    for i, r in enumerate(results, 1):
        output.append({
            "id": i,
            "pathway": "SIT (Situation)",
            "language": r.get("language", "English"),
            "sample_info": r.get("sample_info", {}),
            "dramabox_prompt": r.get("dramabox_prompt", ""),
        })

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(output)} prompts to {OUT_FILE.name}")
    for lang, count in sorted(lang_result_counts.items()):
        print(f"    {lang}: {count}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
