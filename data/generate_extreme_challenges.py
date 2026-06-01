#!/usr/bin/env python3
"""
Generate 600 acting challenges from the Extreme Physical Conditions taxonomy.

6 categories x 10 subcategories x 10 scenarios = 600 acting challenges.
Each challenge: CUT TO: format — starts from sensual/perceptive calm, then erupts.

Uses DeepSeek V4 Flash via Hyprlab API.
Output: acting_challenges_extreme_physical.json
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
TMP_DIR = DATA_DIR / "tmp_extreme_challenges"
TMP_DIR.mkdir(exist_ok=True)

OUT_FILE = DATA_DIR / "acting_challenges_extreme_physical.json"

# EmoNet emotions for sampling
EMONET_EMOTIONS = [
    "Affection", "Amusement", "Anger", "Astonishment/Surprise", "Awe",
    "Bitterness", "Concentration", "Confusion", "Contemplation", "Contempt",
    "Contentment", "Disappointment", "Disgust", "Distress", "Doubt",
    "Elation", "Embarrassment", "Emotional Numbness", "Fatigue/Exhaustion", "Fear",
    "Helplessness", "Hope/Enthusiasm/Optimism", "Impatience/Irritability", "Infatuation",
    "Interest", "Intoxication/Altered States", "Jealousy/Envy", "Longing",
    "Malevolence/Malice", "Pain", "Pleasure/Ecstasy", "Pride", "Relief",
    "Sadness", "Sexual Lust", "Shame", "Sourness", "Teasing",
    "Thankfulness/Gratitude", "Triumph",
]

INTENSITIES = [
    "slightly present",
    "clearly present",
    "extremely present",
    "very intensely present",
]

GENDERS = [
    "a woman with a hyper-feminized voice",
    "a woman with a strongly feminized voice",
    "a woman with a moderately feminized voice",
    "a person with an androgynous, gender-neutral voice",
    "a man with a moderately masculinized voice",
    "a man with a strongly masculinized voice",
    "a man with a hyper-masculinized voice",
]

AGES = [
    "a child",
    "a teenager",
    "a young adult",
    "a middle-aged adult",
    "an older adult",
    "an elderly person",
]

# Progress
progress_lock = threading.Lock()
completed_count = 0
failed_count = 0
total_count = 0

SYSTEM_PROMPT = """\
You are an extreme voice acting coach who writes viscerally intense, physically grounded acting challenges for professional voice actors.

Your challenges must be:
- STANDALONE: fully self-explanatory, no references to acting systems or techniques
- Feature a DRAMATIC ARC: Scene 1 begins in a state of calm sensuality, heightened perceptiveness, and quiet awareness. Then something physically extreme happens. Scene 2 erupts with the full force of the physical sensation.
- Use CUT TO: format — the challenge describes two contrasting moments for the same speaker
- BRUTALLY AUTHENTIC: the physical sensation is EXTREME and must distort the voice — grunting, gasping, screaming, teeth-chattering, gagging, muffled speech, words dissolving into pure vocalization
- NO SOUND EFFECTS or background noise descriptions — just the actor's voice and breath
- Include specific spoken dialogue in the instruction (~40-80 words total across both scenes)
- The spoken words should be organic to the situation — what someone would actually say/scream/gasp in this moment

Each challenge has a "title" (compelling, evocative, 3-8 words) and an "instruction" (4-8 sentences: scenario setup, the calm baseline, the eruption, vocal/physical guidance for both scenes).

Output ONLY a JSON array with exactly 1 object having "title" and "instruction" keys. No markdown, no commentary."""


def call_api(user_prompt: str) -> dict | None:
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
        "max_tokens": 512,
        "temperature": 0.9,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", RETRY_DELAY_BASE * (2 ** attempt)))
                time.sleep(wait)
                continue
            if resp.status_code == 529:
                time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                continue
            if resp.status_code != 200:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                    continue
                return None

            result = resp.json()
            text = result["choices"][0]["message"]["content"].strip()

            # Clean markdown fences
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            challenges = json.loads(text)
            if isinstance(challenges, list) and len(challenges) > 0:
                return challenges[0]
            return None

        except (json.JSONDecodeError, KeyError, IndexError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE)
                continue
            return None

    return None


def get_tmp_path(item_id: str) -> Path:
    return TMP_DIR / f"{item_id}.json"


def is_done(item_id: str) -> bool:
    p = get_tmp_path(item_id)
    if not p.exists() or p.stat().st_size < 10:
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return bool(d.get("title", "").strip())
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

    challenge = call_api(item["user_prompt"])

    if challenge is None:
        with progress_lock:
            failed_count += 1
            if failed_count % 20 == 0:
                print(f"  WARNING: {failed_count} failures", flush=True)
        return None

    result = {
        "title": challenge.get("title", "Untitled"),
        "instruction": challenge.get("instruction", ""),
        "source": "extreme_physical",
        "category_code": item["category_code"],
        "category_name": item["category_name"],
        "subcategory": item["subcategory"],
        "scenario": item["scenario"],
        "emotions_sampled": item["emotions"],
        "speaker_gender": item["gender"],
        "speaker_age": item["age"],
        "item_id": item_id,
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
    print("Extreme Physical Condition Acting Challenge Generator")
    print("=" * 70)

    print("\n[1/4] Loading taxonomy...")
    with open(DATA_DIR / "extreme_physical_taxonomy.json") as f:
        taxonomy = json.load(f)

    print(f"  Categories: {len(taxonomy['categories'])}")
    for cat in taxonomy["categories"]:
        print(f"    {cat['code']}: {cat['name']} ({len(cat['subcategories'])} subcats)")

    print("\n[2/4] Building work items...")
    items = []
    random.seed(42)

    for cat in taxonomy["categories"]:
        cat_code = cat["code"]
        cat_name = cat["name"]
        cat_desc = cat["description"]

        for subcat in cat["subcategories"]:
            subcat_name = subcat["name"]
            subcat_desc = subcat["description"]

            for scenario_idx, scenario in enumerate(subcat["example_scenarios"]):
                # Sample 1-3 emotions
                n_emo = random.randint(1, 3)
                chosen_emos = random.sample(EMONET_EMOTIONS, n_emo)
                emo_parts = [f"{e} ({random.choice(INTENSITIES)})" for e in chosen_emos]
                emotions_str = ", ".join(emo_parts)

                gender = random.choice(GENDERS)
                age = random.choice(AGES)

                item_id = f"extreme_{cat_code}_{subcat_name.lower().replace(' ', '_')[:20]}_{scenario_idx:02d}"

                user_prompt = f"""\
Create 1 extreme acting challenge based on this physical condition scenario.

CATEGORY: {cat_name}
{cat_desc}

SUBCATEGORY: {subcat_name}
{subcat_desc}

SPECIFIC SCENARIO: {scenario}

SPEAKER: {gender}, {age}
EMOTIONS to weave in: {emotions_str}

The challenge MUST:
- Begin in Scene 1 with a state of calm sensuality and heightened perceptiveness — the actor is aware of their body, their surroundings, their breath. The voice is soft, present, alive.
- Then in Scene 2 (after CUT TO:), the extreme physical sensation ERUPTS — the voice transforms under the assault of the physical experience. Words distort, break, dissolve.
- Have approximately 40-80 words of spoken dialogue total (across both scenes)
- Include NO sound effects or background noise — only the actor's voice, breath, and vocalizations
- The physical sensation must be EXTREME — not mild discomfort, but overwhelming physical intensity
- NOT mention acting theory, techniques, or systems

Return ONLY a JSON array with exactly 1 object having "title" and "instruction" keys."""

                items.append({
                    "id": item_id,
                    "category_code": cat_code,
                    "category_name": cat_name,
                    "subcategory": subcat_name,
                    "scenario": scenario,
                    "emotions": emotions_str,
                    "gender": gender,
                    "age": age,
                    "user_prompt": user_prompt,
                })

    total_count = len(items)
    already_done = sum(1 for it in items if is_done(it["id"]))
    print(f"  Total items: {total_count}")
    print(f"  Already done: {already_done}")
    print(f"  Remaining: {total_count - already_done}")

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
                d = json.load(f)
            if d.get("title", "").strip():
                results.append(d)
        except:
            pass

    # Assign IDs
    for i, r in enumerate(results, 1):
        r["id"] = i

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Per-category counts
    cat_counts = {}
    for r in results:
        code = r.get("category_code", "?")
        cat_counts[code] = cat_counts.get(code, 0) + 1

    print(f"  Saved {len(results)} challenges to {OUT_FILE.name}")
    for code, count in sorted(cat_counts.items()):
        print(f"    {code}: {count}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
