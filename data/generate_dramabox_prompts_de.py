#!/usr/bin/env python3
"""
Generate GERMAN DramaBox two-scene (CUT TO:) prompts using DeepSeek V4 Flash via Hyprlab.

Three pathways, 10,000 each:
  1. CC-A  (VoiceNet + EmoNet):  10,000 prompts
  2. CC2-C (Archetype):          10,000 prompts
  3. ACCC  (Acting Challenge):   10,000 prompts (sampled from 18,647)

All spoken dialogue in German (no umlauts: oe/ae/ue instead of ö/ä/ü).
Directions and descriptions remain in English.
~50 spoken words total (~25 per scene).
"""

import json
import os
import sys
import random
import time
import traceback
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

# ─── Add project root to path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dramabox.taxonomy import (
    parse_voicenet_html, load_emonet, load_archetypes,
    load_vocal_bursts, format_vocal_bursts_block,
)
from dramabox.sampling import (
    sample_voicenet, sample_archetype, INTENSITY_LEVELS,
    TEMPO_LABELS, AROUSAL_LABELS, is_prompt_safe,
)
from dramabox.prompts import (
    FLOW_INSTRUCTIONS, ALIGNMENT_INSTRUCTIONS,
    DIRECTION_STYLE_INSTRUCTIONS,
)

# ─── Configuration ───────────────────────────────────────────────────────────
API_KEY = "hypr-lab-dhItb5DFQctQvafMzqgKT3BlbkFJfot58G96B2VMaS4u0015"
API_URL = "https://api.hyprlab.io/v1/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_THREADS = 20
MAX_RETRIES = 5
RETRY_DELAY_BASE = 3

DATA_DIR = Path(__file__).resolve().parent
TMP_DIR = DATA_DIR / "tmp_dramabox_de"
TMP_DIR.mkdir(exist_ok=True)

# Output files
OUT_CCA = DATA_DIR / "dramabox_cca_voicenet_de.json"
OUT_CC2C = DATA_DIR / "dramabox_cc2c_archetype_de.json"
OUT_ACCC = DATA_DIR / "dramabox_accc_acting_challenge_de.json"

# Progress
progress_lock = threading.Lock()
completed_count = 0
failed_count = 0
total_count = 0

# ─── German language instruction (appended to every user prompt) ─────────────

GERMAN_INSTRUCTION = """\
LANGUAGE: German.
ALL spoken dialogue (inside "double quotes") MUST be in German.
CRITICAL: Do NOT use German umlauts (ö, ä, ü, Ö, Ä, Ü). Replace them: ö→oe, ä→ae, ü→ue, Ö→Oe, Ä→Ae, Ü→Ue. For example: "schoene" not "schöne", "ueber" not "über", "Maedchen" not "Mädchen".
Directions (in parentheses) and the speaker description MUST remain in English."""

# ─── System prompt for two-scene DramaBox ────────────────────────────────────

SYSTEM_PROMPT = """\
You write character-consistent two-scene voice performance prompts in DramaBox format for a single speaker.

CRITICAL RULES:
- It is ALWAYS one single person speaking the entire prompt. The same voice, the same actor, from start to finish. Never introduce a second speaker. Explicitly anchor identity: "the same voice", "the same speaker".
- NO markdown. No bold, no stars, no headers, no labels. Just plain text.
- Directions go in (parentheses). Spoken words go in "double quotes". Alternate between them roughly equally. Keep directions SHORT, 5-12 words each.
- The delivery must sound natural, realistic, genuine, spontaneous — like a real human in a real moment, not a stage performance.
- The actor performs with all the little micro-distractions someone in real life in a real situation would have — a natural, authentic sensuality and variance in tone, organically reacting to all the micro-distractions around them. Shifting weight, noticing a sound, losing a thought and finding it again. The performance breathes.
- TOTAL spoken dialogue (inside "double quotes") must be approximately 50 words — roughly 25 words before CUT TO: and 25 words after.
- Do NOT exceed 60 words of dialogue total. Do NOT go below 40 words.
- ALL spoken dialogue MUST be in German. But NEVER use umlauts (ö, ä, ü). Always substitute: oe for ö, ae for ä, ue for ü. This is mandatory.
- Directions (parentheses) and speaker descriptions remain in English.

STRUCTURE (write exactly like this, no labels, no headers):

A [age] [gender] with a [timbre/vocal quality], delivering this high-quality studio voice recording with no background noise.

The same voice is [1 sentence: emotional state for the first moment].

(short direction in English) "German spoken words without umlauts." (short direction) "More German spoken words."

CUT TO:

The same voice now [1 sentence: how the emotion has shifted dramatically].

(short direction in English) "German spoken words without umlauts." (short direction) "More German spoken words."

The performance across both moments should feel [1 sentence].

EMOTION CONTRAST: Maximize emotional distance between scenes. Polarity flips (joy to grief), arousal shifts (screaming to whisper), control shifts (composure to breakdown).

Output ONLY the raw prompt. Nothing else."""


# ─── Load taxonomies ─────────────────────────────────────────────────────────

def load_all_taxonomies():
    dims = parse_voicenet_html(DATA_DIR / "voicenet_ext_taxonomy.html")
    mandatory_codes = {"TEMP", "GEND", "AGEV"}
    mandatory_dims = [d for d in dims if d["code"] in mandatory_codes]
    optional_dims = [d for d in dims if d["code"] not in mandatory_codes]
    temp_dim = next(d for d in dims if d["code"] == "TEMP")
    arou_dim = next(d for d in dims if d["code"] == "AROU")

    emonet = load_emonet(DATA_DIR / "emonet_taxonomy.json")
    emotion_categories = list(emonet.keys())

    archetypes = load_archetypes(DATA_DIR / "archetypes.json")

    vb_taxonomy = load_vocal_bursts(DATA_DIR / "vocal_bursts_taxonomy.json")
    vb_block = format_vocal_bursts_block(vb_taxonomy)

    return {
        "mandatory_dims": mandatory_dims,
        "optional_dims": optional_dims,
        "temp_dim": temp_dim,
        "arou_dim": arou_dim,
        "emotion_categories": emotion_categories,
        "archetypes": archetypes,
        "vb_block": vb_block,
        "vb_taxonomy": vb_taxonomy,
    }


# ─── Sampling configs ────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "sampling": {
        "emotions_min": 1,
        "emotions_max": 3,
        "random_dims_count": 5,
        "tempo_bias_threshold": 3,
        "tempo_bias_weight": 1.5,
        "word_count_min": 20,
        "word_count_max": 30,
        "mandatory_words_count": 0,
        "flow_style_distribution": {"scattered": 0.05, "flowing": 0.55, "mixed": 0.40},
        "emotion_alignment_distribution": {"congruent": 0.30, "neutral": 0.40, "counter-emotional": 0.30},
        "direction_style_distribution": {"literary": 0.50, "tag": 0.50},
        "vocal_bursts_probability": 0.0,
    },
    "_active_languages": ["German"],
    "_language_accents": {"German": []},
}


# ─── Build user prompts for each pathway ─────────────────────────────────────

def build_cca_user_prompt(tax: dict) -> tuple[dict, str]:
    """Build a CC-A (VoiceNet) two-scene prompt — German dialogue."""
    sample = sample_voicenet(
        tax["mandatory_dims"], tax["optional_dims"],
        tax["emotion_categories"], DEFAULT_CONFIG,
    )

    vb_section = ""
    if sample["vocal_bursts_enabled"]:
        vb_section = f"\n{tax['vb_block']}\n"

    prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition.

VOICE ATTRIBUTES (shape the speaker from these):
{sample['attributes_clean']}

EMOTIONS for Scene 1: {sample['emotions']}
(Scene 2 should shift to a contrasting emotional state — you decide what fits.)

{FLOW_INSTRUCTIONS[sample['flow_style']]}
{ALIGNMENT_INSTRUCTIONS[sample['emotion_alignment']]}
{DIRECTION_STYLE_INSTRUCTIONS[sample['direction_style']]}
{vb_section}
TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
{GERMAN_INSTRUCTION}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
The actor performs genuinely with micro-distractions, natural variance, and organic authenticity.
Output ONLY the raw DramaBox prompt."""

    return sample, prompt


def build_cc2c_user_prompt(tax: dict) -> tuple[dict, str]:
    """Build a CC2-C (Archetype) two-scene prompt — German dialogue."""
    sample = sample_archetype(
        tax["archetypes"], tax["temp_dim"], tax["arou_dim"],
        tax["emotion_categories"], DEFAULT_CONFIG,
    )

    prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition.

ARCHETYPE: {sample['_archetype']} (from genre: {sample['_genre']})
EMOTIONS for Scene 1: {sample['emotions']}
TEMPO: {sample['_tempo_desc']}
AROUSAL: {sample['_arousal_desc']}
(Scene 2 should shift to a sharply contrasting emotional state.)

Do NOT reproduce the archetype description literally. Use it as inspiration — the character should be recognizably this archetype but with your own unique spin.

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
{GERMAN_INSTRUCTION}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
The actor performs genuinely with micro-distractions, natural variance, and organic authenticity.
Output ONLY the raw DramaBox prompt."""

    return sample, prompt


def build_accc_user_prompt(challenge: dict) -> tuple[dict, str]:
    """Build an ACCC (Acting Challenge) two-scene prompt — German dialogue."""
    prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition based on this acting challenge:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}

The two scenes should capture two emotionally contrasting moments from this challenge. Scene 1 should establish one emotional state; Scene 2 (after CUT TO:) should shift dramatically.

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
{GERMAN_INSTRUCTION}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
The actor performs genuinely with micro-distractions, natural variance, and organic authenticity.
Choose an appropriate speaker (age, gender, timbre) that fits the challenge.
Output ONLY the raw DramaBox prompt."""

    return {"acting_challenge": challenge["title"]}, prompt


# ─── Umlaut replacement ─────────────────────────────────────────────────────

def replace_umlauts(text: str) -> str:
    """Replace any remaining umlauts in the output."""
    replacements = {
        'ö': 'oe', 'ä': 'ae', 'ü': 'ue',
        'Ö': 'Oe', 'Ä': 'Ae', 'Ü': 'Ue',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


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
            # Post-process: replace any umlauts the model sneaked in
            return replace_umlauts(content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE)
                continue
            return None
    return None


# ─── Item processing ─────────────────────────────────────────────────────────

def make_id(pathway: str, index: int) -> str:
    return f"{pathway}_de_{index:06d}"


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
        "pathway": item["pathway"],
        "language": "German",
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
    print("DramaBox Two-Scene Prompt Generator — GERMAN (no umlauts)")
    print("=" * 70)

    print("\n[1/5] Loading taxonomies...")
    tax = load_all_taxonomies()
    print(f"  VoiceNet: {len(tax['mandatory_dims'])} mandatory + {len(tax['optional_dims'])} optional dims")
    print(f"  EmoNet: {len(tax['emotion_categories'])} emotions")
    print(f"  Archetypes: {sum(len(v) for v in tax['archetypes'].values())} across {len(tax['archetypes'])} genres")

    print("\n[2/5] Building work items...")
    items = []

    # CC-A: 10,000 VoiceNet prompts (German)
    print("  Sampling 10,000 CC-A (VoiceNet) items...")
    random.seed(1337)  # Different seed from English run
    for i in range(10000):
        sample, user_prompt = build_cca_user_prompt(tax)
        items.append({
            "id": make_id("cca", i),
            "pathway": "cca",
            "sample_info": {
                "attributes": sample.get("attributes_clean", ""),
                "emotions": sample.get("emotions", ""),
                "flow_style": sample.get("flow_style", ""),
                "emotion_alignment": sample.get("emotion_alignment", ""),
                "direction_style": sample.get("direction_style", ""),
            },
            "user_prompt": user_prompt,
        })

    # CC2-C: 10,000 Archetype prompts (German)
    print("  Sampling 10,000 CC2-C (Archetype) items...")
    for i in range(10000):
        sample, user_prompt = build_cc2c_user_prompt(tax)
        items.append({
            "id": make_id("cc2c", i),
            "pathway": "cc2c",
            "sample_info": {
                "genre": sample.get("_genre", ""),
                "archetype": sample.get("_archetype", ""),
                "emotions": sample.get("emotions", ""),
                "tempo": sample.get("_tempo_desc", ""),
                "arousal": sample.get("_arousal_desc", ""),
            },
            "user_prompt": user_prompt,
        })

    # ACCC: 10,000 acting challenges (German) — sample from 18,647
    print("  Loading acting challenges...")
    with open(DATA_DIR / "all_acting_challenges.json") as f:
        challenges = json.load(f)
    sampled = random.sample(challenges, 10000)
    print(f"  Sampled 10,000 from {len(challenges)} challenges for ACCC...")
    for i, ch in enumerate(sampled):
        sample, user_prompt = build_accc_user_prompt(ch)
        items.append({
            "id": make_id("accc", i),
            "pathway": "accc",
            "sample_info": {
                "challenge_title": ch.get("title", ""),
                "challenge_id": ch.get("id", i),
            },
            "user_prompt": user_prompt,
        })

    total_count = len(items)
    already_done = sum(1 for it in items if is_done(it["id"]))
    print(f"\n  Total items: {total_count}")
    print(f"  CC-A: 10,000 | CC2-C: 10,000 | ACCC: 10,000")
    print(f"  Already completed (resuming): {already_done}")
    print(f"  Remaining: {total_count - already_done}")

    if already_done == total_count:
        print("\n  All items already done! Skipping to collection.")
    else:
        # Process
        print(f"\n[3/5] Generating with {MAX_THREADS} threads...")
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
    print("\n[4/5] Collecting results...")
    cca_results = []
    cc2c_results = []
    accc_results = []

    for tmp_file in sorted(TMP_DIR.glob("*.json")):
        try:
            with open(tmp_file) as f:
                result = json.load(f)
            if not result.get("dramabox_prompt", "").strip():
                continue
            pathway = result.get("pathway", "")
            if pathway == "cca":
                cca_results.append(result)
            elif pathway == "cc2c":
                cc2c_results.append(result)
            elif pathway == "accc":
                accc_results.append(result)
        except:
            pass

    print(f"  CC-A: {len(cca_results)}")
    print(f"  CC2-C: {len(cc2c_results)}")
    print(f"  ACCC: {len(accc_results)}")

    # Save output files
    print("\n[5/5] Saving output files...")

    def save_pathway(results, out_path, pathway_name):
        output = []
        for i, r in enumerate(results, 1):
            output.append({
                "id": i,
                "pathway": pathway_name,
                "language": "German",
                "sample_info": r.get("sample_info", {}),
                "dramabox_prompt": r.get("dramabox_prompt", ""),
            })
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  {out_path.name}: {len(output)} prompts")

    save_pathway(cca_results, OUT_CCA, "CC-A (VoiceNet)")
    save_pathway(cc2c_results, OUT_CC2C, "CC2-C (Archetype)")
    save_pathway(accc_results, OUT_ACCC, "ACCC (Acting Challenge)")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
