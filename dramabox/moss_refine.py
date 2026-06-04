#!/usr/bin/env python3
"""
DramaBox MOSS Prompt Refinement Pass
======================================

Runs AFTER dramabox_postprocess.py completes. Scans annotation JSONs for
samples where ``moss_refined_prompt_full`` is null, and refines the DramaBox
prompt using MOSS-Audio-8B-Thinking to match the actual performed audio.

Three MOSS inference passes per sample:
  1. Full audio  → refined two-scene prompt matching the actual performance
  2. Part 1 audio → standalone single-scene prompt for scene 1
  3. Part 2 audio → standalone single-scene prompt for scene 2

MOSS listens to each audio clip together with the original prompt + ASR
transcript, then rewrites the DramaBox prompt to match what was actually
performed (not what was originally requested).

Architecture:
  - Coordinator scans postprocess output, distributes work round-robin
  - Worker subprocesses load MOSS in 4-bit (BitsAndBytesConfig) on assigned GPU
  - Updates annotation JSONs in-place with refined prompts

Requires:
  /tmp/moss_venv with transformers==4.57.1 (MOSS is incompatible with
  transformers >= 5.x). The coordinator auto-launches workers using
  /tmp/moss_venv/bin/python if available.

Usage:
    python dramabox_moss_refine.py                        # All GPUs
    python dramabox_moss_refine.py --num-gpus 4           # 4 GPUs
    python dramabox_moss_refine.py --test                 # First 10 samples, 1 GPU
    python dramabox_moss_refine.py --worker --gpu 0 --work-file work_0.json
"""

import os
import re
import sys
import json
import random
import time
import shutil
import logging
import argparse
import subprocess
import threading
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Fix LD_LIBRARY_PATH (conda cuDNN conflicts)
# ---------------------------------------------------------------------------
if "LD_LIBRARY_PATH" in os.environ:
    filtered = ":".join(
        p for p in os.environ["LD_LIBRARY_PATH"].split(":")
        if "ml-general" not in p
    )
    if filtered:
        os.environ["LD_LIBRARY_PATH"] = filtered
    else:
        del os.environ["LD_LIBRARY_PATH"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("moss_refine")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
POSTPROCESS_OUTPUT = BASE_DIR / "dramabox_postprocess_output"
PROGRESS_DIR = POSTPROCESS_OUTPUT / "moss_progress"
STATE_FILE = POSTPROCESS_OUTPUT / "moss_refine_state.json"

MOSS_DIR = "/home/deployer/MOSS-Audio"
MOSS_WEIGHTS = os.path.join(MOSS_DIR, "weights", "MOSS-Audio-8B-Thinking")

# MOSS requires transformers ~4.57.x — use dedicated venv
MOSS_VENV_PYTHON = "/tmp/moss_venv/bin/python"

MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------------------------------------------------------------------------
# MOSS prompt templates
# ---------------------------------------------------------------------------

# ── Taxonomy-derived vocabulary for MOSS inspiration ──────────────────────
# These are presented to MOSS as plain-English descriptive vocabulary, not as
# taxonomy references.  MOSS should use these words naturally in the refined
# prompt without abbreviations, codes, or references to any taxonomy.

_VOICE_QUALITY_VOCABULARY = """\
When describing the speaker's voice, draw freely from this vocabulary of voice qualities and timbre characteristics.  Use whichever terms accurately match what you hear — do not force terms that don't apply.

TIMBRE & TEXTURE: warm, cold, bright, dark, rough, grainy, smooth, crystalline, metallic, brittle, full, rich, thin, hollow, husky, velvety, silky, resonant, airy, breathy, nasal, throaty, chest-resonant, head-resonant, mask-placed
PITCH & REGISTER: Basso Profundo, baritone, tenor, contralto, mezzo-soprano, soprano, high-pitched, low-pitched, deep, elevated, light, grounded
TEMPO & RHYTHM: glacially slow, deliberate, unhurried, conversational tempo, brisk, fast-flowing, hyper-accelerated, choppy, staccato, legato, flowing, smooth transitions
ARTICULATION: crisp, precise, mumbled, slurred, swallowed consonants, hyper-articulated, relaxed articulation, clear diction
EMPHASIS & DYNAMICS: flat delivery, subtle stress, natural emphasis, heavy marking, aggressive punctuation, crescendoing, fading, rising, decelerating, accelerating
PHRASING: fragmented, choppy bursts, natural clauses, extended phrasing, continuous stream, organic breath cycles
TENSION & ENERGY: loose, relaxed, tense, constricted, strangled, pressed, choked, effortless, strained
VOCAL QUALITIES: vocal fry, creaky, raspy, whispery, breathy, clear, ringing, piercing, booming, massive, intimate, vulnerable, guarded, armored"""

_EMOTION_VOCABULARY = """\
When describing the emotions you hear, use precise emotional language.  Pick the most accurate terms — not just broad labels.

POSITIVE: lighthearted fun, amusement, mirth, playfulness, happiness, excitement, joy, exhilaration, delight, jubilation, bliss, ecstasy, rapture, contentment, relaxation, peacefulness, serenity, satisfaction, fulfillment, tranquility, thankfulness, gratitude, appreciation, warmth, trust, caring, tenderness, devotion, reverence, compassion, fondness, adoration, hope, enthusiasm, optimism, anticipation, courage, determination, fervor, inspiration, triumph, pride, dignity, self-confidence
COGNITIVE: interest, fascination, curiosity, intrigue, awe, wonder, astonishment, surprise, amazement, concentration, deep focus, contemplation, thoughtfulness, pondering, reflection, brooding, pensiveness, relief, solace, comfort, liberation
LONGING & DESIRE: yearning, longing, pining, wistfulness, nostalgia, craving, homesickness, saudade, infatuation, romantic desire
NEGATIVE: impatience, irritability, exasperation, doubt, suspicion, skepticism, uncertainty, fear, terror, dread, apprehension, horror, panic, nervousness, worry, anxiety, anguish, trepidation, foreboding, confusion, bewilderment, disorientation, embarrassment, mortification, awkwardness, shame, guilt, remorse, humiliation, disappointment, regret, dismay, sadness, sorrow, grief, melancholy, despair, heartache, mournfulness, misery, resentment, bitterness, cynicism, contempt, scorn, disdain, loathing, disgust, revulsion, anger, rage, fury, wrath, annoyance, spite, malice
PHYSICAL/STATE: physical pain, suffering, agony, helplessness, powerlessness, fatigue, exhaustion, weariness, numbness, detachment, apathy, indifference, stoicism"""

_VOCAL_BURST_VOCABULARY = """\
If you hear any non-verbal vocal sounds, name them precisely in the stage directions.  Examples of vocal bursts:

LAUGHTER: belly laugh, chuckle, giggle, snicker, chortle, guffaw, cackle, nervous laugh, breathless laugh, sarcastic laugh
CRYING: gentle sob, whimpering, sniffling, bawling, stifled sob, trembling exhale, choked-up swallow, mournful keening
PAIN: sharp yelp, low groan, agonized scream, teeth-gritting hiss, pain gasp, straining grunt
FEAR: blood-curdling shriek, startle gasp, frozen breath-hold, panicked hyperventilation, fright yelp, trembling inhale, terrified whimper
ANGER: guttural growl, aggressive snarl, frustrated groan, enraged scream, exasperated sigh, huff of annoyance, dismissive scoff
SURPRISE: realization burst, shocked inhale, confused sound, awe-struck exhale, dropped-jaw breath
PLEASURE: sensual moan, contented sigh, savoring hum, tension-releasing exhale, relaxing exhale, triumphant cheer
CONVERSATIONAL: filler sounds, throat clear, attention-grabbing sounds, lip smack, tongue click
BREATHING: heavy panting, shuddering breath, slow exhale, sharp intake of breath"""

_SPEAKING_STYLE_VOCABULARY = """\
When describing the delivery style, consider these dimensions:

STYLE: casual, conversational, formal, dramatic, theatrical, narrative, storytelling, newsreader, instructional, authoritative, commanding, playful, cartoonish, intimate, whispery, ranting, monologue
MANNER: spontaneous, unrehearsed, rehearsed, polished, raw, genuine, performed, natural, organic, authentic, vulnerable, guarded, open, reserved, assertive, submissive, cooperative, dominant"""

TAXONOMY_INSPIRATION = (
    _VOICE_QUALITY_VOCABULARY + "\n\n"
    + _EMOTION_VOCABULARY + "\n\n"
    + _VOCAL_BURST_VOCABULARY + "\n\n"
    + _SPEAKING_STYLE_VOCABULARY
)

# ── MOSS prompt templates ─────────────────────────────────────────────────

MOSS_REFINE_PROMPT_FULL = """\
You are an expert audio analyst and voice-acting scriptwriter.

Below is a rich vocabulary of voice qualities, emotions, vocal bursts, and speaking styles.  Use these words as inspiration when describing what you hear — pick the terms that genuinely match the audio.  Write everything in clear, plain English that any actor could immediately understand.  Never use abbreviations, codes, or taxonomy references.

""" + TAXONOMY_INSPIRATION + """

---

Now listen carefully to this audio recording.  It contains a two-scene voice performance by a single speaker, separated by a pause (the "CUT TO:" transition).

Here is the ORIGINAL DramaBox prompt that was used to generate this audio:
---
{original_prompt}
---

Here is what ASR (automatic speech recognition) detected in the audio:
Scene 1 transcript: {scene1_transcript}
Scene 2 transcript: {scene2_transcript}

An independent audio analysis model produced this description of the recording:
---
{bude_caption}
---
Incorporate specific observations from this analysis — such as the speaker's approximate age, gender, accent, pitch register, timbre texture, emotional tone, and delivery style — into your refined prompt.  Do not copy sentences verbatim or mention this source, but DO weave its concrete vocal details into your speaker description and stage directions.

Your task: Rewrite the DramaBox prompt so it ACCURATELY describes what was actually performed in this audio.  The original prompt was what we asked for — but the TTS model may have changed words, emotions, pacing, or vocal qualities.  Your refined prompt should match REALITY, not the request.

Rules:
- Keep the DramaBox format: speaker description, then stage directions + dialogue, CUT TO:, second scene.
- SPEAKER DESCRIPTION must be detailed (at least 2-3 sentences).  Include: approximate age, gender, pitch register (e.g. baritone, mezzo-soprano, tenor), timbre texture (e.g. warm, husky, bright, gravelly, smooth, breathy, nasal), resonance quality (e.g. chest-resonant, head-resonant, full, thin), overall energy and tension (e.g. relaxed, tense, effortless), speaking pace (e.g. deliberate, brisk, conversational tempo), volume/projection (e.g. whispering, soft-spoken, conversational, loud, shouting), and "delivering this high-quality studio voice recording with no background noise."
- Use the ASR words for the dialogue (inside "double quotes"), not the original prompt's words.
- Describe the ACTUAL emotions you hear using precise terms from the vocabulary above — not just "sad" or "happy" but specific shades like "wistful melancholy", "quiet awe", "simmering resentment", "tender vulnerability".
- Describe how the voice CHANGES between and within scenes: shifts in pace, volume, pitch, tension, emotional color.  Note if the speaker accelerates, softens, drops to a whisper, rises in intensity, etc.
- Name any vocal bursts (sighs, gasps, laughs, sobs, breath catches, etc.) precisely in the stage directions.
- Describe the speaking style (conversational, dramatic, intimate, theatrical, etc.) and how it shifts between scenes.
- NEVER include timestamps like (00:00), (00:05), (1:23) etc. in the output.  No time codes of any kind.
- Output ONLY the refined DramaBox prompt.  No commentary, no markdown, no labels.
{direction_style_instruction}"""

MOSS_REFINE_PROMPT_PART = """\
You are an expert audio analyst and voice-acting scriptwriter.

Below is a rich vocabulary of voice qualities, emotions, vocal bursts, and speaking styles.  Use these words as inspiration when describing what you hear — pick the terms that genuinely match the audio.  Write everything in clear, plain English that any actor could immediately understand.  Never use abbreviations, codes, or taxonomy references.

""" + TAXONOMY_INSPIRATION + """

---

Now listen carefully to this audio recording.  It contains a single-scene voice performance by one speaker.

This is {part_label} of a two-scene performance.  Here is the ORIGINAL full DramaBox prompt:
---
{original_prompt}
---

Here is what ASR detected in this specific audio clip:
Transcript: {part_transcript}

An independent audio analysis model produced this description of the recording:
---
{bude_caption}
---
Incorporate specific observations from this analysis — such as the speaker's approximate age, gender, accent, pitch register, timbre texture, emotional tone, and delivery style — into your refined prompt.  Do not copy sentences verbatim or mention this source, but DO weave its concrete vocal details into your speaker description and stage directions.

Your task: Write a STANDALONE DramaBox prompt for just this single scene that accurately describes what was actually performed.  This should work as an independent prompt — not referencing any other scene or "CUT TO:".

Rules:
- DramaBox format: speaker description, then stage directions + dialogue.
- SPEAKER DESCRIPTION must be detailed (at least 2-3 sentences).  Include: approximate age, gender, pitch register (e.g. baritone, mezzo-soprano, tenor), timbre texture (e.g. warm, husky, bright, gravelly, smooth, breathy, nasal), resonance quality (e.g. chest-resonant, head-resonant, full, thin), overall energy and tension (e.g. relaxed, tense, effortless), speaking pace (e.g. deliberate, brisk, conversational tempo), volume/projection (e.g. whispering, soft-spoken, conversational, loud, shouting), and "delivering this high-quality studio voice recording with no background noise."
- Use the ASR words for the dialogue (inside "double quotes").
- Describe the ACTUAL emotions you hear using precise terms from the vocabulary above — not just "sad" or "happy" but specific shades like "wistful melancholy", "quiet awe", "simmering resentment", "tender vulnerability".
- Describe how the voice CHANGES within the scene: shifts in pace, volume, pitch, tension, emotional color.  Note if the speaker accelerates, softens, drops to a whisper, rises in intensity, etc.
- Name any vocal bursts (sighs, gasps, laughs, sobs, breath catches, etc.) precisely in the stage directions.
- Describe the speaking style (conversational, dramatic, intimate, theatrical, etc.).
- Do NOT include "CUT TO:" or reference another scene — this is a standalone prompt.
- NEVER include timestamps like (00:00), (00:05), (1:23) etc. in the output.  No time codes of any kind.
- Output ONLY the DramaBox prompt.  No commentary, no markdown, no labels.
{direction_style_instruction}"""

# ── Single-prompt template (no CUT TO:) ──────────────────────────────────

MOSS_REFINE_PROMPT_SINGLE = """\
You are an expert audio analyst and voice-acting scriptwriter.

Below is a rich vocabulary of voice qualities, emotions, vocal bursts, and speaking styles.  Use these words as inspiration when describing what you hear — pick the terms that genuinely match the audio.  Write everything in clear, plain English that any actor could immediately understand.  Never use abbreviations, codes, or taxonomy references.

""" + TAXONOMY_INSPIRATION + """

---

Now listen carefully to this audio recording.  It contains a complete voice performance by a single speaker (one continuous scene, no scene transitions).

Here is the ORIGINAL DramaBox prompt that was used to generate this audio:
---
{original_prompt}
---

Here is what ASR (automatic speech recognition) detected in the audio:
Transcript: {asr_transcript}

An independent audio analysis model produced this description of the recording:
---
{bude_caption}
---
Incorporate specific observations from this analysis — such as the speaker's approximate age, gender, accent, pitch register, timbre texture, emotional tone, and delivery style — into your refined prompt.  Do not copy sentences verbatim or mention this source, but DO weave its concrete vocal details into your speaker description and stage directions.

Your task: Rewrite the DramaBox prompt so it ACCURATELY describes what was actually performed in this audio.  The original prompt was what we asked for — but the TTS model may have changed words, emotions, pacing, or vocal qualities.  Your refined prompt should match REALITY, not the request.

Rules:
- DramaBox format: speaker description, then stage directions + dialogue.  This is a single continuous scene — do NOT include "CUT TO:" or reference multiple scenes.
- SPEAKER DESCRIPTION must be detailed (at least 2-3 sentences).  Include: approximate age, gender, pitch register (e.g. baritone, mezzo-soprano, tenor), timbre texture (e.g. warm, husky, bright, gravelly, smooth, breathy, nasal), resonance quality (e.g. chest-resonant, head-resonant, full, thin), overall energy and tension (e.g. relaxed, tense, effortless), speaking pace (e.g. deliberate, brisk, conversational tempo), volume/projection (e.g. whispering, soft-spoken, conversational, loud, shouting), and "delivering this high-quality studio voice recording with no background noise."
- Use the ASR words for the dialogue (inside "double quotes").
- Describe the ACTUAL emotions you hear using precise terms from the vocabulary above — not just "sad" or "happy" but specific shades like "wistful melancholy", "quiet awe", "simmering resentment", "tender vulnerability".
- Describe how the voice CHANGES within the performance: shifts in pace, volume, pitch, tension, emotional color.
- Name any vocal bursts (sighs, gasps, laughs, sobs, breath catches, etc.) precisely in the stage directions.
- Describe the speaking style (conversational, dramatic, intimate, theatrical, etc.).
- NEVER include timestamps like (00:00), (00:05), (1:23) etc. in the output.  No time codes of any kind.
- Output ONLY the refined DramaBox prompt.  No commentary, no markdown, no labels.
{direction_style_instruction}"""

# ── Direction style instructions (coin-flip: 50% inline, 50% literary) ────

INLINE_DIRECTION_INSTRUCTION = """
IMPORTANT — DIRECTION STYLE: INLINE INTERLEAVED
Write stage directions as SHORT parenthetical tags (AT MOST 6 words each) placed BETWEEN fragments of dialogue.  Each tag is a brief adjective phrase or directive — like a compact label for what the voice is doing RIGHT NOW.  Examples:

(breath shuddering) "I can't..." (voice cracking) "She's so small now."

(tiny wondering laugh) "There you are." (full of love) "You're still in there."

(barely audible) "Okay... nichts." (sniffles, wipes sleeve) "Vielleicht war es nur der Wind."

STRICT RULES for inline directions:
- Maximum 6 words per (parenthetical tag).  Shorter is better.
- Use short adjectives, adverbs, or brief action phrases — like tags: (voice breaking), (slow exhale), (ice-cold), (barely a whisper), (mocking laugh).
- NEVER write full sentences inside parentheses.  No subjects, no verbs-with-objects, no descriptions longer than a brief tag.
- Alternate frequently: direction, speech, direction, speech.
- Do NOT write long descriptive paragraphs — only short interleaved tags.
"""

LITERARY_DIRECTION_INSTRUCTION = """
DIRECTION STYLE: LITERARY / DESCRIPTIVE
Write stage directions as full, evocative sentences or short paragraphs — like a thoughtful director briefing an actor.  Describe the emotional landscape, the vocal quality, and the physical state before or between blocks of dialogue.  The directions should paint a vivid picture of the performance.
"""


# ---------------------------------------------------------------------------
# Reasoning extraction (reused from moss_pipeline.py)
# ---------------------------------------------------------------------------
def _clean_truncation_artifacts(text: str) -> str:
    """Remove artifacts left when MOSS output was truncated at max_new_tokens.

    Handles: partial timestamps like ``(0``, ``(09``, ``(01 tongue click) (``;
    trailing open parens; markdown header artifacts; and incomplete sentences
    after the last proper ending.
    """
    # Strip complete timestamps first (may appear mid-text)
    text = re.sub(r'\s*\(\d{1,2}:\d{2}\)\s*', ' ', text)
    # Strip time-range timestamps like (00:00 - 00:15)
    text = re.sub(r'\s*\(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\)\s*', ' ', text)
    # Strip partial timestamps at end: (0, (09, (12:, (1:2 etc.
    text = re.sub(r'\s*\(\d{1,2}:?\d{0,2}\s*$', '', text)
    # Strip trailing lone open-paren (from truncation mid-direction)
    text = re.sub(r'\s*\(\s*$', '', text)
    # Strip markdown bold/header artifacts like **Title**\n
    text = re.sub(r'^\*\*[^*]+\*\*\s*\n*', '', text)
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def _split_reasoning(text: str) -> tuple[str, str]:
    """Split MOSS output into reasoning trace and final content.

    Handles three scenarios:
    1. Normal: ``<think>reasoning</think>prompt`` → clean split.
    2. Truncated after think: ``<think>reasoning</think>short fragment``
       → if the fragment looks incomplete, try to recover the prompt from
       inside the reasoning (MOSS sometimes drafts the prompt within <think>).
    3. Unclosed think (hit max tokens): ``<think>reasoning + prompt mixed``
       → find the last DramaBox-style prompt inside the text.
    """
    # ── Scenario 1 & 2: closed <think> tag ──────────────────────────────
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        after_think = text[think_match.end():].strip()
        # If the content after </think> looks substantial, use it
        if len(after_think) > 80:
            return reasoning, after_think
        # Otherwise the output was truncated right after </think>.
        # Try to find a DramaBox-style prompt inside the reasoning block —
        # MOSS sometimes writes a full draft inside <think>.
        recovered = _find_prompt_in_text(reasoning)
        if recovered and len(recovered) > len(after_think):
            return reasoning, recovered
        # Fall back to whatever we have after </think>
        if after_think:
            return reasoning, after_think

    # ── Scenario 3: unclosed <think> (max tokens hit during reasoning) ──
    if '<think>' in text and '</think>' not in text:
        inner = text[text.index('<think>') + 7:].strip()
        recovered = _find_prompt_in_text(inner)
        if recovered:
            reasoning = inner[:inner.index(recovered)].strip() if recovered in inner else inner
            return reasoning, recovered
        # Last resort: treat everything after <think> as content
        return "", inner

    # ── No <think> tags at all ──────────────────────────────────────────
    # Try common separator patterns
    for marker in ["Final answer:", "Final description:", "DramaBox prompt:",
                    "Here is the prompt:",
                    "Here's the DramaBox", "Here is the DramaBox"]:
        if marker.lower() in text.lower():
            idx = text.lower().index(marker.lower())
            reasoning = text[:idx].strip()
            final = text[idx + len(marker):].strip()
            return reasoning, final

    # If no clear separation, treat last paragraph as final
    paragraphs = text.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1])
        final = paragraphs[-1]
        return reasoning, final

    return "", text


def _find_prompt_in_text(text: str) -> str | None:
    """Try to locate a DramaBox-style prompt buried inside reasoning text.

    Looks for the characteristic opening pattern: a speaker description
    starting with an age/gender phrase, or a stage direction followed by
    quoted dialogue.
    """
    _GENDER = r'(?:female|male|woman|man|girl|boy)'
    _MIN_LEN = 60  # Minimum length for a recovered prompt to be useful

    # Pattern: "A <age>-year-old <word>..." — typical DramaBox speaker desc
    m = re.search(
        rf'((?:A|An)\s+\d{{1,3}}-year-old\s+\w+\b.*)',
        text, re.DOTALL,
    )
    if m:
        candidate = m.group(1).strip()
        if len(candidate) > _MIN_LEN:
            return candidate

    # Pattern: "A <adjective> ... <gender> voice/speaker..."
    # Use non-greedy .*? to avoid consuming the gender word
    m = re.search(
        rf'((?:A|An)\s+(?:young|middle|elderly|adult|teenage)\b.*?'
        rf'{_GENDER}\s+(?:voice|speaker)\b.*)',
        text, re.DOTALL,
    )
    if m:
        candidate = m.group(1).strip()
        if len(candidate) > _MIN_LEN:
            return candidate

    # Pattern: "<N>-year-old ..." (without leading A/An)
    m = re.search(
        rf'(\d{{1,3}}-year-old\s+\w+\b.*)',
        text, re.DOTALL,
    )
    if m:
        candidate = m.group(1).strip()
        if len(candidate) > _MIN_LEN:
            return candidate

    # Pattern: starts with a stage direction then dialogue
    m = re.search(r'(\([a-z][^)]{2,40}\)\s*".*)', text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) > _MIN_LEN:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Scan for work items
# ---------------------------------------------------------------------------
def scan_for_pending(tars_dir: Path, limit: int = 0) -> list[dict]:
    """Scan postprocess output tars for annotation JSONs needing MOSS refinement.

    Looks inside the staging directories (before tar+cleanup) or extracts from
    the output tars if staging is gone. Returns list of work items.
    """
    items = []

    # Strategy 1: Check staging directories (if postprocess hasn't cleaned up)
    staging_root = POSTPROCESS_OUTPUT / "staging"
    if staging_root.exists():
        for staging_dir in sorted(staging_root.iterdir()):
            if not staging_dir.is_dir():
                continue
            for json_path in sorted(staging_dir.glob("*.json")):
                item = _check_annotation(json_path)
                if item:
                    items.append(item)
                    if limit and len(items) >= limit:
                        return items

    # Strategy 2: Check extracted tars in local output/tars/
    tars_root = POSTPROCESS_OUTPUT / "tars"
    if tars_root.exists() and not items:
        extract_base = POSTPROCESS_OUTPUT / "moss_extracted"
        extract_base.mkdir(parents=True, exist_ok=True)
        for tar_path in sorted(tars_root.glob("*.tar")):
            tar_extract = extract_base / tar_path.stem
            if not tar_extract.exists():
                log.info(f"Extracting {tar_path.name} for MOSS scan...")
                tar_extract.mkdir(parents=True, exist_ok=True)
                import tarfile
                with tarfile.open(tar_path) as tf:
                    tf.extractall(tar_extract, filter="data")
            for json_path in sorted(tar_extract.glob("*.json")):
                item = _check_annotation(json_path)
                if item:
                    items.append(item)
                    if limit and len(items) >= limit:
                        return items

    # Strategy 3: Download annotated tars from HF
    if not items:
        items = _scan_hf_tars(limit)

    # Coin flip: 50% inline directions, 50% literary directions
    for item in items:
        item["direction_style"] = random.choice(["inline", "literary"])

    n_inline = sum(1 for it in items if it["direction_style"] == "inline")
    log.info(f"Found {len(items)} pending MOSS refinement items "
             f"({n_inline} inline, {len(items) - n_inline} literary)")
    return items


DST_REPO = "laion/dramabox-voice-acting-data-annotated"


def _scan_hf_tars(limit: int = 0) -> list[dict]:
    """Download annotated tars from HF and scan for pending refinement."""
    items = []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import tarfile
    except ImportError:
        log.warning("huggingface_hub not available, skipping HF scan")
        return items

    api = HfApi()
    files = api.list_repo_files(DST_REPO, repo_type="dataset")
    tar_names = sorted([f for f in files if f.endswith(".tar")])
    log.info(f"Found {len(tar_names)} annotated tars on HF")

    # Load state to track which tars we've already scanned
    state = load_state()
    scanned_tars = set(state.get("scanned_tars", []))

    extract_base = POSTPROCESS_OUTPUT / "moss_extracted"
    extract_base.mkdir(parents=True, exist_ok=True)

    for tar_name in tar_names:
        if tar_name in scanned_tars:
            continue

        tar_stem = Path(tar_name).stem
        tar_extract = extract_base / tar_stem

        if not tar_extract.exists():
            log.info(f"Downloading {tar_name} from HF...")
            download_dir = POSTPROCESS_OUTPUT / "moss_download"
            download_dir.mkdir(parents=True, exist_ok=True)
            try:
                tar_path = hf_hub_download(
                    repo_id=DST_REPO, filename=tar_name,
                    repo_type="dataset", local_dir=str(download_dir),
                )
                tar_extract.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_path) as tf:
                    tf.extractall(tar_extract, filter="data")
                # Remove downloaded tar to save disk
                try:
                    os.unlink(tar_path)
                except OSError:
                    pass
            except Exception as e:
                log.error(f"Failed to download {tar_name}: {e}")
                continue

        for json_path in sorted(tar_extract.glob("*.json")):
            item = _check_annotation(json_path)
            if item:
                items.append(item)
                if limit and len(items) >= limit:
                    return items

    return items


def _check_annotation(json_path: Path) -> dict | None:
    """Check if annotation JSON needs MOSS refinement. Returns work item or None."""
    try:
        with open(json_path) as f:
            ann = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    # Skip if already refined
    if ann.get("moss_refined_prompt_full") is not None:
        return None

    base_name = json_path.stem
    parent_dir = json_path.parent

    # Detect single-prompt vs two-scene
    is_single = ann.get("split_point_sec") is None or ann.get("pathway") == "single_prompt"

    # Find corresponding audio files
    full_mp3 = parent_dir / f"{base_name}_full.mp3"
    part1_mp3 = parent_dir / f"{base_name}_part1.mp3"
    part2_mp3 = parent_dir / f"{base_name}_part2.mp3"

    if is_single:
        # Single-prompt: only need full audio
        if not full_mp3.exists():
            return None
    else:
        # Two-scene: need at least part1 + part2 (full can be generated from them)
        if not part1_mp3.exists() or not part2_mp3.exists():
            return None

    return {
        "json_path": str(json_path),
        "full_mp3": str(full_mp3) if full_mp3.exists() else None,
        "part1_mp3": str(part1_mp3) if part1_mp3.exists() else None,
        "part2_mp3": str(part2_mp3) if part2_mp3.exists() else None,
        "base_name": base_name,
        "is_single_prompt": is_single,
        "original_prompt": ann.get("original_prompt", ""),
        "scene1_transcript": ann.get("scene1_transcript", ""),
        "scene2_transcript": ann.get("scene2_transcript", ""),
        "asr_transcript": ann.get("asr_transcript", ""),
        "bude_caption_full": ann.get("bude_caption_full", ""),
        "bude_caption_part1": ann.get("bude_caption_part1", ""),
        "bude_caption_part2": ann.get("bude_caption_part2", ""),
    }


# ---------------------------------------------------------------------------
# Worker: loads MOSS and processes items
# ---------------------------------------------------------------------------
def run_worker(gpu_id: int, work_file: str):
    """Process MOSS refinement items on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch

    with open(work_file) as f:
        work_items = json.load(f)

    total = len(work_items)
    log.info(f"[GPU {gpu_id}] {total} items to process. Loading MOSS 4-bit...")
    t0 = time.time()

    # --- Load MOSS-Audio-8B-Thinking in 4-bit ---
    # Add both MOSS_DIR (for 'from src.xxx' imports) and MOSS_DIR/src (for direct imports)
    sys.path.insert(0, MOSS_DIR)
    sys.path.insert(0, os.path.join(MOSS_DIR, "src"))
    from modeling_moss_audio import MossAudioModel
    from processing_moss_audio import MossAudioProcessor, MelConfig
    from audio_io import load_audio
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = MossAudioModel.from_pretrained(
        MOSS_WEIGHTS,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    processor = MossAudioProcessor.from_pretrained(
        MOSS_WEIGHTS,
        trust_remote_code=True,
        enable_time_marker=False,  # Disable timestamp generation
    )

    mel_config = MelConfig()
    load_time = time.time() - t0
    log.info(f"[GPU {gpu_id}] MOSS loaded in {load_time:.1f}s")

    def run_moss(prompt_text: str, audio_path: str) -> tuple[str, str, float, int]:
        """Run a single MOSS inference. Returns (reasoning, output, elapsed, n_tokens)."""
        raw_audio = load_audio(audio_path, sample_rate=mel_config.mel_sr)
        inputs = processor(text=prompt_text, audios=[raw_audio], return_tensors="pt")
        inputs = inputs.to(model.device)

        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(model.dtype)

        audio_input_mask = inputs["input_ids"] == processor.audio_token_id
        inputs["audio_input_mask"] = audio_input_mask

        input_len = inputs["input_ids"].shape[1]

        t_start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                num_beams=1,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=50,
                use_cache=True,
            )
        elapsed = time.time() - t_start

        new_tokens = generated_ids[0, input_len:]
        output_text = processor.decode(new_tokens, skip_special_tokens=True)
        n_tokens = len(new_tokens)

        reasoning, final = _split_reasoning(output_text)
        final = _clean_truncation_artifacts(final)
        return reasoning, final, elapsed, n_tokens

    # --- Process each item ---
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0

    for idx, item in enumerate(work_items):
        base_name = item["base_name"]
        json_path = item["json_path"]

        try:
            log.info(f"[GPU {gpu_id}] [{idx+1}/{total}] Processing {base_name}")

            original_prompt = item["original_prompt"]
            scene1_tr = item.get("scene1_transcript", "")
            scene2_tr = item.get("scene2_transcript", "")
            asr_tr = item.get("asr_transcript", "")
            is_single = item.get("is_single_prompt", False)

            results = {}

            # Direction style: coin flip (50% inline, 50% literary)
            dir_style = item.get("direction_style", "literary")
            if dir_style == "inline":
                dir_instruction = INLINE_DIRECTION_INSTRUCTION
            else:
                dir_instruction = LITERARY_DIRECTION_INSTRUCTION
            log.info(f"  Direction style: {dir_style}, single_prompt: {is_single}")

            # BUD-E-Whisper captions (may be empty if not yet generated)
            bude_full = item.get("bude_caption_full", "")
            bude_p1 = item.get("bude_caption_part1", "")
            bude_p2 = item.get("bude_caption_part2", "")

            if is_single:
                # ── Single-prompt pathway: 1 MOSS pass ──────────────────
                full_mp3_path = item["full_mp3"]
                if not full_mp3_path or not os.path.exists(full_mp3_path):
                    raise FileNotFoundError(f"No full MP3 for single-prompt item: {base_name}")

                prompt_single = MOSS_REFINE_PROMPT_SINGLE.format(
                    original_prompt=original_prompt,
                    asr_transcript=asr_tr,
                    direction_style_instruction=dir_instruction,
                    bude_caption=bude_full or "(not available)",
                )
                reasoning_full, refined_full, elapsed_full, ntok_full = run_moss(
                    prompt_single, full_mp3_path
                )
                results["moss_refined_prompt_full"] = refined_full
                results["moss_reasoning_full"] = reasoning_full
                results["moss_refined_prompt_part1"] = None
                results["moss_refined_prompt_part2"] = None
                results["moss_reasoning_part1"] = None
                results["moss_reasoning_part2"] = None
                results["moss_direction_style"] = dir_style
                log.info(f"  Full (single): {ntok_full} tokens in {elapsed_full:.1f}s")
                torch.cuda.empty_cache()

                total_elapsed = elapsed_full
                total_tok = ntok_full
            else:
                # ── Two-scene pathway: 3 MOSS passes ────────────────────
                # Generate full audio path if not present (old postprocess output)
                full_mp3_path = item["full_mp3"]
                full_mp3_temp = None
                if not full_mp3_path or not os.path.exists(full_mp3_path):
                    full_mp3_temp = item["part1_mp3"].replace("_part1.mp3", "_full_tmp.mp3")
                    log.info(f"  Concatenating part1+part2 → full (no _full.mp3 found)")
                    subprocess.run(
                        ["ffmpeg", "-y",
                         "-i", item["part1_mp3"], "-i", item["part2_mp3"],
                         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
                         "-ac", "1", "-ar", "48000", "-b:a", "256k",
                         full_mp3_temp],
                        capture_output=True, check=True,
                    )
                    full_mp3_path = full_mp3_temp

                # Pass 1: Full audio refinement
                prompt_full = MOSS_REFINE_PROMPT_FULL.format(
                    original_prompt=original_prompt,
                    scene1_transcript=scene1_tr or asr_tr,
                    scene2_transcript=scene2_tr or "",
                    direction_style_instruction=dir_instruction,
                    bude_caption=bude_full or "(not available)",
                )
                reasoning_full, refined_full, elapsed_full, ntok_full = run_moss(
                    prompt_full, full_mp3_path
                )
                results["moss_refined_prompt_full"] = refined_full
                results["moss_reasoning_full"] = reasoning_full
                results["moss_direction_style"] = dir_style
                log.info(f"  Full: {ntok_full} tokens in {elapsed_full:.1f}s")
                torch.cuda.empty_cache()

                # Pass 2: Part 1 refinement
                prompt_p1 = MOSS_REFINE_PROMPT_PART.format(
                    part_label="Scene 1 (Part 1)",
                    original_prompt=original_prompt,
                    part_transcript=scene1_tr or asr_tr,
                    direction_style_instruction=dir_instruction,
                    bude_caption=bude_p1 or "(not available)",
                )
                reasoning_p1, refined_p1, elapsed_p1, ntok_p1 = run_moss(
                    prompt_p1, item["part1_mp3"]
                )
                results["moss_refined_prompt_part1"] = refined_p1
                results["moss_reasoning_part1"] = reasoning_p1
                log.info(f"  Part1: {ntok_p1} tokens in {elapsed_p1:.1f}s")
                torch.cuda.empty_cache()

                # Pass 3: Part 2 refinement
                prompt_p2 = MOSS_REFINE_PROMPT_PART.format(
                    part_label="Scene 2 (Part 2)",
                    original_prompt=original_prompt,
                    part_transcript=scene2_tr or "",
                    direction_style_instruction=dir_instruction,
                    bude_caption=bude_p2 or "(not available)",
                )
                reasoning_p2, refined_p2, elapsed_p2, ntok_p2 = run_moss(
                    prompt_p2, item["part2_mp3"]
                )
                results["moss_refined_prompt_part2"] = refined_p2
                results["moss_reasoning_part2"] = reasoning_p2
                log.info(f"  Part2: {ntok_p2} tokens in {elapsed_p2:.1f}s")
                torch.cuda.empty_cache()

                # Cleanup temp full MP3 if we created one
                if full_mp3_temp and os.path.exists(full_mp3_temp):
                    try:
                        os.unlink(full_mp3_temp)
                    except OSError:
                        pass

                total_elapsed = elapsed_full + elapsed_p1 + elapsed_p2
                total_tok = ntok_full + ntok_p1 + ntok_p2

            # Update annotation JSON in-place
            with open(json_path) as f:
                ann = json.load(f)

            ann.update(results)

            tmp_json = json_path + ".tmp"
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=1)
            os.replace(tmp_json, json_path)

            processed += 1
            log.info(
                f"[GPU {gpu_id}] [{idx+1}/{total}] {base_name} DONE "
                f"({total_tok} tokens, {total_elapsed:.1f}s)"
            )

        except torch.cuda.OutOfMemoryError:
            log.warning(f"[GPU {gpu_id}] OOM on {base_name}, clearing cache and skipping")
            torch.cuda.empty_cache()
            errors += 1

        except Exception as e:
            log.error(f"[GPU {gpu_id}] ERROR on {base_name}: {e}")
            traceback.print_exc()
            errors += 1

        # Clear GPU cache after every item
        torch.cuda.empty_cache()

        # Write progress
        if (idx + 1) % 2 == 0 or idx == total - 1:
            elapsed_total = time.time() - t0 - load_time
            rate = processed / max(elapsed_total, 1)
            remaining = total - idx - 1
            eta_min = remaining / max(rate, 0.001) / 60

            progress_data = {
                "gpu_id": gpu_id,
                "processed": processed,
                "errors": errors,
                "total": total,
                "current": idx + 1,
                "rate_per_min": round(rate * 60, 2),
                "eta_min": round(eta_min, 1),
                "timestamp": time.time(),
            }
            progress_path = PROGRESS_DIR / f"gpu_{gpu_id}.json"
            tmp_p = progress_path.with_suffix(".tmp")
            with open(tmp_p, "w") as f:
                json.dump(progress_data, f)
            tmp_p.rename(progress_path)

    # Final progress
    progress_data = {
        "gpu_id": gpu_id,
        "processed": processed,
        "errors": errors,
        "total": total,
        "current": total,
        "done": True,
        "timestamp": time.time(),
    }
    progress_path = PROGRESS_DIR / f"gpu_{gpu_id}.json"
    tmp_p = progress_path.with_suffix(".tmp")
    with open(tmp_p, "w") as f:
        json.dump(progress_data, f)
    tmp_p.rename(progress_path)

    log.info(f"[GPU {gpu_id}] DONE. processed={processed} errors={errors}")


# =========================================================================
# State Management
# =========================================================================

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "processed_count": 0,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


# =========================================================================
# Progress Monitor
# =========================================================================

MONITOR_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>MOSS Refinement Monitor</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; margin: 0; padding: 20px; }
.container { max-width: 800px; margin: 0 auto; }
h1 { color: #4fc3f7; font-size: 1.4em; border-bottom: 1px solid #333; padding-bottom: 10px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 15px 0; }
.stat-card { background: #1a1a2e; border-radius: 8px; padding: 12px; border: 1px solid #333; }
.stat-value { font-size: 1.6em; font-weight: bold; color: #4fc3f7; }
.stat-label { font-size: 0.8em; color: #888; margin-top: 4px; }
.progress-bar { width: 100%%; height: 28px; background: #1a1a2e; border-radius: 14px; overflow: hidden; margin: 12px 0; }
.progress-fill { height: 100%%; background: linear-gradient(90deg, #0277bd, #4fc3f7); display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85em; min-width: 60px; }
.gpu-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 12px 0; }
.gpu-card { background: #1a1a2e; border-radius: 6px; padding: 8px; text-align: center; border: 1px solid #333; }
.gpu-id { color: #4fc3f7; font-weight: bold; }
.ts { color: #555; font-size: 0.75em; text-align: right; margin-top: 15px; }
</style>
</head><body>
<div class="container">
<h1>MOSS Prompt Refinement Pipeline</h1>
<div id="c">Loading...</div>
</div>
<script>
async function r(){try{const d=(await(await fetch('/api/progress')).json());const p=d.progress_pct;
document.getElementById('c').innerHTML=`
<div class="progress-bar"><div class="progress-fill" style="width:${Math.max(p,1)}%%">${p}%%</div></div>
<div class="stats-grid">
<div class="stat-card"><div class="stat-value">${d.processed}/${d.total}</div><div class="stat-label">Processed</div></div>
<div class="stat-card"><div class="stat-value">${d.errors}</div><div class="stat-label">Errors</div></div>
<div class="stat-card"><div class="stat-value" style="color:#ffd54f">${d.eta||'—'}</div><div class="stat-label">ETA</div></div>
<div class="stat-card"><div class="stat-value">${d.rate_per_hour}/h</div><div class="stat-label">Items/Hour</div></div>
<div class="stat-card"><div class="stat-value">${d.elapsed_h}h</div><div class="stat-label">Elapsed</div></div>
</div>
<h3 style="color:#4fc3f7">GPUs</h3>
<div class="gpu-grid">${Object.entries(d.gpus).map(([id,g])=>
'<div class="gpu-card"><div class="gpu-id">GPU '+id+'</div><div>'+g.processed+'/'+g.total+'</div><div style="color:#888;font-size:0.8em">'+(g.rate_per_min||0)+'/min</div></div>'
).join('')}</div>
<div class="ts">Updated: ${d.timestamp}</div>`;
}catch(e){console.error(e);}}
r(); setInterval(r,5000);
</script></body></html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/progress":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = self.server.get_progress()
            self.wfile.write(json.dumps(data, indent=2).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(MONITOR_HTML.encode())

    def log_message(self, fmt, *args):
        pass


def start_monitor(get_progress_fn, port=8769):
    server = HTTPServer(("0.0.0.0", port), MonitorHandler)
    server.get_progress = get_progress_fn
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info(f"Monitor: http://0.0.0.0:{port}")

    # Cloudflare tunnel
    try:
        tunnel_proc = subprocess.Popen(
            ["/tmp/cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        def read_url():
            for line in iter(tunnel_proc.stdout.readline, ""):
                if "trycloudflare.com" in line:
                    m = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if m:
                        url = m.group(0)
                        log.info(f"MONITOR URL: {url}")
                        (POSTPROCESS_OUTPUT / "moss_monitor_url.txt").write_text(url + "\n")
                        return

        threading.Thread(target=read_url, daemon=True).start()
    except FileNotFoundError:
        log.warning("cloudflared not found, skipping tunnel")
        tunnel_proc = None

    return server, tunnel_proc


# =========================================================================
# Coordinator
# =========================================================================

def get_available_gpus():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [int(l.strip()) for l in r.stdout.strip().split("\n") if l.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def coordinator_main(num_gpus: int, test_mode: bool = False):
    """Main coordinator: scan for pending items, distribute to workers."""
    gpu_ids = get_available_gpus()[:num_gpus]
    num_workers = len(gpu_ids)

    POSTPROCESS_OUTPUT.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    (POSTPROCESS_OUTPUT / "logs").mkdir(parents=True, exist_ok=True)

    state = load_state()
    start_time = time.time()

    # Scan for pending work
    limit = 10 if test_mode else 0
    items = scan_for_pending(POSTPROCESS_OUTPUT / "tars", limit=limit)

    if not items:
        log.info("No pending MOSS refinement items found.")
        return

    if test_mode:
        gpu_ids = gpu_ids[:1]
        num_workers = 1
        log.info(f"TEST MODE: {len(items)} items, 1 GPU")

    total_items = len(items)
    log.info(f"Total items to refine: {total_items} across {num_workers} GPUs")

    # Progress function for monitor
    def get_progress():
        workers = {}
        total_proc = 0
        total_errs = 0
        if PROGRESS_DIR.exists():
            for p in PROGRESS_DIR.glob("gpu_*.json"):
                try:
                    with open(p) as f:
                        wd = json.load(f)
                    gid = str(wd.get("gpu_id", p.stem))
                    workers[gid] = wd
                    total_proc += wd.get("processed", 0)
                    total_errs += wd.get("errors", 0)
                except (json.JSONDecodeError, IOError):
                    pass

        elapsed = time.time() - start_time
        pct = round(total_proc / max(total_items, 1) * 100, 1)
        rate = total_proc / max(elapsed, 1)

        if rate > 0 and elapsed > 60:
            remaining = total_items - total_proc
            eta_h = remaining / rate / 3600
            eta_str = f"{eta_h:.1f}h"
        else:
            eta_str = "calculating..."

        return {
            "processed": total_proc,
            "total": total_items,
            "errors": total_errs,
            "progress_pct": pct,
            "rate_per_hour": round(rate * 3600),
            "eta": eta_str,
            "elapsed_h": round(elapsed / 3600, 2),
            "gpus": workers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Start monitor
    monitor, tunnel = start_monitor(get_progress, port=8769)

    # Clear progress files
    for p in PROGRESS_DIR.glob("gpu_*.json"):
        p.unlink()

    # Distribute round-robin
    shards = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        shards[i % num_workers].append(item)

    # Write work files
    work_dir = POSTPROCESS_OUTPUT / "moss_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_files = []
    for wid, shard in enumerate(shards):
        wf = work_dir / f"work_{gpu_ids[wid]}.json"
        with open(wf, "w") as f:
            json.dump(shard, f)
        work_files.append(wf)
        log.info(f"  Worker GPU {gpu_ids[wid]}: {len(shard)} items")

    # Launch worker subprocesses
    script = str(Path(__file__).resolve())
    processes = []
    log_handles = []

    for wid, gid in enumerate(gpu_ids):
        if not shards[wid]:
            continue

        log_path = POSTPROCESS_OUTPUT / "logs" / f"moss_worker_gpu{gid}.log"
        log_f = open(log_path, "w")
        log_handles.append(log_f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["LD_LIBRARY_PATH"] = ""
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Use MOSS venv (transformers 4.57.x) if available, else system python
        worker_python = MOSS_VENV_PYTHON if os.path.isfile(MOSS_VENV_PYTHON) else sys.executable
        cmd = [
            worker_python, "-u", script,
            "--worker",
            "--gpu", str(gid),
            "--work-file", str(work_files[wid]),
        ]
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        processes.append((wid, gid, proc))
        log.info(f"  Launched worker GPU {gid} (PID {proc.pid})")

    # Wait for all workers to finish
    while True:
        all_done = all(p.poll() is not None for _, _, p in processes)
        if all_done:
            break

        prog = get_progress()
        log.info(
            f"  MOSS refine: {prog['processed']}/{prog['total']} "
            f"({prog['progress_pct']}%) errors={prog['errors']} "
            f"rate={prog['rate_per_hour']}/h"
        )
        time.sleep(30)

    for f in log_handles:
        f.close()

    # Check exit codes
    for wid, gid, proc in processes:
        rc = proc.wait()
        if rc != 0:
            log.warning(f"  Worker GPU {gid} exit code {rc}")

    # Cleanup work files
    for wf in work_files:
        wf.unlink(missing_ok=True)

    # Update state
    prog = get_progress()
    state["processed_count"] = state.get("processed_count", 0) + prog["processed"]
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    log.info(f"\nMOSS REFINEMENT DONE. {prog['processed']}/{total_items} items refined.")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DramaBox MOSS Prompt Refinement")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--test", action="store_true", help="Process 10 items with 1 GPU")

    # Worker mode
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--work-file", type=str)

    args = parser.parse_args()

    if args.worker:
        run_worker(args.gpu, args.work_file)
        return

    coordinator_main(args.num_gpus, test_mode=args.test)


if __name__ == "__main__":
    main()
