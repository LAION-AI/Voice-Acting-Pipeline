#!/usr/bin/env python3
"""
DramaBox End-to-End Voice Acting Data Generation Pipeline.

Unified pipeline: prompt generation -> TTS -> postprocess -> annotated WebDataset TAR.
Configurable via YAML config files. Supports 1-8 GPUs with automatic model loading.

Usage:
    # Full run with config
    python scripts/dramabox_e2e.py --config configs/e2e_local.yaml

    # Test mode (2 prompts, 2 seeds, 1 GPU)
    python scripts/dramabox_e2e.py --config configs/e2e_local.yaml --test

    # Override GPUs and seeds
    python scripts/dramabox_e2e.py --config configs/e2e_local.yaml --gpus 0,1,2,3 --seeds 10

    # Ingest mode (use pre-generated prompts)
    python scripts/dramabox_e2e.py --config configs/e2e_local.yaml --mode ingest
"""

# =============================================================================
# Section 1: Imports and LD_LIBRARY_PATH fix
# =============================================================================
import os
import sys

# Filter out conda ml-general paths that break native libraries
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _ld:
    _filtered = [p for p in _ld.split(":") if "ml-general" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_filtered)

import argparse
import base64
import copy
import hashlib
import io
import json
import logging
import math
import re
import shutil
import signal
import socket
import struct
import subprocess
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# =============================================================================
# Section 2: Config Loading
# =============================================================================

def load_yaml_config(config_path: str) -> dict:
    """Parse YAML config file and resolve relative paths."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)

    # Resolve paths relative to Voice-Acting-Pipeline root
    vap_root = config_path.parent.parent
    config["_vap_root"] = str(vap_root)

    # Resolve data_dir
    data_dir = config.get("data_dir", "data")
    if not os.path.isabs(data_dir):
        data_dir = str(vap_root / data_dir)
    config["data_dir"] = data_dir

    # Resolve output_dir
    output_dir = config.get("output", {}).get("output_dir", "./e2e_output")
    if not os.path.isabs(output_dir):
        output_dir = str(vap_root / output_dir)
    config.setdefault("output", {})["output_dir"] = output_dir

    # Resolve GGUF model path
    pg = config.get("prompt_generation", {})
    gguf_path = pg.get("gguf_model_path", "")
    if gguf_path and not os.path.isabs(gguf_path):
        pg["gguf_model_path"] = str(vap_root / gguf_path)

    # Resolve prompt ingestion file paths
    pi = config.get("prompt_ingestion", {})
    if pi.get("prompt_files"):
        resolved = []
        for pf in pi["prompt_files"]:
            if not os.path.isabs(pf):
                resolved.append(str(Path(data_dir) / pf))
            else:
                resolved.append(pf)
        pi["prompt_files"] = resolved

    return config


def validate_config(config: dict) -> list:
    """Validate config and return list of warnings."""
    warnings = []
    mode = config.get("mode", "generate")
    if mode not in ("generate", "ingest"):
        warnings.append(f"Invalid mode '{mode}', must be 'generate' or 'ingest'")

    if mode == "generate":
        gguf = config.get("prompt_generation", {}).get("gguf_model_path", "")
        if not gguf or not Path(gguf).exists():
            warnings.append(f"GGUF model not found: {gguf}")

    tts = config.get("tts", {})
    dbox_dir = tts.get("dramabox_dir", "")
    if not dbox_dir or not Path(dbox_dir).exists():
        warnings.append(f"DramaBox directory not found: {dbox_dir}")

    storage_mode = config.get("storage", {}).get("mode", "local")
    if storage_mode != "local":
        hf_token = config.get("huggingface", {}).get("token", "")
        env_token = os.environ.get("HF_TOKEN", "")
        if (not hf_token or hf_token == "YOUR_HF_TOKEN_HERE") and not env_token:
            warnings.append("HuggingFace token not set (config or HF_TOKEN env var)")

    return warnings


# =============================================================================
# Section 3: GPU Detection
# =============================================================================

def detect_gpus() -> list:
    """Detect available GPU indices via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        pass
    return [0]


def get_gpu_vram_mb(gpu_id: int) -> int:
    """Get total VRAM in MB for a specific GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
             f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def find_available_port(start_port: int) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 11):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start_port


# =============================================================================
# Section 4: Prompt Generation (GGUF)
# =============================================================================

def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning tags from model output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def load_gguf_llm(model_path: str, n_ctx: int = 8192):
    """Load a GGUF model via llama-cpp-python."""
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)


def generate_prompt_gguf(llm, system_prompt: str, user_prompt: str, config: dict) -> tuple:
    """Generate one prompt using GGUF LLM. Returns (text, elapsed, prompt_tokens, comp_tokens)."""
    pg = config.get("prompt_generation", {})
    max_tokens = pg.get("max_tokens", 4096)
    temperature = pg.get("temperature", 0.85)
    top_p = pg.get("top_p", 0.92)

    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.time() - t0
    text = response["choices"][0]["message"]["content"]
    text = strip_think_tags(text)
    usage = response.get("usage", {})
    return text, elapsed, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def load_taxonomies(config: dict) -> dict:
    """Load all taxonomy data needed for prompt generation."""
    vap_root = config.get("_vap_root", "")
    if vap_root:
        sys.path.insert(0, vap_root)

    from dramabox.taxonomy import (
        parse_voicenet_html, load_emonet, load_archetypes,
        load_vocal_bursts, format_vocal_bursts_block,
    )

    data_dir = Path(config["data_dir"])

    dims = parse_voicenet_html(data_dir / "voicenet_ext_taxonomy.html")
    mandatory_codes = {"TEMP", "GEND", "AGEV"}
    mandatory_dims = [d for d in dims if d["code"] in mandatory_codes]
    optional_dims = [d for d in dims if d["code"] not in mandatory_codes]
    temp_dim = next(d for d in dims if d["code"] == "TEMP")
    arou_dim = next(d for d in dims if d["code"] == "AROU")

    emonet = load_emonet(data_dir / "emonet_taxonomy.json")
    emotion_categories = list(emonet.keys())

    archetypes = load_archetypes(data_dir / "archetypes.json")

    vb_taxonomy = load_vocal_bursts(data_dir / "vocal_bursts_taxonomy.json")
    vb_block = format_vocal_bursts_block(vb_taxonomy)

    with open(data_dir / "all_acting_challenges.json") as f:
        all_challenges = json.load(f)
    with open(data_dir / "acting_challenges_extreme_physical.json") as f:
        extreme_challenges = json.load(f)
    with open(data_dir / "acting_challenges_situation_inspired.json") as f:
        situation_challenges = json.load(f)

    return {
        "all_dims": dims,
        "mandatory_dims": mandatory_dims,
        "optional_dims": optional_dims,
        "temp_dim": temp_dim,
        "arou_dim": arou_dim,
        "emotion_categories": emotion_categories,
        "archetypes": archetypes,
        "vb_block": vb_block,
        "vb_taxonomy": vb_taxonomy,
        "all_challenges": all_challenges,
        "extreme_challenges": extreme_challenges,
        "situation_challenges": situation_challenges,
    }


# --- Sampling configs --------------------------------------------------------

EN_CONFIG = {
    "sampling": {
        "emotions_min": 1, "emotions_max": 3,
        "random_dims_count": 5,
        "tempo_bias_threshold": 3, "tempo_bias_weight": 1.5,
        "word_count_min": 20, "word_count_max": 30,
        "mandatory_words_count": 3,
        "flow_style_distribution": {"scattered": 0.05, "flowing": 0.55, "mixed": 0.40},
        "emotion_alignment_distribution": {"congruent": 0.30, "neutral": 0.40, "counter-emotional": 0.30},
        "direction_style_distribution": {"literary": 0.50, "tag": 0.50},
        "vocal_bursts_probability": 0.50,
    },
    "_active_languages": ["English"],
    "_language_accents": {"English": ["Standard American", "British RP", "Australian", ""]},
}

DE_CONFIG = {
    "sampling": {
        "emotions_min": 1, "emotions_max": 3,
        "random_dims_count": 5,
        "tempo_bias_threshold": 3, "tempo_bias_weight": 1.5,
        "word_count_min": 20, "word_count_max": 30,
        "mandatory_words_count": 0,
        "flow_style_distribution": {"scattered": 0.05, "flowing": 0.55, "mixed": 0.40},
        "emotion_alignment_distribution": {"congruent": 0.30, "neutral": 0.40, "counter-emotional": 0.30},
        "direction_style_distribution": {"literary": 0.50, "tag": 0.50},
        "vocal_bursts_probability": 0.0,
    },
    "_active_languages": ["German"],
    "_language_accents": {"German": []},
}


# --- System prompts ----------------------------------------------------------

CC_SYSTEM_PROMPT = """\
You write character-consistent two-scene voice performance prompts in DramaBox format for a single speaker.

CRITICAL RULES:
- It is ALWAYS one single person speaking the entire prompt. The same voice, the same actor, from start to finish. Never introduce a second speaker. Explicitly anchor identity: "the same voice", "the same speaker".
- NO markdown. No bold, no stars, no headers, no labels. Just plain text.
- Directions go in (parentheses). Spoken words go in "double quotes". Alternate between them roughly equally. Keep directions SHORT, 5-12 words each.
- The delivery must sound natural, realistic, genuine, spontaneous — like a real human in a real moment, not a stage performance.
- The actor performs with all the little micro-distractions someone in real life in a real situation would have — a natural, authentic sensuality and variance in tone, organically reacting to all the micro-distractions around them. Shifting weight, noticing a sound, losing a thought and finding it again. The performance breathes.
- TOTAL spoken dialogue (inside "double quotes") must be approximately 50 words — roughly 25 words before CUT TO: and 25 words after.
- Do NOT exceed 60 words of dialogue total. Do NOT go below 40 words.

STRUCTURE (write exactly like this, no labels, no headers):

A [age] [gender] with a [timbre/vocal quality], delivering this high-quality studio voice recording with no background noise.

The same voice is [1 sentence: emotional state for the first moment].

(short direction) "Spoken words." (short direction) "More spoken words."

CUT TO:

The same voice now [1 sentence: how the emotion has shifted dramatically].

(short direction) "Spoken words." (short direction) "More spoken words."

The performance across both moments should feel [1 sentence].

EMOTION CONTRAST: Maximize emotional distance between scenes. Polarity flips (joy to grief), arousal shifts (screaming to whisper), control shifts (composure to breakdown).

Output ONLY the raw prompt. Nothing else."""

CC_SYSTEM_PROMPT_DE = CC_SYSTEM_PROMPT + """

ALL spoken dialogue MUST be in German. But NEVER use umlauts (ö, ä, ü). Always substitute: oe for ö, ae for ä, ue for ü.
Directions (parentheses) and speaker descriptions remain in English."""

EXTREME_CC_SYSTEM_PROMPT = """\
You write character-consistent two-scene voice performance prompts in DramaBox format for a single speaker.

CRITICAL RULES:
- It is ALWAYS one single person speaking the entire prompt. The same voice, the same actor, from start to finish. Never introduce a second speaker.
- NO markdown. No bold, no stars, no headers, no labels. Just plain text.
- Directions go in (parentheses). Spoken words go in "double quotes". Alternate between them roughly equally. Keep directions SHORT, 5-12 words each.
- The delivery must sound natural, realistic, genuine, spontaneous.
- Scene 1 starts from a place of calm sensuality, heightened perceptiveness — the actor is present, soft, aware of their body and surroundings.
- Scene 2 (after CUT TO:) ERUPTS with the extreme physical sensation — the voice transforms, distorts, breaks under the physical assault. Words dissolve, crack, become raw vocalizations.
- TOTAL spoken dialogue (inside "double quotes") must be approximately 50 words — roughly 25 per scene.
- Do NOT exceed 60 words of dialogue total. Do NOT go below 40 words.
- NO sound effects. Only the actor's voice, breath, and vocalizations.

Output ONLY the raw prompt. Nothing else."""

GERMAN_INSTRUCTION = """\
LANGUAGE: German.
ALL spoken dialogue (inside "double quotes") MUST be in German.
CRITICAL: Do NOT use German umlauts (ö, ä, ü, Ö, Ä, Ü). Replace them: ö→oe, ä→ae, ü→ue.
Directions (in parentheses) and the speaker description MUST remain in English."""


# --- Pathway builders --------------------------------------------------------

def _build_voicenet(tax: dict, lang: str, fmt: str, config: dict) -> tuple:
    """Build a VoiceNet pathway prompt. Returns (system_prompt, user_prompt, info)."""
    vap_root = config.get("_vap_root", "")
    if vap_root:
        sys.path.insert(0, vap_root)

    from dramabox.sampling import sample_voicenet
    from dramabox.prompts import (
        SYSTEM_INSTRUCTION, FLOW_INSTRUCTIONS, ALIGNMENT_INSTRUCTIONS,
        DIRECTION_STYLE_INSTRUCTIONS, SUFFIX_GENUINE, SUFFIX_SPONTANEOUS,
        SUFFIX_QUALITY, build_full_prompt,
    )

    cfg = EN_CONFIG if lang == "English" else DE_CONFIG
    sample = sample_voicenet(
        tax["mandatory_dims"], tax["optional_dims"],
        tax["emotion_categories"], cfg,
    )

    if fmt == "cut_to":
        vb_section = f"\n{tax['vb_block']}\n" if sample["vocal_bursts_enabled"] else ""
        lang_line = GERMAN_INSTRUCTION if lang == "German" else "LANGUAGE: English."
        sys_prompt = CC_SYSTEM_PROMPT_DE if lang == "German" else CC_SYSTEM_PROMPT
        user_prompt = f"""\
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
{lang_line}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
Output ONLY the raw DramaBox prompt."""
    else:
        sys_prompt = SYSTEM_INSTRUCTION
        user_prompt = build_full_prompt(sample, tax["vb_block"])
        if lang == "German":
            user_prompt += f"\n\n{GERMAN_INSTRUCTION}"

    info = {k: str(v)[:120] for k, v in sample.items() if k in (
        "emotions", "attributes_clean", "flow_style", "emotion_alignment",
        "direction_style", "word_count_target",
    )}
    return sys_prompt, user_prompt, info


def _build_archetype(tax: dict, lang: str, fmt: str, config: dict) -> tuple:
    """Build an Archetype pathway prompt."""
    vap_root = config.get("_vap_root", "")
    if vap_root:
        sys.path.insert(0, vap_root)

    from dramabox.sampling import sample_archetype
    from dramabox.prompts import (
        SYSTEM_INSTRUCTION, SUFFIX_GENUINE, SUFFIX_SPONTANEOUS,
        SUFFIX_QUALITY, build_full_prompt,
    )

    cfg = EN_CONFIG if lang == "English" else DE_CONFIG
    sample = sample_archetype(
        tax["archetypes"], tax["temp_dim"], tax["arou_dim"],
        tax["emotion_categories"], cfg,
    )

    if fmt == "cut_to":
        lang_line = GERMAN_INSTRUCTION if lang == "German" else "LANGUAGE: English."
        sys_prompt = CC_SYSTEM_PROMPT_DE if lang == "German" else CC_SYSTEM_PROMPT
        user_prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition.

ARCHETYPE: {sample['_archetype']} (from genre: {sample['_genre']})
EMOTIONS for Scene 1: {sample['emotions']}
TEMPO: {sample['_tempo_desc']}
AROUSAL: {sample['_arousal_desc']}
(Scene 2 should shift to a sharply contrasting emotional state.)

Do NOT reproduce the archetype description literally. Use it as inspiration.

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
{lang_line}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
Output ONLY the raw DramaBox prompt."""
    else:
        sys_prompt = SYSTEM_INSTRUCTION
        user_prompt = build_full_prompt(sample, tax["vb_block"])
        if lang == "German":
            user_prompt += f"\n\n{GERMAN_INSTRUCTION}"

    info = {"genre": sample.get("_genre", ""), "archetype": str(sample.get("_archetype", ""))[:100],
            "emotions": sample.get("emotions", ""), "tempo": sample.get("_tempo_desc", ""),
            "arousal": sample.get("_arousal_desc", "")}
    return sys_prompt, user_prompt, info


def _build_acting_challenge(tax: dict, lang: str, fmt: str, config: dict) -> tuple:
    """Build an Acting Challenge pathway prompt."""
    import random as _rng
    vap_root = config.get("_vap_root", "")
    if vap_root:
        sys.path.insert(0, vap_root)

    from dramabox.prompts import (
        SYSTEM_INSTRUCTION, SUFFIX_GENUINE, SUFFIX_SPONTANEOUS, SUFFIX_QUALITY,
    )

    challenge = _rng.choice(tax["all_challenges"])

    if fmt == "cut_to":
        lang_line = GERMAN_INSTRUCTION if lang == "German" else "LANGUAGE: English."
        sys_prompt = CC_SYSTEM_PROMPT_DE if lang == "German" else CC_SYSTEM_PROMPT
        user_prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition based on this acting challenge:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}

The two scenes should capture two emotionally contrasting moments. Scene 1 establishes one emotional state; Scene 2 (after CUT TO:) shifts dramatically.

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
{lang_line}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
Choose an appropriate speaker (age, gender, timbre) that fits the challenge.
Output ONLY the raw DramaBox prompt."""
    else:
        lang_line = GERMAN_INSTRUCTION if lang == "German" else f"TARGET LANGUAGE for dialogue: {lang}."
        sys_prompt = SYSTEM_INSTRUCTION
        user_prompt = f"""\
Create a single DramaBox-format voice prompt based on this acting challenge:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}

{lang_line}
WORD COUNT for all spoken dialogue combined (inside "..."): approximately 40 words.

Structure:
1. Open with 1-3 English sentences describing the single speaker (age, gender, timbre, voice qualities). Include that this is a pristine, high-quality studio recording with no background noise.
2. Write the performance for ONE speaker: stage directions (English) + dialogue (in quotes).
3. Close with 1-2 sentences of final direction (English).
4. SINGLE SPEAKER only. No dialogue partners.
5. Output ONLY the raw DramaBox prompt string.""" + SUFFIX_GENUINE + SUFFIX_SPONTANEOUS + SUFFIX_QUALITY

    info = {"challenge_title": challenge.get("title", ""),
            "instruction": challenge.get("instruction", "")[:150]}
    return sys_prompt, user_prompt, info


def _build_situation(tax: dict, lang: str, fmt: str, config: dict) -> tuple:
    """Build a Situation pathway prompt."""
    import random as _rng
    vap_root = config.get("_vap_root", "")
    if vap_root:
        sys.path.insert(0, vap_root)

    from dramabox.prompts import (
        SYSTEM_INSTRUCTION, SUFFIX_GENUINE, SUFFIX_SPONTANEOUS, SUFFIX_QUALITY,
    )

    challenge = _rng.choice(tax["situation_challenges"])
    # SIT pathway supports multi-language
    languages = ["English", "French", "Spanish", "German"]
    language = _rng.choice(languages)
    sit_name = challenge.get("situation_name", "")
    sit_dim = challenge.get("situation_dim", "")
    emotions = challenge.get("emotions_sampled", "")

    if fmt == "cut_to":
        lang_instruction = (
            f"ALL spoken dialogue (inside double quotes) MUST be in {language}. "
            "Directions remain in English."
        ) if language != "English" else ""
        sys_prompt = CC_SYSTEM_PROMPT
        user_prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}

SITUATION: {sit_name} (Dimension: {sit_dim})
The speaker is physically IN this situation.

EMOTIONS for Scene 1: {emotions}
(Scene 2 shifts to a contrasting emotional state.)

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
LANGUAGE: {language}. {lang_instruction}
SINGLE SPEAKER throughout. Same voice, same person, two emotional moments.
Output ONLY the raw DramaBox prompt."""
    else:
        lang_instruction = f"ALL dialogue in {language}." if language != "English" else ""
        sys_prompt = SYSTEM_INSTRUCTION
        user_prompt = f"""\
Create a single DramaBox-format voice prompt based on this acting challenge:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}

SITUATION: {sit_name} (Dimension: {sit_dim})
The speaker is physically IN this situation — it naturally affects their voice, breathing, posture.

EMOTIONS: {emotions}
TARGET LANGUAGE for dialogue: {language}. {lang_instruction}
WORD COUNT for all spoken dialogue: approximately 40 words.

Structure:
1. Open with 1-3 English sentences describing the single speaker. Include studio recording quality note.
2. Write the performance for ONE speaker: stage directions (English) + dialogue (in quotes).
3. Close with 1-2 sentences of final direction.
4. SINGLE SPEAKER only. Output ONLY the raw DramaBox prompt string.""" + SUFFIX_GENUINE + SUFFIX_SPONTANEOUS + SUFFIX_QUALITY

    info = {"challenge_title": challenge.get("title", ""), "situation": f"{sit_name} ({sit_dim})",
            "language": language, "emotions": emotions}
    return sys_prompt, user_prompt, info


def _build_extreme_physical(tax: dict, lang: str, fmt: str, config: dict) -> tuple:
    """Build an Extreme Physical pathway prompt."""
    import random as _rng
    vap_root = config.get("_vap_root", "")
    if vap_root:
        sys.path.insert(0, vap_root)

    from dramabox.prompts import (
        SYSTEM_INSTRUCTION, SUFFIX_GENUINE, SUFFIX_SPONTANEOUS, SUFFIX_QUALITY,
    )

    challenge = _rng.choice(tax["extreme_challenges"])
    cat_name = challenge.get("category_name", "")
    subcat = challenge.get("subcategory", "")

    if fmt == "cut_to":
        lang_line = GERMAN_INSTRUCTION if lang == "German" else "LANGUAGE: English."
        sys_prompt = EXTREME_CC_SYSTEM_PROMPT
        if lang == "German":
            sys_prompt += "\n\nALL spoken dialogue MUST be in German. No umlauts — use oe/ae/ue. Directions in English."
        user_prompt = f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}
Category: {cat_name} — {subcat}

Scene 1: calm sensuality and heightened perceptiveness — the actor is soft, present, aware.
Scene 2 (after CUT TO:): the extreme physical sensation erupts — voice transforms, distorts, breaks.

TOTAL SPOKEN WORDS: approximately 50 (roughly 25 per scene).
{lang_line}
NO sound effects — only the actor's voice, breath, and vocalizations.
SINGLE SPEAKER throughout. Same voice, same person, two moments of brutal contrast.
Output ONLY the raw DramaBox prompt."""
    else:
        lang_line = GERMAN_INSTRUCTION if lang == "German" else f"TARGET LANGUAGE for dialogue: {lang}."
        sys_prompt = SYSTEM_INSTRUCTION
        user_prompt = f"""\
Create a single DramaBox-format voice prompt based on this extreme physical acting challenge:

ACTING CHALLENGE: {challenge['title']}
INSTRUCTION: {challenge['instruction']}
Category: {cat_name} — {subcat}

{lang_line}
WORD COUNT for all spoken dialogue: approximately 40 words.
NO sound effects — only the actor's voice, breath, and vocalizations.

Structure:
1. Open with 1-3 English sentences describing the single speaker. Include studio recording quality note.
2. Write the performance for ONE speaker: stage directions (English) + dialogue (in quotes).
3. Close with 1-2 sentences of final direction.
4. SINGLE SPEAKER only. Output ONLY the raw DramaBox prompt string.""" + SUFFIX_GENUINE + SUFFIX_SPONTANEOUS + SUFFIX_QUALITY

    info = {"challenge_title": challenge.get("title", ""), "category": f"{cat_name} — {subcat}"}
    return sys_prompt, user_prompt, info


PATHWAY_BUILDERS = {
    "voicenet": _build_voicenet,
    "archetype": _build_archetype,
    "acting_challenge": _build_acting_challenge,
    "situation": _build_situation,
    "extreme_physical": _build_extreme_physical,
}


def sample_and_build_prompt(pathway: str, language: str, fmt: str,
                            taxonomies: dict, config: dict) -> tuple:
    """Unified dispatcher: returns (system_prompt, user_prompt, sample_info)."""
    builder = PATHWAY_BUILDERS.get(pathway)
    if builder is None:
        raise ValueError(f"Unknown pathway: {pathway}")
    return builder(taxonomies, language, fmt, config)


# =============================================================================
# Section 5: Prompt Ingestion + German Umlaut Fix
# =============================================================================

# --- German umlaut correction -----------------------------------------------

_GERMAN_PROPER_NOUNS = {
    "koeln": "Köln", "muenchen": "München", "duesseldorf": "Düsseldorf",
    "nuernberg": "Nürnberg", "wuerzburg": "Würzburg", "saarbruecken": "Saarbrücken",
    "luebeck": "Lübeck", "goettingen": "Göttingen", "tuebingen": "Tübingen",
    "zuerich": "Zürich", "oesterreich": "Österreich",
}

_GERMAN_FALSE_POSITIVES = {
    "abenteuer", "teuer", "feuer", "ungeheuer", "geheuer", "steuer",
    "euer", "neuer", "neue", "neuen", "neuem", "neues",
    "freuen", "freund", "freundin", "treue", "reu", "reue",
    "museum", "petroleum", "mauern", "lauern", "dauern", "trauern",
    "bedauern", "schauern", "bauer", "bauern", "mauer",
    "hauer", "sauer", "auer", "kauen", "schauen", "brauen",
    "stauen", "tauen", "auen", "blauen", "grauen",
}

_GERMAN_DICT = None
_GERMAN_DICT_LOADED = False


def _load_german_dict():
    """Load German dictionary for umlaut correction."""
    global _GERMAN_DICT, _GERMAN_DICT_LOADED
    if _GERMAN_DICT_LOADED:
        return _GERMAN_DICT
    _GERMAN_DICT_LOADED = True
    dict_path = Path("/tmp/de_DE.dic")
    if not dict_path.exists():
        # Try to download
        try:
            subprocess.run(
                ["bash", "-c",
                 "apt-get download hunspell-de-de 2>/dev/null && "
                 "dpkg-deb -x hunspell-de-de*.deb /tmp/hunspell_de && "
                 "cp /tmp/hunspell_de/usr/share/hunspell/de_DE.dic /tmp/de_DE.dic 2>/dev/null"],
                capture_output=True, timeout=30,
            )
        except Exception:
            pass
    if dict_path.exists():
        words = set()
        for line in dict_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            w = line.split("/")[0].strip().lower()
            if w:
                words.add(w)
        _GERMAN_DICT = words
    return _GERMAN_DICT


def _should_replace_umlaut(word_lower: str, replacement: str, german_dict: set) -> bool:
    """Check if an umlaut replacement is valid."""
    if word_lower in _GERMAN_FALSE_POSITIVES:
        return False
    if german_dict and replacement.lower() in german_dict:
        return True
    # If no dictionary, apply common rules
    return True


def fix_german_quoted_text(text: str) -> str:
    """Fix German umlauts in quoted dialogue text.

    Converts oe->ö, ae->ä, ue->ü within "quoted" text segments,
    with false-positive protection.
    """
    german_dict = _load_german_dict()

    # Fix proper nouns everywhere
    for key, val in _GERMAN_PROPER_NOUNS.items():
        text = re.sub(re.escape(key), val, text, flags=re.IGNORECASE)

    # Only fix within quoted text
    def fix_quoted(m):
        q = m.group(0)
        inner = q[1:-1]

        replacements = [("ue", "ü"), ("ae", "ä"), ("oe", "ö"),
                        ("Ue", "Ü"), ("Ae", "Ä"), ("Oe", "Ö")]
        for old, new in replacements:
            # Word-level replacement with false-positive check
            def replace_in_word(wm):
                word = wm.group(0)
                word_lower = word.lower()
                if word_lower in _GERMAN_FALSE_POSITIVES:
                    return word
                return word.replace(old, new)

            # Find words containing the digraph
            inner = re.sub(r'\b\w*' + re.escape(old) + r'\w*\b', replace_in_word, inner)

        return '"' + inner + '"'

    text = re.sub(r'"[^"]*"', fix_quoted, text)
    return text


# --- Prompt ingestion --------------------------------------------------------

def load_pregenerated_prompts(config: dict) -> list:
    """Load pre-generated prompts from JSON files specified in config."""
    pi = config.get("prompt_ingestion", {})
    prompt_files = pi.get("prompt_files", [])
    quality_prefix = config.get("quality_prefix", "")
    quality_suffix = config.get("quality_suffix", "")

    all_prompts = []
    for pf in prompt_files:
        if not Path(pf).exists():
            logging.warning("Prompt file not found: %s", pf)
            continue

        with open(pf, encoding="utf-8") as f:
            data = json.load(f)

        # Determine pathway from filename
        fname = Path(pf).stem.lower()
        pathway = "unknown"
        if "voicenet" in fname:
            pathway = "cca_voicenet"
        elif "archetype" in fname:
            pathway = "cc2c_archetype"
        elif "acting_challenge" in fname or "accc" in fname:
            pathway = "accc_acting_challenge"
        elif "situation" in fname or "sit" in fname:
            pathway = "sit_situation"
        elif "extreme" in fname:
            pathway = "extreme_physical"

        language = "German" if fname.endswith("_de") else "English"
        has_cut_to = True  # Default to cut_to format

        for item in data:
            prompt_text = item if isinstance(item, str) else item.get("dramabox_prompt", item.get("prompt", ""))
            if not prompt_text:
                continue

            # Apply quality prefix/suffix
            modified = quality_prefix + prompt_text + quality_suffix

            # Fix German umlauts
            if language == "German":
                modified = fix_german_quoted_text(modified)

            has_cut = "CUT TO:" in prompt_text

            all_prompts.append({
                "prompt_id": hashlib.md5(prompt_text.encode()).hexdigest()[:12],
                "original_prompt": prompt_text,
                "modified_prompt": modified,
                "pathway": pathway,
                "language": language,
                "format": "cut_to" if has_cut else "standard",
                "has_cut_to": has_cut,
                "sample_info": item if isinstance(item, dict) else {},
            })

    logging.info("Loaded %d pre-generated prompts from %d files", len(all_prompts), len(prompt_files))
    return all_prompts


def interleave_prompts(prompts: list) -> list:
    """Round-robin interleave prompts across pathways for even progress."""
    by_pathway = {}
    for p in prompts:
        pw = p.get("pathway", "unknown")
        by_pathway.setdefault(pw, []).append(p)

    interleaved = []
    pathway_iters = {k: iter(v) for k, v in by_pathway.items()}
    pathway_keys = list(pathway_iters.keys())
    idx = 0
    exhausted = set()
    while len(exhausted) < len(pathway_keys):
        key = pathway_keys[idx % len(pathway_keys)]
        if key not in exhausted:
            try:
                interleaved.append(next(pathway_iters[key]))
            except StopIteration:
                exhausted.add(key)
        idx += 1
    return interleaved


# =============================================================================
# Section 6: Model Manager
# =============================================================================

class ModelManager:
    """Manages GPU model lifecycle for simultaneous/sequential modes.

    In simultaneous mode all models coexist on one GPU.
    In sequential mode only one stage's models are loaded at a time.
    """

    def __init__(self, config: dict, loading_mode: str = "auto"):
        self.config = config
        self.loading_mode = loading_mode
        self._models = {}  # stage -> dict of models
        self._loaded_stages = set()

    def ensure_stage(self, stage: str) -> dict:
        """Load models for the given stage. Returns dict of model objects.

        stage: "llm" | "tts" | "postprocess"
        In sequential mode, unloads other stages first.
        """
        if stage in self._loaded_stages:
            return self._models[stage]

        if self.loading_mode == "sequential":
            # Unload all other stages
            for s in list(self._loaded_stages):
                if s != stage:
                    self.unload_stage(s)

        if stage == "llm":
            self._models[stage] = self._load_llm()
        elif stage == "tts":
            self._models[stage] = self._load_tts()
        elif stage == "postprocess":
            self._models[stage] = self._load_postprocess()
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self._loaded_stages.add(stage)
        return self._models[stage]

    def unload_stage(self, stage: str):
        """Free models for the given stage."""
        if stage not in self._loaded_stages:
            return

        models = self._models.pop(stage, {})

        if stage == "llm":
            models.clear()
        elif stage == "tts":
            models.clear()
        elif stage == "postprocess":
            models.clear()

        self._loaded_stages.discard(stage)

        # Force GPU memory release
        try:
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass

        logging.info("Unloaded stage '%s'", stage)

    def unload_all(self):
        """Unload all stages."""
        for s in list(self._loaded_stages):
            self.unload_stage(s)

    def _load_llm(self) -> dict:
        """Load GGUF LLM for prompt generation."""
        pg = self.config.get("prompt_generation", {})
        model_path = pg.get("gguf_model_path", "")
        n_ctx = pg.get("n_ctx", 8192)

        logging.info("Loading GGUF LLM: %s", model_path)
        t0 = time.time()
        llm = load_gguf_llm(model_path, n_ctx)
        logging.info("GGUF LLM loaded in %.1fs", time.time() - t0)
        return {"llm": llm}

    def _load_tts(self) -> dict:
        """Load DramaBox TTSServer."""
        tts_cfg = self.config.get("tts", {})
        dramabox_dir = tts_cfg.get("dramabox_dir", "")

        if dramabox_dir:
            src_path = os.path.join(dramabox_dir, "src")
            ltx_path = os.path.join(dramabox_dir, "ltx2")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            if ltx_path not in sys.path:
                sys.path.insert(0, ltx_path)

        from inference_server import TTSServer
        from model_downloader import get_all_paths

        paths = get_all_paths()
        logging.info("Loading DramaBox TTSServer...")
        t0 = time.time()
        server = TTSServer(
            checkpoint=paths["transformer"],
            full_checkpoint=paths["audio_components"],
            gemma_root=paths["gemma_root"],
            device="cuda",
            dtype="bf16",
            compile_model=tts_cfg.get("compile_model", True),
            bnb_4bit=tts_cfg.get("bnb_4bit", True),
        )
        logging.info("TTSServer loaded in %.1fs", time.time() - t0)
        return {"server": server}

    def _load_postprocess(self) -> dict:
        """Load all postprocessing models: RE-USE, LavaSR, Whisper, VoiceCLAP."""
        import torch
        import torchaudio
        import torch.nn as nn
        import warnings
        warnings.filterwarnings("ignore")

        models = {}
        t0 = time.time()

        # --- RE-USE ---
        reuse_dir = self.config.get("postprocess", {}).get("reuse_dir", "/home/deployer/laion/REUSE")
        if reuse_dir not in sys.path:
            sys.path.insert(0, str(reuse_dir))

        from models.stfts import mag_phase_stft, mag_phase_istft
        from models.generator_SEMamba_time_d4 import SEMamba
        from utils.util import load_config, pad_or_trim_to_match

        reuse_cfg_path = os.path.join(
            reuse_dir, "recipes",
            "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
        )
        reuse_cfg = load_config(reuse_cfg_path)
        reuse_model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=reuse_cfg).to("cuda")
        reuse_model.eval()

        models["reuse_model"] = reuse_model
        models["reuse_cfg"] = reuse_cfg
        models["reuse_stft_fns"] = (mag_phase_stft, mag_phase_istft)
        models["reuse_utils"] = pad_or_trim_to_match
        models["reuse_relu"] = nn.ReLU()
        logging.info("RE-USE loaded")

        # --- LavaSR (with vocos monkey-patch) ---
        import vocos.feature_extractors as _vfe
        _OrigMSF = _vfe.MelSpectrogramFeatures

        class _PatchedMSF(_OrigMSF):
            def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256,
                         n_mels=100, padding="center", f_min=None, f_max=None,
                         norm=None, mel_scale=None):
                super(_OrigMSF, self).__init__()
                if padding not in ("center", "same"):
                    raise ValueError("Padding must be 'center' or 'same'.")
                self.padding = padding
                mel_kwargs = dict(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                  n_mels=n_mels, center=(padding == "center"), power=1)
                if f_min is not None:
                    mel_kwargs["f_min"] = f_min
                if f_max is not None:
                    mel_kwargs["f_max"] = f_max
                if norm is not None:
                    mel_kwargs["norm"] = norm
                if mel_scale is not None:
                    mel_kwargs["mel_scale"] = mel_scale
                self.mel_spec = torchaudio.transforms.MelSpectrogram(**mel_kwargs)

        _vfe.MelSpectrogramFeatures = _PatchedMSF

        from LavaSR.model import LavaEnhance2
        lavasr_model = LavaEnhance2("YatharthS/LavaSR", device="cuda:0")
        models["lavasr_model"] = lavasr_model
        logging.info("LavaSR loaded")

        # --- Whisper turbo ---
        import whisper
        whisper_model_name = self.config.get("postprocess", {}).get("whisper_model", "turbo")
        whisper_model = whisper.load_model(whisper_model_name, device="cuda")
        models["whisper_model"] = whisper_model
        logging.info("Whisper %s loaded", whisper_model_name)

        # --- VoiceCLAP Large ---
        from sentence_transformers import SentenceTransformer
        clap_model = SentenceTransformer("laion/voiceclap-large", trust_remote_code=True, device="cuda")

        aesthetics_text = ("Realistic, genuine, spotanoues, authentic, sensual, natural voice "
                           "with all imperfections and organic microdistractions a natural "
                           "situation brings with it")
        aesthetics_emb = clap_model.encode([aesthetics_text], normalize_embeddings=True)[0]

        models["clap_model"] = clap_model
        models["aesthetics_text"] = aesthetics_text
        models["aesthetics_emb"] = aesthetics_emb
        logging.info("VoiceCLAP Large loaded")

        logging.info("All postprocess models loaded in %.1fs", time.time() - t0)
        return models


# =============================================================================
# Section 7: Postprocessing Functions
# =============================================================================

# --- Singing/humming keyword detection ---
SINGING_PATTERN = re.compile(
    r'\b(sing(?:s|ing|er)?|'
    r'hum(?:s|ming|med)|'
    r'whistl(?:e|es|ing|ed)|'
    r'lullaby|chant(?:s|ing|ed)?|serenade[sd]?|'
    r'yodel(?:s|ing|ed)?|croon(?:s|ing|ed)?|'
    r'warbl(?:e|es|ing|ed)?)\b',
    re.IGNORECASE,
)
BARE_HUM_PATTERN = re.compile(
    r'\b(?:begins?\s+to|starts?\s+to|a\s+soft|softly|gently|quietly)\s+hum\b',
    re.IGNORECASE,
)


def is_singing_prompt(prompt_text: str) -> bool:
    """Check if prompt describes singing/humming/whistling performance."""
    return bool(SINGING_PATTERN.search(prompt_text)) or bool(BARE_HUM_PATTERN.search(prompt_text))


def extract_expected_text(prompt: str) -> str:
    """Extract all quoted dialogue from a DramaBox prompt."""
    matches = re.findall(r'"([^"]*)"', prompt)
    if not matches:
        matches = re.findall(r'\u201c([^\u201d]*)\u201d', prompt)
    return " ".join(matches).strip()


def extract_scene_texts(prompt: str) -> tuple:
    """Extract quoted dialogue from before and after CUT TO: in the prompt."""
    parts = re.split(r'\bCUT\s+TO\s*:', prompt, maxsplit=1)
    if len(parts) < 2:
        return "", ""

    def get_quotes(text):
        matches = re.findall(r'"([^"]+)"', text)
        return " ".join(matches)

    return get_quotes(parts[0]), get_quotes(parts[1])


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate via edit distance."""
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n = len(ref_words)
    m = len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[n][m] / n


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def float32_to_base64(arr) -> str:
    """Encode float32 numpy array as base64."""
    import numpy as np
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def find_split_point(word_timestamps: list, total_duration: float) -> tuple:
    """Find CUT TO: transition point via longest silence gap in middle 20-80%."""
    if not word_timestamps or len(word_timestamps) < 4:
        return total_duration / 2.0, "midpoint_fallback"

    gaps = []
    for i in range(1, len(word_timestamps)):
        gap_start = word_timestamps[i - 1]["end"]
        gap_end = word_timestamps[i]["start"]
        gap_len = gap_end - gap_start
        gap_mid = (gap_start + gap_end) / 2.0
        gaps.append((gap_len, gap_mid, i))

    lo = total_duration * 0.20
    hi = total_duration * 0.80
    middle_gaps = [(g, mid, i) for g, mid, i in gaps if lo <= mid <= hi]
    if not middle_gaps:
        lo = total_duration * 0.10
        hi = total_duration * 0.90
        middle_gaps = [(g, mid, i) for g, mid, i in gaps if lo <= mid <= hi]
    if not middle_gaps:
        return total_duration / 2.0, "midpoint_fallback"

    best_gap, best_mid, _ = max(middle_gaps, key=lambda x: x[0])
    if best_gap < 0.3:
        return total_duration / 2.0, "midpoint_fallback"
    return best_mid, "silence_gap"


def apply_fade(audio_np, sr: int, fade_ms: int = 50):
    """Apply fade-in at start and fade-out at end."""
    import numpy as np
    fade_samples = int(sr * fade_ms / 1000)
    if fade_samples >= len(audio_np):
        return audio_np
    audio = audio_np.copy()
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return audio


def split_audio(audio_np, sr: int, split_sec: float):
    """Split mono audio at split_sec, apply 50ms fade."""
    split_sample = int(split_sec * sr)
    split_sample = max(0, min(split_sample, len(audio_np)))
    part1 = apply_fade(audio_np[:split_sample], sr)
    part2 = apply_fade(audio_np[split_sample:], sr)
    return part1, part2


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "256k", sr: int = 48000):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path,
         "-ac", "1", "-ar", str(sr), "-b:a", bitrate, "-f", "mp3", mp3_path],
        capture_output=True, check=True,
    )


def run_reuse(wav_tensor, sr: int, pp_models: dict):
    """Apply RE-USE speech enhancement. wav_tensor: (C, T) on cuda."""
    import torch
    import torch.nn as nn

    reuse_model = pp_models["reuse_model"]
    reuse_cfg = pp_models["reuse_cfg"]
    mag_phase_stft, mag_phase_istft = pp_models["reuse_stft_fns"]
    pad_or_trim = pp_models["reuse_utils"]
    RELU = pp_models["reuse_relu"]

    n_fft = reuse_cfg["stft_cfg"]["n_fft"]
    hop = reuse_cfg["stft_cfg"]["hop_size"]
    win = reuse_cfg["stft_cfg"]["win_size"]
    compress = reuse_cfg["model_cfg"]["compress_factor"]
    reuse_sr = reuse_cfg["stft_cfg"]["sampling_rate"]

    def make_even(v):
        v = int(round(v))
        return v if v % 2 == 0 else v + 1

    with torch.no_grad():
        n_fft_s = make_even(n_fft * sr // reuse_sr)
        hop_s = make_even(hop * sr // reuse_sr)
        win_s = make_even(win * sr // reuse_sr)

        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
            wav_tensor, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
            compress_factor=compress, center=True, addeps=False,
        )
        amp_g, pha_g, _ = reuse_model(noisy_mag, noisy_pha)
        mag = torch.expm1(RELU(amp_g))
        zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0

        audio_g = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, compress)
        audio_g = pad_or_trim(wav_tensor.detach(), audio_g, pad_value=1e-8)
        return audio_g


def run_lavasr(wav_path_in: str, wav_path_out: str, pp_models: dict):
    """Apply LavaSR super-resolution (16kHz -> 48kHz)."""
    import torchaudio
    lavasr = pp_models["lavasr_model"]
    wav, sr_in = lavasr.load_audio(wav_path_in, input_sr=16000)
    output = lavasr.enhance(wav, enhance=True, denoise=False)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    torchaudio.save(wav_path_out, output.cpu(), 48000)


def run_whisper_asr(audio_path: str, pp_models: dict) -> tuple:
    """Run Whisper ASR with word-level timestamps. Returns (text, word_timestamps)."""
    whisper_model = pp_models["whisper_model"]
    result = whisper_model.transcribe(audio_path, word_timestamps=True, language=None)
    text = result.get("text", "").strip()
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })
    return text, words


def get_clap_embeddings_batch(audio_list_16k: list, pp_models: dict):
    """Get VoiceCLAP Large embeddings for a batch of 16kHz mono arrays."""
    import numpy as np
    clap_model = pp_models["clap_model"]
    inputs = [{"array": a.astype(np.float32), "sampling_rate": 16000} for a in audio_list_16k]
    return clap_model.encode(inputs, normalize_embeddings=True, batch_size=len(inputs))


# =============================================================================
# Section 8: Worker Function
# =============================================================================

def run_worker(gpu_id: int, work_file: str, config_path: str, overrides: dict = None):
    """Full pipeline worker for one GPU. Called as subprocess with --worker flag."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config = load_yaml_config(config_path)

    # Apply CLI overrides forwarded from coordinator
    if overrides:
        if "seeds_per_prompt" in overrides:
            config.setdefault("tts", {})["seeds_per_prompt"] = overrides["seeds_per_prompt"]
        if "mode" in overrides:
            config["mode"] = overrides["mode"]

    mode = config.get("mode", "generate")
    quality_prefix = config.get("quality_prefix", "")
    quality_suffix = config.get("quality_suffix", "")

    output_dir = Path(config["output"]["output_dir"])
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_file = progress_dir / f"gpu_{gpu_id}.json"

    # Setup logging for this worker
    log_file = output_dir / f"worker_gpu{gpu_id}.log"
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
        format=f"%(asctime)s [GPU {gpu_id}] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger(f"worker.gpu{gpu_id}")

    with open(work_file) as f:
        work_items = json.load(f)

    total = len(work_items)
    log.info("Worker starting: %d work items", total)

    # Determine loading mode
    loading_mode = config.get("gpu", {}).get("loading_mode", "auto")
    if loading_mode == "auto":
        vram = get_gpu_vram_mb(gpu_id)
        loading_mode = "simultaneous" if vram >= 70000 else "sequential"
        log.info("Auto-detected VRAM: %d MB -> mode: %s", vram, loading_mode)

    manager = ModelManager(config, loading_mode)

    tts_cfg = config.get("tts", {})
    seeds_per_prompt = tts_cfg.get("seeds_per_prompt", 25)
    cfg_scale = tts_cfg.get("cfg_scale", 2.5)
    stg_scale = tts_cfg.get("stg_scale", 1.5)
    dur_mult = tts_cfg.get("duration_multiplier", 1.1)
    watermark = tts_cfg.get("watermark", False)

    pp_cfg = config.get("postprocess", {})
    mp3_bitrate = pp_cfg.get("mp3_bitrate", "256k")
    mp3_sr = pp_cfg.get("mp3_sample_rate", 48000)
    clap_batch_size = pp_cfg.get("clap_batch_size", 16)

    # Progress tracking
    progress = {
        "gpu_id": gpu_id,
        "total": total,
        "completed": 0,
        "errors": 0,
        "current_prompt": "",
        "current_stage": "starting",
        "start_time": time.time(),
        "completed_ids": [],
        "pathway_counts": {},
    }

    def save_progress():
        try:
            tmp = str(progress_file) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(progress, f, indent=1)
            os.replace(tmp, str(progress_file))
        except Exception:
            pass

    save_progress()

    # ── Stage 1: Prompt Generation (if mode=generate) ──
    if mode == "generate":
        log.info("Stage 1: Generating prompts via GGUF LLM")
        progress["current_stage"] = "prompt_generation"
        save_progress()

        vap_root = config.get("_vap_root", "")
        if vap_root:
            sys.path.insert(0, vap_root)

        import random
        random.seed(config.get("seed", 42) + gpu_id)

        llm_models = manager.ensure_stage("llm")
        llm = llm_models["llm"]

        taxonomies = load_taxonomies(config)
        default_format = config.get("prompt_generation", {}).get("default_format", "cut_to")

        for i, item in enumerate(work_items):
            if item.get("modified_prompt"):
                continue  # Already has prompt (resume case)

            pathway = item.get("pathway", "voicenet")
            language = item.get("language", "English")
            fmt = item.get("format", default_format)

            progress["current_prompt"] = f"Generating {i+1}/{total}: {pathway}/{language}"
            save_progress()

            try:
                sys_prompt, user_prompt, sample_info = sample_and_build_prompt(
                    pathway, language, fmt, taxonomies, config,
                )
                text, elapsed, pt, ct = generate_prompt_gguf(llm, sys_prompt, user_prompt, config)
                log.info("Prompt %d/%d: %.1fs (%d+%d tok) [%s/%s]",
                         i + 1, total, elapsed, pt, ct, pathway, language)

                has_cut = "CUT TO:" in text
                modified = quality_prefix + text + quality_suffix
                if language == "German":
                    modified = fix_german_quoted_text(modified)

                item["original_prompt"] = text
                item["modified_prompt"] = modified
                item["has_cut_to"] = has_cut
                item["sample_info"] = sample_info
                item["format"] = "cut_to" if has_cut else "standard"

            except Exception as e:
                log.error("Prompt generation failed for item %d: %s", i, e)
                item["error"] = str(e)
                progress["errors"] += 1

        manager.unload_stage("llm")
        del taxonomies
        log.info("Prompt generation complete")

    # ── Stage 2: TTS Generation ──
    log.info("Stage 2: TTS generation (%d seeds per prompt)", seeds_per_prompt)
    progress["current_stage"] = "tts_generation"
    save_progress()

    tts_models = manager.ensure_stage("tts")
    server = tts_models["server"]

    for i, item in enumerate(work_items):
        if item.get("error"):
            continue
        prompt = item.get("modified_prompt", "")
        if not prompt:
            continue

        prompt_id = item.get("prompt_id", f"p{i:06d}")
        prompt_dir = output_dir / "audio" / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)

        progress["current_prompt"] = f"TTS {i+1}/{total}: {prompt_id}"
        save_progress()

        wav_files = []

        # Duration variation: distribute seeds across 5 duration buckets
        # so the WER-based ranking can select the best-fitting length.
        #   bucket 0 (-20%): seeds 0..4
        #   bucket 1 (-10%): seeds 5..9
        #   bucket 2 (  0%): seeds 10..14
        #   bucket 3 (+10%): seeds 15..19
        #   bucket 4 (+20%): seeds 20..24
        # For non-standard seed counts, cycle through the 5 offsets.
        DURATION_OFFSETS = [-0.20, -0.10, 0.0, 0.10, 0.20]
        bucket_size = max(1, seeds_per_prompt // len(DURATION_OFFSETS))
        seed_dur_mults = {}  # seed_idx -> actual multiplier used

        for seed_idx in range(seeds_per_prompt):
            seed_val = config.get("seed", 42) + seed_idx
            wav_path = prompt_dir / f"seed{seed_idx:02d}.wav"

            bucket = min(seed_idx // bucket_size, len(DURATION_OFFSETS) - 1)
            offset = DURATION_OFFSETS[bucket]
            seed_mult = dur_mult * (1.0 + offset)
            seed_dur_mults[seed_idx] = round(seed_mult, 4)

            if wav_path.exists():
                wav_files.append(str(wav_path))
                continue

            try:
                server.generate_to_file(
                    prompt=prompt,
                    output=str(wav_path),
                    cfg_scale=cfg_scale,
                    stg_scale=stg_scale,
                    duration_multiplier=seed_mult,
                    gen_duration=0.0,
                    seed=seed_val,
                    watermark=watermark,
                )
                wav_files.append(str(wav_path))
                log.info("TTS %s seed%02d done (dur_mult=%.2f, offset=%+.0f%%)",
                         prompt_id, seed_idx, seed_mult, offset * 100)
            except Exception as e:
                log.error("TTS failed %s seed%02d: %s", prompt_id, seed_idx, e)

        item["wav_files"] = wav_files
        item["prompt_dir"] = str(prompt_dir)
        item["seed_dur_mults"] = seed_dur_mults

    if loading_mode == "sequential":
        manager.unload_stage("tts")
    log.info("TTS generation complete")

    # ── Stage 3: Postprocessing ──
    log.info("Stage 3: Postprocessing (RE-USE -> LavaSR -> Whisper -> split -> MP3 -> VoiceCLAP)")
    progress["current_stage"] = "postprocessing"
    save_progress()

    pp_models = manager.ensure_stage("postprocess")

    import torch
    import torchaudio
    import numpy as np

    # CLAP batch accumulator
    clap_batch = []

    for i, item in enumerate(work_items):
        if item.get("error") or not item.get("wav_files"):
            continue

        prompt_id = item.get("prompt_id", f"p{i:06d}")
        prompt_dir = Path(item["prompt_dir"])
        prompt_text = item.get("original_prompt", "")
        singing = is_singing_prompt(prompt_text)
        has_cut = item.get("has_cut_to", False)

        progress["current_prompt"] = f"Postprocess {i+1}/{total}: {prompt_id}"
        save_progress()

        for wav_path_str in item["wav_files"]:
            wav_path = Path(wav_path_str)
            seed_str = wav_path.stem  # e.g., "seed00"
            base_name = f"{prompt_id}_{seed_str}"

            try:
                # Load audio
                wav_tensor, sr = torchaudio.load(str(wav_path))
                wav_tensor = wav_tensor.to("cuda")

                # Ensure mono
                if wav_tensor.shape[0] > 1:
                    wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

                # RE-USE (skip singing)
                reuse_applied = False
                if not singing:
                    wav_tensor = run_reuse(wav_tensor, sr, pp_models)
                    reuse_applied = True

                # Save enhanced WAV for LavaSR input
                enhanced_path = prompt_dir / f"{seed_str}_enhanced.wav"
                torchaudio.save(str(enhanced_path), wav_tensor.cpu(), sr)

                # LavaSR super-resolution
                lavasr_path = prompt_dir / f"{seed_str}_lavasr.wav"
                run_lavasr(str(enhanced_path), str(lavasr_path), pp_models)

                # Load LavaSR output (48kHz)
                lava_wav, lava_sr = torchaudio.load(str(lavasr_path))
                mono_np = lava_wav.squeeze(0).numpy()
                total_duration = len(mono_np) / lava_sr

                # Whisper ASR
                asr_text, word_timestamps = run_whisper_asr(str(lavasr_path), pp_models)

                # Split detection and audio split
                split_sec, split_method = None, None
                part1_np, part2_np = None, None
                scene1_text, scene2_text = "", ""
                wer_part1, wer_part2 = None, None

                if has_cut:
                    split_sec, split_method = find_split_point(word_timestamps, total_duration)
                    part1_np, part2_np = split_audio(mono_np, lava_sr, split_sec)
                    ref1, ref2 = extract_scene_texts(prompt_text)

                    # Transcribe parts for WER
                    scene1_text = " ".join(w["word"] for w in word_timestamps if w["end"] <= split_sec)
                    scene2_text = " ".join(w["word"] for w in word_timestamps if w["start"] >= split_sec)

                    if ref1:
                        wer_part1 = round(compute_wer(scene1_text, ref1), 4)
                    if ref2:
                        wer_part2 = round(compute_wer(scene2_text, ref2), 4)

                # Full WER
                full_ref = extract_expected_text(prompt_text)
                wer_full = round(compute_wer(asr_text, full_ref), 4) if full_ref else None

                # MP3 conversion
                full_mp3 = prompt_dir / f"{base_name}_full.mp3"
                wav_to_mp3(str(lavasr_path), str(full_mp3), mp3_bitrate, mp3_sr)

                if has_cut and part1_np is not None and part2_np is not None:
                    p1_wav = prompt_dir / f"{seed_str}_part1.wav"
                    p2_wav = prompt_dir / f"{seed_str}_part2.wav"
                    torchaudio.save(str(p1_wav), torch.from_numpy(part1_np).unsqueeze(0), lava_sr)
                    torchaudio.save(str(p2_wav), torch.from_numpy(part2_np).unsqueeze(0), lava_sr)

                    p1_mp3 = prompt_dir / f"{base_name}_part1.mp3"
                    p2_mp3 = prompt_dir / f"{base_name}_part2.mp3"
                    wav_to_mp3(str(p1_wav), str(p1_mp3), mp3_bitrate, mp3_sr)
                    wav_to_mp3(str(p2_wav), str(p2_mp3), mp3_bitrate, mp3_sr)

                # Resample to 16kHz for CLAP
                def resample_16k(audio_np, current_sr):
                    if current_sr == 16000:
                        return audio_np
                    t = torch.from_numpy(audio_np).unsqueeze(0).to("cuda")
                    t_16k = torchaudio.functional.resample(t, current_sr, 16000)
                    return t_16k.squeeze(0).cpu().numpy()

                full_16k = resample_16k(mono_np, lava_sr)

                # Build annotation
                annotation = {
                    "prompt_id": prompt_id,
                    "seed": int(seed_str.replace("seed", "")),
                    "pathway": item.get("pathway", "unknown"),
                    "language": item.get("language", "English"),
                    "format": item.get("format", "cut_to"),
                    "original_prompt": prompt_text,
                    "modified_prompt": item.get("modified_prompt", ""),
                    "singing_flag": singing,
                    "reuse_applied": reuse_applied,
                    "has_cut_to": has_cut,
                    "asr_transcript": asr_text,
                    "word_timestamps": word_timestamps,
                    "split_point_sec": round(split_sec, 3) if split_sec else None,
                    "split_method": split_method,
                    "scene1_transcript": scene1_text,
                    "scene2_transcript": scene2_text,
                    "full_duration_sec": round(total_duration, 3),
                    "part1_duration_sec": round(split_sec, 3) if split_sec else None,
                    "part2_duration_sec": round(total_duration - split_sec, 3) if split_sec else None,
                    "wer_full": wer_full,
                    "wer_part1": wer_part1,
                    "wer_part2": wer_part2,
                    "sample_info": item.get("sample_info", {}),
                    "cfg_scale": cfg_scale,
                    "stg_scale": stg_scale,
                    "duration_multiplier": item.get("seed_dur_mults", {}).get(
                        int(seed_str.replace("seed", "")), dur_mult),
                }

                # Queue for CLAP batch
                clap_item = {
                    "base_name": base_name,
                    "output_dir": str(prompt_dir),
                    "has_cut_to": has_cut,
                    "full_16k": full_16k,
                    "annotation": annotation,
                }
                if has_cut and part1_np is not None:
                    clap_item["part1_16k"] = resample_16k(part1_np, lava_sr)
                    clap_item["part2_16k"] = resample_16k(part2_np, lava_sr)

                clap_batch.append(clap_item)

                # Flush CLAP batch if full
                if len(clap_batch) >= clap_batch_size:
                    _flush_clap_batch(clap_batch, pp_models)
                    clap_batch.clear()

                # Cleanup intermediate WAVs
                for tmp in [enhanced_path, lavasr_path]:
                    if tmp.exists():
                        tmp.unlink()
                if has_cut:
                    for tmp in [prompt_dir / f"{seed_str}_part1.wav",
                                prompt_dir / f"{seed_str}_part2.wav"]:
                        if tmp.exists():
                            tmp.unlink()

            except Exception as e:
                log.error("Postprocess failed %s: %s\n%s", base_name, e, traceback.format_exc())
                progress["errors"] += 1

        # Mark prompt as complete
        progress["completed"] += 1
        pw = item.get("pathway", "unknown")
        progress["pathway_counts"][pw] = progress["pathway_counts"].get(pw, 0) + 1
        progress["completed_ids"].append(prompt_id)
        save_progress()

    # Flush remaining CLAP batch
    if clap_batch:
        _flush_clap_batch(clap_batch, pp_models)
        clap_batch.clear()

    manager.unload_all()

    progress["current_stage"] = "done"
    progress["end_time"] = time.time()
    save_progress()
    log.info("Worker GPU %d finished: %d completed, %d errors",
             gpu_id, progress["completed"], progress["errors"])


def _flush_clap_batch(batch: list, pp_models: dict):
    """Process accumulated CLAP embeddings in batch, compute scores, write annotations."""
    import numpy as np

    if not batch:
        return

    clap_inputs = []
    index_map = []
    for i, d in enumerate(batch):
        if d["has_cut_to"] and "part1_16k" in d:
            clap_inputs.append(d["part1_16k"])
            index_map.append((i, "part1"))
            clap_inputs.append(d["part2_16k"])
            index_map.append((i, "part2"))
        clap_inputs.append(d["full_16k"])
        index_map.append((i, "full"))

    try:
        embeddings = get_clap_embeddings_batch(clap_inputs, pp_models)
        emb_dicts = [{} for _ in batch]
        for idx, (item_idx, key) in enumerate(index_map):
            emb_dicts[item_idx][key] = embeddings[idx]

        aesthetics_emb = pp_models["aesthetics_emb"]
        aesthetics_text = pp_models["aesthetics_text"]

        for i, d in enumerate(batch):
            ann = d["annotation"]
            embs = emb_dicts[i]

            ann["voiceclap_embedding_full"] = float32_to_base64(embs["full"])
            sim_full = cosine_sim(embs["full"], aesthetics_emb)
            ann["clap_cosine_similarity_full"] = round(sim_full, 6)

            if d["has_cut_to"] and "part1" in embs:
                ann["voiceclap_embedding_part1"] = float32_to_base64(embs["part1"])
                ann["voiceclap_embedding_part2"] = float32_to_base64(embs["part2"])
                ann["clap_cosine_similarity_part1"] = round(cosine_sim(embs["part1"], aesthetics_emb), 6)
                ann["clap_cosine_similarity_part2"] = round(cosine_sim(embs["part2"], aesthetics_emb), 6)
            else:
                ann["voiceclap_embedding_part1"] = None
                ann["voiceclap_embedding_part2"] = None
                ann["clap_cosine_similarity_part1"] = None
                ann["clap_cosine_similarity_part2"] = None

            ann["clap_aesthetics_text"] = aesthetics_text

            def _reward(sim, wer):
                if sim is None or wer is None:
                    return None
                return round(sim * (1.0 - min(wer, 1.0)), 6)

            ann["reward_full"] = _reward(ann["clap_cosine_similarity_full"], ann.get("wer_full"))
            ann["reward_part1"] = _reward(ann.get("clap_cosine_similarity_part1"), ann.get("wer_part1"))
            ann["reward_part2"] = _reward(ann.get("clap_cosine_similarity_part2"), ann.get("wer_part2"))

            # Write annotation JSON
            ann_path = os.path.join(d["output_dir"], f"{d['base_name']}.json")
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=1)

    except Exception as e:
        logging.error("CLAP batch failed: %s\n%s", e, traceback.format_exc())


# =============================================================================
# Section 9: WebDataset TAR Creation + Upload
# =============================================================================

def create_webdataset_tar(completed_groups: list, tar_index: int, config: dict) -> Path:
    """Create a WebDataset-format TAR from completed prompt groups.

    Each sample in the TAR contains:
    - {prompt_id}_seed{NN}_full.mp3
    - {prompt_id}_seed{NN}_part1.mp3 (if CUT TO:)
    - {prompt_id}_seed{NN}_part2.mp3 (if CUT TO:)
    - {prompt_id}_seed{NN}.json (annotation)
    """
    output_dir = Path(config["output"]["output_dir"])
    tar_dir = output_dir / "tars"
    tar_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tar_dir / f"shard_{tar_index:06d}.tar"

    with tarfile.open(str(tar_path), "w") as tar:
        for group in completed_groups:
            prompt_id = group["prompt_id"]
            prompt_dir = Path(group["prompt_dir"])

            if not prompt_dir.exists():
                continue

            # Add all MP3 files and JSON annotations
            for f in sorted(prompt_dir.glob("*.mp3")):
                tar.add(str(f), arcname=f.name)
            for f in sorted(prompt_dir.glob("*.json")):
                tar.add(str(f), arcname=f.name)

    logging.info("Created TAR shard_%06d.tar (%d groups, %.1f MB)",
                 tar_index, len(completed_groups),
                 tar_path.stat().st_size / (1024 * 1024))
    return tar_path


def upload_tar_to_hf(tar_path: Path, config: dict) -> bool:
    """Upload a TAR file to HuggingFace Hub."""
    hf_cfg = config.get("huggingface", {})
    token = hf_cfg.get("token", "")
    if not token or token == "YOUR_HF_TOKEN_HERE":
        token = os.environ.get("HF_TOKEN", "")
    if not token:
        logging.error("No HuggingFace token available for upload")
        return False

    repo_id = hf_cfg.get("repo_id", "")
    if not repo_id:
        logging.error("No HuggingFace repo_id configured")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=f"data/{tar_path.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logging.info("Uploaded %s to %s", tar_path.name, repo_id)
        return True
    except Exception as e:
        logging.error("Upload failed for %s: %s", tar_path.name, e)
        return False


def tar_upload_cleanup_cycle(state: dict, config: dict):
    """Check for complete groups, create TARs, upload, and optionally delete."""
    output_dir = Path(config["output"]["output_dir"])
    groups_per_tar = config.get("batching", {}).get("groups_per_tar", 10)
    storage_mode = config.get("storage", {}).get("mode", "local")

    # Find completed groups (those with annotation JSONs)
    audio_dir = output_dir / "audio"
    if not audio_dir.exists():
        return

    completed_groups = []
    for prompt_dir in sorted(audio_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_id = prompt_dir.name
        if prompt_id in state.get("tarred_prompt_ids", set()):
            continue

        # Check if this group has at least one annotation JSON
        jsons = list(prompt_dir.glob("*.json"))
        mp3s = list(prompt_dir.glob("*.mp3"))
        if jsons and mp3s:
            completed_groups.append({
                "prompt_id": prompt_id,
                "prompt_dir": str(prompt_dir),
            })

    if len(completed_groups) < groups_per_tar:
        return

    # Process complete groups in chunks of groups_per_tar
    while len(completed_groups) >= groups_per_tar:
        chunk = completed_groups[:groups_per_tar]
        completed_groups = completed_groups[groups_per_tar:]

        tar_index = state.get("tar_index", 0)
        tar_path = create_webdataset_tar(chunk, tar_index, config)
        state["tar_index"] = tar_index + 1

        # Track tarred prompt IDs
        if "tarred_prompt_ids" not in state:
            state["tarred_prompt_ids"] = set()
        for g in chunk:
            state["tarred_prompt_ids"].add(g["prompt_id"])

        # Upload if configured
        if storage_mode in ("upload_and_delete", "upload_and_keep"):
            success = upload_tar_to_hf(tar_path, config)
            if success:
                state.setdefault("uploaded_tars", []).append(tar_path.name)

                if storage_mode == "upload_and_delete":
                    # Delete local audio dirs
                    for g in chunk:
                        d = Path(g["prompt_dir"])
                        if d.exists():
                            shutil.rmtree(str(d))
                    # Delete tar
                    if tar_path.exists():
                        tar_path.unlink()
                    logging.info("Deleted local files for %d groups", len(chunk))


# =============================================================================
# Section 10: State Management
# =============================================================================

def load_state(config: dict) -> dict:
    """Load pipeline state from JSON file, or return fresh state."""
    output_dir = Path(config["output"]["output_dir"])
    state_file = output_dir / "pipeline_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            # Convert tarred_prompt_ids back to set
            if "tarred_prompt_ids" in state:
                state["tarred_prompt_ids"] = set(state["tarred_prompt_ids"])
            logging.info("Loaded state: %d completed prompts, tar_index=%d",
                         len(state.get("completed_prompt_ids", [])),
                         state.get("tar_index", 0))
            return state
        except Exception as e:
            logging.warning("Failed to load state: %s", e)

    return {
        "completed_prompt_ids": [],
        "tarred_prompt_ids": set(),
        "tar_index": 0,
        "uploaded_tars": [],
        "total_generated": 0,
        "total_errors": 0,
        "start_time": time.time(),
    }


def save_state(state: dict, config: dict):
    """Atomic save of pipeline state."""
    output_dir = Path(config["output"]["output_dir"])
    state_file = output_dir / "pipeline_state.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert sets to lists for JSON serialization
    save_dict = dict(state)
    if isinstance(save_dict.get("tarred_prompt_ids"), set):
        save_dict["tarred_prompt_ids"] = list(save_dict["tarred_prompt_ids"])

    tmp = str(state_file) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(save_dict, f, indent=2)
    os.replace(tmp, str(state_file))


# =============================================================================
# Section 11: Monitoring Dashboard
# =============================================================================

MONITOR_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DramaBox E2E Pipeline Monitor</title>
<meta http-equiv="refresh" content="5">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { color: #58a6ff; font-size: 1.8em; margin-bottom: 10px; }
  h2 { color: #79c0ff; font-size: 1.2em; margin: 20px 0 10px; }
  .progress-bar { background: #21262d; border-radius: 8px; height: 30px; overflow: hidden; margin: 10px 0; }
  .progress-fill { background: linear-gradient(90deg, #238636, #2ea043); height: 100%; border-radius: 8px;
                   transition: width 0.5s; display: flex; align-items: center; justify-content: center;
                   color: white; font-weight: bold; font-size: 0.9em; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 15px 0; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
  .stat-label { font-size: 0.8em; color: #8b949e; }
  .stat-value { font-size: 1.4em; color: #58a6ff; font-weight: 600; }
  .gpu-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; margin: 15px 0; }
  .gpu-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
  .gpu-title { color: #d2a8ff; font-weight: 600; margin-bottom: 8px; }
  .gpu-stage { color: #8b949e; font-size: 0.85em; }
  .gpu-progress { color: #58a6ff; font-size: 0.9em; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th, td { padding: 8px 12px; border-bottom: 1px solid #21262d; text-align: left; }
  th { color: #8b949e; font-size: 0.85em; }
  td { color: #c9d1d9; }
  .error { color: #f85149; }
  .ok { color: #3fb950; }
</style>
</head>
<body>
<div class="container">
<h1>DramaBox E2E Pipeline Monitor</h1>
<p id="update-time" style="color:#8b949e;margin-bottom:15px"></p>
<div id="content">Loading...</div>
</div>
<script>
async function refresh() {
    try {
        const r = await fetch('/api/progress');
        const d = await r.json();
        document.getElementById('update-time').textContent = 'Last update: ' + new Date().toLocaleTimeString();

        let pct = d.total_prompts > 0 ? (d.completed_prompts / d.total_prompts * 100).toFixed(1) : 0;
        let elapsed = d.elapsed_sec || 0;
        let rate = elapsed > 0 ? (d.completed_prompts / elapsed * 3600).toFixed(0) : 0;
        let eta = d.completed_prompts > 0 ? ((d.total_prompts - d.completed_prompts) / (d.completed_prompts / elapsed) / 60).toFixed(0) : '?';

        let html = `
        <div class="progress-bar"><div class="progress-fill" style="width:${pct}%">${pct}%</div></div>
        <div class="stats">
            <div class="stat"><div class="stat-label">Completed</div><div class="stat-value">${d.completed_prompts}/${d.total_prompts}</div></div>
            <div class="stat"><div class="stat-label">Errors</div><div class="stat-value ${d.total_errors > 0 ? 'error' : 'ok'}">${d.total_errors}</div></div>
            <div class="stat"><div class="stat-label">Rate</div><div class="stat-value">${rate}/hr</div></div>
            <div class="stat"><div class="stat-label">ETA</div><div class="stat-value">${eta} min</div></div>
            <div class="stat"><div class="stat-label">TARs Created</div><div class="stat-value">${d.tars_created || 0}</div></div>
            <div class="stat"><div class="stat-label">TARs Uploaded</div><div class="stat-value">${d.tars_uploaded || 0}</div></div>
            <div class="stat"><div class="stat-label">Disk Used</div><div class="stat-value">${d.disk_used_gb || '?'} GB</div></div>
            <div class="stat"><div class="stat-label">Elapsed</div><div class="stat-value">${(elapsed/60).toFixed(0)} min</div></div>
        </div>`;

        if (d.gpus && d.gpus.length) {
            html += '<h2>GPU Workers</h2><div class="gpu-grid">';
            for (const g of d.gpus) {
                let gpct = g.total > 0 ? (g.completed / g.total * 100).toFixed(0) : 0;
                html += `<div class="gpu-card">
                    <div class="gpu-title">GPU ${g.gpu_id}</div>
                    <div class="gpu-progress">${g.completed}/${g.total} (${gpct}%)</div>
                    <div class="gpu-stage">Stage: ${g.current_stage || '?'}</div>
                    <div class="gpu-stage" style="margin-top:4px;font-size:0.8em;color:#6e7681">${g.current_prompt || ''}</div>
                    <div class="gpu-stage">Errors: <span class="${g.errors > 0 ? 'error' : ''}">${g.errors || 0}</span></div>
                </div>`;
            }
            html += '</div>';
        }

        if (d.pathways && Object.keys(d.pathways).length) {
            html += '<h2>Pathway Progress</h2><table><tr><th>Pathway</th><th>Completed</th></tr>';
            for (const [k,v] of Object.entries(d.pathways)) {
                html += `<tr><td>${k}</td><td>${v}</td></tr>`;
            }
            html += '</table>';
        }

        document.getElementById('content').innerHTML = html;
    } catch(e) {
        document.getElementById('content').innerHTML = '<p class="error">Failed to fetch progress: ' + e + '</p>';
    }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class MonitorHandler(BaseHTTPRequestHandler):
    """HTTP handler for the monitoring dashboard."""

    get_progress_fn = None  # Set by start_monitor()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(MONITOR_HTML.encode("utf-8"))
        elif self.path == "/api/progress":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            progress = self.get_progress_fn() if self.get_progress_fn else {}
            self.wfile.write(json.dumps(progress).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress access logs


def start_monitor(get_progress_fn, config: dict) -> tuple:
    """Start the monitoring HTTP server and optional Cloudflare tunnel.

    Returns (server, port, tunnel_process_or_None).
    """
    monitor_cfg = config.get("monitor", {})
    if not monitor_cfg.get("enabled", True):
        return None, 0, None

    start_port = monitor_cfg.get("port", 8766)
    port = find_available_port(start_port)

    MonitorHandler.get_progress_fn = staticmethod(get_progress_fn)
    server = HTTPServer(("0.0.0.0", port), MonitorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info("Monitor dashboard: http://localhost:%d", port)

    # Cloudflare tunnel
    tunnel_proc = None
    if monitor_cfg.get("cloudflare_tunnel", False):
        try:
            tunnel_proc = subprocess.Popen(
                ["/tmp/cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            deadline = time.time() + 20
            for line in tunnel_proc.stdout:
                if "trycloudflare.com" in line:
                    m = re.search(r'(https://[^\s]+trycloudflare\.com[^\s]*)', line)
                    if m:
                        logging.info("Monitor tunnel: %s", m.group(1))
                        print(f"\n  MONITOR URL: {m.group(1)}\n", flush=True)
                        break
                if time.time() > deadline:
                    break
        except Exception as e:
            logging.warning("Cloudflare tunnel failed: %s", e)

    return server, port, tunnel_proc


# =============================================================================
# Section 12: Coordinator
# =============================================================================

def coordinator_main(config: dict):
    """Main coordinator: distribute work, launch workers, monitor, package."""
    import random
    random.seed(config.get("seed", 42))

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = config.get("mode", "generate")
    auto_resume = config.get("auto_resume", True)

    # Detect GPUs
    gpu_cfg = config.get("gpu", {})
    devices = gpu_cfg.get("devices", "auto")
    if devices == "auto":
        gpus = detect_gpus()
    elif isinstance(devices, list):
        gpus = devices
    else:
        gpus = [int(x) for x in str(devices).split(",")]

    logging.info("Using GPUs: %s", gpus)

    # Load or create state
    state = load_state(config) if auto_resume else {
        "completed_prompt_ids": [], "tarred_prompt_ids": set(),
        "tar_index": 0, "uploaded_tars": [], "total_generated": 0,
        "total_errors": 0, "start_time": time.time(),
    }
    completed_ids = set(state.get("completed_prompt_ids", []))

    # Prepare work items
    work_items = []

    if mode == "ingest":
        logging.info("Mode: ingest — loading pre-generated prompts")
        prompts = load_pregenerated_prompts(config)
        prompts = interleave_prompts(prompts)
        for p in prompts:
            if auto_resume and p["prompt_id"] in completed_ids:
                continue
            work_items.append(p)

    elif mode == "generate":
        logging.info("Mode: generate — sampling pathway/language assignments")
        batching = config.get("batching", {})
        prompts_per_group = batching.get("prompts_per_group", 25)
        groups_per_tar = batching.get("groups_per_tar", 10)
        total_prompts = prompts_per_group * groups_per_tar * max(len(gpus), 1)

        pathway_pcts = config.get("pathway_percentages", {
            "voicenet": 0.25, "acting_challenge": 0.25, "situation": 0.25,
            "archetype": 0.15, "extreme_physical": 0.10,
        })
        lang_pcts = config.get("language_percentages", {"English": 0.50, "German": 0.50})
        default_format = config.get("prompt_generation", {}).get("default_format", "cut_to")

        # Normalize percentages
        pw_total = sum(pathway_pcts.values())
        lang_total = sum(lang_pcts.values())

        pathways = list(pathway_pcts.keys())
        pw_weights = [pathway_pcts[p] / pw_total for p in pathways]
        languages = list(lang_pcts.keys())
        lang_weights = [lang_pcts[l] / lang_total for l in languages]

        for idx in range(total_prompts):
            pw = random.choices(pathways, weights=pw_weights, k=1)[0]
            lang = random.choices(languages, weights=lang_weights, k=1)[0]
            prompt_id = f"e2e_{idx:06d}"

            if auto_resume and prompt_id in completed_ids:
                continue

            work_items.append({
                "prompt_id": prompt_id,
                "pathway": pw,
                "language": lang,
                "format": default_format,
                "original_prompt": None,
                "modified_prompt": None,
                "has_cut_to": None,
                "sample_info": {},
            })

    logging.info("Total work items: %d (after resume filtering)", len(work_items))

    if not work_items:
        logging.info("No work items to process — pipeline may already be complete")
        return

    # Distribute work items across GPUs (round-robin)
    gpu_work = {g: [] for g in gpus}
    for i, item in enumerate(work_items):
        gpu_work[gpus[i % len(gpus)]].append(item)

    for g in gpus:
        logging.info("GPU %d: %d work items", g, len(gpu_work[g]))

    # Write per-GPU work files
    work_dir = output_dir / "work_files"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_files = {}
    for g in gpus:
        wf = work_dir / f"work_gpu_{g}.json"
        with open(wf, "w") as f:
            json.dump(gpu_work[g], f, indent=1)
        work_files[g] = str(wf)

    # Start monitoring dashboard
    start_time = time.time()

    def get_progress():
        """Aggregate progress from all GPU workers."""
        progress_dir = output_dir / "progress"
        gpu_progress = []
        total_completed = 0
        total_errors = 0
        pathway_counts = {}

        for g in gpus:
            pf = progress_dir / f"gpu_{g}.json"
            try:
                if pf.exists():
                    with open(pf) as f:
                        gp = json.load(f)
                    gpu_progress.append(gp)
                    total_completed += gp.get("completed", 0)
                    total_errors += gp.get("errors", 0)
                    for pw, cnt in gp.get("pathway_counts", {}).items():
                        pathway_counts[pw] = pathway_counts.get(pw, 0) + cnt
                else:
                    gpu_progress.append({"gpu_id": g, "total": len(gpu_work.get(g, [])),
                                         "completed": 0, "errors": 0, "current_stage": "waiting"})
            except Exception:
                gpu_progress.append({"gpu_id": g, "total": 0, "completed": 0,
                                     "errors": 0, "current_stage": "unknown"})

        # Disk usage
        disk_gb = "?"
        try:
            total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
            disk_gb = f"{total_bytes / (1024**3):.1f}"
        except Exception:
            pass

        return {
            "total_prompts": len(work_items),
            "completed_prompts": total_completed,
            "total_errors": total_errors,
            "elapsed_sec": time.time() - start_time,
            "tars_created": state.get("tar_index", 0),
            "tars_uploaded": len(state.get("uploaded_tars", [])),
            "disk_used_gb": disk_gb,
            "gpus": gpu_progress,
            "pathways": pathway_counts,
        }

    monitor_server, monitor_port, tunnel_proc = start_monitor(get_progress, config)

    # Launch worker subprocesses
    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    config_path = config["_config_path"]

    # Build list of CLI overrides to forward to workers
    cli_overrides = []
    seeds_override = config.get("_cli_seeds_override")
    if seeds_override is not None:
        cli_overrides += ["--seeds", str(seeds_override)]
    mode_override = config.get("_cli_mode_override")
    if mode_override is not None:
        cli_overrides += ["--mode", str(mode_override)]

    processes = []
    for i, g in enumerate(gpus):
        cmd = [
            python_exe, "-u", script_path,
            "--config", config_path,
            "--worker",
            "--gpu", str(g),
            "--work-file", work_files[g],
        ] + cli_overrides
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)

        logging.info("Launching worker GPU %d (stagger %ds)", g, i * 2)
        if i > 0:
            time.sleep(2)

        proc = subprocess.Popen(cmd, env=env)
        processes.append((g, proc))

    logging.info("All %d workers launched", len(processes))

    # Monitor loop
    try:
        while True:
            # Check if all workers have finished
            all_done = all(proc.poll() is not None for _, proc in processes)
            if all_done:
                break

            # Print progress to terminal
            progress = get_progress()
            pct = (progress["completed_prompts"] / max(progress["total_prompts"], 1)) * 100
            print(f"\r  Progress: {progress['completed_prompts']}/{progress['total_prompts']} "
                  f"({pct:.1f}%) | Errors: {progress['total_errors']} | "
                  f"TARs: {progress['tars_created']}",
                  end="", flush=True)

            # Run tar/upload/cleanup cycle
            tar_upload_cleanup_cycle(state, config)
            save_state(state, config)

            time.sleep(60)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt — sending SIGTERM to workers")
        for g, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for g, proc in processes:
            proc.wait(timeout=30)

    # Wait for all workers
    for g, proc in processes:
        if proc.poll() is None:
            proc.wait(timeout=300)
        rc = proc.returncode
        if rc != 0:
            logging.warning("Worker GPU %d exited with code %d", g, rc)

    print()  # Newline after progress line

    # Final flush: create TARs from any remaining complete groups
    logging.info("Final TAR flush...")
    # Temporarily lower the groups_per_tar threshold for final flush
    orig_groups = config.get("batching", {}).get("groups_per_tar", 10)
    config.setdefault("batching", {})["groups_per_tar"] = 1
    tar_upload_cleanup_cycle(state, config)
    config["batching"]["groups_per_tar"] = orig_groups

    save_state(state, config)

    # Update completed prompt IDs from worker progress
    completed_all = set(state.get("completed_prompt_ids", []))
    for g in gpus:
        pf = output_dir / "progress" / f"gpu_{g}.json"
        if pf.exists():
            with open(pf) as f:
                gp = json.load(f)
            for pid in gp.get("completed_ids", []):
                completed_all.add(pid)
    state["completed_prompt_ids"] = list(completed_all)
    save_state(state, config)

    # Print summary
    progress = get_progress()
    elapsed = progress["elapsed_sec"]
    print("\n" + "=" * 70)
    print("  DramaBox E2E Pipeline — Complete")
    print("=" * 70)
    print(f"  Prompts: {progress['completed_prompts']}/{progress['total_prompts']}")
    print(f"  Errors:  {progress['total_errors']}")
    print(f"  TARs:    {state.get('tar_index', 0)} created, {len(state.get('uploaded_tars', []))} uploaded")
    print(f"  Time:    {elapsed/60:.1f} min")
    if progress['completed_prompts'] > 0:
        rate = progress['completed_prompts'] / elapsed * 3600
        print(f"  Rate:    {rate:.0f} prompts/hr")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    # Cleanup
    if monitor_server:
        monitor_server.shutdown()
    if tunnel_proc:
        tunnel_proc.terminate()


# =============================================================================
# Section 13: CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DramaBox End-to-End Voice Acting Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 2 prompts, 2 seeds, 1 GPU")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Override GPU list: '0,1,2,3'")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override seeds_per_prompt")
    parser.add_argument("--mode", choices=["generate", "ingest"], default=None,
                        help="Override pipeline mode")

    # Internal worker mode (launched by coordinator)
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--work-file", type=str, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Worker mode: run on a single GPU
    if args.worker:
        if args.gpu is None or args.work_file is None:
            parser.error("--worker requires --gpu and --work-file")
        # Collect CLI overrides to pass to worker
        overrides = {}
        if args.seeds is not None:
            overrides["seeds_per_prompt"] = args.seeds
        if args.mode is not None:
            overrides["mode"] = args.mode
        run_worker(args.gpu, args.work_file, args.config, overrides=overrides)
        return

    # Coordinator mode
    config = load_yaml_config(args.config)

    # Setup logging
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_cfg = config.get("logging", {})
    log_file = output_dir / log_cfg.get("file", "pipeline.log")
    log_level = getattr(logging, log_cfg.get("level", "INFO"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # Apply CLI overrides
    if args.mode:
        config["mode"] = args.mode
        config["_cli_mode_override"] = args.mode
    if args.gpus:
        config.setdefault("gpu", {})["devices"] = [int(x) for x in args.gpus.split(",")]
    if args.seeds:
        config.setdefault("tts", {})["seeds_per_prompt"] = args.seeds
        config["_cli_seeds_override"] = args.seeds

    # Test mode overrides
    if args.test:
        logging.info("TEST MODE: 2 prompts, 2 seeds, 1 GPU")
        config.setdefault("gpu", {})["devices"] = [detect_gpus()[0]]
        config.setdefault("tts", {})["seeds_per_prompt"] = 2
        config["_cli_seeds_override"] = 2
        config.setdefault("batching", {})["prompts_per_group"] = 2
        config.setdefault("batching", {})["groups_per_tar"] = 1
        config.setdefault("monitor", {})["enabled"] = False

    # Validate config
    warnings = validate_config(config)
    for w in warnings:
        logging.warning("Config: %s", w)

    print("=" * 70)
    print("  DramaBox End-to-End Voice Acting Data Generation Pipeline")
    print("=" * 70)
    print(f"  Mode:      {config.get('mode', 'generate')}")
    print(f"  Config:    {config['_config_path']}")
    print(f"  Output:    {config['output']['output_dir']}")
    print(f"  Storage:   {config.get('storage', {}).get('mode', 'local')}")
    gpu_devs = config.get("gpu", {}).get("devices", "auto")
    if gpu_devs == "auto":
        gpu_devs = detect_gpus()
    print(f"  GPUs:      {gpu_devs}")
    print(f"  Seeds:     {config.get('tts', {}).get('seeds_per_prompt', 25)}")
    print(f"  Loading:   {config.get('gpu', {}).get('loading_mode', 'auto')}")
    print("=" * 70)
    print()

    coordinator_main(config)


if __name__ == "__main__":
    main()
