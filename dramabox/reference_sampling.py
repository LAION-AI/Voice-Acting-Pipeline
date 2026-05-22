"""Path C: Reference audio pipeline with timbre whisper annotations.

Samples situation-dependent VoiceNet dimensions (excluding identity-related
attributes like age, gender, timbre, resonance) and combines them with
the timbre whisper caption from a reference audio to produce DramaBox prompts
that can be used together with the reference audio for TTS.
"""
import json
import random
from pathlib import Path

from .sampling import INTENSITY_LEVELS, TEMPO_LABELS
from .utils import weighted_choice

# VoiceNet dimension codes that are IDENTITY-RELATED and should be excluded
# from situation-dependent sampling (they describe the speaker's inherent voice,
# which is already captured by the timbre whisper caption).
IDENTITY_DIM_CODES = {
    # Speaker physical characteristics
    "GEND",  # Perceived Gender
    "AGEV",  # Perceived Age
    # Timbral qualities (inherent to the voice)
    "BRGT",  # Brightness
    "ROUG",  # Roughness
    "HARM",  # Harmonicity
    "FULL",  # Fullness
    "WARM",  # Warmth
    "METL",  # Metallic quality
    "ESTH",  # Aesthetic quality
    # Resonance placement (physical vocal tract)
    "R_CHST", "R_THRT", "R_ORAL", "R_MASK", "R_NASL", "R_HEAD", "R_MIXD",
    # Recording/technical (not performance-related)
    "RCQL",  # Recording Quality
    "BKGN",  # Background Noise
    "EXPL",  # Explicitness
}

# All other dimensions are SITUATION-DEPENDENT — they describe performance,
# delivery, emotional state, and speaking style that can vary across utterances.
# Examples: Tempo, Cognitive Load, Articulation, Disfluency, Speaking Styles
# (ASMR, Newsreader, Rant), Arousal, Tension, Stance, etc.

# System instruction for Path C prompts
SYSTEM_INSTRUCTION_PATHC = """\
You are a scriptwriter who creates voice-performance prompts in the DramaBox format for a single speaker.

DramaBox format rules:
- Everything is ONE speaker. Never introduce a second character or a dialogue partner.
- Start with a speaker description (age, gender, timbre, voice quality) — always in English.
- Then alternate between stage directions (in English, outside quotes) and spoken dialogue (in the target language, inside double quotes "...").
- Stage directions describe actions, pauses, emotional shifts, vocal changes — NEVER spoken aloud.
- Direct speech MUST be in the specified target language.
- Do NOT put sound effects like "sigh", "gasp", "cough" inside quotes — keep those as stage directions.
- Phonetic vocalizations like "Hahaha", "Mmm", "Ugh" CAN go inside quotes.

Recording environment assumption:
- The output prompt MUST include, as part of the opening speaker description, a statement that this is a high-quality studio voice recording with no background noise.

IMPORTANT: You will be given a TIMBRE DESCRIPTION of the speaker's voice. Your speaker description MUST be consistent with this timbre — same gender, similar age range, matching vocal qualities. Do NOT contradict the timbre information.

The content should be DRAMATICALLY INTERESTING — with a sense of story, situation, or emotional arc.

You must produce EXACTLY ONE complete DramaBox prompt string. Nothing else — no markdown, no commentary, no labels.
"""


def load_reference_metadata(json_path: str | Path) -> dict:
    """Load reference audio metadata from a JSON file.

    Expected format matches the Emolia dataset: contains timbre_caption,
    timbre_tags, emotion_annotation, speaker info, etc.
    """
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def get_timbre_caption(metadata: dict, caption_key: str = "timbre_caption") -> str:
    """Extract the timbre caption from reference metadata.

    Args:
        metadata: Reference audio metadata dict.
        caption_key: Key name for the timbre annotation field.

    Returns:
        Timbre caption string.
    """
    caption = metadata.get(caption_key, "")
    if not caption:
        # Try alternative keys
        for alt_key in ["timbre_full_prediction", "timbre_description", "caption"]:
            caption = metadata.get(alt_key, "")
            if caption:
                break
    return caption


def get_situation_dependent_dims(all_dims: list[dict]) -> list[dict]:
    """Filter VoiceNet dimensions to only situation-dependent ones.

    Excludes identity-related dimensions (age, gender, timbre, resonance,
    recording quality) since those are captured by the reference audio's
    timbre whisper caption.
    """
    return [d for d in all_dims if d["code"] not in IDENTITY_DIM_CODES]


def sample_reference_path(
    timbre_caption: str,
    situation_dims: list[dict],
    emotion_categories: list[str],
    config: dict,
    reference_audio_path: str = "",
    reference_metadata: dict | None = None,
) -> dict:
    """Sample attributes for Path C (reference audio pipeline).

    Args:
        timbre_caption: Timbre whisper caption describing the speaker's voice.
        situation_dims: List of situation-dependent VoiceNet dimension dicts.
        emotion_categories: List of emotion category names.
        config: Full configuration dict.
        reference_audio_path: Path to the reference audio file.
        reference_metadata: Full metadata dict from the reference JSON.

    Returns:
        Dict with all sampled attributes for prompt construction.
    """
    sampling = config.get("sampling", {})
    ref_cfg = config.get("reference_audio", {})

    # Sample language and accent
    languages = config["_active_languages"]
    accents_map = config["_language_accents"]
    language = random.choice(languages)
    accents = accents_map.get(language, [])
    accent = random.choice(accents) if accents else ""

    # Sample 1-3 emotions
    n_min = sampling.get("emotions_min", 1)
    n_max = sampling.get("emotions_max", 3)
    n_emotions = random.randint(n_min, n_max)
    chosen = random.sample(emotion_categories, min(n_emotions, len(emotion_categories)))
    emotion_parts = [f"{emo} ({random.choice(INTENSITY_LEVELS)})" for emo in chosen]
    emotions_str = ", ".join(emotion_parts)

    # Sample tempo with bias
    tempo_bias_threshold = sampling.get("tempo_bias_threshold", 3)
    tempo_bias_weight = sampling.get("tempo_bias_weight", 1.5)

    temp_dim = None
    for d in situation_dims:
        if d["code"] == "TEMP":
            temp_dim = d
            break

    tempo_val = 3
    tempo_desc = "standard, unremarkable tempo"
    if temp_dim:
        weights = [tempo_bias_weight if lv["val"] >= tempo_bias_threshold else 1.0
                   for lv in temp_dim["levels"]]
        level = random.choices(temp_dim["levels"], weights=weights, k=1)[0]
        tempo_val = level["val"]
        tempo_desc = TEMPO_LABELS.get(tempo_val, level["desc"][:60])

    # Sample 5 situation-dependent VoiceNet dimensions (excluding TEMP which we already sampled)
    n_situation = ref_cfg.get("situation_dims_count", 5)
    eligible = [d for d in situation_dims if d["code"] != "TEMP"]
    chosen_dims = random.sample(eligible, min(n_situation, len(eligible)))

    sampled_attrs = []
    for dim in chosen_dims:
        level = random.choice(dim["levels"])
        sampled_attrs.append({
            "code": dim["code"], "name": dim["name"],
            "level_val": level["val"], "level_desc": level["desc"],
        })

    # Build attributes string
    attrs_parts = [f"Tempo: {tempo_desc} (level {tempo_val})"]
    for a in sampled_attrs:
        attrs_parts.append(f"{a['name']}: {a['level_desc']}")
    attributes_display = " | ".join(attrs_parts)

    attrs_raw_parts = [f"Tempo [TEMP] (level {tempo_val}): {tempo_desc}"]
    for a in sampled_attrs:
        attrs_raw_parts.append(
            f"{a['name']} [{a['code']}] (level {a['level_val']}): {a['level_desc']}"
        )
    attributes_raw = " | ".join(attrs_raw_parts)

    # Word count
    word_min = sampling.get("word_count_min", 5)
    word_max = sampling.get("word_count_max", 60)
    word_count = random.randint(word_min, word_max)

    return {
        "sampling_path": "reference",
        "archetype_info": "",
        "language": language,
        "accent": accent,
        "emotions": emotions_str,
        "attributes_clean": attributes_display,
        "attributes_raw": attributes_raw,
        "flow_style": "",
        "flow_forced_by_voicenet": False,
        "emotion_alignment": "",
        "direction_style": "",
        "vocal_bursts_enabled": False,
        "word_count_target": word_count,
        "must_include_words": [],
        "reference_audio": reference_audio_path,
        "timbre_caption": timbre_caption,
        "_sampled_dims": sampled_attrs,
        "_tempo_val": tempo_val,
        "_tempo_desc": tempo_desc,
    }


def build_reference_prompt(sample: dict) -> str:
    """Build the LLM user prompt for Path C (reference audio pipeline)."""
    lang = sample["language"]
    accent = sample["accent"]
    accent_line = f"ACCENT / DIALECT: {accent}" if accent else ""

    return f"""\
Create a single DramaBox-format voice prompt for a speaker whose voice is described below.

SPEAKER TIMBRE (from reference audio — your speaker description MUST be consistent with this):
{sample['timbre_caption']}

TARGET LANGUAGE for all dialogue (inside "..."): {lang}
{accent_line}
EMOTIONS conveyed: {sample['emotions']}

SITUATION-DEPENDENT VOICE ATTRIBUTES (apply these to the performance style, NOT to the speaker's inherent voice):
{sample['attributes_clean']}

WORD COUNT for all spoken dialogue combined (inside "..."): approximately {sample['word_count_target']} words.

Instructions:
- Your speaker description (age, gender, timbre) MUST match the timbre description above.
- The situation-dependent attributes describe HOW the speaker performs in this particular scene — tempo, cognitive load, articulation style, speaking style, etc.
- Write a dramatically interesting scene with a sense of story and emotional arc.
- Incorporate the emotions meaningfully into the performance.

Structure:
1. Open with 1-3 English sentences describing the single speaker (age, gender, timbre, voice qualities). IMPORTANT: Make this consistent with the timbre description. Include that this is a pristine, high-quality studio recording with no background noise.
2. Write the performance for ONE speaker: stage directions (English) + dialogue ("{lang}", in quotes).
3. Close with 1-2 sentences of final direction (English).
4. SINGLE SPEAKER only. No dialogue partners.
5. All dialogue in {lang}. Directions in clear, plain English.
6. Output ONLY the raw DramaBox prompt string."""


def build_reference_full_prompt(sample: dict) -> str:
    """Build complete Path C prompt with suffixes."""
    from .prompts import SUFFIX_GENUINE, SUFFIX_SPONTANEOUS, SUFFIX_QUALITY
    return build_reference_prompt(sample) + SUFFIX_GENUINE + SUFFIX_SPONTANEOUS + SUFFIX_QUALITY
