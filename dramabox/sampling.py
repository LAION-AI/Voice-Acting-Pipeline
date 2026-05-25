"""Attribute sampling for DramaBox prompts.

Path A (VoiceNet-based): Full VoiceNet dimension sampling with flow/alignment/direction.
Path B (Archetype-based): Genre archetype sampling with tempo + arousal only.
"""
import random
import re

from .utils import weighted_choice

# Intensity levels for sampled emotions
INTENSITY_LEVELS = [
    "slightly present",
    "clearly present",
    "extremely present",
    "very intensely present",
]

# Tempo descriptors (for archetype path)
TEMPO_LABELS = {
    0: "glacially slow",
    1: "unusually deliberate and labored",
    2: "relaxed and unhurried",
    3: "standard, unremarkable tempo",
    4: "brisk, elevated momentum",
    5: "noticeably compressed and fast",
    6: "absolute maximum speed",
}

# Arousal descriptors (for archetype path)
AROUSAL_LABELS = {
    0: "absolute minimum physiological activation",
    1: "extremely low vocal energy, sedate",
    2: "awake but consciously restrained",
    3: "standard baseline energy",
    4: "clearly elevated physiological activation",
    5: "high autonomic activation, intense",
    6: "maximum physiological mobilization",
}

# VoiceNet dimensions that indicate fragmented/choppy delivery
FRAGMENTATION_DIMS = {"CHNK", "SMTH", "DFLU", "VFLX", "VOLT"}

# NSFW safety: minor-indicating patterns and emotions to strip
MINOR_PATTERNS = re.compile(
    r"\b(toddler|child|kid|baby|infant|teenager|teen|adolescent"
    r"|young\s*child|little\s*boy|little\s*girl)\b",
    re.IGNORECASE,
)
NSFW_EMOTION_CATS = {"Sexual Lust", "Infatuation"}

# Broad safety filter for combined prompt text
MINOR_WORDS = {"kid", "teen", "boy", "girl", "child", "toddler", "baby",
               "infant", "minor", "underage", "youth", "neonatal",
               "pre-pubescent", "adolescent", "childhood"}
ADULT_WORDS = {"porn", "sex", "fuck", "lust", "passion", "gore",
               "erotic", "naked", "nude", "explicit"}


def is_prompt_safe(text: str) -> bool:
    """Return False if text combines minor-indicating AND adult words."""
    words = set(re.findall(r'[a-z]+(?:-[a-z]+)*', text.lower()))
    return not (bool(words & MINOR_WORDS) and bool(words & ADULT_WORDS))


def _sample_emotions(emotion_categories: list[str], config: dict) -> str:
    """Sample 1-3 emotions with intensity levels."""
    sampling = config.get("sampling", {})
    n_min = sampling.get("emotions_min", 1)
    n_max = sampling.get("emotions_max", 3)
    n_emotions = random.randint(n_min, n_max)
    chosen = random.sample(emotion_categories, min(n_emotions, len(emotion_categories)))
    parts = [f"{emo} ({random.choice(INTENSITY_LEVELS)})" for emo in chosen]
    return ", ".join(parts)


def _sample_language(config: dict) -> tuple[str, str]:
    """Sample a language and optional accent from config."""
    languages = config["_active_languages"]
    accents_map = config["_language_accents"]
    language = random.choice(languages)
    accents = accents_map.get(language, [])
    accent = random.choice(accents) if accents else ""
    return language, accent


def sample_voicenet(mandatory_dims: list[dict], optional_dims: list[dict],
                    emotion_categories: list[str], config: dict,
                    wordlist_fn=None) -> dict:
    """Sample a full VoiceNet-based prompt specification (Path A).

    Args:
        mandatory_dims: List of mandatory VoiceNet dimension dicts.
        optional_dims: List of optional VoiceNet dimension dicts.
        emotion_categories: List of emotion category names.
        config: Full configuration dict.
        wordlist_fn: Callable(language) -> list[str] for mandatory words.

    Returns:
        Dict with all sampled attributes for prompt construction.
    """
    sampling = config.get("sampling", {})
    language, accent = _sample_language(config)
    emotions_str = _sample_emotions(emotion_categories, config)

    # Mandatory dimensions (with tempo bias)
    tempo_bias_threshold = sampling.get("tempo_bias_threshold", 3)
    tempo_bias_weight = sampling.get("tempo_bias_weight", 1.5)

    mandatory_attrs = []
    for dim in mandatory_dims:
        if dim["code"] == "TEMP":
            weights = [tempo_bias_weight if lv["val"] >= tempo_bias_threshold else 1.0
                       for lv in dim["levels"]]
            level = random.choices(dim["levels"], weights=weights, k=1)[0]
        else:
            level = random.choice(dim["levels"])
        mandatory_attrs.append({
            "code": dim["code"], "name": dim["name"],
            "level_val": level["val"], "level_desc": level["desc"],
        })

    # Random optional dimensions
    n_random = sampling.get("random_dims_count", 5)
    chosen_optional = random.sample(optional_dims, min(n_random, len(optional_dims)))
    random_attrs = []
    for dim in chosen_optional:
        level = random.choice(dim["levels"])
        random_attrs.append({
            "code": dim["code"], "name": dim["name"],
            "level_val": level["val"], "level_desc": level["desc"],
        })

    all_attrs = mandatory_attrs + random_attrs
    attributes_clean = " | ".join(f"{a['name']}: {a['level_desc']}" for a in all_attrs)
    attributes_raw = " | ".join(
        f"{a['name']} [{a['code']}] (level {a['level_val']}): {a['level_desc']}"
        for a in all_attrs
    )

    # NSFW safety: if age suggests a minor, patch
    has_minor = bool(MINOR_PATTERNS.search(attributes_raw))
    if has_minor:
        attributes_clean = MINOR_PATTERNS.sub("adult", attributes_clean)
        attributes_raw = MINOR_PATTERNS.sub("adult", attributes_raw)
        for cat in NSFW_EMOTION_CATS:
            emotions_str = re.sub(rf"{re.escape(cat)}\s*\([^)]*\),?\s*", "", emotions_str)
        emotions_str = emotions_str.strip().rstrip(",").strip()
        if not emotions_str:
            emotions_str = "Contentment (clearly present)"

    # Flow style
    flow_dist = sampling.get("flow_style_distribution",
                             {"scattered": 0.05, "flowing": 0.55, "mixed": 0.40})
    flow_options = [(k, v) for k, v in flow_dist.items()]

    flow_forced = False
    for a in all_attrs:
        code, lv = a["code"], a["level_val"]
        if code in FRAGMENTATION_DIMS:
            if code in ("CHNK", "SMTH") and lv <= 1:
                flow_forced = True
            elif code in ("DFLU", "VFLX", "VOLT") and lv >= 5:
                flow_forced = True
    flow_style = "scattered" if flow_forced else weighted_choice(flow_options)

    # Other sampling decisions
    align_dist = sampling.get("emotion_alignment_distribution",
                              {"congruent": 0.30, "neutral": 0.40, "counter-emotional": 0.30})
    dir_dist = sampling.get("direction_style_distribution",
                            {"literary": 0.50, "tag": 0.50})
    vb_prob = sampling.get("vocal_bursts_probability", 0.50)

    emotion_alignment = weighted_choice([(k, v) for k, v in align_dist.items()])
    direction_style = weighted_choice([(k, v) for k, v in dir_dist.items()])
    vocal_bursts = random.random() < vb_prob

    word_min = sampling.get("word_count_min", 5)
    word_max = sampling.get("word_count_max", 60)
    word_count = random.randint(word_min, word_max)

    # Mandatory words
    must_words = []
    n_must = sampling.get("mandatory_words_count", 3)
    if wordlist_fn is not None:
        wl = wordlist_fn(language)
        must_words = random.sample(wl, min(n_must, len(wl)))

    return {
        "sampling_path": "voicenet",
        "archetype_info": "",
        "language": language,
        "accent": accent,
        "emotions": emotions_str,
        "attributes_clean": attributes_clean,
        "attributes_raw": attributes_raw,
        "flow_style": flow_style,
        "flow_forced_by_voicenet": flow_forced,
        "emotion_alignment": emotion_alignment,
        "direction_style": direction_style,
        "vocal_bursts_enabled": vocal_bursts,
        "word_count_target": word_count,
        "must_include_words": must_words,
    }


def sample_archetype(archetypes: dict[str, list[str]],
                     temp_dim: dict, arou_dim: dict,
                     emotion_categories: list[str],
                     config: dict) -> dict:
    """Sample an archetype-based prompt specification (Path B).

    Args:
        archetypes: Dict mapping genre names to lists of archetype descriptions.
        temp_dim: The TEMP VoiceNet dimension dict (for tempo sampling).
        arou_dim: The AROU VoiceNet dimension dict (for arousal sampling).
        emotion_categories: List of emotion category names.
        config: Full configuration dict.

    Returns:
        Dict with all sampled attributes for prompt construction.
    """
    sampling = config.get("sampling", {})

    # Safety-filtered archetype sampling
    for _attempt in range(50):
        genre = random.choice(list(archetypes.keys()))
        archetype = random.choice(archetypes[genre])
        if is_prompt_safe(f"{genre} {archetype}"):
            break

    language, accent = _sample_language(config)
    emotions_str = _sample_emotions(emotion_categories, config)

    # Tempo with bias
    tempo_bias_threshold = sampling.get("tempo_bias_threshold", 3)
    tempo_bias_weight = sampling.get("tempo_bias_weight", 1.5)
    temp_weights = [tempo_bias_weight if lv["val"] >= tempo_bias_threshold else 1.0
                    for lv in temp_dim["levels"]]
    temp_level = random.choices(temp_dim["levels"], weights=temp_weights, k=1)[0]
    tempo_val = temp_level["val"]
    tempo_desc = TEMPO_LABELS.get(tempo_val, temp_level["desc"][:60])

    # Arousal (uniform)
    arou_level = random.choice(arou_dim["levels"])
    arousal_val = arou_level["val"]
    arousal_desc = AROUSAL_LABELS.get(arousal_val, arou_level["desc"][:60])

    word_min = sampling.get("word_count_min", 5)
    word_max = sampling.get("word_count_max", 60)
    word_count = random.randint(word_min, word_max)

    attributes_raw = (
        f"Tempo [TEMP] (level {tempo_val}): {temp_level['desc']} | "
        f"Arousal [AROU] (level {arousal_val}): {arou_level['desc']}"
    )

    return {
        "sampling_path": "archetype",
        "archetype_info": f"{genre} | {archetype}",
        "language": language,
        "accent": accent,
        "emotions": emotions_str,
        "attributes_clean": "",
        "attributes_raw": attributes_raw,
        "flow_style": "",
        "flow_forced_by_voicenet": False,
        "emotion_alignment": "",
        "direction_style": "",
        "vocal_bursts_enabled": False,
        "word_count_target": word_count,
        "must_include_words": [],
        "_genre": genre,
        "_archetype": archetype,
        "_tempo_desc": tempo_desc,
        "_arousal_desc": arousal_desc,
    }
