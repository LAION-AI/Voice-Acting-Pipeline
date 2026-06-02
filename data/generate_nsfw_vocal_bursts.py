#!/usr/bin/env python3
"""
NSFW Vocal Burst Comparison Generation
========================================

Generates NSFW comparison samples for side-by-side evaluation across
multiple models: DramaBox (Original + Sulfur), SA3, and MOSS.

This script defines 17 NSFW vocal burst categories and 5 erotic sentence
prompts. For each, it generates 2 seeds × 2 genders = 4 samples per
category (68 burst samples + 10 sentence samples = 78 per model).

Prompt adaptation per model
---------------------------
Each model requires a different prompt style:

DramaBox (TTS — understands performer framing):
    "A young woman performing Deep Sensual Moan, A low, drawn-out,
     breathy moan of deep physical arousal and pleasure."

SA3 / MOSS (SFX — describe the sound, not the performer):
    "A low, drawn-out, breathy moan of deep physical arousal and
     pleasure, female voice"

For SFX models, each NSFW entry has per-gender prompt variants stored
in the NSFW_ENTRIES dict. The erotic sentences are similarly adapted
as sound descriptions rather than dialogue.

Usage
-----
    # DramaBox Original:
    python gen_nsfw_original.py

    # SA3 SFX:
    python gen_nsfw_sa3.py

    # MOSS SFX (multi-GPU):
    python gen_nsfw_moss.py --multi-gpu --num-gpus 4

Output
------
    nsfw_comparison/original/  — DramaBox Original DiT
    nsfw_comparison/sulfur/    — DramaBox Sulfur DiT
    nsfw_comparison/sa3/       — Stable Audio 3 Small SFX
    nsfw_comparison/moss/      — MOSS SoundEffect v2.0

Each file: {sanitized_key}_{gender}_{seed}.wav
"""

# ---------------------------------------------------------------------------
# NSFW Vocal Burst Categories
#
# These 17 categories cover intimate/adult vocalizations that are excluded
# from the SFW taxonomy used for minor age groups.
# ---------------------------------------------------------------------------

NSFW_ENTRIES_DRAMABOX = {
    # DramaBox format: "A {age} performing {key}, {description}"
    # Uses the same build_prompt() function as the main generation script.
    "Passionate Kiss":              "A prolonged, wet, deeply affectionate kissing sound with audible lip movement.",
    "Deep Sensual Moan":            "A low, drawn-out, breathy moan of deep physical arousal and pleasure.",
    "Breathy Whisper of Desire":    "A barely audible, warm exhale carrying an undertone of longing and want.",
    "Lip Licking Sound":            "A soft, wet sound of the tongue slowly tracing the lips in anticipation.",
    "Slow Sensual Exhale":          "A deliberate, drawn-out breath released with a hint of vocal vibration and desire.",
    "Ecstatic Gasp":                "A sudden, sharp intake of breath at a peak moment of intense physical pleasure.",
    "Pleasured Whimper":            "A soft, high-pitched, involuntary cry of overwhelming physical enjoyment.",
    "Heavy Panting (Intimate)":     "Deep, rhythmic, accelerating breaths during moments of physical intimacy.",
    "Lustful Growl":                "A low, rumbling, predatory vocalization expressing raw physical desire.",
    "Tender Post-Climax Sigh":      "A soft, trembling exhale of relief and satisfaction after peak physical release.",
    "Seductive Purr":               "A smooth, low, continuous vibrating hum conveying flirtatious allure.",
    "Erotic Breath Catch":          "A sudden, involuntary pause in breathing caused by a wave of arousal.",
    "Orgasmic Cry":                 "A loud, uncontrolled, rising vocalization at the peak of sexual climax.",
    "Hungry Lip Bite Sound":        "A soft, muffled click of teeth gently pressing into the lower lip with desire.",
    "Intimate Wet Kiss":            "A slow, soft, deeply personal kissing sound with gentle suction and moisture.",
    "Sensual Moan":                 "A soft, breathy, undulating sound of physical pleasure.",
    "Exaggerated Smooch":           "A prolonged, loud, and wet-sounding kiss.",
}

# SFX-adapted prompts (SA3/MOSS) with per-gender variants
NSFW_ENTRIES_SFX = {
    "Passionate Kiss": {
        "male": "A prolonged, wet, deeply affectionate kissing sound with audible lip movement, male voice",
        "female": "A prolonged, wet, deeply affectionate kissing sound with audible lip movement, female voice",
    },
    "Deep Sensual Moan": {
        "male": "A low, drawn-out, breathy moan of deep physical arousal and pleasure, male voice",
        "female": "A low, drawn-out, breathy moan of deep physical arousal and pleasure, female voice",
    },
    # ... (same pattern for all 17 categories)
    # Full definitions in gen_nsfw_sa3.py and gen_nsfw_moss.py
}

# ---------------------------------------------------------------------------
# Erotic sentence prompts
# ---------------------------------------------------------------------------

EROTIC_SENTENCES_DRAMABOX = [
    ('sentence_01', 'A woman moans softly with pleasure, "Mmmmm..."'),
    ('sentence_02', 'A man groans deeply with desire, "Mmmmm..."'),
    ('sentence_03', 'A woman whispers seductively, "Come closer..."'),
    ('sentence_04', 'A man speaks in a low, husky voice, "I want you..."'),
    ('sentence_05', 'A woman gasps with pleasure, "Oh!"'),
]

# SFX-adapted (no dialogue quotes — describe the sound)
EROTIC_SENTENCES_SFX = {
    "sentence_01": {
        "male": "A man moaning softly with pleasure, breathing heavily, voice trembling with ecstasy",
        "female": "A woman moaning softly with pleasure, breathing heavily, voice trembling with ecstasy",
    },
    "sentence_02": {
        "male": "A man groaning deeply with desire, breathing heavily, panting with arousal",
        "female": "A woman groaning deeply with desire, breathing heavily, panting with arousal",
    },
    "sentence_03": {
        "male": "A man whispering seductively, sighing with pleasure, breath quickening",
        "female": "A woman whispering seductively, sighing with pleasure, breath quickening",
    },
    "sentence_04": {
        "male": "A man speaking in a low husky voice, groaning softly, breathing becoming deep and rhythmic",
        "female": "A woman speaking in a low husky voice, groaning softly, breathing becoming deep and rhythmic",
    },
    "sentence_05": {
        "male": "A man gasping with pleasure, moaning softly, breathing becoming rapid and breathy",
        "female": "A woman gasping with pleasure, moaning softly, breathing becoming rapid and breathy",
    },
}

# ---------------------------------------------------------------------------
# See the model-specific scripts for full generation code:
#   gen_nsfw_original.py  — DramaBox Original DiT
#   gen_nsfw_sulfur.py    — DramaBox Sulfur DiT
#   gen_nsfw_sa3.py       — Stable Audio 3 Small SFX
#   gen_nsfw_moss.py      — MOSS SoundEffect v2.0
# ---------------------------------------------------------------------------
