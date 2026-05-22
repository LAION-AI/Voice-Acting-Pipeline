"""LLM prompt construction for DramaBox format."""

SYSTEM_INSTRUCTION = """\
You are a scriptwriter who creates voice-performance prompts in the DramaBox format for a single speaker.

DramaBox format rules:
- Everything is ONE speaker. Never introduce a second character or a dialogue partner.
- Start with a speaker description (age, gender, timbre, voice quality) — always in English.
- Then alternate between stage directions (in English, outside quotes) and spoken dialogue (in the target language, inside double quotes "...").
- Stage directions describe actions, pauses, emotional shifts, vocal changes — NEVER spoken aloud.
- Direct speech MUST be in the specified target language.
- Do NOT put sound effects like "sigh", "gasp", "cough" inside quotes — keep those as stage directions.
- Phonetic vocalizations like "Hahaha", "Mmm", "Ugh" CAN go inside quotes.
- NEVER use abbreviations, codes, or shorthand from any taxonomy in the directions. All stage directions must be written in clear, plain, self-contained English that any actor could immediately understand without reference material.

Recording environment assumption:
- The output prompt MUST include, as part of the opening speaker description, a statement that this is a high-quality studio voice recording with no background noise — as if captured in a professional voice acting / audio studio environment.

The content of what the speaker says should be DRAMATICALLY INTERESTING — it should have a sense of story, situation, or emotional arc. It does not need to be over-the-top; it can be quiet, whispered, intimate, or subtle — but the words and the performance should feel like they belong to a compelling scene.

You must produce EXACTLY ONE complete DramaBox prompt string. Nothing else — no markdown, no commentary, no labels.
"""

# Flow style sub-instructions
FLOW_INSTRUCTIONS = {
    "scattered": """\
SPEECH FLOW STYLE: SCATTERED
- Break dialogue into very short fragments (1–4 words each).
- Insert a stage direction between nearly every fragment.
- The result should feel choppy, halting, interrupted.
""",
    "flowing": """\
SPEECH FLOW STYLE: FLOWING / CONTINUOUS
- Write dialogue as long, unbroken blocks (8–25+ words per quoted segment).
- Use very few stage directions — at most 2–4 across the entire performance.
- Prioritize natural flow. Let the speaker talk.
""",
    "mixed": """\
SPEECH FLOW STYLE: MIXED / ORGANIC
- Alternate between longer passages (5–15 words) and shorter fragments (2–5 words).
- Insert stage directions naturally at emotional or physical shift points.
- Vary segment lengths — avoid regularity.
""",
}

# Emotion alignment sub-instructions
ALIGNMENT_INSTRUCTIONS = {
    "congruent": """\
EMOTION–TEXT ALIGNMENT: CONGRUENT
- The CONTENT of what is said should directly match and reinforce the emotions.
""",
    "neutral": """\
EMOTION–TEXT ALIGNMENT: NEUTRAL TEXT / EMOTIONAL DELIVERY
- The CONTENT should be emotionally neutral or mundane.
- The stage directions should instruct the performer to deliver with the specified emotions.
""",
    "counter-emotional": """\
EMOTION–TEXT ALIGNMENT: COUNTER-EMOTIONAL
- The CONTENT should actively contradict the emotions in the directions.
- If sad, the words should be cheerful. If angry, the words should be gentle. Etc.
""",
}

# Direction writing style sub-instructions
DIRECTION_STYLE_INSTRUCTIONS = {
    "literary": """\
DIRECTION WRITING STYLE: LITERARY
- Write stage directions as a thoughtful director would brief an actor.
- Use full, evocative sentences.
""",
    "tag": """\
DIRECTION WRITING STYLE: TAG / CONDENSED
- Write stage directions as ultra-brief tags: "Softly.", "Voice breaking.", "Turns away."
- 1–5 words maximum per tag. This applies ONLY to directions, not to dialogue.
""",
}

# Suffixes appended to every prompt
SUFFIX_GENUINE = (
    "\n\nThe voice must sound completely genuine and human — "
    "not robotic, not exaggerated, not 'acted.' It should feel like "
    "a real person in a real moment."
)
SUFFIX_SPONTANEOUS = (
    "\n\nThe delivery should feel spontaneous and unrehearsed — "
    "like the words are being discovered in the moment, not recited."
)
SUFFIX_QUALITY = (
    "\n\nThis is a pristine, high-quality studio recording with "
    "no background noise."
)


def build_llm_prompt(sample: dict, vb_block: str) -> str:
    """Construct the user prompt for Path A (VoiceNet-based) generation."""
    lang = sample["language"]
    accent = sample["accent"]
    accent_line = f"ACCENT / DIALECT: {accent}" if accent else ""
    vb_section = f"\n{vb_block}\n" if sample["vocal_bursts_enabled"] else ""

    return f"""\
Create a single DramaBox-format voice prompt with these constraints:

TARGET LANGUAGE for all dialogue (inside "..."): {lang}
{accent_line}
EMOTIONS conveyed: {sample['emotions']}
VOICE ATTRIBUTES (use the most striking 5–10 of these to shape the speaker description):
{sample['attributes_clean']}

MANDATORY WORDS that must appear naturally in the spoken dialogue: {', '.join(sample['must_include_words'])}
WORD COUNT for all spoken dialogue combined (inside "..."): approximately {sample['word_count_target']} words.

{FLOW_INSTRUCTIONS[sample['flow_style']]}

{ALIGNMENT_INSTRUCTIONS[sample['emotion_alignment']]}

{DIRECTION_STYLE_INSTRUCTIONS[sample['direction_style']]}
{vb_section}
Structure:
1. Open with 1–3 English sentences describing the single speaker (age, gender, timbre, voice qualities{', accent' if accent else ''}). IMPORTANT: include that this is a pristine, high-quality studio recording with no background noise.
2. Write the performance for ONE speaker: stage directions (English) + dialogue ("{lang}", in quotes).
3. Close with 1–2 sentences of final direction (English).
4. SINGLE SPEAKER only. No dialogue partners.
5. All dialogue in {lang}. Directions in clear, plain English — no abbreviations or codes.
6. Output ONLY the raw DramaBox prompt string."""


def build_archetype_prompt(sample: dict) -> str:
    """Construct the user prompt for Path B (archetype-based) generation."""
    lang = sample["language"]
    accent = sample["accent"]
    accent_line = f"ACCENT / DIALECT: {accent}" if accent else ""

    return f"""\
Create a single DramaBox-format voice prompt for this character archetype:

ARCHETYPE: {sample['_archetype']} (from genre: {sample['_genre']})
TARGET LANGUAGE for all dialogue (inside "..."): {lang}
{accent_line}
EMOTIONS conveyed: {sample['emotions']}
TEMPO: {sample['_tempo_desc']}
AROUSAL: {sample['_arousal_desc']}
WORD COUNT for all spoken dialogue combined (inside "..."): approximately {sample['word_count_target']} words.

Instructions:
- Do NOT reproduce the archetype description literally. Use it as inspiration and vary it creatively — the character should be recognizably this archetype but with your own unique spin.
- Write a dramatically interesting scene with a sense of story and emotional arc.
- Naturally reflect the given tempo and arousal level in both the stage directions and the character's delivery.
- Incorporate the emotions meaningfully into the performance.

Structure:
1. Open with 1-3 English sentences describing the single speaker (age, gender, timbre, voice qualities). IMPORTANT: include that this is a pristine, high-quality studio recording with no background noise.
2. Write the performance for ONE speaker: stage directions (English) + dialogue ("{lang}", in quotes).
3. Close with 1-2 sentences of final direction (English).
4. SINGLE SPEAKER only. No dialogue partners.
5. All dialogue in {lang}. Directions in clear, plain English.
6. Output ONLY the raw DramaBox prompt string."""


def build_full_prompt(sample: dict, vb_block: str) -> str:
    """Build complete user prompt with suffixes for either sampling path."""
    if sample["sampling_path"] == "archetype":
        prompt = build_archetype_prompt(sample)
    else:
        prompt = build_llm_prompt(sample, vb_block)
    return prompt + SUFFIX_GENUINE + SUFFIX_SPONTANEOUS + SUFFIX_QUALITY
