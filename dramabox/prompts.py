"""LLM prompt construction for DramaBox format.

Supports two prompt pathways:
- **Standard**: Single-scene performance with one emotional arc.
- **CUT TO:**: Two-scene performance for the same speaker, contrasting
  two dramatically different emotional states separated by ``CUT TO:``.

Recommended model: ``google/gemma-4-E4B-it`` (official) or
``HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`` (GGUF Q8,
uncensored — produces bolder, more varied creative output).
GGUF weights: https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
"""

# ═══════════════════════════════════════════════════════════════════════════
# Format examples — used both inside system prompts and for documentation
# ═══════════════════════════════════════════════════════════════════════════

STANDARD_FORMAT_EXAMPLE = """\
A middle-to-late aged androgynous speaker with a grainy, massive, Basso \
Profondo voice, delivering this high-quality studio voice recording with \
no background noise.

The voice carries a quiet, simmering intensity — someone replaying a \
moment they can't let go of.

(Low, deliberate.) "I told myself I wouldn't come back here." \
(A slow exhale.) "But the door was open. And the light was on."

(Voice dropping further, almost to a whisper.) "You left the coffee out. \
Two cups. Like you knew."

A long pause. The weight of the room settles.

(Barely audible, rough at the edges.) "I sat down. I drank it cold. \
And I waited for something that was never going to happen."

The final words land without force — just the gravity of someone who \
has accepted what they already knew."""

CC_FORMAT_EXAMPLE = """\
A 40-year-old woman with a warm, slightly husky voice, delivering this \
high-quality studio voice recording with no background noise.

The same voice is choked with grief, barely able to speak, pulling away \
from the sight in front of her.

(a breath shuddering) "I can't... I can't watch this anymore." \
(voice cracking) "She's so small now. So thin."

CUT TO:

The same voice now softens into quiet awe, anchored by a single look \
that transforms her despair into unexpected peace.

(a tiny, wondering laugh) "There you are. That same sparkle from when \
I was five." (voice full of love) "You're still in there. Thank you."

The performance across both moments should feel like a real person \
moving from fleeing a painful room to being fully present, held by one \
memory in her mother's eyes."""

# ═══════════════════════════════════════════════════════════════════════════
# System instructions
# ═══════════════════════════════════════════════════════════════════════════

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

The content of what the speaker says should be DRAMATICALLY INTERESTING and SENSIBLE — it must form coherent, meaningful sentences that tell a story fragment, reveal character, or capture a vivid emotional moment. The dialogue should make logical sense as something a real person would actually say in a specific situation. Avoid gibberish, isolated words, or fragmentary phrases that lack context. Even short prompts should convey a complete thought or scene. It does not need to be over-the-top; it can be quiet, whispered, intimate, or subtle — but the words and the performance should feel like they belong to a compelling scene.

MINIMUM DIALOGUE: The spoken dialogue (inside quotes) must contain at least 10 words total. Even if a low word count is specified, ensure the dialogue is substantial enough to be meaningful.

You must produce EXACTLY ONE complete DramaBox prompt string. Nothing else — no markdown, no commentary, no labels.

Here is an example of a correctly formatted standard DramaBox prompt — study the formatting carefully and follow it exactly:

---EXAMPLE START---
""" + STANDARD_FORMAT_EXAMPLE + """
---EXAMPLE END---

CRITICAL FORMATTING RULES:
- All spoken dialogue uses DOUBLE QUOTES "like this" — NEVER single quotes 'like this'.
- Stage directions go in (parentheses) or as plain text paragraphs.
- Opens with a speaker description including recording quality.
- Alternates between directions and dialogue naturally.
- Output ONLY the raw prompt text. No markdown, no labels, no commentary.
"""

CC_SYSTEM_INSTRUCTION = """\
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

Output ONLY the raw prompt. Nothing else.

Here is an example of a correctly formatted CUT TO: DramaBox prompt:

---EXAMPLE START---
""" + CC_FORMAT_EXAMPLE + """
---EXAMPLE END---

CRITICAL FORMATTING RULES:
- All spoken dialogue uses DOUBLE QUOTES "like this" — NEVER single quotes.
- Scene 1 establishes one emotional state. CUT TO: marks the transition. Scene 2 is a dramatically different emotional state.
- Same speaker throughout — anchor with "the same voice".
- Output ONLY the raw prompt text. No markdown, no labels, no commentary.
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


def build_cc_prompt(sample: dict) -> str:
    """Construct the user prompt for CUT TO: (character-consistent) generation.

    This produces a two-scene prompt where the same speaker performs in
    two dramatically contrasting emotional states separated by ``CUT TO:``.
    """
    lang = sample["language"]
    accent = sample.get("accent", "")
    accent_line = f"ACCENT / DIALECT: {accent}" if accent else ""

    return f"""\
Create a character-consistent TWO-SCENE DramaBox prompt with CUT TO: transition.

TARGET LANGUAGE for all dialogue (inside "..."): {lang}
{accent_line}
EMOTIONS for Scene 1: {sample['emotions']}
VOICE ATTRIBUTES (use the most striking 5-10 to shape the speaker description):
{sample['attributes_clean']}

WORD COUNT for all spoken dialogue combined: approximately {sample['word_count_target']} words (split roughly equally across both scenes).

Instructions:
- Scene 1: Establish the character in one emotional state using the emotions above.
- CUT TO: marks the transition.
- Scene 2: The SAME speaker in a dramatically DIFFERENT emotional state — maximize the contrast.
- Same voice, same actor throughout. Anchor identity with "the same voice".

Output ONLY the raw DramaBox prompt."""


def build_full_prompt(sample: dict, vb_block: str) -> str:
    """Build complete user prompt with suffixes for either sampling path."""
    if sample["sampling_path"] == "cc":
        prompt = build_cc_prompt(sample)
    elif sample["sampling_path"] == "archetype":
        prompt = build_archetype_prompt(sample)
    else:
        prompt = build_llm_prompt(sample, vb_block)
    return prompt + SUFFIX_GENUINE + SUFFIX_SPONTANEOUS + SUFFIX_QUALITY
