# Path SIT — Situation-Driven Acting

Situation-driven performance generation. The actor is physically and socially embedded in a specific situation from the **Situation Taxonomy** — their body posture, physical activity, social context, environment, health, or pain state naturally affects how they speak and perform.

## Situation Taxonomy

The [Situation Taxonomy](../data/situation_taxonomy.json) extends the core [VoiceNet taxonomy](voicenet_taxonomy.md) (57 voice attribute dimensions) with **11 situation-dependent dimensions** containing **289 total situations** that describe how a speaker's physical state and environment alter their vocal output.

Based on the [Extended VoiceNet Taxonomy](https://projects.laion.ai/Voice-Acting-Pipeline/voicenet_extension_taxonomy.html) from [Schuhmann et al., 2025](https://arxiv.org/abs/2505.20033).

| Dimension | Code | Levels | Description |
|-----------|------|--------|-------------|
| Body Posture & Gravitational Alignment | POSE | 32 | How gravity, skeletal alignment, and thoracic compression alter the vocal tract and diaphragm |
| Physical Activity & Dynamic Load | ACTV | 69 | How metabolic demand, physical movement, and interaction with objects compete with the speech signal |
| Speaking Target & Projection | TRGT | 25 | Who or what the speaker is addressing — determines throw of voice, register, and feedback loop |
| Social Situation & Context | SOCT | 56 | Social dynamics, power relations, and contextual norms that shape vocal behavior |
| Environment & Acoustic Space | ENVI | 22 | Physical space and its acoustic properties affecting the voice |
| Health & Physiological Condition | HLTH | 18 | Medical conditions, illnesses, and physiological states that alter vocal production |
| Face & Head Obstructions | GEAR | 14 | Physical obstructions (masks, helmets, food) that filter or modify the voice |
| Climate & Atmospheric Conditions | CLMT | 10 | Temperature, humidity, and weather effects on vocal tract and breathing |
| Substance & Chemical Influence | SUBST | 12 | Chemical substances affecting vocal control, coordination, and quality |
| Fatigue, Sleep & Energy State | FATG | 19 | Sleep deprivation, exhaustion, and energy levels impacting vocal effort |
| Pain & Physical Distress | PAIN | 12 | Active pain states and their effect on breathing, tension, and vocal production |

Each situation level includes:
- **name**: Human-readable situation label
- **group**: Category within the dimension
- **acoustic_signature**: How this situation specifically affects the voice acoustically

## Acting Challenge Database (Situation-Inspired)

5,749 acting challenges generated from the situation taxonomy (289 situations x 20 variants each, with a small number of API failures). Each variant samples:

- **1-3 EmoNet emotions** with intensity levels (slightly/clearly/extremely/very intensely present)
- **Speaker gender** from 7 VoiceNet GEND levels
- **Speaker age** from 6 VoiceNet AGEV levels

The challenges place the actor genuinely IN the situation — the physical/social context naturally affects the voice, breathing, and emotional delivery.

## SIT — Standalone Situation

### Sampling Strategy

1. **Situation**: Random selection from 289 situations across 11 dimensions
2. **Emotions**: 1-3 random emotions from EmoNet (40 categories) with random intensity
3. **Speaker Gender**: Random from 7 VoiceNet GEND levels
4. **Speaker Age**: Random from 6 AGEV levels
5. **Word Count**: 40-80 words of spoken dialogue

### Key Characteristics

- **Physically grounded** — the actor's body, environment, or health state is part of the performance
- **Acoustic authenticity** — the situation should naturally affect vocal quality (e.g., lying down compresses the diaphragm, freezing cold tightens the throat)
- **Dynamic emotional arc** — emotion at the start transforms by the end
- **Micro-distractions** — organic reactions to the physical situation (shivering, panting, wincing, shifting weight)

---

## SIT-CC — Character Consistent

### Two-Scene Format

Same actor, same situation, **two different emotional moments** separated by "CUT TO:". The speaker's physical situation stays identical — they're still lying down, still in the boardroom, still freezing — but the emotional delivery shifts dramatically.

### Sampling Strategy

Same as standalone SIT, plus:
- Word count: ~50 total, split roughly evenly (~25 per scene)
- Scene 2 emotion contrasts sharply with Scene 1

### Scene Structure

```
[Speaker description — age, gender, timbre — applies to BOTH scenes]

[Situation context: the actor is IN this physical/social situation]

[Scene 1: emotional moment with situation-appropriate vocal effects]

CUT TO:

[Same situation, but dramatic emotional shift]

[Scene 2: contrasting emotional moment, same physical constraints]
```
