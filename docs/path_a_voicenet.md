# Path A — VoiceNet Sampling

**The most granular sampling path.** Uses the full 57-dimension VoiceNet taxonomy to construct highly specific voice performance prompts with fine-grained control over vocal characteristics.

## Sampling Strategy

Path A is the default sampling path (80% of generated prompts) and offers the most detailed control over voice performance attributes.

### Step-by-Step

1. **Language + Accent**: Random language from enabled languages (English, German, French, Spanish) with optional accent/dialect variant
2. **Emotions**: Sample 1-3 emotions from [EmoNet](emonet_taxonomy.md) (40 categories) with intensity levels (1-4)
3. **Mandatory VoiceNet Dimensions**: Always sample these 3:
   - **Tempo** (TMPO): Speaking rate from extremely slow (0) to extremely fast (6)
   - **Perceived Gender** (GEND): Hyper-feminized (0) to hyper-masculinized (6)
   - **Voice Age** (AGEV): Infant/toddler (0) to advanced age (6)
4. **Random VoiceNet Dimensions**: Sample 5 additional dimensions from the remaining 54
5. **Flow Style**: Determined by sampled attributes:
   - `scattered` — fragmented, staccato delivery
   - `flowing` — smooth, connected delivery
   - `mixed` — natural variation
6. **Emotion Alignment**: How emotions relate to vocal attributes:
   - `congruent` — emotions match vocal energy
   - `neutral` — no deliberate alignment
   - `counter-emotional` — intentional tension between emotion and delivery
7. **Direction Style**: Either `literary` (prose-like stage directions) or `tag` (brief parenthetical tags)
8. **Vocal Bursts**: Optionally include the [120 vocal bursts taxonomy](vocal_bursts_taxonomy.md) — sobs, laughs, gasps, growls, etc.
9. **Mandatory Words**: 3 words from language-specific word list that must appear naturally in dialogue

### VoiceNet Taxonomy

The full VoiceNet taxonomy covers 57 dimensions organized into 10 attribute groups:

| Group | Dimensions | Examples |
|-------|-----------|----------|
| Rhythm & Timing | 8 | Tempo, Chunking, Smoothness, Clarity, Pitch Range, Emphasis, Disfluency, Structure |
| Social & Interpersonal | 3 | Stance, Focus, Vulnerability |
| Speaker Identity | 3 | Gender, Voice Age, Register |
| Emotion & Affect | 3 | Valence, Arousal, Volatility |
| Physical Production | 4 | Respiration, Tension, Cognitive Load, Attack |
| Spectral & Timbral | 7 | Brightness, Roughness, Harmonicity, Fullness, Warmth, Metallic, Esthetics |
| Temporal Dynamics | 4 | Velocity Flux, Dynamic Arc, Arousal Shift, Valence Shift |
| Language & Recording | 3 | Recording Quality, Background Noise, Content Appropriateness |
| Resonance Placement | 7 | Chest, Throat, Oral, Mask, Nasal, Head, Mixed |
| Speaking Style | 15 | Casual, Conversational, Formal, Dramatic, Narrator, ASMR, Whisper, etc. |

Each dimension has 7 ordinal levels (0-6) with detailed descriptions. See [voicenet_taxonomy.md](voicenet_taxonomy.md) for the full taxonomy.

## LLM Prompt Construction

The sampled attributes are assembled into a structured prompt for Gemma 4 E4B-it, which generates a DramaBox-format script containing:

- **Speaker description** (1-3 sentences): Age, gender, timbre, voice qualities
- **Stage directions** (English): Actions, pauses, emotional shifts — never spoken aloud
- **Spoken dialogue** (target language): Inside double quotes, the actual words to be synthesized

## Output Format

```
A deep-voiced man in his late forties with a gravelly baritone and slight
Midwestern drawl stands in a pristine recording studio.

He pauses, staring at the letter in his hands, his jaw tightening.

"I told you this would happen. I told you, and you didn't listen."

His voice cracks, barely above a whisper now.

"But I'm still here. I'm always still here."
```

## Audio Processing

1. **DramaBox TTS**: 22B DiT transformer synthesizes raw audio
2. **RE-USE Enhancement**: nvidia/RE-USE speech enhancement (direct, single pass)
3. **Best-of-N Scoring**: 3 candidates scored by WER + content enjoyment

## Demo

Listen to Path A samples in the [main demo grid](https://projects.laion.ai/Voice-Acting-Pipeline/demo/).
