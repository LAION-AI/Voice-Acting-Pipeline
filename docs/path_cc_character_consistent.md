# CC — Character Consistent v1

**Two-scene character consistency testing.** Generates two scenes with the **same speaker** in **contrasting emotional states**, separated by a "CUT TO:" marker. The speaker's fundamental voice characteristics (age, gender, timbre) remain identical — only the emotional delivery changes.

## Purpose

Character consistent paths test whether DramaBox TTS can maintain speaker identity across dramatically different emotional performances. A good CC sample should sound like the same person in both scenes, despite the emotional contrast.

## Three Sampling Variants

CC v1 has three sampling variants that mirror the standalone paths A, B, C:

### CC-A — VoiceNet

Full 57-dimension VoiceNet sampling (same as [Path A](path_a_voicenet.md)):
- 3 mandatory dims (Tempo, Gender, Age) + 5 random dims
- Flow style, emotion alignment, direction style
- Mandatory words from language word list
- Optional vocal bursts

**Plus character consistency additions:**
- Scene 1 emotions: Sampled normally (1-3 from EmoNet)
- Scene 2 emotions: **Contrasting** — if Scene 1 is positive → Scene 2 is negative (and vice versa)
- Word count: 50-80 total (~25-40 per scene)

### CC-B — Archetype

Archetype-based sampling (same as [Path B](path_b_archetype.md)):
- Random genre + archetype from 920 options
- Tempo and arousal
- Do NOT reproduce archetype literally — use as inspiration

**Plus character consistency:**
- Contrasting emotions between scenes
- Same character identity across both scenes

### CC-C — Archetype Named

Named archetype (same as [Path C](path_c_archetype_named.md)):
- Same as CC-B but archetype is explicitly named in the script
- Character role woven into speaker description and stage directions

## Emotion Contrast Logic

The emotion contrast system ensures the two scenes feel emotionally distinct:

| Scene 1 Emotions | Scene 2 Sampling |
|-------------------|------------------|
| Positive (joy, affection, hope...) | Sample from **negative** pool (fear, sadness, anger...) |
| Negative (fear, anger, shame...) | Sample from **positive** pool (joy, contentment, pride...) |
| Both positive + negative | Sample from combined pool, excluding Scene 1 emotions |
| Neutral (surprise, relief, fatigue...) | Sample from positive or negative (50/50 chance) |

### Emotion Categories

**Positive:** Amusement, Elation, Pleasure/Ecstasy, Contentment, Gratitude, Affection, Infatuation, Hope/Optimism, Triumph, Pride, Interest, Awe

**Negative:** Fear, Distress, Sadness, Anger, Disgust, Contempt, Bitterness, Shame, Embarrassment, Disappointment, Helplessness, Pain, Malevolence/Malice

**Neutral:** Surprise, Concentration, Contemplation, Relief, Longing, Teasing, Doubt, Confusion, Fatigue, Emotional Numbness, Impatience, Jealousy, Sexual Lust, Sourness, Intoxication

## Scene Structure

```
[Speaker description — age, gender, timbre, voice qualities]

[Scene 1: Stage directions + dialogue in target language]

CUT TO:

[Scene 2: Stage directions + dialogue in target language — same speaker, different emotion]
```

## Audio Processing

1. **DramaBox TTS**: Synthesizes one continuous audio file containing both scenes
2. **RE-USE Enhancement**: Chunked processing (15s chunks, 1s overlap, cross-faded)
3. **Best-of-N Scoring**: 3 candidates per prompt
4. **Audio Splitting**: Qwen3-ASR with forced aligner finds the "CUT TO:" boundary in the audio, splits into Scene 1 + Scene 2 with 100ms fades

## Demo

Listen to CC-A/B/C samples with Scene 1/Scene 2 players: [Character Consistent demo](https://projects.laion.ai/Voice-Acting-Pipeline/demo/cc.html)

## See Also

- [CC2 v2](path_cc2_character_consistent_v2.md) — improved version with enhanced prompting
- [ACCC](path_ac_acting_challenge.md#accc-character-consistent) — challenge-driven character consistency
