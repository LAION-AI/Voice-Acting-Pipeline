# CC2 — Character Consistent v2

**Improved two-scene character consistency.** Same concept as [CC v1](path_cc_character_consistent.md) — same speaker, contrasting emotions, "CUT TO:" separator — but with significantly enhanced LLM prompting that produces more dramatic, more human-sounding performances.

## What Changed from CC v1

CC2 adds three key improvements to the LLM system instruction:

### 1. Explicit Emotional Scene Setup

**Before Scene 1 dialogue**, the LLM must write 1-2 sentences that vividly set the emotional situation:
- Describe the social context and what is happening
- Describe the speaker's state of mind and emotional energy driving the speech
- "Paint the emotional landscape" before the dialogue begins

**CC v1:** Scene 1 jumps straight into dialogue
**CC2:** Scene 1 is preceded by vivid emotional context

### 2. Explicit Dramatic Transition

**After "CUT TO:"**, the LLM must write 1-3 sentences explicitly describing the dramatic shift:
- How the same actor's emotional tone, talking style, and delivery change
- The new situation and contrasting emotional state
- Make clear this is the same person in a completely different emotional reality

**CC v1:** "CUT TO:" followed directly by Scene 2 dialogue
**CC2:** "CUT TO:" followed by explicit description of the emotional shift

### 3. Genuine Human Performance Emphasis

CC2 adds a strong performance quality suffix emphasizing:
- **Genuine**: Raw human emotion, not performed or theatrical
- **Spontaneous**: Words feel thought of and spoken in the moment, not recited
- **Authentic**: Every emotional shift comes from a real place
- **High Quality**: Pristine studio recording, no artifacts

> The emotional contrast between the two scenes should be VISCERAL and UNMISTAKABLE — the listener should immediately feel the dramatic shift in the speaker's inner world the moment Scene 2 begins, even though the voice is recognizably the same person.

## Three Sampling Variants

### CC2-A — VoiceNet

Full VoiceNet sampling (same attributes as [CC-A](path_cc_character_consistent.md)):
- 57 VoiceNet dimensions with flow style, emotion alignment, vocal bursts
- 5-10 striking dimensions highlighted for the LLM
- Mandatory words in target language
- Contrasting emotions between scenes

### CC2-B — Archetype

Archetype-based (same as CC-B):
- Random genre + archetype from 920 options
- Tempo + arousal
- Archetype as inspiration, not literal reproduction
- Contrasting emotions between scenes

### CC2-C — Archetype Named

Named archetype (same as CC-C):
- Archetype explicitly named in speaker description and stage directions
- Character role clearly recognizable (e.g. "battle-hardened noble knight")
- Contrasting emotions between scenes

## Scene Structure (CC2 Format)

```
A weathered woman in her early sixties with a rich contralto and slight
tremor of age stands in a pristine recording studio.

She stands at the kitchen window, watching the last golden light of
afternoon fade. Her hands are still dusted with flour from the pie she
baked — his favorite — though he won't be coming home again.

"I kept setting the table for two. Every night. Can you believe that?"

Her voice breaks softly.

"Force of habit, I suppose. Twenty-three years of force of habit."

CUT TO:

The same woman, now at the front door, her voice suddenly steel-bright
and clipped. A neighbor has come to offer condolences, but the pity
in their eyes has lit something fierce inside her.

"Don't you dare stand there looking at me like that. I'm not broken."

She draws herself up to her full height, jaw set.

"He's the one who left. I'm the one who's still standing."
```

## Emotion Contrast Logic

Same as CC v1 — see [CC emotion contrast logic](path_cc_character_consistent.md#emotion-contrast-logic).

## Audio Processing

Same pipeline as CC v1:
1. **DramaBox TTS**: Continuous audio synthesis
2. **RE-USE Enhancement**: Chunked (15s chunks, 1s overlap)
3. **Best-of-N Scoring**: 3 candidates
4. **Audio Splitting**: Qwen3-ASR word timestamps → Scene 1 + Scene 2

## Demo

Listen to CC2-A/B/C samples with Scene 1/Scene 2 players: [Character Consistent v2 demo](https://projects.laion.ai/Voice-Acting-Pipeline/demo/cc2.html)

## See Also

- [CC v1](path_cc_character_consistent.md) — original version
- [ACCC](path_ac_acting_challenge.md#accc-character-consistent) — challenge-driven character consistency
