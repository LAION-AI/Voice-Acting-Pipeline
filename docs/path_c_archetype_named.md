# Path C — Archetype Named

**Explicit archetype naming path.** Same sampling as Path B, but with a critical addition: the LLM is instructed to **explicitly name the character's archetype role** in the DramaBox script's speaker description and stage directions.

## How It Differs from Path B

In Path B, the archetype serves as creative inspiration — the LLM generates a performance *in the style of* the archetype but doesn't necessarily name it. In Path C, the archetype identity is woven directly into the script:

**Path B output** (archetype as inspiration):
```
A gruff man with a deep, commanding voice stands in a recording studio.
```

**Path C output** (archetype explicitly named):
```
A battle-hardened noble knight with a deep, commanding baritone stands
in a pristine recording studio, his voice carrying the weight of decades
of warfare and duty.
```

This gives DramaBox TTS a stronger character signal — the model can leverage the explicit character description to produce more characterful performances.

## Sampling Strategy

Identical to [Path B](path_b_archetype.md):

1. Pick a random genre and archetype from 920 options
2. Sample language + accent
3. Sample 1-3 emotions with intensity
4. Sample Tempo (with fast bias) and Arousal (uniform)

### Additional LLM Instruction

The key addition in the LLM prompt:

> MUST explicitly name the character's role/archetype in the opening speaker description and/or stage directions. The archetype identity should be clearly recognizable so the voice model knows WHAT KIND OF CHARACTER is speaking. Weave the archetype naturally into both the speaker description AND at least one stage direction.

## Audio Processing

Same as Paths A and B:
1. **DramaBox TTS**: Raw audio synthesis
2. **RE-USE Enhancement**: Speech enhancement
3. **Best-of-N Scoring**: 3 candidates, select best

## Demo

Listen to Path C samples in the [main demo grid](https://projects.laion.ai/Voice-Acting-Pipeline/demo/).
