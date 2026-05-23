# Path B — Archetype Sampling

**Character-driven sampling path.** Instead of specifying individual vocal dimensions, Path B selects a character archetype from 920 options across 92 genres. The archetype guides the LLM to create a character-appropriate performance.

## Sampling Strategy

Path B is the secondary sampling path (20% of generated prompts) and focuses on character identity over individual vocal parameters.

### Step-by-Step

1. **Archetype**: Pick a random genre (92 available) and archetype (10 per genre = 920 total)
   - Examples: "Film Noir | World-Weary Private Detective", "Anime | Hot-Blooded Rival", "Medical Drama | Exhausted Resident"
2. **Language + Accent**: Random from enabled languages with optional accent
3. **Emotions**: Sample 1-3 emotions from EmoNet with intensity levels
4. **Tempo**: Sampled with a fast bias (archetypes tend to have more dynamic pacing)
5. **Arousal**: Sampled uniformly across 7 levels

### Key Differences from Path A

| Aspect | Path A (VoiceNet) | Path B (Archetype) |
|--------|-------------------|---------------------|
| Primary input | 57 vocal dimensions | Character archetype |
| Flow style | Explicitly sampled | Not constrained |
| Emotion alignment | Explicitly sampled | Not constrained |
| Direction style | Literary or tag | Not constrained |
| Mandatory words | 3 words required | None |
| Vocal bursts | Optionally included | Not included |

## LLM Prompt Construction

The prompt tells Gemma 4 to create a performance **inspired by** the archetype — not a literal reproduction. The archetype serves as creative direction, not a rigid template.

The instruction includes:
- Genre and archetype name
- Sampled emotions with intensity
- Tempo and arousal level
- Language for dialogue

## Archetype Taxonomy

920 character voice archetypes across 92 genres. See [archetypes.md](archetypes.md) for the full list.

Example genres and archetypes:
- **Fantasy RPG**: Wise Old Sage, Cocky Rogue, Battle-Hardened Commander
- **Sitcom**: Sarcastic Best Friend, Overbearing Parent, Awkward Nerd
- **Horror**: Creepy Child, Sinister Cult Leader, Terrified Survivor
- **Documentary**: Authoritative Narrator, Passionate Activist, Weathered Expert

## Audio Processing

Same as Path A:
1. **DramaBox TTS**: Raw audio synthesis
2. **RE-USE Enhancement**: Speech enhancement
3. **Best-of-N Scoring**: 3 candidates, select best

## Demo

Listen to Path B samples in the [main demo grid](https://projects.laion.ai/Voice-Acting-Pipeline/demo/).
