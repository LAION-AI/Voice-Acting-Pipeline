# Path AC — Acting Challenge

Acting challenge-driven performance generation. Samples from **1,478 structured challenge scenarios** to create audition-style method acting performances. Two variants: standalone (AC) and character consistent (ACCC).

## Acting Challenge Database

The challenge database contains 1,478 scenarios from three sources:

| Source | Count | Description |
|--------|-------|-------------|
| LAION's Got Talent | 278 | Original acting challenges with diverse emotional and situational contexts |
| Haiku Batch | 1,000 | AI-generated challenges with wide coverage |
| VoiceNet Extension | 200 | Challenges designed around VoiceNet vocal dimensions |

Each challenge has a **title** and **instruction** describing the scenario:

**Example challenge:**
> **Title:** The Goodbye That Wasn't Planned
>
> **Instruction:** You're at a train station, saying goodbye to someone you didn't expect to see again. The train is already boarding. Express the collision of surprise, relief, and the pain of another imminent separation.

Browse 100 random samples: [Acting Challenges Preview](https://projects.laion.ai/Voice-Acting-Pipeline/acting_challenges_preview.html)

## AC — Standalone Acting Challenge

### Sampling Strategy

1. **Acting Challenge**: Random selection from 1,478 challenges (title + instruction)
2. **Speaker Gender**: Random from VoiceNet GEND dimension (7 levels):
   - 0: Hyper-feminized voice
   - 1: Strongly feminized
   - 2: Moderately feminized
   - 3: Androgynous/neutral
   - 4: Moderately masculinized
   - 5: Strongly masculinized
   - 6: Hyper-masculinized voice
3. **Speaker Age**: Random from VoiceNet AGEV dimension (7 levels):
   - 0: Infant/toddler
   - 1: Child
   - 2: Teenager/adolescent
   - 3: Young adult
   - 4: Middle-aged adult
   - 5: Older adult
   - 6: Advanced age/elderly
4. **Word Count**: Random 40-80 words of spoken dialogue

### Key Characteristics

- **No self-introduction** — the actor simply begins performing, no "Hi, I'm..."
- **Dynamic emotional arc** — at least one turning point, new insight, or emotional shift
- **Naturalistic method acting** — sensory-aware, emotionally honest, genuine delivery
- **Diverse delivery** — whispered, loud, sensual, ranting — all valid if authentic
- **No pre-specified emotions** — the challenge itself drives the emotional approach; the LLM decides

### Performance Quality Suffix

Every AC prompt is appended with:

> Realistic, spontaneous, genuine, authentic voice acting performance. The actor is a seasoned professional who has deeply internalized this challenge. Their delivery should feel like it's happening for the first time — raw, immediate, and alive. Not a polished reading, but a genuine human moment captured in a pristine studio environment. The performance should have genuine imperfections — slight stumbles in thought, natural breath patterns, moments of searching for the right word — that make it feel unmistakably real and human.

### Audio Processing

1. **DramaBox TTS**: Raw audio synthesis
2. **RE-USE Enhancement**: Direct (single pass, audio is short at 40-80 words)
3. **Best-of-N Scoring**: 3 candidates, select best by WER + content enjoyment

---

## ACCC — Character Consistent

### Two-Scene Format

Same actor, same challenge, **two different emotional moments** separated by "CUT TO:". The speaker's fundamental voice (age, gender, timbre) stays identical — only the emotional delivery, talking style, dynamics, and pace change dramatically.

### Sampling Strategy

Same as standalone AC:
1. Random acting challenge
2. Random gender + age
3. Word count: 40-80 total, split roughly evenly (~20-40 per scene, minimum 10 per scene)

### Scene Structure

```
[Speaker description — applies to BOTH scenes]

[Scene 1 setup: 1-2 sentences setting emotional situation and actor's state]

[Scene 1 performance: dialogue ~half of total word count]

CUT TO:

[Scene 2 transition: 1-3 sentences describing DRAMATIC shift in emotional tone,
 talking style, delivery]

[Scene 2 performance: dialogue ~half of total word count]
```

### Audio Processing

1. **DramaBox TTS**: Raw audio synthesis (produces one continuous audio file)
2. **RE-USE Enhancement**: Chunked (15s chunks, 1s overlap) — ACCC audio is longer
3. **Best-of-N Scoring**: 3 candidates, select best
4. **Audio Splitting**: Qwen3-ASR word-level timestamps find the "CUT TO:" boundary, then the audio is split into Scene 1 and Scene 2 with 100ms fades

### How Splitting Works

1. **Qwen3-ASR** transcribes the full audio with word-level timestamps using the forced aligner
2. The DramaBox prompt is parsed to find the first 2-3 words of Scene 2 dialogue (after "CUT TO:")
3. These words are matched against the ASR transcript timestamps
4. The audio is split at the matched timestamp (100ms before the Scene 2 words) with cross-fades

## Demo

- **AC standalone**: [Acting Challenge demo](https://projects.laion.ai/Voice-Acting-Pipeline/demo/ac.html) — before/after RE-USE comparison
- **ACCC two-scene**: Same page — Scene 1 + Scene 2 split players
