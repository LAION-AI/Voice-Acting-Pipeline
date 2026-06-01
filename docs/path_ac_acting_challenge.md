# Path AC — Acting Challenge

Acting challenge-driven performance generation. Samples from **18,647 structured challenge scenarios** to create audition-style method acting performances. Two variants: standalone (AC) and character consistent (ACCC).

## Acting Challenge Database

The challenge database contains 18,647 scenarios from six sources:

| Source | Count | Description |
|--------|-------|-------------|
| LAION's Got Talent | 278 | Original acting challenges with diverse emotional and situational contexts |
| Haiku Batch | 1,000 | AI-generated challenges with wide coverage |
| VoiceNet Extension | 200 | Challenges designed around VoiceNet vocal dimensions |
| Eric Morris Method-Inspired | 4,030 | Generated from 806 concepts, techniques, exercises, and recommendations extracted from the [Eric Morris method acting system Unified Knowledge Base](https://github.com/christophschuhmann/eric_morris-method-acting-system). Each of the 806 items produced 5 distinct challenges inspired by its underlying emotional and psychological principles, covering deathbed farewells, confrontations, quiet vulnerability, intense sensory experiences, and contradictory emotions. |
| Existing Challenge Variants | 7,390 | Generated from the original 1,478 challenges. Each challenge produced 5 new variants in different contexts with deeper emotional stakes, interpersonal confrontations, quiet vulnerability, explosive intensity, and contradictory emotions. |
| Situation-Inspired | 5,749 | Generated from 289 situations across 11 dimensions of the [Situation Taxonomy](../data/situation_taxonomy.json) (body posture, physical activity, speaking target, social context, environment, health, gear, climate, substances, fatigue, pain). Each situation produced 20 variants with randomly sampled EmoNet emotions, gender, and age. |

All 18,647 challenges are shuffled together in the final database. Each challenge has a **title** and **instruction** describing the scenario, plus a **source** tag and an optional **inspired_by** field linking back to the originating concept or challenge.

**Example challenge (original):**
> **Title:** The Goodbye That Wasn't Planned
>
> **Instruction:** You're at a train station, saying goodbye to someone you didn't expect to see again. The train is already boarding. Express the collision of surprise, relief, and the pain of another imminent separation.

**Example challenge (Eric Morris-inspired):**
> **Title:** The Last Words You Never Heard
>
> **Instruction:** Your father is dying in a hospital bed, and for the first time in your adult life, he's trying to tell you something he's never said before. Begin with your arms crossed, deflecting — you've heard it all before. But as his words land, something cracks open. Let the shift from guarded resentment to helpless, raw gratitude happen without planning it.

**Example challenge (existing challenge variant):**
> **Title:** The Last Recipe Before the Memory Fades
>
> **Instruction:** You're teaching your adult child to make your signature family dish, but you're in the early stages of memory loss. Start with warmth and confidence, but let the moments where you lose the thread become real — the confusion, the flash of fear, the stubborn return to the task. End somewhere between dignity and devastation.

The subset files are also available separately:
- [`acting_challenges_eric_morris_inspired.json`](https://github.com/LAION-AI/Voice-Acting-Pipeline/blob/main/data/acting_challenges_eric_morris_inspired.json) — 4,030 Eric Morris-inspired challenges
- [`acting_challenges_existing_inspired.json`](https://github.com/LAION-AI/Voice-Acting-Pipeline/blob/main/data/acting_challenges_existing_inspired.json) — 7,390 existing challenge variants
- [`acting_challenges_situation_inspired.json`](https://github.com/LAION-AI/Voice-Acting-Pipeline/blob/main/data/acting_challenges_situation_inspired.json) — 5,749 situation-inspired challenges

Browse 100 random samples: [Acting Challenges Preview](https://projects.laion.ai/Voice-Acting-Pipeline/acting_challenges_preview.html)

## AC — Standalone Acting Challenge

### Sampling Strategy

1. **Acting Challenge**: Random selection from 18,647 challenges (title + instruction)
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
