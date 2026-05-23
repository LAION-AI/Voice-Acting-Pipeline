# Voice Acting Pipeline
*By <a href="https://scholar.google.com/citations?user=EvrlaSAAAAAJ">Christoph Schuhmann</a>*

End-to-end voice prompt generation and audio synthesis using the [DramaBox](https://huggingface.co/ResembleAI/Dramabox) TTS model (22B DiT transformer) and structured voice taxonomy sampling. Based on the voice taxonomy research from [Schuhmann et al., 2025](https://arxiv.org/abs/2505.20033).

This pipeline generates richly annotated voice performance prompts in the **DramaBox format** — single-speaker scenes with stage directions (English) and spoken dialogue (target language) — then synthesizes them into audio. Each prompt is procedurally constructed by sampling from structured taxonomies, then expanded by an LLM (Gemma 4 E4B-it) into a full performance script.

## All Paths at a Glance

The pipeline supports **12 generation paths** organized into three families. Each path uses a different sampling strategy to produce diverse voice acting data.

### Standalone Paths (Single Scene)

| Path | Sampling | Description | Details |
|------|----------|-------------|---------|
| **A** (VoiceNet) | 57 VoiceNet dims + EmoNet + Vocal Bursts | Full taxonomy sampling: 3 mandatory dims (Tempo, Gender, Age) + 5 random, 1-3 emotions, flow style, mandatory words | [Path A Details](docs/path_a_voicenet.md) |
| **B** (Archetype) | 920 archetypes × 92 genres | Genre/character archetype-based: random archetype + emotions + Tempo/Arousal | [Path B Details](docs/path_b_archetype.md) |
| **C** (Archetype Named) | Same as B + explicit naming | Archetype with explicit role naming in the DramaBox script (e.g. "a battle-hardened noble knight") | [Path C Details](docs/path_c_archetype_named.md) |
| **D** (Reference Audio) | Timbre whisper + VoiceNet + Chatterbox VC | Reference audio pipeline: timbre caption guides prompt, DramaBox TTS + voice conversion to match reference speaker | [Path D Details](docs/path_d_reference.md) |
| **AC** (Acting Challenge) | 1478 acting challenges + VoiceNet gender/age | Audition-style method acting from challenge scenarios — naturalistic, genuine, dynamic emotional arc | [AC Details](docs/path_ac_acting_challenge.md) |

### Character Consistent Paths (Two Scenes — "CUT TO:")

All CC paths generate two scenes with the **same speaker** in **contrasting emotional states**, separated by a "CUT TO:" marker. The speaker's fundamental voice (age, gender, timbre) stays identical — only the emotional delivery changes. Audio is later split into Scene 1 / Scene 2 using Qwen3-ASR word-level timestamps.

| Path | Sampling | Key Improvement | Details |
|------|----------|-----------------|---------|
| **CC-A** (VoiceNet) | VoiceNet + contrasting emotions | Original two-scene format | [CC Details](docs/path_cc_character_consistent.md) |
| **CC-B** (Archetype) | Archetype + contrasting emotions | Original two-scene format | [CC Details](docs/path_cc_character_consistent.md) |
| **CC-C** (Archetype Named) | Archetype named + contrasting emotions | Original two-scene format | [CC Details](docs/path_cc_character_consistent.md) |
| **CC2-A** (VoiceNet v2) | VoiceNet + contrasting emotions | Enhanced: explicit emotional scene setup + dramatic transition descriptions | [CC2 Details](docs/path_cc2_character_consistent_v2.md) |
| **CC2-B** (Archetype v2) | Archetype + contrasting emotions | Enhanced: genuine/spontaneous/authentic delivery emphasis | [CC2 Details](docs/path_cc2_character_consistent_v2.md) |
| **CC2-C** (Archetype Named v2) | Archetype named + contrasting emotions | Enhanced: visceral emotional contrast, human-sounding | [CC2 Details](docs/path_cc2_character_consistent_v2.md) |
| **ACCC** (Acting Challenge CC) | Acting challenge + VoiceNet gender/age | Challenge-driven two-scene format — same actor, same challenge, contrasting emotional moments | [ACCC Details](docs/path_ac_acting_challenge.md#accc-character-consistent) |

### Processing Pipeline (All Paths)

```
Sampling → Gemma 4 LLM → DramaBox TTS → RE-USE Enhancement → Best-of-N Scoring
                                                                    ↓
                                                    Parakeet ASR (WER) + Empathic Insight (enjoyment)
                                                    reward = (1 - WER) × content_enjoyment
```

For CC/CC2/ACCC paths, an additional step splits the two-scene audio:
```
RE-USE audio → Qwen3-ASR (word timestamps) → Find "CUT TO:" boundary → Split into Scene 1 + Scene 2
```

## Demo

Listen to generated samples from all paths:

| Demo | Description |
|------|-------------|
| **[Main Grid (Paths A/B/C/D)](https://projects.laion.ai/Voice-Acting-Pipeline/demo/)** | 40 prompts across 4 standalone paths, 3 candidates each, Best-of-3 scoring |
| **[RE-USE Enhancement](https://projects.laion.ai/Voice-Acting-Pipeline/demo/reuse.html)** | Before/after RE-USE speech enhancement comparison |
| **[Character Consistent v1](https://projects.laion.ai/Voice-Acting-Pipeline/demo/cc.html)** | CC-A/B/C: two-scene pairs with Scene 1 + Scene 2 split players |
| **[Character Consistent v2](https://projects.laion.ai/Voice-Acting-Pipeline/demo/cc2.html)** | CC2-A/B/C: improved prompting with emotional scene setup |
| **[Acting Challenge](https://projects.laion.ai/Voice-Acting-Pipeline/demo/ac.html)** | AC standalone + ACCC two-scene acting challenges |

## Taxonomies & Data

The pipeline samples from several structured taxonomies to create diverse, controlled voice performances:

| Taxonomy | Size | Format | Documentation |
|----------|------|--------|---------------|
| **VoiceNet** | 57 dimensions × 7 levels | HTML | [Taxonomy docs](docs/voicenet_taxonomy.md) · [Interactive viewer](https://projects.laion.ai/Voice-Acting-Pipeline/voicenet_extension_taxonomy.html) |
| **VoiceNet Extension** | Situation-dependent dims | HTML | [Interactive viewer](https://projects.laion.ai/Voice-Acting-Pipeline/voicenet_extension_taxonomy.html) |
| **EmoNet** | 40 emotions × 4 intensity levels | JSON | [Taxonomy docs](docs/emonet_taxonomy.md) |
| **Vocal Bursts** | 120 non-linguistic sounds | JSON | [Taxonomy docs](docs/vocal_bursts_taxonomy.md) |
| **Character Archetypes** | 920 archetypes × 92 genres | JSON | [Taxonomy docs](docs/archetypes.md) |
| **Acting Challenges** | 1,478 challenge scenarios | JSON | [Preview (100 samples)](https://projects.laion.ai/Voice-Acting-Pipeline/acting_challenges_preview.html) |
| **Situation Taxonomy** | Poses, activities, social contexts | JSON | [Data file](data/situation_taxonomy.json) |

Paper reference: [Schuhmann et al., 2025 — arXiv:2505.20033](https://arxiv.org/abs/2505.20033). See [docs/paper_reference.md](docs/paper_reference.md) for citation and BibTeX.

---

## Standalone Paths — Details

### Path A — VoiceNet (default 80%)

Full 57-dimension voice attribute sampling. The most granular control over voice performance.

1. Sample language + accent
2. Sample 1-3 emotions from EmoNet with intensity
3. Sample 3 mandatory VoiceNet dims (Tempo, Gender, Age) + 5 random from 54 remaining
4. Determine flow style (scattered/flowing/mixed), emotion alignment, direction style
5. Optionally include vocal bursts taxonomy
6. Inject 3 mandatory words from language-specific word list
7. Construct structured LLM prompt with all constraints → Gemma 4 generates DramaBox script

See [docs/path_a_voicenet.md](docs/path_a_voicenet.md) for full details.

### Path B — Archetype (default 20%)

Genre/character archetype-based sampling. Focuses on character identity over individual vocal dimensions.

1. Pick a random genre and archetype from 920 options
2. Sample language + accent
3. Sample 1-3 emotions with intensity
4. Sample Tempo (with fast bias) and Arousal (uniform)
5. Construct archetype-focused LLM prompt — no flow/alignment/direction constraints

See [docs/path_b_archetype.md](docs/path_b_archetype.md) for full details.

### Path C — Archetype Named

Same as Path B but with **explicit instruction to name the archetype role** in the DramaBox script output (e.g. "a battle-hardened noble knight" in the speaker description and stage directions). This gives DramaBox TTS a stronger character signal.

See [docs/path_c_archetype_named.md](docs/path_c_archetype_named.md) for full details.

### Path D — Reference Audio

The most promising path for voice cloning. Uses reference audio's timbre whisper caption to guide prompt generation, then voice-converts the DramaBox TTS output to match the reference speaker.

1. Load reference audio metadata (timbre whisper caption)
2. Generate timbre caption on-the-fly if missing (via `laion/timbre-whisper`)
3. Filter VoiceNet dimensions to situation-dependent only (exclude identity: age, gender, timbre, resonance)
4. Sample 1-3 emotions + tempo + 5 situation-dependent dimensions
5. Construct LLM prompt with timbre caption + sampled performance attributes
6. **Synthesize with DramaBox TTS (text-only, no voice reference)** — passing `voice_ref` directly to DramaBox leads to unstable/garbled generations
7. **Voice-convert generated audio to match reference via Chatterbox VC**
8. Score and rank with Best-of-N

> **Why text-only TTS + VC?** The timbre whisper caption gives Gemma 4 a rich description of the target speaker's vocal qualities, which guides the LLM to produce a speaker-consistent DramaBox script. Chatterbox VC then handles the actual voice transfer. This two-stage approach is far more stable than passing `voice_ref` directly to DramaBox, which causes garbled or incoherent audio output.

See [docs/path_d_reference.md](docs/path_d_reference.md) for full details.

### Path AC — Acting Challenge

Audition-style method acting performances driven by acting challenge scenarios. Samples from 1,478 structured challenges covering diverse emotional and situational contexts.

1. Sample a random acting challenge (title + instruction) from the [challenge database](https://projects.laion.ai/Voice-Acting-Pipeline/acting_challenges_preview.html)
2. Sample speaker gender (VoiceNet GEND dimension, 7 levels) and age (AGEV dimension, 7 levels)
3. Sample word count (40-80 words)
4. Gemma 4 generates a DramaBox prompt — actor performs the challenge naturalistically
5. DramaBox TTS → RE-USE enhancement → Best-of-N scoring

Key characteristics:
- **No self-introduction** — the actor simply begins performing
- **Dynamic emotional arc** with at least one turning point or new insight
- **Naturalistic, genuine, spontaneous** delivery — method acting, not theatrical performance
- **Diverse delivery** — whispered, loud, sensual, ranting, all valid if authentic

See [docs/path_ac_acting_challenge.md](docs/path_ac_acting_challenge.md) for full details.

---

## Character Consistent Paths — Details

All CC paths produce **two scenes with the same speaker** in contrasting emotional states, separated by a "CUT TO:" marker.

### CC v1 (A/B/C) — Character Consistent

The original two-scene format. Three sampling variants matching standalone Paths A, B, C:

- **CC-A** (VoiceNet): Full 57-dim sampling + contrasting emotions between scenes
- **CC-B** (Archetype): Archetype-based + contrasting emotions
- **CC-C** (Archetype Named): Named archetype + contrasting emotions

**Emotion contrast logic:** If Scene 1 has positive emotions → Scene 2 samples from negative emotions (and vice versa). Word count: 50-80 total (~25-40 per scene).

See [docs/path_cc_character_consistent.md](docs/path_cc_character_consistent.md) for full details.

### CC2 v2 (A/B/C) — Character Consistent v2

Improved version of CC with enhanced LLM prompting:

- **Scene 1 setup:** Before the first dialogue, 1-2 sentences vividly set the emotional situation — social context, speaker's state of mind, emotional energy
- **Scene 2 transition:** After "CUT TO:", 1-3 sentences explicitly describe the dramatic shift in emotional tone, talking style, delivery, and pace
- **Performance quality emphasis:** Delivery must sound like a real, living, breathing human being — genuine, spontaneous, authentic, with natural hesitations and organic pacing

See [docs/path_cc2_character_consistent_v2.md](docs/path_cc2_character_consistent_v2.md) for full details.

### ACCC — Acting Challenge Character Consistent

Challenge-driven two-scene format: same actor performing the same acting challenge at two different emotional moments with dramatically shifted delivery.

1. Sample acting challenge + gender + age (same as standalone AC)
2. Sample word count (40-80 total, split ~evenly between scenes)
3. Gemma 4 generates two contrasting scenes from the same challenge
4. DramaBox TTS → chunked RE-USE enhancement → Best-of-N scoring
5. Qwen3-ASR word timestamps → split into Scene 1 + Scene 2

See [docs/path_ac_acting_challenge.md#accc-character-consistent](docs/path_ac_acting_challenge.md#accc-character-consistent) for full details.

---

## Audio Processing

### RE-USE Speech Enhancement

All standalone paths (A, B, C, AC) and character consistent paths use [nvidia/RE-USE](https://huggingface.co/nvidia/RE-USE) (SEMamba) for speech enhancement:
- **Standalone/short audio:** Direct enhancement (single pass)
- **CC/CC2/ACCC (long audio):** Chunked enhancement (15s chunks, 1s overlap, cross-faded)

### Audio Splitting (CC/CC2/ACCC)

Two-scene audio is split using [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) with forced alignment:
1. Transcribe with word-level timestamps
2. Parse the DramaBox prompt to find first words of Scene 2 dialogue
3. Match ASR timestamps to find the split boundary
4. Split with 100ms fades at the boundary

### Best-of-N Ranking

1. Generate N candidate audio samples (default 3)
2. Score each with:
   - **WER** (Word Error Rate): [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) ASR transcription vs expected dialogue
   - **Content Enjoyment**: [Empathic Insight Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) (BUD-E-Whisper encoder + MLP)
3. Composite reward: `(1 - min(WER, 1.0)) × content_enjoyment`
4. Select the candidate with the highest reward

---

## Quick Start

### Installation

```bash
git clone https://github.com/LAION-AI/Voice-Acting-Pipeline.git
cd Vocalino-V0.1-Voice-Acting-Pipeline
pip install -e .
```

For TTS synthesis (requires GPU with ~24GB VRAM):
```bash
pip install -e ".[tts]"
```

For audio refinement and scoring:
```bash
pip install -e ".[refinement,scoring]"
```

### Generate Prompts (Mode 1)

```bash
# Generate 1000 DramaBox prompts using GPUs 0 and 1
dramabox generate-prompts --config config.json --total 1000 --gpus 0,1
```

### Synthesize Audio (Mode 2)

```bash
# Synthesize audio from an existing CSV
dramabox synthesize --csv output/dramabox_chunk_000.csv --gpus 0,1,2,3
```

### End-to-End (Mode 3)

```bash
# Generate prompts and immediately synthesize audio
dramabox run --config config.json --total 1000 --gpus 0,1,2,3
```

### Reference Audio Pipeline — Path D (Mode 4)

```bash
dramabox reference --config config.json --ref-dir /path/to/references --total 10 --gpus 6,7
```

### Demo Grid (Mode 5)

```bash
# Full 4-path demo: A + B + C + D, 10 prompts each, best-of-3 scoring
dramabox demo --config config.json --full --n-prompts 10 --best-of-n 3 --gpus 6,7
```

### Score Audio

```bash
dramabox score --audio output/audio/sample_000000_raw.wav --prompt "prompt text" --gpu 0
```

## Configuration

All parameters are in [`config.json`](config.json). See [`config_schema.md`](config_schema.md) for full documentation of every field.

### Key Settings

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `prompt_generation` | `llm_model` | `google/gemma-4-E4B-it` | LLM for prompt generation |
| `prompt_generation` | `total_prompts` | `100000` | Number of prompts to generate |
| `sampling` | `archetype_ratio` | `0.20` | Fraction using archetype path |
| `sampling` | `word_count_min/max` | `10 / 60` | Target dialogue word count range |
| `tts` | `cfg_scale` | `2.0` | Classifier-free guidance scale |
| `tts` | `steps` | `30` | Euler flow matching steps |
| `best_of_n` | `n_candidates` | `3` | Candidates per Best-of-N ranking |

### Adding Languages

Languages are configured in `config.json`. Currently active: English, German, French, Spanish. Ready to enable: Italian, Dutch, Russian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Polish, Swedish.

## Models Used

| Model | Purpose | VRAM |
|-------|---------|------|
| [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | DramaBox prompt generation | ~16GB |
| [`ResembleAI/Dramabox`](https://huggingface.co/ResembleAI/Dramabox) | TTS synthesis (22B DiT) | ~24GB |
| [`nvidia/RE-USE`](https://huggingface.co/nvidia/RE-USE) | Speech enhancement (SEMamba) | ~1GB |
| [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Word-level timestamps for audio splitting | ~4GB |
| [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ASR for WER scoring | ~2GB |
| [`laion/Empathic-Insight-Voice-Plus`](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) | Content enjoyment scoring | ~2GB |
| [`laion/timbre-whisper`](https://huggingface.co/laion/timbre-whisper) | On-the-fly timbre captioning (Path D) | ~2GB |
| [Chatterbox VC](https://github.com/resemble-ai/chatterbox) | Voice conversion (Path D) | ~4GB |

## Project Structure

```
dramabox-pipeline/
├── config.json                 # All configurable parameters
├── config_schema.md            # Documentation for config fields
├── pyproject.toml              # Python packaging
├── data/
│   ├── voicenet_ext_taxonomy.html   # VoiceNet (57 dims)
│   ├── all_acting_challenges.json   # 1,478 acting challenge scenarios
│   ├── situation_taxonomy.json      # Situation taxonomy (poses, activities, contexts)
│   ├── emonet_taxonomy.json         # EmoNet (40 emotions)
│   ├── vocal_bursts_taxonomy.json   # Vocal bursts (120 types)
│   ├── archetypes.json              # Archetypes (92 genres × 10)
│   └── wordlists/                   # Per-language word lists
├── dramabox/
│   ├── cli.py                  # CLI entry point
│   ├── config_loader.py        # Config loading and validation
│   ├── taxonomy.py             # Taxonomy parsers and loaders
│   ├── sampling.py             # Path A + Path B sampling
│   ├── reference_sampling.py   # Path D: reference audio sampling
│   ├── prompts.py              # LLM prompt construction
│   ├── prompt_generator.py     # Multi-GPU LLM batch generation
│   ├── tts_synthesizer.py      # Multi-GPU DramaBox TTS
│   ├── reuse_enhance.py        # RE-USE speech enhancement
│   ├── scoring.py              # ASR WER + content enjoyment scoring
│   ├── demo_grid.py            # HTML demo grid generator
│   └── pipeline.py             # Mode 1–6 orchestrator
├── docs/
│   ├── voicenet_taxonomy.md              # VoiceNet 57-dim taxonomy
│   ├── voicenet_extension_taxonomy.html  # Interactive VoiceNet viewer
│   ├── emonet_taxonomy.md                # EmoNet 40 emotions
│   ├── vocal_bursts_taxonomy.md          # 120 vocal bursts
│   ├── archetypes.md                     # 920 archetypes
│   ├── acting_challenges_preview.html    # Acting challenge preview (100 samples)
│   ├── paper_reference.md                # Citation and BibTeX
│   ├── path_a_voicenet.md                # Path A detailed docs
│   ├── path_b_archetype.md               # Path B detailed docs
│   ├── path_c_archetype_named.md         # Path C detailed docs
│   ├── path_d_reference.md               # Path D detailed docs
│   ├── path_ac_acting_challenge.md       # AC + ACCC detailed docs
│   ├── path_cc_character_consistent.md   # CC v1 detailed docs
│   ├── path_cc2_character_consistent_v2.md  # CC2 v2 detailed docs
│   └── demo/                             # HTML demo grids with audio
│       ├── index.html                    # Main 4-path grid
│       ├── reuse.html                    # RE-USE before/after
│       ├── cc.html                       # Character Consistent v1
│       ├── cc2.html                      # Character Consistent v2
│       └── ac.html                       # Acting Challenge
└── examples/                   # Example prompts
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Prompt generation | 1 GPU, 16GB VRAM | 4+ GPUs, 16GB+ each |
| TTS synthesis | 1 GPU, 24GB VRAM | 4+ GPUs, 24GB+ each |
| Refinement + scoring | 1 GPU, 8GB VRAM | 1 GPU, 16GB+ |
| RE-USE enhancement | CPU or GPU | 1 GPU |
| RAM | 32GB | 64GB+ |

---

## License

- This pipeline code — [Apache 2.0](LICENSE)
- [DramaBox](https://huggingface.co/ResembleAI/Dramabox) — see model card
