# Voice Acting Pipeline
*By <a href="https://scholar.google.com/citations?user=EvrlaSAAAAAJ">Christoph Schuhmann</a>*

**Open-weights voice acting data pipeline combining structured taxonomy sampling, DramaBox TTS synthesis, Sidon speech restoration, and ChatterboxVC augmentation with best-of-N ranking across 46 scoring methods.**

> **Live Demo:** [Sidon+VC Sample Groups](https://projects.laion.ai/Voice-Acting-Pipeline/demo/sidon_vc_sample_groups.html) — 20 groups x 25 candidates with LLM-guided CUT TO: splitting, Whisper turbo ASR, and Gemma 4 re-annotation.

> **Benchmarks** ([landing page](https://projects.laion.ai/Voice-Acting-Pipeline/)):
> - 🎬 [**Vanilla DramaBox TTS — generate + reward-rank**](https://projects.laion.ai/Voice-Acting-Pipeline/dramabox.html) — raw two-scene `CUT TO:` DramaBox prompts fed directly to the 8B MOSS voice-acting TTS, 4 seeds, scored + reward-ranked; listenable takes, best-of-k quality/compute trade-off, and the full k=1..32 seed-scaling walltime table.
> - ⚡ [**Local-LLM DramaBox prompt-generation throughput**](https://projects.laion.ai/Voice-Acting-Pipeline/prompts.html) — per-pathway/language token + throughput stats with a 1M-prompt estimate, plus the DramaBox TTS seed-scaling table for best-of-k planning.
> - 📖 [**How the DramaBox prompt dataset is sampled &amp; generated**](https://projects.laion.ai/Voice-Acting-Pipeline/sampling.html) — a plain-English, reproducible walkthrough of the [`laion/dramabox-cutscene-prompts`](https://huggingface.co/datasets/laion/dramabox-cutscene-prompts) dataset: the 5 sampling pathways, every taxonomy it draws from (VoiceNet, 40 EmoNet emotions, archetypes, situations, 180 vocal bursts), the exact prompts the model receives, and real generated examples with their sampled metadata.

> **Dataset Plan:** See the full technical white paper — [Towards an Emotionally Expressive Audio Omni-Model](LAION-Voice-Whitepaper.md) — for the complete LAION Voice and LAION Voice Acting dataset construction plan, model inventory, and annotation strategy.

---

# DramaBox Voice Acting Data Pipeline

End-to-end voice prompt generation and audio synthesis using the [DramaBox](https://huggingface.co/ResembleAI/Dramabox) TTS model (22B DiT transformer) and structured voice taxonomy sampling. Based on the voice taxonomy research from [Schuhmann et al., 2025](https://arxiv.org/abs/2505.20033) and [EmoNet-Voice (Schuhmann et al., 2025)](https://arxiv.org/abs/2506.09827).

This pipeline generates richly annotated voice performance prompts in the **DramaBox format** — single-speaker scenes with stage directions (English) and spoken dialogue (target language) — then synthesizes them into audio. Each prompt is procedurally constructed by sampling from structured taxonomies, then expanded by an LLM (Gemma 4 E4B-it) into a full performance script.

## Current Pipeline: DramaBox + Sidon + ChatterboxVC

The full audio processing chain:

```
Taxonomy Sampling          LLM Prompt Gen         DramaBox TTS (22B)
(VoiceNet/Archetype/       (Gemma 4 E4B-it)       CFG=2.5, STG=1.5
 Situation/ActingChall)                            25 candidates/prompt
        |                        |                        |
        +------------------------+                        |
                                                          v
                                              +---------------------------+
                                              |  Per-Sample Augment       |
                                              |                           |
                                              |  Path A: Sidon only       |
                                              |    (16kHz -> 48kHz)       |
                                              |  Path B: ChatterboxVC     |
                                              |    + Sidon (VC -> restore)|
                                              |                           |
                                              |  Pick best by             |
                                              |  DNS-MOS OVR score        |
                                              +-------------+-------------+
                                                            |
                                              +-------------v-------------+
                                              |  Whisper Turbo ASR        |
                                              |  (word-level timestamps)  |
                                              +-------------+-------------+
                                                            |
                                              +-------------v-------------+
                                              |  LLM-Guided CUT TO:      |
                                              |  Split (Gemma 4 E4B-it)  |
                                              |  + quiet-spot detection   |
                                              |  + LLM fade strategy     |
                                              +-------------+-------------+
                                                            |
                                              +-------------v-------------+
                                              |  Best-of-25 Ranking       |
                                              |  (WER, VoiceCLAP,         |
                                              |   EmoNet, Content)        |
                                              +-------------+-------------+
                                                            |
                                              +-------------v-------------+
                                              |  Gemma 4 Re-annotation    |
                                              |  (ASR -> refined prompt)  |
                                              +---------------------------+
```

### Augmentation Sub-Pipeline

For each raw TTS candidate, two enhancement paths run and the best is selected:

```
Raw TTS Audio ──┬──► Sidon Speech Restoration ──► DNS-MOS ──┐
(from DramaBox)  │    (w2v-BERT LoRA + DAC)                   ├──► Pick higher OVR
                 │    (16kHz → 48kHz)                         │
                 └──► ChatterboxVC ──► Sidon ──► DNS-MOS ────┘
                      (S3Gen flow-matching VC,
                       self-VC or ref-VC)
```

### Two-Part CUT TO: Pipeline

For two-scene audio, a self-VC of the full audio provides the VC target for speaker consistency:

```
Full Audio ──► Self-VC ──► Sidon ──► full_enhanced (VC target)
                                           │
           ┌───────────────────────────────┘
           │
Part 1 ──┬──► Sidon only ──► DNS-MOS ──┐
         │                              ├──► Pick best
         └──► VC(→full_enhanced) + Sidon ──► DNS-MOS ──┘

Part 2 ──┬──► Sidon only ──► DNS-MOS ──┐
         │                              ├──► Pick best
         └──► VC(→full_enhanced) + Sidon ──► DNS-MOS ──┘
```

**Scoring methods (46 total):**
- **6 quality/CLAP methods** — combine WER, VoiceCLAP similarity (Large + Small), content enjoyment, with (1-WER) multiplicative factor
- **40 EmoNet emotion methods** — one per emotion dimension (Empathic Insight Plus), with WER < 10% hard cutoff

## Demo Grids

### Sidon + ChatterboxVC Pipeline (Current)

Listen to the latest Sidon+VC experiment — 20 groups x 25 candidates = 500 audio clips with LLM-guided CUT TO: splitting and Gemma 4 re-annotation.

| Demo | Description | Link |
|------|-------------|------|
| **Sidon+VC Sample Groups** | 20 groups, best-of-25, LLM-guided splits | [sidon_vc_sample_groups.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/sidon_vc_sample_groups.html) |

### ACCC LavaSR Experiment (Legacy)

50 groups x 25 candidates = 1,250 audio clips across 10 pages. Each page has an interactive ranking dropdown with 46 methods.

| Page | Groups | Link |
|------|--------|------|
| Page 1 | Groups 0-4 | [accc_lavasr_p1.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p1.html) |
| Page 2 | Groups 5-9 | [accc_lavasr_p2.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p2.html) |
| Page 3 | Groups 10-14 | [accc_lavasr_p3.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p3.html) |
| Page 4 | Groups 15-19 | [accc_lavasr_p4.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p4.html) |
| Page 5 | Groups 20-24 | [accc_lavasr_p5.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p5.html) |
| Page 6 | Groups 25-29 | [accc_lavasr_p6.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p6.html) |
| Page 7 | Groups 30-34 | [accc_lavasr_p7.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p7.html) |
| Page 8 | Groups 35-39 | [accc_lavasr_p8.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p8.html) |
| Page 9 | Groups 40-44 | [accc_lavasr_p9.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p9.html) |
| Page 10 | Groups 45-49 | [accc_lavasr_p10.html](https://projects.laion.ai/Voice-Acting-Pipeline/demo/accc_lavasr_p10.html) |

**Also available:**
- [Pitch Analysis](https://projects.laion.ai/Voice-Acting-Pipeline/demo/pitch_analysis.html) — F0 contour analysis across candidates

---

## All Paths at a Glance

The pipeline supports **12 generation paths** organized into three families. Each path uses a different sampling strategy to produce diverse voice acting data.

### Standalone Paths (Single Scene)

| Path | Sampling | Description | Details |
|------|----------|-------------|---------|
| **A** (VoiceNet) | 57 VoiceNet dims + EmoNet + Vocal Bursts | Full taxonomy sampling: 3 mandatory dims (Tempo, Gender, Age) + 5 random, 1-3 emotions, flow style, mandatory words | [Path A Details](docs/path_a_voicenet.md) |
| **B** (Archetype) | 920 archetypes x 92 genres | Genre/character archetype-based: random archetype + emotions + Tempo/Arousal | [Path B Details](docs/path_b_archetype.md) |
| **C** (Archetype Named) | Same as B + explicit naming | Archetype with explicit role naming in the DramaBox script (e.g. "a battle-hardened noble knight") | [Path C Details](docs/path_c_archetype_named.md) |
| **D** (Reference Audio) | Timbre whisper + VoiceNet + Chatterbox VC | Reference audio pipeline: timbre caption guides prompt, DramaBox TTS + voice conversion to match reference speaker | [Path D Details](docs/path_d_reference.md) |
| **AC** (Acting Challenge) | 19,247 acting challenges + VoiceNet gender/age | Audition-style method acting from challenge scenarios — naturalistic, genuine, dynamic emotional arc | [AC Details](docs/path_ac_acting_challenge.md) |
| **SIT** (Situation) | 289 situations x EmoNet emotions | Situation-driven acting: actor is physically/socially IN a specific situation from the [Situation Taxonomy](data/situation_taxonomy.json) (body posture, activity, social context, environment, health, climate, fatigue, pain) with sampled emotions | [SIT Details](docs/path_sit_situation.md) |

### Character Consistent Paths (Two Scenes — "CUT TO:")

All CC paths generate two scenes with the **same speaker** in **contrasting emotional states**, separated by a "CUT TO:" marker. The speaker's fundamental voice (age, gender, timbre) stays identical — only the emotional delivery changes. Audio is split into Scene 1 / Scene 2 using LLM-guided splitting (Gemma 4 E4B-it + Whisper turbo word-level timestamps + quiet-spot detection).

| Path | Sampling | Key Improvement | Details |
|------|----------|-----------------|---------|
| **CC-A** (VoiceNet) | VoiceNet + contrasting emotions | Original two-scene format | [CC Details](docs/path_cc_character_consistent.md) |
| **CC-B** (Archetype) | Archetype + contrasting emotions | Original two-scene format | [CC Details](docs/path_cc_character_consistent.md) |
| **CC-C** (Archetype Named) | Archetype named + contrasting emotions | Original two-scene format | [CC Details](docs/path_cc_character_consistent.md) |
| **CC2-A** (VoiceNet v2) | VoiceNet + contrasting emotions | Enhanced: explicit emotional scene setup + dramatic transition descriptions | [CC2 Details](docs/path_cc2_character_consistent_v2.md) |
| **CC2-B** (Archetype v2) | Archetype + contrasting emotions | Enhanced: genuine/spontaneous/authentic delivery emphasis | [CC2 Details](docs/path_cc2_character_consistent_v2.md) |
| **CC2-C** (Archetype Named v2) | Archetype named + contrasting emotions | Enhanced: visceral emotional contrast, human-sounding | [CC2 Details](docs/path_cc2_character_consistent_v2.md) |
| **ACCC** (Acting Challenge CC) | Acting challenge + VoiceNet gender/age | Challenge-driven two-scene format — same actor, same challenge, contrasting emotional moments | [ACCC Details](docs/path_ac_acting_challenge.md#accc-character-consistent) |
| **SIT-CC** (Situation CC) | Situation + EmoNet + contrasting emotions | Two-scene situation-driven format — same actor IN the same situation, two contrasting emotional moments (5,749 pre-generated prompts in en/fr/es/de) | [SIT-CC Details](docs/path_sit_situation.md#sit-cc-character-consistent) |

---

## Audio Processing

### Sidon Speech Restoration

All paths use [Sidon](https://huggingface.co/sarulab-speech/sidon_raw_weight) (w2v-BERT LoRA encoder + DAC decoder) for speech restoration:
- Input 16 kHz -> output 48 kHz (simultaneous enhancement + super-resolution)
- LoRA-adapted w2v-BERT 2.0 extracts clean SSL features from noisy/degraded input
- DAC-based vocoder synthesizes high-quality 48 kHz audio from the clean features

### ChatterboxVC Augmentation

Optionally, [Chatterbox VC](https://github.com/resemble-ai/chatterbox) (S3Gen flow-matching VC) is applied before Sidon:
- Self-VC (voice-converts to itself) removes TTS artifacts while preserving speaker identity
- Reference-VC (voice-converts to a reference speaker) enables voice cloning in Path D
- Both paths are scored with DNS-MOS OVR; the higher-scoring result is kept

### DNS-MOS Quality Scoring

Each enhanced candidate is scored using a native PyTorch DNS-MOS model:
- Predicts SIG (signal quality), BAK (background quality), OVR (overall quality) on a 1-5 scale
- Used to select between Sidon-only and VC+Sidon augmentation paths
- Audio is chunked into 9-second windows and scores are averaged

### LLM-Guided Audio Splitting (CC/CC2/ACCC)

Two-scene audio is split using a three-phase LLM-guided pipeline:

1. **Whisper Turbo ASR** — transcribe with word-level timestamps
2. **LLM split point** (Gemma 4 E4B-it) — reads the DramaBox prompt + ASR word timestamps to identify the exact CUT TO: scene transition timestamp. Matches Scene 1/Scene 2 dialogue to ASR words.
3. **Quiet-spot detection** — finds the nearest inter-word silence gap within +/-1.5s of the LLM timestamp, then picks the quietest sample (by RMS energy) within that gap. A 15ms margin from word edges guarantees cuts never land inside a word.
4. **LLM fade strategy** (Gemma 4 E4B-it) — chooses fade-out/fade-in durations (0-200ms) and optional silence gap (0-500ms) based on the emotional contrast between scenes. Options: hard cut, fade, or crossfade.
5. **Fallback** — if the LLM fails, falls back to longest silence gap in the middle 20-80% of the audio.

### Best-of-N Ranking (46 Methods)

For each group of candidates, 46 ranking methods are available:

**Quality/CLAP methods (6):**

| Method | Formula |
|--------|---------|
| **v_snr_L** (default) | (1 - WER) x (san_L - neg_san_L + 2) |
| **v_snr_S** | (1 - WER) x (san_S - neg_san_S + 2) |
| **v_san_L** | (1 - WER) x (san_L + 1) |
| **v_san_S** | (1 - WER) x (san_S + 1) |
| **Content Enjoyment** | Raw Empathic Insight Plus score |
| **Standard** | (1 - WER) x Content Enjoyment |

Where:
- **WER** — Word Error Rate from [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) ASR vs expected dialogue
- **san_L / san_S** — VoiceCLAP cosine similarity (Large / Small) between audio and sanitized prompt (stage directions only, dialogue removed)
- **neg_san_L / neg_san_S** — VoiceCLAP similarity to negative text ("robotic, distorted, uncanny, distorted, distortion")

**EmoNet emotion methods (40):**

Each of the 40 EmoNet emotion dimensions from [Empathic Insight Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) is a separate ranking method. Audio is encoded with [BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper) (768-dim), pooled (mean+min+max+std = 3072-dim), then scored by 40 specialized MLP expert heads.

Emotion rankings use a **WER < 10% hard cutoff** — only candidates that said the right words qualify. Within qualifying candidates, they are ranked by descending emotion score.

The 40 emotions: Affection, Amusement, Anger, Astonishment/Surprise, Awe, Bitterness, Concentration, Confusion, Contemplation, Contempt, Contentment, Disappointment, Disgust, Distress, Doubt, Elation, Embarrassment, Emotional Numbness, Fatigue/Exhaustion, Fear, Helplessness, Hope/Enthusiasm/Optimism, Impatience/Irritability, Infatuation, Interest, Intoxication/Altered States, Jealousy/Envy, Longing, Malevolence/Malice, Pain, Pleasure/Ecstasy, Pride, Relief, Sadness, Sexual Lust, Shame, Sourness, Teasing, Thankfulness/Gratitude, Triumph.

---

## Taxonomies & Data

The pipeline samples from several structured taxonomies to create diverse, controlled voice performances:

| Taxonomy | Size | Format | Documentation |
|----------|------|--------|---------------|
| **VoiceNet** | 57 dimensions x 7 levels | HTML | [Taxonomy docs](docs/voicenet_taxonomy.md) / [Interactive viewer](https://projects.laion.ai/Voice-Acting-Pipeline/voicenet_extension_taxonomy.html) |
| **VoiceNet Extension** | Situation-dependent dims | HTML | [Interactive viewer](https://projects.laion.ai/Voice-Acting-Pipeline/voicenet_extension_taxonomy.html) |
| **EmoNet** | 40 emotions x 4 intensity levels | JSON | [Taxonomy docs](docs/emonet_taxonomy.md) |
| **Vocal Bursts** | 120 non-linguistic sounds | JSON | [Taxonomy docs](docs/vocal_bursts_taxonomy.md) |
| **Character Archetypes** | 920 archetypes x 92 genres | JSON | [Taxonomy docs](docs/archetypes.md) |
| **Acting Challenges** | 19,247 challenge scenarios | JSON | [Preview (100 samples)](https://projects.laion.ai/Voice-Acting-Pipeline/acting_challenges_preview.html) |
| **Extreme Physical** | 6 categories, 60 subcategories, 600 challenges | JSON | Tension (100), Breathlessness (100), Pain (100), Temperature (100), Taste (100), Surprise (100) |
| **Situation Taxonomy** | 11 dimensions, 289 situations | JSON | [Data file](data/situation_taxonomy.json) — Body posture (32), physical activity (69), speaking target (25), social context (56), environment (22), health (18), face/head gear (14), climate (10), substances (12), fatigue (19), pain (12) |

Paper references:
- [Schuhmann et al., 2025 — arXiv:2505.20033](https://arxiv.org/abs/2505.20033) (EmoNet-Face, VoiceNet, taxonomies)
- [EmoNet-Voice — arXiv:2506.09827](https://arxiv.org/abs/2506.09827) (Empathic Insight Voice Plus)
- See [docs/paper_reference.md](docs/paper_reference.md) for citation and BibTeX.

### Pre-Generated DramaBox Prompts

79,087 ready-to-use DramaBox two-scene CUT TO: prompts across all pathways and languages:

| File | Pathway | Count | Language | Examples |
|------|---------|-------|----------|----------|
| [`dramabox_cca_voicenet.json`](data/dramabox_cca_voicenet.json) | CC-A (VoiceNet) | 19,332 | English | [Examples](docs/dramabox_cca_voicenet_examples.md) |
| [`dramabox_cc2c_archetype.json`](data/dramabox_cc2c_archetype.json) | CC2-C (Archetype) | 9,999 | English | [Examples](docs/dramabox_cc2c_archetype_examples.md) |
| [`dramabox_accc_acting_challenge.json`](data/dramabox_accc_acting_challenge.json) | ACCC (Acting Challenge) | 12,893 | English | [Examples](docs/dramabox_accc_acting_challenge_examples.md) |
| [`dramabox_sit_situation.json`](data/dramabox_sit_situation.json) | SIT (Situation) | 5,749 | en/fr/es/de | [Examples](docs/dramabox_sit_situation_examples.md) |
| [`dramabox_extreme_physical.json`](data/dramabox_extreme_physical.json) | Extreme Physical | 600 | English | [Examples](docs/dramabox_extreme_physical_examples.md) |
| [`dramabox_cca_voicenet_de.json`](data/dramabox_cca_voicenet_de.json) | CC-A (VoiceNet) | 9,983 | German | [Examples](docs/dramabox_cca_voicenet_de_examples.md) |
| [`dramabox_cc2c_archetype_de.json`](data/dramabox_cc2c_archetype_de.json) | CC2-C (Archetype) | 9,983 | German | [Examples](docs/dramabox_cc2c_archetype_de_examples.md) |
| [`dramabox_accc_acting_challenge_de.json`](data/dramabox_accc_acting_challenge_de.json) | ACCC (Acting Challenge) | 9,948 | German | [Examples](docs/dramabox_accc_acting_challenge_de_examples.md) |
| [`dramabox_extreme_physical_de.json`](data/dramabox_extreme_physical_de.json) | Extreme Physical | 600 | German | [Examples](docs/dramabox_extreme_physical_de_examples.md) |

German prompts use oe/ae/ue instead of umlauts (ö/ä/ü). Directions and speaker descriptions are in English; only spoken dialogue (in "double quotes") is in the target language.

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
7. Construct structured LLM prompt with all constraints -> Gemma 4 generates DramaBox script

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

Audition-style method acting performances driven by acting challenge scenarios. Samples from 19,247 structured challenges covering diverse emotional and situational contexts.

1. Sample a random acting challenge (title + instruction) from the [challenge database](https://projects.laion.ai/Voice-Acting-Pipeline/acting_challenges_preview.html)
2. Sample speaker gender (VoiceNet GEND dimension, 7 levels) and age (AGEV dimension, 7 levels)
3. Sample word count (40-80 words)
4. Gemma 4 generates a DramaBox prompt — actor performs the challenge naturalistically
5. DramaBox TTS -> Sidon+VC augmentation -> Best-of-N scoring

Key characteristics:
- **No self-introduction** — the actor simply begins performing
- **Dynamic emotional arc** with at least one turning point or new insight
- **Naturalistic, genuine, spontaneous** delivery — method acting, not theatrical performance
- **Diverse delivery** — whispered, loud, sensual, ranting, all valid if authentic

See [docs/path_ac_acting_challenge.md](docs/path_ac_acting_challenge.md) for full details.

### Path SIT — Situation

Situation-driven acting challenges where the actor is physically and socially embedded in a specific situation from the [Situation Taxonomy](data/situation_taxonomy.json). The taxonomy covers 11 dimensions with 289 total situations describing how body posture, physical activity, speaking target, social context, environment, health conditions, face/head gear, climate, substances, fatigue, and pain affect the voice.

1. Sample a situation from one of 289 levels across 11 dimensions (e.g. "Lying flat on back", "Eating crunchy food", "Boardroom negotiation", "Freezing cold")
2. Sample 1-3 emotions from EmoNet with intensity levels
3. Sample speaker gender (7 levels) and age (6 levels)
4. DeepSeek V4 Flash generates an acting challenge that places the actor IN this situation — the physical/social context naturally affects the voice and performance
5. Each challenge has an emotional arc with micro-distractions and organic authenticity

The situation taxonomy is based on the [Extended VoiceNet Taxonomy](data/voicenet_ext_taxonomy.html) ([interactive viewer](https://projects.laion.ai/Voice-Acting-Pipeline/voicenet_extension_taxonomy.html)) which extends the core 57 VoiceNet voice attribute dimensions with situation-dependent dimensions that describe how the speaker's physical state and environment affect their vocal output.

See [docs/path_sit_situation.md](docs/path_sit_situation.md) for full details.

---

## Character Consistent Paths — Details

All CC paths produce **two scenes with the same speaker** in contrasting emotional states, separated by a "CUT TO:" marker.

### CC v1 (A/B/C) — Character Consistent

The original two-scene format. Three sampling variants matching standalone Paths A, B, C:

- **CC-A** (VoiceNet): Full 57-dim sampling + contrasting emotions between scenes
- **CC-B** (Archetype): Archetype-based + contrasting emotions
- **CC-C** (Archetype Named): Named archetype + contrasting emotions

**Emotion contrast logic:** If Scene 1 has positive emotions -> Scene 2 samples from negative emotions (and vice versa). Word count: 50-80 total (~25-40 per scene).

See [docs/path_cc_character_consistent.md](docs/path_cc_character_consistent.md) for full details.

### CC2 v2 (A/B/C) — Character Consistent v2

Improved version of CC with enhanced LLM prompting:

- **Scene 1 setup:** Before the first dialogue, 1-2 sentences vividly set the emotional situation — social context, speaker's state of mind, emotional energy
- **Scene 2 transition:** After "CUT TO:", 1-3 sentences explicitly describe the dramatic shift in emotional tone, talking style, delivery, and pace
- **Performance quality emphasis:** Delivery must sound like a real, living, breathing human being — genuine, spontaneous, authentic, with natural hesitations and organic pacing

See [docs/path_cc2_character_consistent_v2.md](docs/path_cc2_character_consistent_v2.md) for full details.

### ACCC — Acting Challenge Character Consistent

Challenge-driven two-scene format: same actor performing the same acting challenge at two different emotional moments with dramatically shifted delivery. This is the **primary path** used in the current experiment.

1. Sample acting challenge + gender + age (same as standalone AC)
2. Sample word count (40-80 total, split ~evenly between scenes)
3. Gemma 4 generates two contrasting scenes from the same challenge
4. DramaBox TTS -> Sidon+VC augmentation -> Best-of-25 scoring
5. LLM-guided split: Whisper turbo ASR word timestamps + Gemma 4 split-point detection + quiet-spot energy analysis -> Scene 1 + Scene 2

See [docs/path_ac_acting_challenge.md#accc-character-consistent](docs/path_ac_acting_challenge.md#accc-character-consistent) for full details.

---

## Quick Start

### Installation

```bash
git clone https://github.com/LAION-AI/Voice-Acting-Pipeline.git
cd Voice-Acting-Pipeline
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
| [`sarulab-speech/sidon_raw_weight`](https://huggingface.co/sarulab-speech/sidon_raw_weight) | Speech restoration (w2v-BERT LoRA + DAC, 16kHz->48kHz) | ~4GB |
| [Chatterbox VC](https://github.com/resemble-ai/chatterbox) | Voice conversion (S3Gen flow-matching VC) | ~4GB |
| DNS-MOS (PyTorch) | Quality scoring (SIG/BAK/OVR, 1-5 scale) | ~0.1GB |
| [`laion/VoiceCLAP`](https://huggingface.co/laion/VoiceCLAP) | Audio-text similarity scoring (Large 3584-dim + Small 768-dim) | ~2GB |
| [`laion/Empathic-Insight-Voice-Plus`](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) | 40 EmoNet emotion scoring + content enjoyment (BUD-E-Whisper + MLP) | ~2GB |
| [`laion/BUD-E-Whisper`](https://huggingface.co/laion/BUD-E-Whisper) | Audio encoder for emotion scoring (768-dim embeddings) | ~1GB |
| [Whisper turbo](https://github.com/openai/whisper) | Word-level timestamps for ASR + audio splitting | ~3GB |
| [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ASR for WER scoring | ~2GB |
| [`laion/timbre-whisper`](https://huggingface.co/laion/timbre-whisper) | On-the-fly timbre captioning (Path D) | ~2GB |
| [MOSS-Audio-8B-Thinking](https://huggingface.co/ICTNLP/MOSS-Audio-8B-Thinking) | Audio-guided prompt re-annotation (3 passes per sample) | ~8GB |

## Project Structure

```
Voice-Acting-Pipeline/
├── README.md                          # This file
├── LAION-Voice-Whitepaper.md          # Dataset plan: LAION Voice + Voice Acting corpus
├── config.json                        # All configurable parameters
├── config_schema.md                   # Documentation for config fields
├── pyproject.toml                     # Python packaging
├── run_sample_groups.py               # Full pipeline: TTS + Sidon/VC + ASR + LLM splitting + scoring + HTML
├── run_reannotate.py                  # Standalone Gemma 4 re-annotation + HTML report rebuild
├── run_resplit.py                     # Standalone LLM-guided CUT TO: re-splitting
├── data/
│   ├── voicenet_ext_taxonomy.html     # VoiceNet (57 dims x 7 levels)
│   ├── all_acting_challenges.json     # 19,247 acting challenge scenarios
│   ├── acting_challenges_situation_inspired.json  # 5,749 situation-inspired challenges
│   ├── acting_challenges_eric_morris_inspired.json # 4,030 Eric Morris-inspired challenges
│   ├── acting_challenges_existing_inspired.json    # 7,390 existing challenge variants
│   ├── dramabox_cca_voicenet.json     # 19,332 pre-generated CC-A DramaBox prompts
│   ├── dramabox_cc2c_archetype.json   # 9,999 pre-generated CC2-C DramaBox prompts
│   ├── dramabox_accc_acting_challenge.json # 12,893 pre-generated ACCC DramaBox prompts
│   ├── dramabox_sit_situation.json   # 5,749 pre-generated SIT DramaBox prompts (en/fr/es/de)
│   ├── dramabox_cca_voicenet_de.json # 9,983 German CC-A DramaBox prompts (no umlauts)
│   ├── dramabox_cc2c_archetype_de.json # 9,983 German CC2-C DramaBox prompts (no umlauts)
│   ├── dramabox_accc_acting_challenge_de.json # 9,948 German ACCC DramaBox prompts (no umlauts)
│   ├── dramabox_extreme_physical.json # 600 extreme physical DramaBox prompts
│   ├── dramabox_extreme_physical_de.json # 600 German extreme physical DramaBox prompts
│   ├── acting_challenges_extreme_physical.json # 600 extreme physical challenges
│   ├── extreme_physical_taxonomy.json # 6 categories x 10 subcategories taxonomy
│   ├── situation_taxonomy.json        # Situation taxonomy (poses, activities, contexts)
│   ├── emonet_taxonomy.json           # EmoNet (40 emotions x 4 intensity levels)
│   ├── vocal_bursts_taxonomy.json     # Vocal bursts (120 types)
│   ├── archetypes.json                # Archetypes (920 x 92 genres)
│   └── wordlists/                     # Per-language word lists
├── dramabox/
│   ├── cli.py                         # CLI entry point
│   ├── config_loader.py               # Config loading and validation
│   ├── taxonomy.py                    # Taxonomy parsers and loaders
│   ├── sampling.py                    # Path A + Path B sampling
│   ├── reference_sampling.py          # Path D: reference audio sampling
│   ├── prompts.py                     # LLM prompt construction
│   ├── prompt_generator.py            # Multi-GPU LLM batch generation
│   ├── tts_synthesizer.py             # Multi-GPU DramaBox TTS
│   ├── sidon_enhance.py               # Sidon + ChatterboxVC augmentation (replaces RE-USE + LavaSR)
│   ├── reuse_enhance.py               # RE-USE speech enhancement (legacy)
│   ├── moss_refine.py                 # MOSS-Audio re-annotation (audio-guided prompt rewriting)
│   ├── moss_pipeline.py               # MOSS orchestrator (multi-GPU job distribution)
│   ├── scoring.py                     # ASR WER + content enjoyment + EmoNet scoring
│   ├── demo_grid.py                   # HTML demo grid generator
│   └── pipeline.py                    # Mode 1-6 orchestrator
├── scripts/
│   ├── _accc_lavasr_pipeline.py       # ACCC LavaSR experiment pipeline (current)
│   ├── _emonet_worker.py              # EmoNet 40-emotion GPU worker
│   ├── _score_worker.py               # WER + content enjoyment GPU worker
│   ├── _lavasr_worker.py              # LavaSR BWE GPU worker
│   └── _lavasr_clap_worker.py         # VoiceCLAP scoring GPU worker
├── docs/
│   ├── voicenet_taxonomy.md           # VoiceNet 57-dim taxonomy
│   ├── voicenet_extension_taxonomy.html  # Interactive VoiceNet viewer
│   ├── emonet_taxonomy.md             # EmoNet 40 emotions
│   ├── vocal_bursts_taxonomy.md       # 120 vocal bursts
│   ├── archetypes.md                  # 920 archetypes
│   ├── acting_challenges_preview.html # Acting challenge preview (100 samples)
│   ├── paper_reference.md             # Citation and BibTeX
│   ├── path_a_voicenet.md             # Path A detailed docs
│   ├── path_b_archetype.md            # Path B detailed docs
│   ├── path_c_archetype_named.md      # Path C detailed docs
│   ├── path_d_reference.md            # Path D detailed docs
│   ├── path_ac_acting_challenge.md    # AC + ACCC detailed docs
│   ├── path_sit_situation.md          # SIT + SIT-CC situation pathway docs
│   ├── path_cc_character_consistent.md   # CC v1 detailed docs
│   ├── path_cc2_character_consistent_v2.md  # CC2 v2 detailed docs
│   ├── dramabox_cca_voicenet_examples.md        # CC-A English examples
│   ├── dramabox_cc2c_archetype_examples.md      # CC2-C English examples
│   ├── dramabox_accc_acting_challenge_examples.md # ACCC English examples
│   ├── dramabox_sit_situation_examples.md       # SIT multilingual examples
│   ├── dramabox_cca_voicenet_de_examples.md     # CC-A German examples
│   ├── dramabox_cc2c_archetype_de_examples.md   # CC2-C German examples
│   ├── dramabox_accc_acting_challenge_de_examples.md # ACCC German examples
│   ├── dramabox_extreme_physical_examples.md    # Extreme Physical English examples
│   ├── dramabox_extreme_physical_de_examples.md # Extreme Physical German examples
│   └── demo/                          # HTML demo grids with embedded audio
│       ├── sidon_vc_sample_groups.html # Sidon+VC experiment (20 groups, LLM splits)
│       ├── accc_lavasr.html           # ACCC LavaSR index (redirects to page 1)
│       ├── accc_lavasr_p1.html        # ACCC LavaSR grid pages 1-10
│       ├── ...
│       └── pitch_analysis.html        # Pitch analysis
└── examples/
    └── example_prompt.txt             # Sample DramaBox prompt
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Prompt generation | 1 GPU, 16GB VRAM | 4+ GPUs, 16GB+ each |
| TTS synthesis | 1 GPU, 24GB VRAM | 4+ GPUs, 24GB+ each |
| Sidon + ChatterboxVC augmentation | 1 GPU, 12GB VRAM | 8 GPUs (parallel workers) |
| MOSS re-annotation | 1 GPU, 8GB VRAM (4-bit) | 8 GPUs (parallel workers) |
| VoiceCLAP + EmoNet scoring | 1 GPU, 4GB VRAM | 8 GPUs (parallel workers) |
| RAM | 32GB | 64GB+ |

---

## MOSS Re-Annotation (Audio-Guided Prompt Rewriting)

After post-processing, the pipeline runs a MOSS re-annotation pass (`dramabox/moss_refine.py`) that closes the loop between intended and actual performance.

### Concept

Original DramaBox prompts are **directions** — the TTS model interprets them, and the actual audio may differ from what was requested. MOSS-Audio-8B-Thinking listens to each generated audio clip together with the original prompt and ASR transcript, then **rewrites the prompt to match what was actually performed**.

### Three Inference Passes Per Sample

1. **Full audio** -> refined two-scene prompt matching the actual performance
2. **Part 1 audio** -> standalone single-scene prompt for Scene 1
3. **Part 2 audio** -> standalone single-scene prompt for Scene 2

### Architecture

```
Original DramaBox Prompt
        +
ASR Transcript (from Whisper)
        +
Generated Audio (MP3)
        |
        v
+---------------------------+
|  MOSS-Audio-8B-Thinking   |
|  (4-bit, per-GPU worker)  |
|                           |
|  Listens to audio +       |
|  reads text context        |
|                           |
|  Rewrites prompt to       |
|  match actual performance  |
+---------------------------+
        |
        v
moss_refined_prompt_full
moss_refined_prompt_part1
moss_refined_prompt_part2
```

### Why Re-Annotation Matters

- Original prompts are **suggestions**, not guarantees
- Audio may differ from prompt due to TTS interpretation
- MOSS-refined prompts become **ground truth** for annotations
- Enables training on **what models actually do**, not what we asked for

### Usage

```bash
# Run MOSS re-annotation on all post-processed samples
python dramabox/moss_refine.py                    # All GPUs
python dramabox/moss_refine.py --num-gpus 4       # 4 GPUs
python dramabox/moss_refine.py --test             # First 10 samples, 1 GPU
```

Requires `/tmp/moss_venv` with `transformers==4.57.1` (MOSS is incompatible with transformers >= 5.x).

---

## License

- This pipeline code — [Apache 2.0](LICENSE)
- [DramaBox](https://huggingface.co/ResembleAI/Dramabox) — see model card
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Apache 2.0
- [Seed-VC](https://github.com/Plachtaa/seed-vc) — MIT
