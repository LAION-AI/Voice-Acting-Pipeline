# DramaBox Pipeline

End-to-end voice prompt generation and audio synthesis using the [DramaBox](https://huggingface.co/ResembleAI/Dramabox) TTS model and structured voice taxonomy sampling.

Based on the voice taxonomy research from [Schumann et al., 2025](https://arxiv.org/abs/2505.20033).

## Overview

This pipeline generates richly annotated voice performance prompts in the **DramaBox format** — single-speaker scenes with stage directions (English) and spoken dialogue (target language) — then synthesizes them into audio using the DramaBox TTS model.

Each prompt is procedurally constructed by sampling from four structured taxonomies:

- **VoiceNet**: 57 voice performance dimensions with 7 ordinal levels each
- **EmoNet**: 40 emotion categories with 4 intensity levels
- **Vocal Bursts**: 120 non-linguistic vocal sounds (sobs, laughs, gasps, etc.)
- **Archetypes**: 920 character voice archetypes across 92 genres

## Features

- **Four sampling paths**:
  - **Path A** (VoiceNet): Full 57-dimension voice attribute sampling (default 80%)
  - **Path B** (Archetype): Genre/character archetype-based sampling (default 20%)
  - **Path C** (Reference Audio): Timbre whisper + situation-dependent VoiceNet dims
  - **Path D** (MOSS Audio Thinking): Direct audio analysis with chain-of-thought reasoning
- **Multi-GPU batch processing** for prompt generation and TTS synthesis
- **Audio refinement**: Resemble Enhance (denoise/super-resolution) + Chatterbox Voice Conversion
- **Best-of-N ranking**: Composite reward scoring using Parakeet v3 ASR (WER) + Empathic Insight Plus (content enjoyment)
- **On-the-fly timbre captioning**: Automatic timbre description via `laion/timbre-whisper`
- **16 languages** preconfigured (4 active, 12 ready to enable)
- **DramaBox TTS** with `torch.compile`, 30-step Euler flow matching, CFG 2.0
- **Demo grid generation**: HTML comparison pages with audio players
- **Configurable everything**: `config.json` controls all parameters — no code changes needed

## Quick Start

### Installation

```bash
git clone https://github.com/<org>/dramabox-pipeline.git
cd dramabox-pipeline
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

Output: CSV chunk files in `./output/`.

### Synthesize Audio (Mode 2)

```bash
# Synthesize audio from an existing CSV
dramabox synthesize --csv output/dramabox_chunk_000.csv --gpus 0,1,2,3
```

Output: WAV files in `./output/audio/`.

### End-to-End (Mode 3)

```bash
# Generate prompts and immediately synthesize audio
dramabox run --config config.json --total 1000 --gpus 0,1,2,3
```

### Reference Audio Pipeline — Path C (Mode 4)

```bash
# Generate prompts using reference audio with timbre annotations
dramabox reference --config config.json --ref-dir /path/to/emolia/references --total 10 --gpus 6,7
```

Uses reference audio's timbre whisper caption + situation-dependent VoiceNet dimensions to create prompts that match the reference speaker's voice characteristics.

### MOSS Audio Thinking — Path D (Mode 5)

```bash
# Generate prompts by having MOSS Audio listen to reference audio
dramabox moss --config config.json --ref-dir /path/to/audio/files --total 10 --gpus 6,7
```

MOSS-Audio-4B-Thinking directly analyzes the audio and generates DramaBox prompts with chain-of-thought reasoning traces.

### Demo Grid (Mode 6)

```bash
# Generate HTML demo grid with 5 references × 10 configs
dramabox demo --config config.json --gpus 6,7
```

### Score Audio

```bash
# Score an audio file against its DramaBox prompt
dramabox score --audio output/audio/sample_000000_raw.wav --prompt "prompt text or file path" --gpu 0
```

## Configuration

All parameters are in [`config.json`](config.json). See [`config_schema.md`](config_schema.md) for full documentation of every field.

### Key Settings

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `prompt_generation` | `llm_model` | `google/gemma-4-E4B-it` | LLM for prompt generation |
| `prompt_generation` | `total_prompts` | `100000` | Number of prompts to generate |
| `sampling` | `archetype_ratio` | `0.20` | Fraction using archetype path |
| `sampling` | `word_count_min/max` | `5 / 60` | Target dialogue word count range |
| `tts` | `cfg_scale` | `2.0` | Classifier-free guidance scale |
| `tts` | `compile` | `true` | Enable torch.compile for faster inference |
| `tts` | `steps` | `30` | Euler flow matching steps |
| `output` | `chunk_size` | `5000` | Prompts per CSV chunk file |
| `reference_audio` | `situation_dims_count` | `5` | Situation-dependent dims per Path C sample |
| `best_of_n` | `n_candidates` | `3` | Candidates per Best-of-N ranking |
| `refinement` | `enhance_mode` | `"enhance"` | Resemble Enhance mode |

### Adding Languages

Languages are configured in the `languages` section of `config.json`. To enable a new language:

1. Set `"_enabled": true` for the language entry
2. Add a word list at `data/wordlists/<language>.json` (JSON array of words)
3. Optionally add accent variants to the `accents` array

Currently active: English, German, French, Spanish. Ready to enable: Italian, Dutch, Russian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Polish, Swedish.

## Architecture

### Pipeline Modes

| Mode | Command | Input | Output | Description |
|------|---------|-------|--------|-------------|
| 1 | `generate-prompts` | config.json | CSV chunks | Generate prompts only |
| 2 | `synthesize` | CSV file | WAV files | Synthesize audio from CSV |
| 3 | `run` | config.json | CSV + WAV | End-to-end pipeline |
| 4 | `reference` | ref audio dir | JSON + prompts | Path C: reference audio pipeline |
| 5 | `moss` | audio dir | JSON + prompts | Path D: MOSS Audio thinking |
| 6 | `demo` | config.json | HTML grid | Demo comparison grid |
| — | `score` | audio + prompt | scores | ASR WER + content enjoyment |

### Sampling Path A — VoiceNet (default 80%)

1. Sample language + accent
2. Sample 1–3 emotions from EmoNet with intensity
3. Sample 3 mandatory VoiceNet dims (Tempo, Gender, Age) + 5 random from 54 remaining
4. Determine flow style (scattered/flowing/mixed), emotion alignment, direction style
5. Optionally include vocal bursts taxonomy
6. Inject 3 mandatory words from language-specific word list
7. Construct structured LLM prompt with all constraints

### Sampling Path B — Archetype (default 20%)

1. Pick a random genre and archetype from 920 options
2. Sample language + accent
3. Sample 1–3 emotions with intensity
4. Sample Tempo (with fast bias) and Arousal (uniform)
5. Construct archetype-focused LLM prompt — no flow/alignment/direction constraints

### Sampling Path C — Reference Audio

1. Load reference audio metadata (timbre whisper caption)
2. Generate timbre caption on-the-fly if missing (via `laion/timbre-whisper`)
3. Filter VoiceNet dimensions to situation-dependent only (exclude identity: age, gender, timbre, resonance)
4. Sample 1–3 emotions + tempo + 5 situation-dependent dimensions
5. Construct LLM prompt with timbre caption + sampled performance attributes
6. Synthesize with DramaBox TTS using reference audio for voice matching
7. Optionally refine with Resemble Enhance + Chatterbox VC

### Sampling Path D — MOSS Audio Thinking

1. Feed reference audio to MOSS-Audio-4B-Thinking model
2. Model reasons about the speaker's voice characteristics (chain-of-thought)
3. Generates a complete DramaBox prompt based on what it hears
4. Saves reasoning trace + final prompt for analysis

### Audio Refinement Pipeline

For reference audio or generated output:
1. **Resemble Enhance**: Denoise or full enhancement (super-resolution to 48kHz)
2. **Chatterbox VC**: Voice conversion to match reference (self-VC or ref-VC)

### Best-of-N Ranking

1. Generate N candidate audio samples (default 3)
2. Score each with:
   - **WER** (Word Error Rate): Parakeet v3 ASR transcription vs expected text from DramaBox prompt quotes
   - **Content Enjoyment**: Empathic Insight Plus (BUD-E-Whisper encoder + MLP expert)
3. Composite reward: `(1 - min(WER, 1.0)) × content_enjoyment`
4. Select the candidate with the highest reward

### DramaBox TTS

- **Model**: [`ResembleAI/Dramabox`](https://huggingface.co/ResembleAI/Dramabox) — 22B DiT transformer
- **Text Encoder**: [`unsloth/gemma-3-12b-it-bnb-4bit`](https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit) — 4-bit quantized Gemma 3
- **Scheduler**: LTX2Scheduler with 30 Euler flow matching steps
- **Guidance**: CFG scale 2.0, STG scale 1.5
- **torch.compile**: Enabled by default for faster inference after initial warmup
- **Self-VC**: Optional second pass using raw output as voice reference for consistency
- **VRAM**: ~24GB per GPU

## Taxonomy Documentation

- [VoiceNet Taxonomy](docs/voicenet_taxonomy.md) — 57 voice performance dimensions, 7 levels each
- [EmoNet Taxonomy](docs/emonet_taxonomy.md) — 40 emotion categories with intensity levels
- [Vocal Bursts](docs/vocal_bursts_taxonomy.md) — 120 non-linguistic vocal sound types
- [Character Archetypes](docs/archetypes.md) — 920 archetypes across 92 genres
- [Paper Reference](docs/paper_reference.md) — Citation and context

## CSV Output Format

Each generated CSV contains these columns:

| Column | Description |
|--------|-------------|
| `global_idx` | Sequential index across all chunks |
| `sampling_path` | `"voicenet"`, `"archetype"`, or `"reference"` |
| `archetype_info` | `"Genre \| Archetype"` for path B, empty otherwise |
| `language` | Target language for dialogue |
| `accent` | Accent/dialect variant (if any) |
| `emotions` | Sampled emotions with intensity levels |
| `word_count_target` | Target word count for dialogue |
| `must_include_words` | Mandatory words (path A only) |
| `flow_style` | `scattered`, `flowing`, or `mixed` (path A only) |
| `flow_forced_by_voicenet` | Whether flow was forced by extreme VoiceNet values |
| `emotion_alignment` | `congruent`, `neutral`, or `counter-emotional` (path A only) |
| `direction_style` | `literary` or `tag` (path A only) |
| `vocal_bursts_enabled` | Whether vocal bursts taxonomy was included |
| `attributes_raw` | Raw VoiceNet attributes with codes and levels |
| `dramabox_prompt` | The generated DramaBox prompt text |
| `reference_audio` | Path to reference audio (path C/D only) |
| `timbre_caption` | Timbre description from reference (path C/D only) |

## Project Structure

```
dramabox-pipeline/
├── config.json                 # All configurable parameters
├── config_schema.md            # Documentation for config fields
├── pyproject.toml              # Python packaging
├── requirements.txt            # Dependencies
├── data/
│   ├── voicenet_ext_taxonomy.html   # VoiceNet (57 dims)
│   ├── emonet_taxonomy.json         # EmoNet (40 emotions)
│   ├── vocal_bursts_taxonomy.json   # Vocal bursts (120 types)
│   ├── archetypes.json              # Archetypes (92 genres × 10)
│   └── wordlists/                   # Per-language word lists
├── dramabox/
│   ├── cli.py                  # CLI entry point (7 commands)
│   ├── config_loader.py        # Config loading and validation
│   ├── taxonomy.py             # Taxonomy parsers and loaders
│   ├── sampling.py             # Path A + Path B sampling
│   ├── reference_sampling.py   # Path C: reference audio sampling
│   ├── prompts.py              # LLM prompt construction
│   ├── wordlists.py            # Per-language word lists
│   ├── utils.py                # Utility functions
│   ├── csv_io.py               # CSV reader/writer
│   ├── prompt_generator.py     # Multi-GPU LLM batch generation
│   ├── tts_synthesizer.py      # Multi-GPU DramaBox TTS
│   ├── timbre_whisper.py       # On-the-fly timbre captioning
│   ├── audio_refine.py         # Resemble Enhance + Chatterbox VC
│   ├── scoring.py              # ASR WER + content enjoyment scoring
│   ├── moss_pipeline.py        # Path D: MOSS Audio Thinking
│   ├── demo_grid.py            # HTML demo grid generator
│   └── pipeline.py             # Mode 1–6 orchestrator
├── docs/                       # Taxonomy documentation (markdown)
└── examples/                   # Example prompts
```

## Models Used

| Model | Purpose | VRAM |
|-------|---------|------|
| [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) | DramaBox prompt generation | ~16GB |
| [`ResembleAI/Dramabox`](https://huggingface.co/ResembleAI/Dramabox) | TTS synthesis (22B DiT) | ~24GB |
| [`laion/timbre-whisper`](https://huggingface.co/laion/timbre-whisper) | On-the-fly timbre captioning | ~2GB |
| [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ASR for WER scoring | ~2GB |
| [`laion/Empathic-Insight-Voice-Plus`](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) | Content enjoyment scoring | ~2GB |
| [Chatterbox VC](https://github.com/resemble-ai/chatterbox) | Voice conversion | ~4GB |
| [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) | Audio denoising/enhancement | ~2GB |
| MOSS-Audio-4B-Thinking | Audio analysis with reasoning | ~10GB |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Prompt generation | 1 GPU, 16GB VRAM | 4+ GPUs, 16GB+ each |
| TTS synthesis | 1 GPU, 24GB VRAM | 4+ GPUs, 24GB+ each |
| Refinement + scoring | 1 GPU, 8GB VRAM | 1 GPU, 16GB+ |
| MOSS Audio (Path D) | 1 GPU, 16GB VRAM | 1 GPU, 24GB+ |
| RAM | 32GB | 64GB+ |

## Paper Reference

This pipeline builds on the voice taxonomy research presented in:

> Christoph Schumann et al.
> arXiv:2505.20033, 2025
> https://arxiv.org/abs/2505.20033

See [docs/paper_reference.md](docs/paper_reference.md) for the full citation and BibTeX.

## License

This pipeline code is licensed under [Apache 2.0](LICENSE). The DramaBox model has its own license — see the [model card](https://huggingface.co/ResembleAI/Dramabox) for details.
