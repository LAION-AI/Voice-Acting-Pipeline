# Configuration Reference

All parameters for the DramaBox pipeline are in `config.json`. This document explains every field.

Fields prefixed with `_` (e.g., `_comment`, `_enabled`, `_docs`) are metadata — they are ignored by the pipeline code and exist purely for documentation.

---

## `prompt_generation`

Settings for the LLM that generates DramaBox prompts.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_model` | string | `"google/gemma-4-E4B-it"` | HuggingFace model ID for the text generation LLM. |
| `llm_dtype` | string | `"bfloat16"` | Model dtype. `"bfloat16"` or `"float16"`. |
| `batch_size` | int | `16` | Number of prompts processed per GPU batch. Higher = faster but uses more VRAM. |
| `max_tokens` | int | `1024` | Maximum tokens to generate per prompt. |
| `temperature` | float | `1.0` | Sampling temperature. Higher = more creative/random. |
| `top_p` | float | `0.95` | Nucleus sampling threshold. |
| `total_prompts` | int | `100000` | Total number of prompts to generate. |
| `seed` | int | `600` | Random seed for reproducibility. |
| `gpus` | list[int] | `[0, 1, 2, 3]` | GPU IDs to use. Each gets one model instance. |

**VRAM requirement**: ~16GB per GPU for Gemma 4 E4B-it in BF16.

---

## `languages`

Each key is a language name. The value is an object with:

| Field | Type | Description |
|-------|------|-------------|
| `_enabled` | bool | `true` to include this language in generation, `false` to skip it. |
| `accents` | list[str] | Optional accent/dialect variants. Empty list = no accent variation. |
| `_comment` | string | (Optional) Notes about when to enable this language. |

**To add a new language:**
1. Set `"_enabled": true` in the config.
2. Add a word list file at `data/wordlists/<language_lowercase>.json` — a JSON array of words. If no word list file exists, English words are used as fallback.
3. Optionally add accent variants to the `accents` array.

**Currently active (with bundled word lists):** English, German, French, Spanish.

**Ready to enable (add word list first):** Italian, Dutch, Russian, Portuguese, Chinese (Mandarin), Japanese, Korean, Arabic, Hindi, Turkish, Polish, Swedish.

---

## `sampling`

Controls how prompt attributes are procedurally sampled.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `archetype_ratio` | float | `0.20` | Fraction of prompts using Path B (archetype-based). Rest use Path A (VoiceNet). |
| `word_count_min` | int | `5` | Minimum target word count for spoken dialogue. |
| `word_count_max` | int | `60` | Maximum target word count for spoken dialogue. |
| `mandatory_dims` | list[str] | `["TEMP", "GEND", "AGEV"]` | VoiceNet dimension codes always sampled in Path A. |
| `random_dims_count` | int | `5` | Number of additional random VoiceNet dimensions in Path A. |
| `emotions_min` | int | `1` | Minimum number of emotion categories per prompt. |
| `emotions_max` | int | `3` | Maximum number of emotion categories per prompt. |
| `mandatory_words_count` | int | `3` | Number of mandatory words injected into Path A prompts. |
| `flow_style_distribution` | object | `{"scattered": 0.05, ...}` | Probability weights for flow style. Must sum to 1.0. |
| `emotion_alignment_distribution` | object | `{"congruent": 0.30, ...}` | Probability weights for emotion–text alignment. |
| `direction_style_distribution` | object | `{"literary": 0.50, ...}` | Probability weights for direction writing style. |
| `vocal_bursts_probability` | float | `0.50` | Probability that vocal bursts taxonomy is included in the LLM prompt. |
| `tempo_bias_threshold` | int | `3` | TEMP levels >= this value get extra sampling weight. |
| `tempo_bias_weight` | float | `1.5` | Weight multiplier for TEMP levels >= threshold (1.0 = no bias). |

### Sampling Path A (VoiceNet-based, default 80%)

- Samples 3 mandatory VoiceNet dimensions + 5 random from 54 remaining
- Includes flow style, emotion alignment, direction style, vocal bursts, mandatory words
- Full taxonomy-driven speaker attribute specification

### Sampling Path B (Archetype-based, default 20%)

- Picks a random genre and archetype from 920 character archetypes
- Only samples TEMP (tempo) and AROU (arousal) from VoiceNet
- No mandatory words, flow style, or direction style — lets the LLM decide naturally
- Provides the archetype description as creative inspiration

---

## `output`

Settings for CSV output.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunk_size` | int | `5000` | Number of prompts per CSV chunk file. |
| `output_dir` | string | `"./output"` | Directory for output files (CSV + audio). Relative to config file. |
| `csv_prefix` | string | `"dramabox"` | Filename prefix for CSV chunks (e.g., `dramabox_chunk_000.csv`). |

---

## `tts`

Settings for DramaBox TTS audio synthesis. Requires the [DramaBox](https://huggingface.co/ResembleAI/Dramabox) model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | string | `"ResembleAI/Dramabox"` | HuggingFace model ID for the DramaBox DiT transformer. |
| `text_encoder_id` | string | `"unsloth/gemma-3-12b-it-bnb-4bit"` | HuggingFace model ID for the text encoder. |
| `cfg_scale` | float | `2.0` | Classifier-free guidance scale. Higher = more prompt adherence. |
| `stg_scale` | float | `1.5` | Spatiotemporal guidance scale. |
| `duration_multiplier` | float | `1.1` | Multiplier applied to estimated speech duration. |
| `compile` | bool | `true` | Enable `torch.compile` for the transformer (faster inference after warmup). |
| `compile_mode` | string | `"default"` | torch.compile mode (`"default"`, `"reduce-overhead"`, `"max-autotune"`). |
| `compile_dynamic` | bool | `true` | Enable dynamic shapes in torch.compile. |
| `steps` | int | `30` | Number of Euler flow matching denoising steps. |
| `seed` | int | `42` | Random seed for reproducible audio generation. |
| `gpus` | list[int] | `[0, 1, 2, 3]` | GPU IDs for TTS. Each gets one TTSServer instance. |
| `bnb_4bit` | bool | `true` | Use bitsandbytes 4-bit quantization for Gemma text encoder. |
| `self_vc` | bool | `true` | Generate a second "self-voice-cloned" pass using raw output as reference. |
| `ref_duration` | float | `10.0` | Duration (seconds) of voice reference clip for self-VC. |
| `watermark` | bool | `false` | Apply Perth audio watermark to outputs. |
| `output_format` | string | `"wav"` | Audio output format. |
| `stagger_start_seconds` | int | `2` | Delay between GPU worker launches to avoid download races. |
| `dramabox_dir` | string | `""` | Path to DramaBox repository root (if not pip-installed). |

**VRAM requirement**: ~24GB per GPU with BF16 + BNB-4bit Gemma.

---

## `reference_audio`

Settings for Path D: reference audio pipeline.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `situation_dims_count` | int | `5` | Number of situation-dependent VoiceNet dimensions to sample (excluding identity dims). |
| `timbre_whisper_model` | string | `"laion/timbre-whisper"` | HuggingFace model ID for on-the-fly timbre captioning. |
| `timbre_whisper_max_tokens` | int | `440` | Maximum tokens for timbre caption generation. |

### Identity vs Situation-Dependent Dimensions

Path D excludes **identity-related** VoiceNet dimensions from sampling (they're captured by the timbre whisper caption):
- Speaker characteristics: `GEND` (Gender), `AGEV` (Age)
- Timbral qualities: `BRGT`, `ROUG`, `HARM`, `FULL`, `WARM`, `METL`, `ESTH`
- Resonance placement: `R_CHST`, `R_THRT`, `R_ORAL`, `R_MASK`, `R_NASL`, `R_HEAD`, `R_MIXD`
- Technical/recording: `RCQL`, `BKGN`, `EXPL`

The remaining ~38 **situation-dependent** dimensions (tempo, cognitive load, articulation, speaking styles, arousal, valence, etc.) are eligible for sampling.

---

## `refinement`

Audio refinement pipeline settings (Resemble Enhance + Chatterbox VC).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enhance_mode` | string | `"enhance"` | `"denoise"` for denoising only, `"enhance"` for full enhancement (denoise + super-resolution to 48kHz). |
| `refine_reference` | bool | `true` | Whether to enhance/clean reference audio before use. |
| `refine_reference_self_vc` | bool | `true` | Whether to apply self-VC to enhanced reference audio. |
| `refine_output` | bool | `true` | Whether to enhance generated TTS output. |
| `refine_output_vc_to_ref` | bool | `true` | Whether to VC the enhanced output to match the reference voice. |

### Refinement Pipeline

**For reference audio:**
1. Resemble Enhance (denoise or full enhance) -> cleaned reference
2. (Optional) Chatterbox self-VC -> stabilized reference

**For generated output:**
1. Resemble Enhance -> cleaned output
2. (Optional) Chatterbox VC to reference -> voice-matched output

---

## `best_of_n`

Best-of-N ranking with composite reward scoring.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Whether to enable Best-of-N ranking. |
| `n_candidates` | int | `3` | Number of candidate samples to generate per prompt. |
| `asr_model` | string | `"nvidia/parakeet-tdt-0.6b-v3"` | ASR model for transcription and WER computation. |
| `enjoyment_model` | string | `"laion/Empathic-Insight-Voice-Plus"` | Model for content enjoyment scoring. |

### Scoring Formula

```
reward = (1 - min(WER, 1.0)) * content_enjoyment
```

- **WER** (Word Error Rate): Computed by transcribing the audio with Parakeet v3 ASR and comparing against the expected text extracted from double-quoted segments in the DramaBox prompt.
- **Content Enjoyment**: Scored by Empathic Insight Plus (BUD-E-Whisper encoder embeddings + MLP expert).
- The candidate with the highest composite reward is selected.

---

## `demo`

Settings for demo grid generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `emolia_dir` | string | (see config) | Path to Emolia reference audio directory (.json + .mp3 pairs). |
| `n_references` | int | `5` | Number of reference speakers to include in the demo. |
| `n_configs_per_ref` | int | `10` | Number of configuration variants per reference. |
| `output_dir` | string | `"/tmp/dramabox_demo"` | Output directory for demo files (HTML grid + audio). |
| `gpus` | list[int] | `[6, 7]` | GPU IDs for demo generation. |

### Demo Configuration Variants

The demo generates 10 variants per reference:
1. `path_d_default` — Default Path C settings
2. `path_d_high_tempo` — Biased toward fast speech
3. `path_d_low_tempo` — Biased toward slow speech
4. `path_d_high_emotion` — 3 emotions at once
5. `path_d_scattered_flow` — Choppy, fragmented delivery
6. `path_d_flowing` — Long, continuous speech blocks
7. `path_d_german` — German dialogue
8. `path_d_french` — French dialogue
9. `path_d_spanish` — Spanish dialogue
10. `path_d_multi_emotion` — 2–3 emotions

---

## `data_paths`

Paths to taxonomy data files, relative to the config file location.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `voicenet_html` | string | `"data/voicenet_ext_taxonomy.html"` | VoiceNet taxonomy HTML file. |
| `emonet_json` | string | `"data/emonet_taxonomy.json"` | EmoNet emotion taxonomy JSON. |
| `vocal_bursts_json` | string | `"data/vocal_bursts_taxonomy.json"` | Vocal bursts taxonomy JSON. |
| `archetypes_json` | string | `"data/archetypes.json"` | Voice archetypes taxonomy JSON. |
| `wordlists_dir` | string | `"data/wordlists"` | Directory containing per-language word list JSONs. |
