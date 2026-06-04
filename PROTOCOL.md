# DramaBox Prompt Generation Protocol

Testing results for LLM-based DramaBox prompt generation, covering model selection, system prompt design, and pipeline integration.

## Models Tested

### Gemma-4-E4B-Uncensored (GGUF Q8)

- **Model**: [HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive) (Q8_K_P quantization)
- **File size**: 7.6 GB GGUF
- **Inference**: llama-cpp-python with `n_ctx=8192, n_gpu_layers=-1`
- **VRAM**: ~9-10 GB
- **Result**: Excellent. 10/10 prompts succeeded (5 Standard, 5 CUT TO:)

Compared to the official `google/gemma-4-E4B-it`, the uncensored variant produces bolder, more varied creative output. It does not refuse emotional or dramatic content that the censored model sometimes flags.

### MOSS-Audio-8B-Thinking (text-only mode)

- **Model**: OpenMOSS-Team/MOSS-Audio-8B-Thinking (Qwen3-8B backbone)
- **VRAM (bf16)**: ~18 GB model + KV cache → OOM on 80GB A100 with long prompts
- **VRAM (4-bit via BitsAndBytesConfig)**: ~13.5 GB model, but KV cache grows to 45-58 GB with long inputs
- **Result**: Not suitable for text-only prompt generation. 3/10 OOM, 7/10 produced 8-45 tokens of output

MOSS is designed as an audio understanding model. When used text-only (without audio input), it produces very short, fragmentary outputs — not usable DramaBox prompts. It is suitable for audio-grounded tasks like prompt refinement (listening to audio and rewriting prompts to match actual performance).

**4-bit quantization note**: Must use `quantization_config=BitsAndBytesConfig(...)` as a named parameter, not `load_in_4bit=True` directly — `MossAudioModel.__init__()` does not accept `load_in_4bit` as a keyword argument.

## Generation Results Summary

### Gemma-4-E4B-Uncensored

| Type | Count | Avg Time | Avg Output Tokens | Avg Output Chars | Success |
|------|-------|----------|-------------------|------------------|---------|
| Standard | 5 | 12.5s | 293 | 1471 | 5/5 |
| CUT TO: | 5 | 7.7s | 193 | 859 | 5/5 |

- Standard prompts with vocal burst instructions (~6500 input tokens) take ~13s
- Standard prompts without vocal bursts (~1700 input tokens) take ~10s
- CUT TO: prompts (~1500 input tokens) are fastest at ~7-8s
- All outputs are well-formatted, follow DramaBox structure, and contain meaningful dramatic content

### MOSS-Audio-8B-Thinking (text-only)

| Type | Count | OOM | Avg Tokens (success) | Avg Chars (success) |
|------|-------|-----|----------------------|---------------------|
| Standard | 5 | 3 | 14 | 46 |
| CUT TO: | 5 | 0 | 19 | 53 |

- Standard prompts with long system prompts (6500+ tokens) cause OOM even in 4-bit
- When generation succeeds, output is 8-45 tokens — essentially a single sentence fragment
- MOSS uses ~1000 tokens for `<think>` reasoning before producing minimal output
- Conclusion: MOSS is not viable for text-only DramaBox prompt generation

## System Prompt Design

### Standard (Single-Scene)

The system prompt includes:
1. DramaBox format rules (single speaker, alternating directions/dialogue, stage direction conventions)
2. Recording environment assumption (studio quality, no background noise)
3. Content quality guidance (dramatically interesting, coherent, minimum 10 words dialogue)
4. A baked-in format example showing correct formatting
5. Critical formatting rules (double quotes, parenthetical directions, raw output only)

### CUT TO: (Two-Scene, Character-Consistent)

The CUT TO: system prompt includes:
1. Single-speaker constraint with identity anchoring ("the same voice")
2. No-markdown rule
3. Direction format guidance (parentheses, 5-12 words each)
4. Natural/spontaneous delivery guidance
5. Word count targets (~50 words total, ~25 per scene)
6. Structural template (speaker description → scene 1 → CUT TO: → scene 2 → closing)
7. Emotion contrast maximization guidance
8. A baked-in format example

### User Prompt Construction

User prompts include sampled attributes from the VoiceNet/EmoNet taxonomies:
- **Path A (VoiceNet)**: Target language, accent, emotions, voice attributes, mandatory words, word count, flow style, emotion alignment, direction style, optional vocal bursts
- **Path B (Archetype)**: Archetype description, genre, tempo, arousal level, emotions, word count
- **Path CC (CUT TO:)**: Emotions for scene 1, voice attributes, word count split across scenes

Three suffixes are appended to every user prompt:
- Genuine delivery ("not robotic, not exaggerated, not acted")
- Spontaneous delivery ("words discovered in the moment")
- Studio quality ("pristine, high-quality studio recording")

## GPU Memory Requirements

| Component | VRAM |
|-----------|------|
| Gemma-4-E4B Q8 GGUF (llama-cpp) | ~9-10 GB |
| MOSS-Audio-8B bf16 | ~18 GB (+ KV cache) |
| MOSS-Audio-8B 4-bit | ~13.5 GB (+ KV cache) |
| DramaBox TTS (ResembleAI/Dramabox + Gemma-3-12B-4bit encoder) | ~24 GB |
| RE-USE speech enhancement | ~3 GB |
| LavaSR super-resolution | ~4 GB |
| Whisper turbo ASR | ~1.5 GB |
| VoiceCLAP Large | ~2 GB |

## Pipeline Architecture

The end-to-end pipeline runs on air-gapped supercomputers with no web API access. All models are loaded from local paths.

```
Phase 1: Prompt Generation (dramabox/prompt_generator.py)
  → Gemma-4-E4B-Uncensored generates DramaBox prompts
  → Output: CSV files with prompts

Phase 2: TTS Synthesis (dramabox_pipeline.py)
  → DramaBox TTS generates audio from prompts
  → Output: WAV + JSON sidecar per prompt, tarred and uploaded

Phase 3: Post-Processing (dramabox_postprocess.py)
  → RE-USE enhancement → LavaSR → Whisper ASR → CUT TO: split
  → Part1/Part2/Full MP3 conversion + VoiceCLAP embeddings
  → Output: Annotated MP3 + JSON per sample

Phase 4: MOSS Refinement (dramabox_moss_refine.py)
  → MOSS-Audio-8B listens to each audio + reads original prompt + ASR
  → Rewrites DramaBox prompt to match actual performance (3 passes: full, part1, part2)
  → Output: Updated annotation JSON with refined prompts
  → Requires: /tmp/moss_venv with transformers==4.57.1
```

### MOSS Venv Setup

MOSS-Audio requires transformers ~4.57.x (transformers 5.x produces degenerate output from the Qwen3 backbone). Create a dedicated venv:

```bash
python -m venv /tmp/moss_venv
/tmp/moss_venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
/tmp/moss_venv/bin/pip install transformers==4.57.1 'huggingface_hub>=0.34,<0.35' \
    bitsandbytes accelerate soundfile scipy numpy
```

The MOSS refinement script auto-detects this venv and launches workers using `/tmp/moss_venv/bin/python`. If the venv is not found, it falls back to `sys.executable` (which will fail if transformers >= 5.x is installed).

## Configuration Reference

Key `config.json` parameters for prompt generation:

```json
{
  "prompt_generation": {
    "llm_model": "google/gemma-4-E4B-it",
    "_comment": "For GGUF: HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive (Q8)",
    "batch_size": 16,
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.95
  },
  "sampling": {
    "cc_ratio": 0.30,
    "archetype_ratio": 0.20,
    "vocal_bursts_probability": 0.50
  }
}
```

With `cc_ratio=0.30` and `archetype_ratio=0.20`:
- 30% of prompts are CUT TO: (character-consistent two-scene)
- 20% are archetype-based (single-scene from character archetypes)
- 50% are VoiceNet-based (full dimension sampling with voice attributes)
