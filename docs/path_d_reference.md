# Path D — Reference Audio

**The most promising path for voice cloning.** Uses a reference audio clip to guide both prompt generation (via timbre captioning) and final voice matching (via voice conversion). The result sounds like the reference speaker performing with the sampled emotional/stylistic direction.

## Why Two Stages?

Passing `voice_ref` directly to DramaBox TTS leads to unstable, garbled generations. Instead, Path D uses a two-stage approach:

1. **Text-only DramaBox TTS** — guided by a timbre whisper caption that describes the reference speaker's vocal qualities
2. **Chatterbox Voice Conversion** — converts the TTS output to match the reference speaker's actual voice

This produces stable, high-quality voice matching while preserving the emotional performance.

## Sampling Strategy

1. **Reference Audio**: Load a reference clip (5-30 seconds of clean, single-speaker audio)
2. **Timbre Caption**: Generate or load a timbre description via [`laion/timbre-whisper`](https://huggingface.co/laion/timbre-whisper)
   - Example: *"A warm, resonant baritone with slight breathiness and a smooth, unhurried delivery"*
3. **Situation-Dependent VoiceNet Dims**: Filter the 57 VoiceNet dimensions to exclude identity-related ones (age, gender, timbre, resonance). Sample 5 from the remaining situation-dependent dimensions.
4. **Emotions**: Sample 1-3 emotions from EmoNet with intensity
5. **Tempo**: Sampled from VoiceNet

### Why Filter VoiceNet Dimensions?

Identity-related dimensions (voice age, perceived gender, timbre qualities, resonance placement) would conflict with the reference speaker's actual characteristics. Only situation-dependent dimensions (emotional delivery, pacing, intensity, speaking style) are sampled — these describe *how* the speaker performs, not *who* they are.

## LLM Prompt Construction

The prompt includes:
- **Timbre caption** — describes the reference speaker's vocal qualities
- **Sampled situation-dependent dimensions** — performance attributes
- **Sampled emotions** — emotional coloring
- **Language** — target language for dialogue

Gemma 4 generates a DramaBox script that is vocally consistent with the timbre description while incorporating the sampled performance attributes.

## Audio Processing

1. **DramaBox TTS**: Text-only synthesis (no `voice_ref` parameter) — the timbre caption in the script guides the vocal quality
2. **Chatterbox Voice Conversion**: Convert TTS output to match reference speaker
3. **Best-of-N Scoring**: Score by WER + content enjoyment, select best candidate

## Demo

Listen to Path D samples in the [main demo grid](https://projects.laion.ai/Voice-Acting-Pipeline/demo/).
