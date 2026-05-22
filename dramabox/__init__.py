"""DramaBox Pipeline — voice prompt generation and audio synthesis.

Sampling paths:
  A: VoiceNet-based (full dimension sampling)
  B: Archetype-based (genre/character archetype)
  C: Reference audio (timbre whisper + situation-dependent dims)
  D: MOSS Audio Thinking (direct audio analysis)
"""
__version__ = "0.2.0"
