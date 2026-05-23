"""DramaBox Pipeline — voice prompt generation and audio synthesis.

Sampling paths:
  A: VoiceNet-based (full dimension sampling)
  B: Archetype-based (genre/character archetype)
  C: Archetype-based with explicit naming (archetype role in DramaBox script)
  D: Reference audio (timbre whisper + situation-dependent dims + Chatterbox VC)
"""
__version__ = "0.2.0"
