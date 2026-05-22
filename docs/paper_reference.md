# Paper Reference

The VoiceNet and EmoNet taxonomies used in the DramaBox pipeline are based on research by Christoph Schuhmann et al.

## Paper

**Title:** EmoNet-Face: An Expert-Annotated Benchmark for Synthetic Emotion Recognition

**Link:** [https://arxiv.org/abs/2505.20033](https://arxiv.org/abs/2505.20033)

## BibTeX

```bibtex
@article{schuhmann2025emonet,
  title={EmoNet-Face: An Expert-Annotated Benchmark for Synthetic Emotion Recognition},
  author={Schuhmann, Christoph and Kaczmarczyk, Robert and Rabby, Gollam and Friedrich, Felix and Kraus, Maurice and Kalyan, Krishna and Nadi, Kourosh and Nguyen, Huu and Kersting, Kristian and Auer, S{\"o}ren},
  journal={arXiv preprint arXiv:2505.20033},
  year={2025}
}
```

## VoiceNet and EmoNet in DramaBox

**EmoNet** is a taxonomy of 40 emotion categories, each with synonyms and intensity levels, designed for fine-grained annotation of emotional expression. In DramaBox, the EmoNet taxonomy is used to label and generate emotional voice performances across a structured spectrum of human feelings -- from amusement and elation to anger, fear, and grief -- at four distinct intensity levels.

**VoiceNet** extends this framework to cover non-linguistic vocal expressions (vocal bursts) such as laughter, crying, gasps, growls, and other paralinguistic sounds. The DramaBox pipeline uses the VoiceNet vocal bursts taxonomy of 120 distinct non-speech vocalizations to enrich generated audio drama scripts with realistic non-verbal vocal cues.

Together, these taxonomies provide DramaBox with a principled, research-grounded vocabulary for describing and generating the full range of human vocal expression in dramatic performance contexts.
