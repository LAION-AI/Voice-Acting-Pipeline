# Vocal Bursts Taxonomy & Generation

This taxonomy defines **202 non-linguistic vocal sounds** organised by emotional and functional category. These vocal bursts represent the full range of human non-speech vocalisations — from laughter and crying to physiological reflexes, animal imitations, and intimate vocalisations.

A **SFW version** (180 entries, NSFW categories removed) is available for use with minor age groups.

## Prompt Format

### DramaBox TTS (text-to-speech)

DramaBox understands performer-framing prompts — it synthesises the vocalisation with vocal qualities matching the described age and gender. **No voice reference audio is needed.**

```
A {age_descriptor} performing {burst_key}, {burst_description}
```

**Examples:**
```
A toddler girl performing Belly Laugh, A deep, uncontrollable laugh that involves the whole body, originating from the diaphragm.
A young man performing Orgasmic Cry, A loud, uncontrolled, rising vocalization at the peak of sexual climax.
A elderly woman performing Exasperated Sigh, A heavy, audible exhale expressing deep frustration or weariness.
A middle-aged man performing Guttural Growl, A very low, harsh, continuous sound of deep hostility.
```

**Parameters:** cfg=2.5, stg=1.5, 30 steps, duration 3–12s random, no voice reference, watermark=False

### Stable Audio 3 Small SFX (text-to-audio)

SA3 is a sound effects model — it does **not** understand performer framing. Prompts describe the sound directly, with the age/gender as a secondary modifier:

```
{burst_description}, {age_descriptor}
```

**Examples:**
```
A deep, uncontrollable laugh that involves the whole body, toddler girl
A loud, uncontrolled, rising vocalization at the peak of sexual climax, young man
A heavy, audible exhale expressing deep frustration or weariness, elderly woman
```

**Parameters:** steps=8, cfg_scale=1.0, 44.1kHz

### MOSS SoundEffect v2.0 (text-to-audio)

Same prompt adaptation as SA3 — description first, age/gender second:

```
{burst_description}, {age_descriptor}
```

**Parameters:** num_inference_steps=100, cfg_scale=4.0, bfloat16, 48kHz

## Demographics

Samples span **16 age/gender groups** (8 per gender):

| Female | Male | Age Range | Minor? |
|--------|------|-----------|--------|
| toddler girl | toddler boy | 2–6 | Yes (SFW only) |
| pre-puberty girl | pre-puberty boy | 7–12 | Yes (SFW only) |
| teenage girl | teenage boy | 13–17 | Yes (SFW only) |
| young woman | young man | 18–30 | No |
| middle-aged woman | middle-aged man | 31–50 | No |
| mature woman | mature man | 51–65 | No |
| elderly woman | elderly man | 66–80 | No |
| senescent woman | senescent man | 80+ | No |

Minor age groups draw from the **SFW taxonomy only** (180 entries). Adult groups draw from the **full extended taxonomy** (202 entries including NSFW).

## Dataset

2000 samples per model (1000 male + 1000 female), 125 per age group. Taxonomy entries are shuffled per age group and cycled to fill all 125 slots.

**Available at:** [huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw](https://huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw)

## Generation Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `data/generate_vocal_bursts.py` | DramaBox TTS | Main generation (2000 samples, multi-GPU) |
| `data/generate_vocal_bursts_sa3.py` | Stable Audio 3 Small SFX | SA3 generation with prompt adaptation |
| `data/generate_vocal_bursts_moss.py` | MOSS SoundEffect v2.0 | MOSS generation (full or grid-only mode) |
| `data/generate_nsfw_vocal_bursts.py` | All models | NSFW comparison prompts & structure |
| `data/postprocess_reuse.py` | NVIDIA RE-USE | Speech enhancement postprocessing |

All scripts support `--multi-gpu` for parallel generation across N GPUs.

## RE-USE Postprocessing

All outputs are optionally enhanced with [NVIDIA RE-USE](https://huggingface.co/nvidia/RE-USE) (9.6M param speech enhancement, SEMamba architecture). RE-USE operates in the STFT domain, processing magnitude and phase through a state-space model to remove noise and artifacts while preserving vocal characteristics.

---

## Full Taxonomy (202 entries)

### Laughter & Amusement (8)

| Name | Description |
|------|-------------|
| Belly Laugh | A deep, uncontrollable laugh that involves the whole body, originating from the diaphragm. |
| Chuckle | A soft, restrained, low-pitched laugh, often with the mouth closed. |
| Giggle | A high-pitched, rapid, and often playful series of short laughs. |
| Cackle | A shrill, harsh, staccato laugh, often perceived as wicked. |
| Snicker | A half-suppressed, nasal laugh expressing disrespect or schadenfreude. |
| Guffaw | A sudden, loud, boisterous burst of laughter. |
| Snort-Laugh | Laughter interrupted by an involuntary nasal snort. |
| Wheezing Laugh | Breathless, high-pitched squeaking when laughing too hard to breathe. |

### Crying & Distress (10)

| Name | Description |
|------|-------------|
| Gentle Sob | A quiet, singular tearful inhalation. |
| Hysterical Wailing | Loud, sustained, erratic vocalisations of deep grief. |
| Whimpering | Soft, high-pitched, broken vocalisations of distress. |
| Bawling | Loud, open-mouthed, unconstrained crying. |
| Mournful Keening | A high-pitched, melodic wailing of mourning. |
| Stifled Sob | A sudden, muffled choking sound when trying to hide crying. |
| Silent Tearful Gasp | A sharp intake of breath with trembling lip, holding back tears. |
| Sniffling (Emotional) | Rapid nasal intakes to hold back mucus/tears. |
| Trembling Exhale | A shaky, vibrating outward breath after crying. |
| Choked-up Swallow | An audible gulp when trying not to cry. |

### Breathing & Sighs (11)

| Name | Description |
|------|-------------|
| Exasperated Sigh | A heavy, forceful exhale expressing frustration. |
| Contented Sigh | A soft, gentle exhale expressing deep satisfaction. |
| Relieved Exhale | A long, smooth outward breath after stress ends. |
| Shuddering Breath | A stuttering, involuntary breath from cold or emotion. |
| Heavy Panting | Rapid, deep breathing from physical exertion. |
| Yawn | A deep, involuntary inhalation with wide-open mouth. |
| Breath-Hold Release | A sudden, explosive exhale after holding breath. |
| Nostril Flare Exhale | A sharp, forceful breath through flared nostrils. |
| Huff of Annoyance | A sharp, forceful exhale through nose or lips. |
| Meditative Deep Breath | A slow, conscious, deep inhalation and exhalation. |
| Winded Gasp | Desperate, laboured breathing after being hit in the abdomen. |

### Surprise & Shock (6)

| Name | Description |
|------|-------------|
| Startled Yelp | A short, sharp, involuntary cry of surprise. |
| Dramatic Gasp | A loud, theatrical inhalation of astonishment. |
| Startle Grunt | A sudden, low-pitched vocalisation when physically startled. |
| Squeak of Fright | A tiny, highly constricted vocalisation of sudden panic. |
| Frozen Breath-Hold | Audible stopping and holding of breath from sudden fear. |
| Breathy "Oh No" | A whispered, almost non-verbal realisation of impending danger. |

### Disgust & Disapproval (8)

| Name | Description |
|------|-------------|
| Disgusted "Ugh" | A guttural, nasal expression of revulsion. |
| Retching / Gag Reflex | Involuntary convulsive sounds triggered by nausea. |
| Scoff | A breathy, slightly voiced nasal sound expressing disdain. |
| Tsk / Tongue Click | A sharp suction sound against the teeth expressing disapproval. |
| Derisive Snort | A forceful nasal exhale expressing contempt. |
| Spitting Sound | An explosive bilabial sound of disgust or contempt. |
| Raspberry / Bronx Cheer | A loud, vibrating sound made with the tongue and lips. |
| Dismissive "Pfft" | A quick, breathy bilabial fricative of disdain. |

### Pain & Discomfort (8)

| Name | Description |
|------|-------------|
| Sharp Yelp | A very brief, high-pitched cry of sudden, sharp pain. |
| Prolonged Groan | A deep, sustained guttural sound of dull, chronic pain. |
| Agonised Scream | A loud, sustained, harsh vocalisation of extreme physical agony. |
| Teeth-Gritting Hiss | A sharp intake of air through closed teeth in reaction to pain. |
| Pain Gasp | A sudden, sharp inhalation reacting to a spike in pain. |
| Chronic Pain Whimper | A soft, continuous, helpless sound during sustained discomfort. |
| Exhausted Panting | Rapid, heavy breathing from physical depletion. |
| Sickly Cough | Involuntary vocalisation triggered by respiratory illness. |

### Effort & Exertion (8)

| Name | Description |
|------|-------------|
| Heavy Lifting Grunt | A short, explosive vocalisation when moving a heavy object. |
| Martial Arts Kiai | A sharp, focused shout used to channel energy in combat. |
| Tennis Grunt | A rhythmic, explosive exhale during a powerful stroke. |
| Straining Hiss | A long, tight exhale through clenched teeth during sustained effort. |
| Battle Cry | A loud, sustained, aggressive yell to intimidate. |
| Exhaustion Sigh | A deep, deflating exhale after strenuous activity. |
| Jumping Effort Sound | A short burst vocalisation when leaping or vaulting. |
| Pushing / Pulling Strain | Rhythmic grunts during sustained push/pull effort. |

### Communication Signals (11)

| Name | Description |
|------|-------------|
| Shush / Shh | A sustained, voiceless fricative demanding silence. |
| Psst | A sharp, whispered fricative to attract attention. |
| Wolf Whistle | A two-tone whistle directed at someone, typically flirtatious. |
| Attention-Getting Cough | A deliberate, light cough to signal one's presence. |
| "Ahem" Throat Clear | A deliberate vocal clearing to get attention. |
| Disapproving "Hmm" | A closed-mouth nasal hum expressing scepticism. |
| Agreeable "Mm-hmm" | A rising then falling nasal hum expressing agreement. |
| Thinking "Uhh" / "Hmm" | A sustained, mid-pitched vocalisation while processing. |
| Call / Holler | A loud, sustained, open-vowel shout to someone at distance. |
| Baby Talk / Cooing At | High-pitched, melodic vocalisations directed at an infant. |
| Encouraging "Come On!" | An energetic, rising-pitch vocalisation urging action. |

### Eating & Drinking (6)

| Name | Description |
|------|-------------|
| Slurping | A loud, sustained sucking sound when drinking. |
| Lip Smacking | Repetitive, moist mouth sounds of enjoyment. |
| Satisfied "Ahh" (Post-drink) | A voiced, open-mouthed sigh of satisfaction after drinking. |
| Crunching Sound | Audible grinding and breaking from eating something crispy. |
| Gulping | Audible swallowing sounds during rapid drinking. |
| Chewing with Mouth Open | Rhythmic, moist, smacking sounds of open-mouth eating. |

### Sleep & Unconscious (5)

| Name | Description |
|------|-------------|
| Snoring | Rhythmic, harsh, vibrating sounds produced during sleep. |
| Sleep Talking / Mumbling | Incoherent, slurred murmuring during sleep. |
| Waking-Up Groan | A low, reluctant vocalisation upon being woken. |
| Sleep Whimper / Cry | Soft, involuntary distressed sounds during a nightmare. |
| Hypnic Jerk Gasp | A sudden, sharp gasp upon the involuntary jolt of falling asleep. |

### Animal Imitations (6)

| Name | Description |
|------|-------------|
| Growling (Animal-like) | A sustained, low, guttural sound imitating a threatening animal. |
| Purring (Cat-like) | A soft, continuous, vibrating hum imitating a contented cat. |
| Hissing (Snake-like) | A sharp, sustained fricative imitating a snake or angry cat. |
| Howling (Wolf-like) | A long, rising, sustained, open vocalisation imitating a wolf. |
| Barking (Dog-like) | Short, sharp, explosive vocalisations imitating a dog. |
| Roaring (Lion-like) | A loud, deep, sustained, open-throated vocalisation. |

### Musical & Rhythmic (7)

| Name | Description |
|------|-------------|
| Humming a Tune | A closed-mouth, melodic, sustained vocal line. |
| Beatboxing | Percussive vocal sounds imitating drums and instruments. |
| Vocal Trill / Vibrato | A rapid oscillation in pitch produced by the voice. |
| Scat Singing | Improvised, nonsensical melodic syllables. |
| Lullaby Humming | A soft, slow, repetitive melodic hum to soothe. |
| Rhythmic Chanting | A repetitive, monotone group vocalisation. |
| Whistling a Tune | A clear, pitched, melodic sound produced by the lips. |

### Nervous & Anxious (7)

| Name | Description |
|------|-------------|
| Nervous Laughter | Erratic, slightly forced laughter to diffuse tension. |
| Teeth Chattering | Rapid, involuntary clicking of teeth from cold or fear. |
| Stammering / Stuttering | Involuntary repetition of syllables from anxiety. |
| Audible Swallow (Nervous) | A loud, dry gulp when anxious. |
| Hyperventilation | Rapid, shallow, panicked breathing out of control. |
| Shaky Voice Crack | A sudden break in pitch from emotional or nervous strain. |
| Nervous Whistling in the Dark | A shaky, slightly off-key whistling to mask fear. |

### Age-Specific (6)

| Name | Description |
|------|-------------|
| Baby Cooing | Soft, melodic vowel-like sounds of a content infant. |
| Toddler Babbling | Repetitive, rhythmic consonant-vowel syllable chains. |
| Elderly Wheeze | A thin, raspy, laboured breathing sound. |
| Child's Tantrum Scream | An extremely loud, high-pitched, sustained scream. |
| Adolescent Voice Crack | An involuntary, sudden shift in pitch mid-sentence. |
| Senescent Vocal Tremor | A wavering, unsteady vocal quality from age-related muscle changes. |

### Bodily Functions (9)

| Name | Description |
|------|-------------|
| Hiccup | A sudden, involuntary diaphragmatic contraction. |
| Sneeze | A sudden, explosive expulsion of air through nose and mouth. |
| Burp / Belch | A guttural release of gas from the stomach. |
| Stomach Growl (Hunger) | A rumbling, gurgling sound from the abdomen. |
| Throat Clearing | A short, sharp vocalisation to clear mucus. |
| Coughing Fit | A series of rapid, involuntary coughs. |
| Dry Heave | A convulsive gagging motion without vomiting. |
| Post-Sneeze Sigh | A relieved exhale immediately following a sneeze. |
| Nose Blowing | A forceful expulsion of air through the nose. |

### Whistling (6)

| Name | Description |
|------|-------------|
| Casual Whistle | A relaxed, aimless melodic whistle. |
| Attention Whistle | A short, sharp whistle to get someone's attention. |
| Impressed Whistle | A rising-falling "whew" whistle expressing admiration. |
| Referee / Signal Whistle (Sharp) | A piercing, sustained whistle for signalling. |
| Creepy Whistling | A slow, deliberate, eerie melodic whistle. |
| Whistling Through Teeth (Cold) | A thin, airy whistle through clenched teeth from cold. |

### Oral / Mouth Sounds (8)

| Name | Description |
|------|-------------|
| Tongue Click (Rhythmic) | A sharp palatal or lateral click made rhythmically. |
| Teeth Sucking | A short, sharp intake of air through the teeth. |
| Lip Pop | A short, percussive sound made by quickly opening the lips. |
| Cheek Pop | A popping sound made by inflating and releasing the cheek. |
| Mouth Click | A quiet, wet clicking sound from the mouth. |
| Jaw Crack | An audible popping or cracking of the jaw joint. |
| Lip Trill / Motorboat | A sustained vibration of the lips, like an engine. |
| Drooling / Slobbering Sound | Wet, uncontrolled mouth sounds. |

### Throat Sounds (7)

| Name | Description |
|------|-------------|
| Throat Clearing (Habitual) | A gentle, repetitive clearing of the throat. |
| Gargling | A sustained, bubbling sound of liquid in the throat. |
| Vocal Fry | A low, creaky, popping vocal register. |
| Glottal Stop | A brief, sharp closure of the vocal folds. |
| Choking / Sputtering | Gasping, gagging sounds from an obstructed airway. |
| Hawking (Phlegm) | A deep, guttural gathering of phlegm in the throat. |
| Raspy Croak | A dry, rough, frog-like vocal quality from strain or illness. |

### Nasal Sounds (5)

| Name | Description |
|------|-------------|
| Sniffling (Cold) | Repeated, rapid nasal inhalations from a runny nose. |
| Snorting | A sudden, forceful exhale through the nose. |
| Nasal Snore (Awake) | A gentle, nasal vibration while breathing. |
| Nose Whistle | A high-pitched, thin sound from air passing through the nose. |
| Congested Breathing | Heavy, laboured nasal breathing through blocked passages. |

### Vocal Tics & Reflexes (6)

| Name | Description |
|------|-------------|
| Involuntary Yelp | A sudden, uncontrolled vocalisation from a physical tic. |
| Repetitive Throat Sound | A habitual, rhythmic vocal clearing or grunting. |
| Echolalia-like Repetition | Automatic, involuntary echoing of a heard word or sound. |
| Sudden Bark (Vocal Tic) | A short, sharp, dog-like vocalisation from a tic disorder. |
| Palilalia-like Repetition | Involuntary repetition of one's own words or sounds. |
| Suppressed Tic Sound | A muffled, strained sound from trying to suppress a vocal tic. |

### Temperature & Environment Reactions (4)

| Name | Description |
|------|-------------|
| Shivering Chatter | Rapid, involuntary teeth clicking and trembling vocalisations from cold. |
| Heat Exhaustion Panting | Slow, heavy, open-mouthed breathing from overheating. |
| Teeth Chattering (Cold) | Rapid, involuntary jaw movement producing a clicking sound. |
| "Brrr" Shiver Sound | A voiced, trembling vocalisation expressing cold. |

### Expressive Interjections (7)

| Name | Description |
|------|-------------|
| Eureka Exclamation | An excited, rising-pitch burst of realisation. |
| Frustrated "Argh" | A guttural, strained vocalisation of blocked goals. |
| Triumphant "Yes!" | A loud, punchy, celebratory exclamation. |
| Defeated "Oh No" | A descending, breathy vocalisation of dread. |
| Sarcastic "Oh Really" | A drawn-out, flat, ironic questioning vocalisation. |
| Impatient "Tch" | A sharp dental click expressing annoyance. |
| Wistful "Ahh" | A soft, descending sigh of nostalgia or longing. |

### NSFW / Intimate (22)

*These entries are excluded from the SFW taxonomy used for minor age groups.*

| Name | Description |
|------|-------------|
| Passionate Kiss | A prolonged, wet, deeply affectionate kissing sound. |
| Deep Sensual Moan | A low, drawn-out, breathy moan of deep physical arousal. |
| Breathy Whisper of Desire | A barely audible, warm exhale carrying longing. |
| Lip Licking Sound | A soft, wet sound of the tongue tracing the lips. |
| Slow Sensual Exhale | A deliberate, drawn-out breath with vocal vibration. |
| Ecstatic Gasp | A sudden, sharp intake of breath at peak pleasure. |
| Pleasured Whimper | A soft, high-pitched, involuntary cry of enjoyment. |
| Heavy Panting (Intimate) | Deep, rhythmic, accelerating breaths during intimacy. |
| Lustful Growl | A low, rumbling vocalisation of raw desire. |
| Tender Post-Climax Sigh | A soft, trembling exhale after peak release. |
| Seductive Purr | A smooth, low, continuous vibrating hum. |
| Erotic Breath Catch | A sudden, involuntary pause in breathing from arousal. |
| Orgasmic Cry | A loud, uncontrolled, rising vocalisation at climax. |
| Hungry Lip Bite Sound | A soft click of teeth pressing into the lower lip. |
| Intimate Wet Kiss | A slow, soft kissing sound with gentle suction. |
| Sensual Moan | A soft, breathy, undulating sound of pleasure. |
| Exaggerated Smooch | A prolonged, loud, wet-sounding kiss. |
| Post-Coital Murmur | A soft, incoherent, satisfied vocalisation. |
| Aroused Breathing | Gradually deepening, becoming ragged breathing. |
| Playful Bedroom Giggle | A soft, intimate, teasing laugh. |
| Dominant Command Tone | A low, authoritative, breathy vocal quality. |
| Submissive Whine | A high-pitched, yielding, pleading vocalisation. |
