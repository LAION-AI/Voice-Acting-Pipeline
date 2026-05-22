# VoiceNet Extended Taxonomy

A comprehensive rubric of **57 voice performance dimensions**, each scored on
**7 ordinal levels (0--6)**, designed for fine-grained speech understanding and
annotation. The taxonomy is organized into **10 attribute groups** covering
rhythm, social dynamics, speaker identity, emotion, physical production,
spectral quality, temporal dynamics, recording conditions, resonance placement,
and speaking style.

Based on the VoiceNet-Ext framework from Schumann et al. (2025).
Paper: <https://arxiv.org/abs/2505.20033>

---

## Table of Contents

- **[RHYTHM & TIMING](#rhythm--timing)** (8 dimensions)
  - [`TEMP`](#temp--tempo) Tempo
  - [`CHNK`](#chnk--chunking) Chunking
  - [`SMTH`](#smth--smoothness) Smoothness
  - [`CLRT`](#clrt--articulation-clarity) Articulation Clarity
  - [`RANG`](#rang--pitch-range) Pitch Range
  - [`EMPH`](#emph--emphasis) Emphasis
  - [`DFLU`](#dflu--disfluency) Disfluency
  - [`STRU`](#stru--structure) Structure
- **[SOCIAL & INTERPERSONAL](#social--interpersonal)** (3 dimensions)
  - [`STNC`](#stnc--stance) Stance
  - [`FOCS`](#focs--focus) Focus
  - [`VULN`](#vuln--vulnerability) Vulnerability
- **[SPEAKER IDENTITY](#speaker-identity)** (3 dimensions)
  - [`GEND`](#gend--perceived-gender) Perceived Gender
  - [`AGEV`](#agev--voice-age) Voice Age
  - [`REGS`](#regs--register) Register
- **[EMOTION & AFFECT](#emotion--affect)** (3 dimensions)
  - [`VALN`](#valn--valence) Valence
  - [`AROU`](#arou--arousal) Arousal
  - [`VOLT`](#volt--volatility) Volatility
- **[PHYSICAL PRODUCTION](#physical-production)** (4 dimensions)
  - [`RESP`](#resp--respiration) Respiration
  - [`TENS`](#tens--tension) Tension
  - [`COGL`](#cogl--cognitive-load) Cognitive Load
  - [`ATCK`](#atck--attack) Attack
- **[SPECTRAL & TIMBRAL](#spectral--timbral)** (7 dimensions)
  - [`BRGT`](#brgt--brightness) Brightness
  - [`ROUG`](#roug--roughness) Roughness
  - [`HARM`](#harm--harmonicity) Harmonicity
  - [`FULL`](#full--fullness) Fullness
  - [`WARM`](#warm--warmth) Warmth
  - [`METL`](#metl--metallic-character) Metallic Character
  - [`ESTH`](#esth--esthetics) Esthetics
- **[TEMPORAL DYNAMICS](#temporal-dynamics)** (4 dimensions)
  - [`VFLX`](#vflx--velocity-flux) Velocity Flux
  - [`DARC`](#darc--dynamic-arc) Dynamic Arc
  - [`ARSH`](#arsh--arousal-shift) Arousal Shift
  - [`VALS`](#vals--valence-shift) Valence Shift
- **[LANGUAGE & RECORDING](#language--recording)** (3 dimensions)
  - [`RCQL`](#rcql--recording-quality) Recording Quality
  - [`BKGN`](#bkgn--background-noise) Background Noise
  - [`EXPL`](#expl--content-appropriateness-3-point-scale) Content Appropriateness (3-point Scale)
- **[RESONANCE PLACEMENT](#resonance-placement)** (7 dimensions)
  - [`R_CHST`](#r_chst--chest-resonance) Chest Resonance
  - [`R_THRT`](#r_thrt--throat-resonance) Throat Resonance
  - [`R_ORAL`](#r_oral--oral-resonance) Oral Resonance
  - [`R_MASK`](#r_mask--mask-resonance) Mask Resonance
  - [`R_NASL`](#r_nasl--nasal-resonance) Nasal Resonance
  - [`R_HEAD`](#r_head--head-resonance) Head Resonance
  - [`R_MIXD`](#r_mixd--mixed-resonance) Mixed Resonance
- **[SPEAKING STYLE](#speaking-style)** (15 dimensions)
  - [`S_CASU`](#s_casu--casual-style) Casual Style
  - [`S_CONV`](#s_conv--conversational-style) Conversational Style
  - [`S_FORM`](#s_form--formal-style) Formal Style
  - [`S_DRAM`](#s_dram--dramatic-style) Dramatic Style
  - [`S_NARR`](#s_narr--narrator-style) Narrator Style
  - [`S_NEWS`](#s_news--newsreader-style) Newsreader Style
  - [`S_TECH`](#s_tech--teacher-didactic-style) Teacher/Didactic Style
  - [`S_AUTH`](#s_auth--authoritative-style) Authoritative Style
  - [`S_PLAY`](#s_play--playful-style) Playful Style
  - [`S_CART`](#s_cart--cartoonish-style) Cartoonish Style
  - [`S_ASMR`](#s_asmr--asmr-style) ASMR Style
  - [`S_WHIS`](#s_whis--whisper-talk-style) Whisper-Talk Style
  - [`S_RANT`](#s_rant--ranting-angry-style) Ranting/Angry Style
  - [`S_STRY`](#s_stry--storytelling-style) Storytelling Style
  - [`S_MONO`](#s_mono--monologue-style) Monologue Style

---

## RHYTHM & TIMING

### `TEMP` -- Tempo

_Measures the mechanical speed of word and syllable production_

| Level | Description |
|:-----:|:------------|
| 0 | The speech is glacially slow, with syllables stretched to their absolute breaking point to create a near-static acoustic environment. Each isolated word floats in heavy silence, lacking standard linguistic momentum. The fundamental velocity of the vocal apparatus is labored and drawn out, making the linguistic content nearly unrecognizable. |
| 1 | The word production is unusually deliberate and labored, characterized by excessive time taken to resolve every single phoneme. The pace feels incredibly heavy and sedate, creating a dragging, low-density acoustic stream with greatly elongated vowel durations. It sounds like someone reading a solemn decree or struggling through deep lethargy. |
| 2 | The vocal delivery is relaxed and unhurried, moving at a pace slightly below the conversational average but maintaining functional momentum. Syllables are articulated comfortably without any sense of compression, rush, or artificial urgency. The overall acoustic speed is comparable to a gentle storyteller or a patient teacher explaining a concept. |
| 3 | The audio features a standard, unremarkable human speech tempo typical of an everyday, efficient conversation. The linguistic density and syllable rate are perfectly balanced, creating a comfortable and natural acoustic flow. There is no perceived stretching or squeezing of the phonemes. |
| 4 | The speech has a brisk, elevated momentum that suggests high engagement and a healthy forward conversational push. Syllables are delivered with slight forward energy without sounding panicked or artificially rushed. The linguistic stream moves swiftly and efficiently, keeping the listener highly engaged. |
| 5 | Words and syllables are noticeably compressed, running into each other to create a fast, high-density stream of acoustic information. The speaker is vibrating with rapid physical energy, forcing the vocal apparatus to move at a highly accelerated velocity. This rapidly trims all excess duration from the vowels, making the listener work hard to track the phonemes. |
| 6 | The tempo reaches the absolute human limit of linguistic speed, resulting in a blistering, hyper-accelerated wall of sound. Phonation is extremely dense, causing individual word boundaries to completely blur into a continuous, high-velocity acoustic vibration. The delivery sounds like a professional auctioneer or an individual in a state of high-speed panic. |

### `CHNK` -- Chunking

_Describes how the speaker groups words into breath units and pause frequency_

| Level | Description |
|:-----:|:------------|
| 0 | The speech is severely fragmented into single syllables or isolated words, broken apart by massive, jarring gaps of silence. It sounds like a malfunctioning vocal mechanism or someone physically struggling to piece together a single thought. There is absolutely no continuous respiratory flow or cohesive acoustic phrasing present. |
| 1 | The audio consists of very short, choppy two-word bursts that create a percussive, stop-and-go rhythmic structure. Every minor thought is separated by distinct, unnecessary pauses that completely interrupt the natural breathing cycle. The overall acoustic effect is highly disjointed and lacks any continuous flow. |
| 2 | Breath units are consistently shorter than natural phrasing, characterized by cautious, highly separated groupings of words. The speaker frequently pauses after small clauses, breaking the sentence into small, easily digestible acoustic blocks. This results in a pedantic, heavily segmented delivery style that feels artificially paced. |
| 3 | The speech is grouped into natural, medium-length sentence-sized clauses that align perfectly with standard human breath cycles. Pauses occur naturally at grammatical boundaries, creating an easy-to-follow, balanced conversational rhythm. The respiratory management and acoustic spacing feel completely unremarkable and organically structured. |
| 4 | The phrasing is noticeably extended, often linking multiple sentences together into a single, continuous, sweeping breath unit. The speaker utilizes high lung capacity to bridge normal stopping points, creating a smooth, long-range acoustic flow. Pauses are infrequent but deliberate, supporting a large-scale structural delivery. |
| 5 | The audio features very long, dense streams of words poured out with only the briefest of snatched inhalations. The linguistic density between pauses is incredibly high, driven by an urgent need to finish large thoughts without resting. This creates a relentless, unbroken wave of speech that stretches the limits of normal breath support. |
| 6 | The speaker produces a massive, continuous wall of words that lasts the entire duration of the audio clip without a single pause or breath. It is a relentless, unrelenting flood of dense linguistic output that offers absolutely no acoustic gaps for rest or processing. The listener is overwhelmed by the sheer, uninterrupted continuity of the vocalization. |

### `SMTH` -- Smoothness

_Gauges the regularity of timing grid and fluidity of transitions between linguistic units_

| Level | Description |
|:-----:|:------------|
| 0 | The rhythmic timing of the speech is completely chaotic and spasmodic, filled with random, jarring glitches and acoustic micro-stops. The vocal delivery moves in disconnected, highly irregular pulses that suggest severe vocal instability or mechanical breakdown. There is an absolute lack of continuous, predictable acoustic flow. |
| 1 | Every syllable is sharply detached from its neighbor, creating a percussive, staccato, machine-gun-like acoustic effect. The timing is extremely rigid, with a total lack of fluid connection or legato transition between sounds. The delivery feels intensely harsh and mathematically segmented into isolated acoustic hits. |
| 2 | The rhythmic grid is uneven and bumpy, characterized by frequent micro-hesitations and clumsy acoustic transitions. The speech lacks a steady, predictable pulse, sounding like an unprepared or organically flawed read. The acoustic envelope stumbles awkwardly from word to word without natural grace. |
| 3 | The audio features the standard, slightly flexible rhythm of natural human speech, complete with minor, organic timing imperfections. It flows smoothly but retains the natural breathing and slight elasticity characteristic of spontaneous thought processes. The timing is comfortable, balanced, and conventionally fluid. |
| 4 | The speech timing is noticeably consistent, practiced, and regular, locking into a predictable and pleasing rhythmic pulse. It sounds like the highly controlled delivery of a professional narrator who smoothly connects phrases with practiced ease. The transitions between words are clean, measured, and highly stable. |
| 5 | The vocal delivery flows like silk, featuring a beautiful, continuous, wave-like legato quality in the acoustic signal. Every word melts seamlessly into the next with exceptional grace and perfectly modulated timing transitions. The rhythmic structure is highly pleasing, fluid, and entirely uninterrupted by micro-stops. |
| 6 | The rhythm is mathematically perfect and entirely metronomic, completely devoid of natural human timing deviations. The delivery sounds highly mechanical, like a high-end digital synthesizer or an AI voice strictly locked to a digital quantization grid. It possesses an inhuman, flawless acoustic regularity that feels completely sterile. |

### `CLRT` -- Articulation Clarity

_Measures the precision with which consonants and vowels are produced_

| Level | Description |
|:-----:|:------------|
| 0 | The speech is an indecipherable, blurry hum where distinct consonants and vowel boundaries are completely invisible in the acoustic signal. It sounds like a heavily muffled voice heard through a thick wall, lacking any high-frequency transient bites. The listener cannot extract any clear phonemes from this continuous, mushy sound wave. |
| 1 | Consonants are severely swallowed and lack all precision, resulting in a heavily mumbled, fuzzy acoustic texture. The word boundaries are entirely soft, sounding like the speaker is heavily sedated or speaking with a highly restricted oral cavity. The severe lack of articulatory effort makes acoustic transcription incredibly difficult. |
| 2 | The articulation is consistently soft and overly relaxed, lacking the crisp transient edges typical of active communication. While most words are intelligible, the overall delivery is mushy, similar to tired, domestic mumble-core speech. The vocal apparatus is applying minimal physical effort to shape the sound waves. |
| 3 | The audio exhibits a neutral, standard level of articulation perfectly suited for everyday, clear acoustic communication. Words are easily separable with well-defined vowel shapes and appropriately struck consonants. The speaker puts in just enough mechanical effort to be naturally understood without sounding overly precise. |
| 4 | Every syllable is pronounced with crisp, distinct clarity, ensuring highly intelligible and clean phoneme boundaries. The consonants have a clear, precise bite, typical of a high-quality, professional narrator focused on maximum audio legibility. The delivery is sharp and clean without crossing into artificial exaggeration. |
| 5 | The articulation is incredibly precise and deliberate, with consonants carved out using high physical energy and perfectly shaped vowels. It has the sharp, commanding acoustic clarity of a classically trained stage actor or an elite broadcaster. Every acoustic boundary between sounds is meticulously managed and highly defined. |
| 6 | The phonemes are hyper-articulated to an extreme, unnatural degree, completely over-enunciating every single consonant and vowel. The delivery sounds like a rigid caricature of a formal speaker, with exaggerated plosive pops and sharply cut word endings. This level of mechanical precision is highly distracting and completely abandons organic human blending. |

### `RANG` -- Pitch Range

_Describes the vertical movement of fundamental frequency across the segment_

| Level | Description |
|:-----:|:------------|
| 0 | The fundamental frequency remains on a perfectly flat, horizontal line with absolutely zero micro-tonal deviation. The voice is completely devoid of melodic movement, sounding like a singular synthetic beep or a completely lifeless drone. It represents the absolute mathematical minimum of pitch variance in an acoustic signal. |
| 1 | The pitch exhibits only microscopic, severely suppressed movements, hovering heavily around a single monotonous baseline note. The vocal melody feels clinical, drained of energy, and aggressively flattened out. It is a highly restricted acoustic profile that completely lacks natural frequency inflection. |
| 2 | The intonation moves within a very narrow, highly controlled, and restrained fundamental frequency window. Pitch rises and falls are present but intentionally minimized, typical of a speaker remaining intensely calm or delivering dry technical instructions. The acoustic melody is functional but kept strictly clamped down. |
| 3 | The audio features the moderate, standard intonation range of a typical human conversation, complete with organic pitch rises and falls. The fundamental frequency moves smoothly across a standard melodic window, emphasizing thoughts naturally. It is a balanced, conventionally expressive acoustic contour. |
| 4 | The pitch moves expressively across a wide range of notes, actively utilizing significant fundamental frequency highs and lows to project personality. The voice sounds lively, colorful, and deeply engaged, creating a dynamic and engaging melodic map. The sweeping pitch jumps clearly highlight the emotional intent of the speaker. |
| 5 | The delivery is highly melodic, characterized by sweeping, dramatic pitch jumps and beautifully pronounced sing-song contours. The voice utilizes nearly its full musical range, bouncing rapidly between high acoustic peaks and deep valleys. It possesses the vibrant, animated theatricality of a professional voice actor or storyteller. |
| 6 | The pitch swings wildly and hyperbolically between the absolute extreme highs and lows of the speaker's vocal register. It is an aggressively stylized, operatic delivery that sounds larger than life and entirely theatrical. This massive variance in fundamental frequency creates a highly exaggerated, nearly musical vocal performance. |

### `EMPH` -- Emphasis

_Refers to how the speaker creates informational hierarchy by stressing specific words or syllables_

| Level | Description |
|:-----:|:------------|
| 0 | Every single word is delivered with identical acoustic weight, perfectly flat volume, and exactly uniform duration. There is absolutely no informational hierarchy, making the speech sound like an unanchored, monotonous list of syllables. The complete absence of dynamic stress renders the sentence entirely flat. |
| 1 | The emphasis is incredibly weak, with only the faintest, nearly undetectable shifts in volume or pitch to mark key words. The hierarchy of information is deeply buried, making it difficult to discern the core focus of the sentence through the audio alone. The dynamic delivery feels aimless and severely under-stressed. |
| 2 | Important words are only lightly and subtly marked by minor increases in vocal energy or slight pitch bumps. This gentle emphasis provides a basic, low-stakes structural map for the listener to follow the logic. It is a soft, polite acoustic hierarchy without any aggressive highlighting. |
| 3 | The speech features a clear, natural emphasis pattern where key informational anchors are distinctly and comfortably stressed. The listener can immediately identify the core message through organic, well-placed spikes in acoustic volume and pitch. The dynamic weighting is perfectly balanced for everyday comprehension. |
| 4 | The speaker heavily and effectively marks important words, creating a strong, highly structured acoustic hierarchy. Key concepts are underlined with distinct vocal force, elongated syllable duration, and clear pitch spikes. The delivery sounds like a highly capable teacher giving crucial, high-stakes instructions. |
| 5 | Content words are punched with aggressive, intense energy that aggressively forces the listener's attention to specific points. The emphasis pattern is highly percussive, utilizing sharp, forceful bursts of volume to highlight critical concepts. It is a commanding, highly active dynamic structure that bites through the audio mix. |
| 6 | The emphasis is applied with violent, explosive intensity, creating massive acoustic disparities between stressed and unstressed words. Highlighted syllables erupt with jarring volume and prolonged duration, resembling a high-pressure sales pitch or a furious argument. The dynamic contrast within the sentence is stretched to its absolute extreme limit. |

### `DFLU` -- Disfluency

_Tracks the presence of linguistic noise including fillers, false starts, and self-corrections_

| Level | Description |
|:-----:|:------------|
| 0 | The audio clip is a pristine, flawless stream of speech with absolutely zero fillers, hesitations, or vocal restarts. It sounds like a perfectly edited studio recording or a masterfully rehearsed script reading. The linguistic signal is completely devoid of any cognitive or mechanical interference. |
| 1 | The delivery is highly polished and professional, containing perhaps only a single, negligible micro-hesitation in a long stream of speech. The linguistic output is exceptionally clean, indicating a highly prepared and focused speaker. There is almost no acoustic junk interrupting the primary message. |
| 2 | The speech is highly fluent but contains an occasional, organic filler sound that naturally bridges a thought without causing distraction. It represents the smooth, intelligent flow of a highly articulate spontaneous conversation. The tiny acoustic imperfections feel deeply human but do not degrade the legibility. |
| 3 | The audio contains a standard, entirely natural amount of non-lexical fillers like 'um' or 'uh', along with minor self-corrections. The delivery feels highly authentic and spontaneous while remaining perfectly easy to comprehend. It represents the exact baseline of unscripted, everyday human processing noise. |
| 4 | The speech is noticeably hesitant, with vocal fillers and noticeable processing pauses occurring in nearly every acoustic phrase. The speaker frequently interrupts their own flow to search for words, causing a distinctly staggered and bumpy acoustic output. The high density of linguistic noise significantly slows down the transfer of information. |
| 5 | The delivery is overwhelmingly messy, dominated by frequent, distracting restarts, stammers, and heavy usage of vocal fillers. The core message is buried under a thick layer of cognitive interference and broken audio sentences. The listener must actively filter out the high volume of acoustic junk to understand the speaker. |
| 6 | The speech is entirely shattered into a chaotic, virtually incoherent web of stammers, abandoned words, and continuous filler sounds. The speaker completely fails to maintain a stable linguistic thread, making it physically difficult to extract any meaning from the audio wave. It is a pure acoustic manifestation of profound cognitive or linguistic breakdown. |

### `STRU` -- Structure

_Looks at macro-level organization of ideas within a clip_

| Level | Description |
|:-----:|:------------|
| 0 | The speech lacks any logical connection between words, phrases, or ideas, presenting as an entirely scattered acoustic word salad. It consists of completely unrelated, jarring outbursts that possess zero macro-level organization. The listener cannot map any linear progression or coherent thought process in the audio. |
| 1 | The delivery is incredibly jumpy, with the speaker constantly abandoning topics mid-sentence and pivoting without any acoustic transitions. It is an erratic, highly disorganized stream of audio that totally fails to connect point A to point B. The cognitive map of the speech is fractured and nearly impossible to track. |
| 2 | The audio features a very loose, rambling structure where a main idea is heavily buried under layers of irrelevant tangents and side-thoughts. The delivery sounds like an unorganized, wandering stream-of-consciousness that struggles to reach an acoustic destination. The flow is meandering, lacking clear boundaries or logical guideposts. |
| 3 | The speech maintains a basic, coherent logical flow that successfully moves from one idea to the next without losing the listener. The speaker utilizes standard conversational markers to link thoughts together in a naturally progressing chain. It is a functional, easily summarizable delivery of audio information. |
| 4 | The delivery is distinctly linear and well-organized, marching along a highly predictable and clear path from introduction to conclusion. The speaker expertly avoids tangents, making the logical explanation effortlessly easy to follow. It has the distinct acoustic markers of a well-planned, sequential thought process. |
| 5 | The speech is exceptionally organized, utilizing very clear verbal transitions, strong signposting, and disciplined returns to the main topic. It sounds like a highly prepared, professional address that effortlessly manages complex acoustic information. The macro-level structure is robust, elegant, and highly effective. |
| 6 | The delivery exhibits perfect, high-level architectural clarity, functioning like a flawlessly structured masterclass or formal oral essay. The speaker sets up clear acoustic introductions, deliberate developmental steps, and highly defined summary closures. The logical map of the audio is an absolute masterpiece of communicative engineering. |

---

## SOCIAL & INTERPERSONAL

### `STNC` -- Stance

_Describes the speaker's social posture and behavior in the conversation space_

| Level | Description |
|:-----:|:------------|
| 0 | The voice sounds incredibly tiny, submissive, and heavily apologetic, actively attempting to take up as little acoustic space as possible. It lacks all projection and confidence, signaling a complete surrender of social power to the listener. The vocal delivery feels like the speaker is trying to make themselves entirely invisible. |
| 1 | The audio features a very reserved, cautious, and quiet delivery that respectfully waits for long silences before initiating sound. The phrase endings are deeply softened and drop in volume, indicating a distinct hesitation to claim the conversational floor. It is a highly guarded, socially deferential acoustic posture. |
| 2 | The speaker adopts a highly cooperative, friendly, and open tone that actively invites back-and-forth exchange. The acoustic markers suggest a balanced, polite awareness of a conversational partner, utilizing warm intonation to share the space. It is an inviting, egalitarian vocal stance. |
| 3 | The delivery utilizes a neutral, standard interactional style focused entirely on relaying information rather than establishing social hierarchy. The speaker confidently occupies their space without either imposing on or deferring to the listener. It is a balanced, highly functional interpersonal acoustic baseline. |
| 4 | The voice is highly assertive, confident, and direct, utilizing strong projection that signals an absolute expectation to be heard. The speaker firmly claims the acoustic floor and drives the interaction forward without hesitation. It is a proactive, secure, and commanding conversational posture. |
| 5 | The speech carries a heavy, authoritative weight, sounding exactly like a person in absolute command of the acoustic environment. The phrasing is intensely final, utilizing loud, downward-inflected endings that allow virtually no room for doubt or interruption. It is a deeply dominant, uncompromising display of vocal power. |
| 6 | The speaker completely dominates and monopolizes the acoustic space, delivering a relentless, aggressive lecture that talks entirely at the listener. There is absolutely zero room left for a response, effectively crushing a dialogue into a forcefully imposed wall of sound. It is an overwhelming, highly aggressive display of pure interpersonal dominance. |

### `FOCS` -- Focus

_Measures the direction of the speaker's interpersonal attention_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic energy is entirely internalized and heavily dissociated, sounding exactly like a person sleep-talking or mumbling strictly to themselves. The speaker demonstrates zero awareness of an external audience, 'dropping' their words aimlessly rather than projecting them outward. It is a profoundly isolated, completely inward-facing vocal geometry. |
| 1 | The delivery is completely emotionally detached and highly impersonal, resembling someone coldly reading a script into an empty void. While the voice is technically audible, the 'soul' of the speaker is entirely disconnected from the listener, producing a chillingly distant acoustic profile. There is absolutely zero interpersonal 'reach' in the audio wave. |
| 2 | The speaker's attention is clearly split, resulting in a distracted, slightly 'hollow' acoustic delivery where their mind is visibly wandering. They are physically speaking to someone, but the vocal projection lacks true interpersonal grounding, creating a slightly unfocused and floaty audio presence. The communicative intent is weak and continually drifting. |
| 3 | The audio captures a standard, polite interactional focus where the speaker is casually aware of the listener without applying intense psychological pressure. The vocal energy is appropriately directed outward, maintaining a comfortable, socially standard interpersonal connection. The acoustic projection is functional, balanced, and appropriately present. |
| 4 | The speaker employs clear, highly engaged 'Direct Address,' actively ensuring their acoustic energy firmly lands on the listener. The voice possesses a strong forward projection that metaphorically 'looks' right at the audience, demanding their attention and cognitive participation. The interpersonal connection is tight, highly active, and deeply communicative. |
| 5 | The vocal delivery utilizes an intense, highly precise laser-focus, locking onto the listener with overwhelming interpersonal intent. Every single syllable is aggressively 'thrown' at the audience to persuade, threaten, or seduce, creating a deeply penetrating acoustic presence. The psychological weight of the audio is completely inescapable. |
| 6 | The speaker's focus becomes deeply obsessive and almost uncomfortably hypnotic, treating the listener as the sole existing object in their entire reality. The acoustic delivery is overwhelmingly intimate and trance-like, completely invading the listener's psychological space with an unblinking, hyper-focused vocal intensity. It is an extreme, profoundly overwhelming manifestation of directed presence. |

### `VULN` -- Vulnerability

_Measures the 'openness' or 'permeability' of the speaker_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic profile is completely impenetrable and heavily armored, projecting an absolute, unyielding emotional defense wall. The tone is hard, shiny, and highly guarded, mimicking the untouchable vocal posture of a mob boss or a drill sergeant where absolutely nothing emotional 'leaks out.' The psychological barrier is actively hostile to any connection. |
| 1 | The delivery is highly guarded, deeply cautious, and strictly professional, meticulously maintaining a hard interpersonal boundary. The speaker tightly controls their acoustic footprint, sharing only the absolute minimum amount of personal resonance required to communicate. The psychological 'mask' is firmly and deliberately strapped on. |
| 2 | The voice utilizes a highly composed, standard adult emotional mask that keeps deep personal vulnerability politely and safely hidden. The tone is perfectly functional and socially appropriate, providing a smooth, well-maintained acoustic surface that prevents the listener from peering too deeply inside. The emotional core remains safely guarded. |
| 3 | The acoustic delivery is open, friendly, and highly accessible, featuring a distinct 'soft' edge to the vocal resonance that invites connection. The speaker sounds entirely willing to be emotionally affected by the interaction, lowering their vocal defenses to create a warm, unguarded acoustic space. The psychological boundary is comfortably lowered. |
| 4 | The speaker's voice is highly empathetic and distinctly permeable, sounding as if they are actively absorbing and reacting to the emotional energy of the listener. The acoustic profile is deeply responsive, stripped of standard defensive armor, and highly sensitive to the interpersonal environment. The psychological state is deeply engaged and highly unprotected. |
| 5 | The speaker's emotional mask is completely removed, resulting in an exposed, deeply defenseless acoustic profile often heard during profound confessions or moments of genuine awe. The voice is entirely stripped of all social protection, allowing raw, unfiltered emotional resonance to completely dictate the pitch and volume. The listener is granted absolute, unhindered access to the speaker's core state. |
| 6 | The acoustic signal is profoundly raw, completely naked, and overwhelmingly permeable, completely lacking any psychological filter or protective barrier. The speaker's sensitivity is dialed so high that the mere act of speaking seems to cause physical emotional pain, resulting in a deeply trembling, completely shattered vocal output. It is the absolute extreme of terrifying, unshielded emotional exposure. |

---

## SPEAKER IDENTITY

### `GEND` -- Perceived Gender

_Measures the perceived gender of the voice based on psychoacoustic interaction of pitch and resonance_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic profile features an extremely high fundamental frequency combined with the bright, small-cavity resonance characteristic of a hyper-feminized vocal tract. The voice carries a very light physical weight and delicate harmonic texture, entirely devoid of low-end chest frequencies. It sounds like a high soprano, projecting a bright, purely head-dominant acoustic signature. |
| 1 | The audio presents a clearly feminine vocal signature with standard female fundamental pitch ranges and proportionally scaled formant frequencies. The tonal balance is strongly centered in the head and oral cavities, lacking heavy lower-harmonic rumble. The acoustic weight and spectral footprint are unmistakably characteristic of a typical adult woman. |
| 2 | The voice is predominantly feminine but exhibits subtle neutral or slightly darkened resonant qualities in its acoustic profile. It operates at a slightly lower fundamental frequency, resembling a female alto or a speaker with a marginally larger perceived vocal tract. The tonal weight is slightly thicker than standard feminine speech, adding a gentle acoustic warmth. |
| 3 | The acoustic signature sits perfectly centered on the androgynous spectrum, making it completely ambiguous and difficult to categorize by traditional gender markers. The audio lacks both the sharp, high-frequency brightness of typical feminine speech and the heavy, thick chest rumble of masculine speech. It is a highly balanced, perfectly neutral vocal tract manifestation. |
| 4 | The vocal profile is predominantly masculine but features a slightly elevated fundamental pitch and a somewhat smaller resonant cavity than a typical male. The harmonic weight is relatively light, producing a bright, tenor-like acoustic quality that lacks heavy sub-harmonic rumble. It sounds like a youthful male or a man speaking with a noticeably lifted, head-forward resonance. |
| 5 | The audio delivers a standard masculine vocal signature characterized by a naturally low fundamental frequency and distinct, robust chest resonance. The spectral balance features strong low-mid frequency support, creating a thick, authoritative acoustic weight. It is the unmistakable, highly resonant sound profile of a typical adult man. |
| 6 | The vocal tract produces a hyper-masculinized acoustic profile, heavily dominated by an extremely low fundamental frequency and massive, booming chest resonance. The vocal folds sound incredibly thick, generating an overpowering, low-frequency rumble that fills the entire acoustic space. It mimics the larger-than-life, deeply resonant depth of a cinematic hero or a classic sub-bass radio announcer. |

### `AGEV` -- Voice Age

_Tracks the perceived biological age of the speaker based on vocal maturity, elasticity, and wear_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic signal is dominated by the extremely small vocal tract and unstable, high-pitched fundamental frequency of an infant or toddler. The audio entirely lacks complex linguistic structure, consisting primarily of pure, high-frequency biological resonance and primitive vocalizations. The physical elasticity of the vocal folds produces bright, highly piercing neonatal sound waves. |
| 1 | The voice carries the clear, bright, and highly energetic acoustic signature of a pre-pubescent child. The resonant cavity is distinctly small, yet the articulation shows developing structure, producing a lightweight, highly elastic harmonic profile. The audio entirely lacks the grounded, heavy chest frequencies characteristic of adult physical maturity. |
| 2 | The vocal profile exhibits the transitional acoustics of an adolescent, blending remnants of childhood brightness with newly developing lower-frequency depth. The fundamental pitch may occasionally destabilize or 'crack,' revealing the rapid biological scaling of the vocal tract. The overall resonance is thickening, but has not yet settled into full adult acoustic weight. |
| 3 | The audio showcases a voice at peak biological vigor, producing perfectly elastic, incredibly clean harmonic definition across all frequencies. There is absolutely no acoustic evidence of vocal fatigue, muscular wear, or aging-related texture in the signal. The fundamental pitch is rock-solid, and the resonance is highly resilient, fresh, and perfectly supported. |
| 4 | The acoustic signature reflects a fully developed, mature adult, characterized by a highly settled, grounded, and authoritative resonant weight. The vocal folds exhibit a very slight, natural thickening, which adds a warm, robust low-mid frequency presence to the audio. The overall physical support is incredibly stable, projecting a commanding and complete spectral profile. |
| 5 | The voice begins to display the distinct acoustic markers of middle-to-late aging, including a slight loss of respiratory support and a narrowing of the functional pitch range. A noticeable 'grainy' or weathered texture permeates the audio, resulting from the natural, slow desiccation of the vocal folds over decades. The physical momentum of the speech is perceptibly heavier and slightly less elastic. |
| 6 | The acoustic profile is heavily defined by extreme biological senescence, characterized by a pronounced, uncontrollable physiological tremor or 'wobble' in the fundamental frequency. The resonance sounds highly ossified, thin, and reedy, with significant breathiness indicating a massive loss of muscular vocal support. The audio captures the labored, fragile phonation typical of very advanced age. |

### `REGS` -- Register

_Classifies the dominant musical tessitura or pitch-class of the voice using classical domain terms_

| Level | Description |
|:-----:|:------------|
| 0 | The fundamental frequency resides at the extreme low end of human phonation, producing a heavy, cavernous Basso Profondo acoustic profile. The audio is overwhelmingly dominated by thick, low-frequency chest vibrations that create a dark, rumbling physical weight. The vocal baseline sits so low it almost borders on continuous vocal fry. |
| 1 | The vocal gravity sits comfortably in the standard Baritone range, blending a rich, grounded low-end depth with highly functional mid-range clarity. The acoustic profile is deeply warm and stabilizing, producing a robust harmonic footprint that easily cuts through a mix. It is a highly common, pleasing male musical register. |
| 2 | The pitch center is elevated into the Tenor register, characterized by a light, agile, and noticeably 'bright' fundamental frequency. The audio entirely lacks heavy sub-harmonic chest rumble, instead relying on a highly projected, ringing presence in the upper-mid frequencies. The voice possesses a buoyant, physically lifted acoustic quality. |
| 3 | The acoustic signature occupies the ambiguous 'bridge' register, manifesting either as a uniquely deep, husky female Contralto or a high, head-voiced male Countertenor. The fundamental pitch hovers in a highly distinct transitional zone, blending upper-chest warmth with lower-head resonance. The resulting audio footprint is rich, complex, and deeply textured. |
| 4 | The vocal center of gravity naturally rests in the standard female Mezzo-Soprano range, offering a perfectly balanced, versatile mid-frequency tone. The audio contains more physical 'body' and lower-harmonic warmth than a high soprano, yet remains cleanly situated in the upper-mid acoustic spectrum. The resonance is highly adaptable and functionally clear. |
| 5 | The fundamental frequency is brightly elevated into the Soprano register, generating a soaring, highly brilliant acoustic profile. The audio is overwhelmingly centered in the upper head resonance, entirely stripping away lower-frequency chest weight to produce a light, airy sound. The voice shines with high-frequency harmonic presence. |
| 6 | The pitch sits at the absolute upper limit of biological phonation, producing an impossibly thin, piercing Coloratura or 'whistle' register. The acoustic signal is intensely bright and completely devoid of lower harmonics, mimicking the high-frequency chirp of a bird. The vocal baseline operates entirely outside the boundaries of normal conversational norms. |

---

## EMOTION & AFFECT

### `VALN` -- Valence

_Measures the global emotional polarity of the performance_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic signal is completely saturated with pure, unadulterated emotional suffering, broadcasting extreme hostility, deep agony, or pathological bitterness. Every micro-inflection is heavily weighed down by absolute aversion, completely stripping the audio of any warmth or comfort. It is the raw, highly disturbing sound of a speaker trapped in a state of total negativity. |
| 1 | The vocal delivery is clearly anchored in an unhappy, frustrated, or heavily saddened emotional state. A distinct negative 'weight' pulls down the pitch contours, coloring every single phoneme with an acoustic sense of gloom and deep dissatisfaction. The audio profile is dark, heavily burdened, and distinctly unwelcoming. |
| 2 | The speech exhibits a subtle, negative-leaning emotional bias, marked by faint acoustic traces of boredom, mild disappointment, or simmering unease. While functioning near the baseline, the vocal tone lacks any upward energy, maintaining a slightly heavy, 'scowling' micro-expression. The psychological state feels slightly closed off and subtly uncooperative. |
| 3 | The audio presents a perfectly balanced, emotionally neutral baseline completely devoid of any detectable positive or negative affective bias. The speaker delivers the linguistic content with a highly objective, purely informative acoustic profile that lacks any personal emotional involvement. It is a sterile, functionally flat emotional landscape. |
| 4 | The voice demonstrates a positive-leaning, highly polite, and mildly interested emotional state. The overall acoustic delivery sounds perceptibly 'lighter' and more open than the standard baseline, often featuring subtle upward pitch inflections that signal approachability. It is a warm, gently inviting, and entirely unthreatening vocal posture. |
| 5 | The acoustic profile is infused with a clear, undeniable 'smile,' radiating a cheerful, happy, and highly warm emotional state. The fundamental frequency dances with positive, upward-leaning energy, creating a highly engaging and deeply inviting auditory atmosphere. The positivity is physically audible in the brightened shape of the vowels. |
| 6 | The vocal delivery explodes with pure, euphoric, and deeply ecstatic emotional joy. The voice sounds as if it is completely overflowing with manic pleasure, utilizing incredibly bright resonance and soaring, uninhibited pitch peaks. It is a wildly unconstrained, highly contagious acoustic manifestation of absolute happiness. |

### `AROU` -- Arousal

_Measures the physiological 'engine speed' of the speaker_

| Level | Description |
|:-----:|:------------|
| 0 | The physiological engine of the speaker is operating at an absolute minimum, producing a severely lethargic, nearly comatose acoustic signal. The voice physically drags, lacking the fundamental breath energy required to sustain clear phonation, resulting in heavily faded, trailing audio. It is the sound of total physiological exhaustion or deep drowsiness. |
| 1 | The vocal energy is extremely low, characterized by a highly sedate, physically relaxed, and carefully muted acoustic delivery. Everything in the biological apparatus is slowed down, mirroring a deep meditative state or an incredibly quiet, hushed 'library' environment. The overall sound wave is deeply subdued and gentle. |
| 2 | The speaker is entirely awake but is consciously holding back their physiological energy, maintaining a highly reserved and strictly controlled vocal output. The intensity of the acoustic signal is deliberately suppressed, typical of a cautious, highly formal, or hyper-professional setting. The physical drive of the voice is kept on a very tight leash. |
| 3 | The audio captures the standard, baseline physiological energy level of a fully alert human engaged in a normal conversation. The speaker sounds entirely present, utilizing a healthy, balanced physical force that perfectly matches the conversational context. The autonomic nervous system is active, stable, and completely unexceptional. |
| 4 | The physiological activation is clearly elevated, driving the vocal apparatus with an energized, slightly urgent physical momentum. A perceptible 'hum' of high alertness buzzes through the audio, making the speech sound highly insistent, excited, or mildly anxious. The acoustic transients hit with noticeable, fast-paced physical force. |
| 5 | The autonomic nervous system is in a state of high activation, pushing the vocal folds with intense physical pressure and heavily increased acoustic volume. The energy borders on shouting, demonstrating an aggressive, high-stakes physical drive that begins to actively strain the biological limits of the throat. The audio is saturated with highly mobilized, forceful vocal power. |
| 6 | The vocal apparatus is pushed into maximum physiological mobilization, resulting in a hysterical, violently extreme acoustic output. The energy is so overwhelmingly highâ€”resembling a full panic attack or a blood-curdling screamâ€”that the biological system sounds on the verge of total failure. The sound wave is a highly alarming, explosive manifestation of absolute fight-or-flight survival. |

### `VOLT` -- Volatility

_Measures how much the emotional state shifts or vibrates within the clip_

| Level | Description |
|:-----:|:------------|
| 0 | The emotional state is absolutely static and heavily frozen, presenting as an unyielding block of psychological ice. The voice exhibits zero micro-shimmer or affective reaction to the actual words being spoken, resulting in a disturbingly flat and highly rigid acoustic profile. The emotional trajectory is a completely dead, unmoving flatline. |
| 1 | The vocal delivery is highly steady and exceptionally consistent, locking into a single, unvarying mood with intense professional discipline. The affective tone is deeply grounded and completely immune to external or internal disruption, producing a highly reliable and unshakeable acoustic signal. The psychological baseline is rock solid. |
| 2 | The audio displays a standard, highly stable level of human emotional consistency, featuring only minor, entirely logical affective shifts based directly on the text. The overall psychological state remains deeply firm, completely avoiding any erratic or unpredictable acoustic behavior. The emotional baseline is perfectly anchored. |
| 3 | The speech features the natural, organic 'breathing' of standard human emotion, allowing the affective state to dynamically flex and respond to the conversation in real-time. The emotional shifts feel highly spontaneous and healthy, creating a gently undulating, realistic acoustic landscape. The psychological volatility is perfectly balanced. |
| 4 | Minor, uncontrollable 'cracks' begin to appear in the speaker's emotional facade, introducing a distinct psychological 'flickering' into the acoustic signal. The speaker is actively attempting to remain steady, but underlying surges of emotion cause rapid micro-shifts in pitch and volume that leak into the audio. The emotional stability is noticeably straining under pressure. |
| 5 | The vocal delivery is highly unstable, characterized by jarring, unpredictable shifts in affective tone that drastically alter the acoustic profile. The speaker sounds emotionally untethered, sounding as if they could violently snap from laughing to crying in a fraction of a second. The psychological vector is highly erratic and physically difficult to predict. |
| 6 | The emotional state is completely chaotic and deeply shattered, featuring violent, rapid-fire affective switching that fundamentally destabilizes the entire acoustic signal. The speaker cycles through manic, extreme emotional extremes every few seconds, entirely destroying any sense of a consistent psychological baseline. The audio is a terrifying, highly volatile map of total emotional collapse. |

---

## PHYSICAL PRODUCTION

### `RESP` -- Respiration

_Measures the audibility and laboriousness of the speaker's breathing_

| Level | Description |
|:-----:|:------------|
| 0 | The speaker's respiratory cycle is completely invisible and inaudible within the acoustic signal, allowing for impossibly long, seamless linguistic phrases. They possess seemingly infinite breath control, characteristic of a highly trained singer or an elite public speaker operating with absolute physiological efficiency. The 'wind' of the lungs never once interrupts the 'music' of the voice. |
| 1 | Breathing is strictly controlled and occurs only as highly efficient, barely perceptible micro-inhalations taken at long, calculated intervals. The respiratory function operates entirely in the background of the audio mix, never pulling focus from the acoustic clarity of the words. It is a highly disciplined, remarkably quiet respiratory profile. |
| 2 | The audio features standard, completely natural conversational breathing where inhalations occur at logical grammatical pauses. The sound of the breath is mildly audible but entirely unobtrusive, perfectly blending into the organic rhythm of the speech. The respiratory engine sounds healthy, unforced, and entirely typical. |
| 3 | The speaker's breathing is noticeably elevated, with inhalations occurring more frequently and audibly than strictly necessary for basic phonation. The slightly heavier air intake suggests a subtle increase in physical engagement, mild psychological stress, or active physical movement. The sound of the respiratory cycle begins to actively assert itself into the audio stream. |
| 4 | The respiratory cycle is highly labored, characterized by deep, clearly audible gasps and heavy heaving between sentences. The speaker is rapidly running out of air, forcing them to physically truncate their phrase lengths to accommodate their desperate need for oxygen. The acoustic signal is frequently and aggressively interrupted by the struggle to breathe. |
| 5 | The audio is dominated by rapid, highly shallow gasping that violently and unpredictably chops the linguistic stream into fragmented pieces. The speaker sounds heavily winded, as if they have just finished a full-speed sprint or are experiencing severe, acute physical distress. The violent rush of air is almost louder than the actual phonation. |
| 6 | The physical ability to speak is almost entirely destroyed by catastrophic respiratory failure, resulting in an agonal, suffocating acoustic profile. The audio captures the terrifying sounds of choking, hyperventilation, or a desperate fight for survival where air cannot be processed into words. It is an extreme, highly distressing biological breakdown of the breathing mechanism. |

### `TENS` -- Tension

_Measures the perceived muscular 'tightness' in the throat, jaw, and neck_

| Level | Description |
|:-----:|:------------|
| 0 | The vocal apparatus exhibits absolutely zero muscular resistance, producing an impossibly 'floppy,' wide-open, and entirely tension-free acoustic signal. The voice sounds profoundly relaxed, mirroring the deep physiological release of a post-massage state or heavy sedation. The resulting sound wave is incredibly soft, sprawling, and physically uninhibited. |
| 1 | The throat and jaw muscles are highly loose and comfortable, producing a deeply warm, physically effortless vocal tone. There is absolutely no noticeable constriction in the biological pipeline, allowing the audio to resonate with a highly pleasing, 'open' acoustic geometry. The speech is physically incredibly easy to produce. |
| 2 | The voice utilizes standard, neutral conversational muscle tone, applying just enough physical firmness to clearly articulate without squeezing the vocal tract. There is no distracting or uncomfortable physical tension audible in the acoustic signal. The throat remains appropriately relaxed but functionally active. |
| 3 | A faint acoustic 'edge' or mild muscular squeeze becomes audible, indicating a subtle tightening of the neck or jaw muscles. This mild tension profile often suggests a speaker who is deeply concentrating, slightly annoyed, or operating under minor psychological stress. The voice loses a fraction of its natural warmth due to this slight physical constriction. |
| 4 | The vocal tone sounds noticeably pressed and restricted, as if the audio is being physically pushed through a narrowed, tightly clenched biological pipe. The speaker is clearly carrying heavy stress in their jaw and neck, resulting in a firm, mildly metallic, and highly constrained acoustic signal. The lack of open resonance is physically palpable. |
| 5 | The muscles surrounding the vocal folds are heavily strained, producing a hard, grinding, and intensely uncomfortable acoustic texture. The sheer physical effort required to force sound through the tightly constricted throat is blatantly audible, robbing the voice of all organic softness. The audio feels highly stressful and physically painful to listen to. |
| 6 | The somatic tension is so extreme that the vocal musculature is entirely rigid and locked, actively crushing the vocal folds and causing the voice to crack or break. The delivery sounds brutally strangled, heavily distorted by the immense external pressure applied by the neck and jaw. It is the acoustic manifestation of a biological system paralyzed by high-stakes stress. |

### `COGL` -- Cognitive Load

_Tracks the mental friction involved in formulating speech_

| Level | Description |
|:-----:|:------------|
| 0 | The speech output is perfectly fluid and completely effortless, possessing absolutely zero cognitive hesitation or processing friction. It sounds as though the speaker has completely memorized the text and is reciting it from deep neural pathways without a single micro-second of real-time thought. The thought-to-speech pipeline is flawlessly executed. |
| 1 | The cognitive delivery is highly articulate and incredibly fast, demonstrating a near-perfect, highly efficient mental processing speed. The speaker easily translates complex ideas into a perfectly sequenced acoustic stream with almost zero visible mental labor. The intellectual machinery operates with highly impressive, uninterrupted smoothness. |
| 2 | The audio features standard, healthy conversational flow where the speaker experiences only occasional, completely natural micro-pauses to search for the correct word. The mental processing is entirely manageable and feels organically easy, creating a comfortable, naturally paced acoustic rhythm. The brain-to-mouth connection is stable and typical. |
| 3 | The speaker's real-time mental processing becomes audibly noticeable, creating an uneven acoustic rhythm as they actively build their sentences block by block. There is a distinct 'searching' quality in the voice, accompanied by minor hesitations that reveal the brain working to organize the logic. The cognitive load is elevated but successfully managed. |
| 4 | The cognitive machinery is heavily burdened, resulting in a delivery weighed down by frequent vocalized pauses, long gaps, and a distinct acoustic 'heaviness.' The speaker is visibly struggling to formulate their thoughts, heavily slowing down the linguistic output as their brain attempts to solve the communication problem. The mental effort significantly degrades the pacing of the audio. |
| 5 | The speaker's mental capacity is severely overloaded, causing them to frequently lose their train of thought, self-correct, and acoustically backtrack. The heavy cognitive friction makes the internal struggle blatantly obvious, heavily fracturing the flow of information and causing the speaker to stumble over their own logic. The brain is clearly failing to keep up with the mouth. |
| 6 | The cognitive burden is so catastrophically highâ€”due to extreme shock, complex fabrication, or intellectual overloadâ€”that the speaker's linguistic output completely breaks down. The brain entirely fails to string together a coherent sentence, resulting in a shattered, deeply confused, and logically incoherent acoustic wave. The mental operating system has essentially crashed. |

### `ATCK` -- Attack

_Measures onset characteristics as sound initiates from silence_

| Level | Description |
|:-----:|:------------|
| 0 | The voice emerges from silence with the softest, most delicate possible onset, barely disturbing the acoustic field. The consonants at the start of words are whispered rather than struck, creating an incredibly gentle, almost imperceptible beginning to the phonation. The initial transient is minimally energetic, lacking all percussive bite. |
| 1 | The acoustic onset is exceptionally soft and rounded, featuring a slow, extremely gradual rise in amplitude from silence to full phonation. There is virtually no sharp transient edge, with the sound gently materializing out of the acoustic void like a feather landing. The initial envelope is highly smooth and completely non-aggressive. |
| 2 | The speech begins with a standard, unremarkable soft onset typical of natural, unstressed word beginnings. The consonant is lightly articulated, creating a gentle but clearly perceptible entry into phonation. The initial transient has minimal acoustic bite but is clearly present and organically natural. |
| 3 | The voice initiates with a neutral, standard acoustic onset that is neither particularly soft nor particularly hard. The consonants are cleanly articulated, producing a clearly defined but not exaggerated beginning to the phonation. The onset characteristics are functionally normal and conventionally balanced. |
| 4 | The speech begins with a noticeably energetic, crisp, and clearly defined acoustic onset. The consonants are sharply articulated, creating a distinct percussive 'bite' to the beginning of words. The transient is clearly present and actively reinforces the sense of forward momentum and assertive energy. |
| 5 | The voice attacks from silence with considerable force and acoustic sharpness, featuring a hard, crisp, and distinctly percussive initial transient. The consonant is powerfully struck, creating an aggressive, physically forceful beginning to the phonation. The onset characteristics project intensity, authority, and high physical engagement. |
| 6 | The acoustic onset is violently explosive and extremely sharp, characterized by an immediate, jarring eruption of sound from complete silence. The consonants are struck with extreme force, creating an aggressively percussive, nearly violent beginning to the phonation. The initial transient is physically harsh and demands immediate, startled attention. |

---

## SPECTRAL & TIMBRAL

### `BRGT` -- Brightness

_Refers to the prominence of high-frequency energy within the voice_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic signal is completely devoid of high-frequency energy, sounding heavily muffled as if submerged underwater or blocked by a thick physical barrier. Transient consonants and sibilance are entirely absorbed and invisible, leaving only a dark, formless blob of audio. The spectral balance is severely rolled off at the top, resulting in a muddy low-end rumble. |
| 1 | The treble frequencies are significantly reduced, creating a woolly, dark acoustic texture that entirely lacks a sharp articulatory edge. Sibilant sounds are soft and heavily suppressed, rendering the vocal signal distinctly dull and top-heavy. While marginally more intelligible than a total high-frequency cutoff, the voice remains heavily veiled and muddy. |
| 2 | High-frequency energy is present but gently attenuated, resulting in a warm, rounded, and earthy acoustic profile. This slight treble suppression creates a pleasingly smooth, vintage sound that lacks modern analytical crispness but is exceptionally easy on the ears. Sibilance is naturally controlled and softly absorbed into the frequency mix. |
| 3 | The spectral balance represents the perfect, neutral baseline of natural human speech with an entirely even distribution of high and low frequencies. There is a highly functional mix of low-end body and high-end clarity that ensures perfect legibility without harshness. Sibilants and transients hit with standard, unexaggerated acoustic presence. |
| 4 | The high frequencies are distinctly well-defined and highly articulate, giving the voice a modern, crisp, and extremely present production quality. Every consonant slices through the audio mix with sharp transient clarity and beautifully distinct sibilance. The acoustic profile feels bright and forward without ever causing auditory fatigue. |
| 5 | Treble frequencies are heavily emphasized, adding a brilliant, airy shimmer to the absolute top of the vocal spectrum. The voice sits extremely forward in the mix, characterized by highly pronounced sibilance and a sparkling, ultra-clear acoustic sheen. The continuous high-frequency excitation creates a dazzling, though potentially fatiguing, listening experience. |
| 6 | The high-frequency energy is overwhelmingly harsh, resulting in a sharply piercing, sizzling acoustic output that feels physically painful to hear. Sibilant consonants act like breaking glass, dominating the entire spectrum with aggressive, ear-drilling treble spikes. The signal is entirely unbalanced, relying exclusively on an abrasive, highly metallic top-end. |

### `ROUG` -- Roughness

_Measures the presence of rapid amplitude modulations or grit within the vocal fold vibration_

| Level | Description |
|:-----:|:------------|
| 0 | The vocal fold vibration is impossibly smooth and pure, closely resembling the clean acoustic geometry of a pure sine wave or a synthetic flute. The signal contains absolutely zero aperiodic noise, biological grit, or vocal fry, resulting in a perfectly clean, uninterrupted tonal stream. It is a flawless, glassy acoustic emission completely free of organic friction. |
| 1 | Phonation is exceptionally consistent and velvety, exhibiting only the absolute faintest microscopic traces of organic human texture. The vocal folds vibrate with incredibly healthy, polished regularity, generating a highly soothing and buttery acoustic output. The sound wave glides effortlessly with almost zero internal biological friction. |
| 2 | The voice features the standard, healthy level of organic texture typical of everyday human speech and natural biology. A slight, highly natural acoustic 'bite' prevents the tone from sounding synthetic, rooting the vocal production in grounded human reality. There is no distracting grit, fry, or harshness present in the highly stable signal. |
| 3 | A distinct, grainy texture noticeably permeates the vocal tone, introducing rapid amplitude modulations into the fundamental frequency. The voice sounds as though it contains a layer of acoustic sand, typical of mild vocal fry or slightly desiccated vocal folds. This gentle grit adds a rough, tactile characteristic to the otherwise stable phonation. |
| 4 | The acoustic signal is heavily raspy and scratchy, revealing a highly weathered biological apparatus operating with significant internal friction. The vocal folds vibrate with a continuous, grinding noise floor, sounding like a speaker with a mild cold or a long-term smoking habit. This heavy grit is an inescapable, defining feature of the overall sonic footprint. |
| 5 | The vocal texture is aggressively harsh and distinctly growling, driven by intense physical friction and heavy aperiodic interference at the vocal folds. The fundamental tone is deeply disrupted by a heavy, rattling rumble that heavily distorts the biological sound wave. It is a highly abrasive, physically forceful acoustic grind that dominates the spectrum. |
| 6 | The fundamental tone is almost completely obliterated by violent, aperiodic noise and extreme biological crunch. The vocal folds are in a state of chaotic, shredding friction, producing an acoustic output akin to a death-metal growl or severe physical tissue breakdown. It is a jarring, maximally rough audio signal utterly devoid of clean periodicity. |

### `HARM` -- Harmonicity

_Measures the ratio of periodic signal to aperiodic noise_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic signal is entirely composed of broadband aperiodic noise, lacking even the absolute faintest trace of a discernible fundamental pitch. It represents a totally unvoiced, absolute whisper where all acoustic energy is derived from air turbulence rather than vocal fold vibration. The sound wave is a ghost made purely of friction and white-noise static. |
| 1 | A microscopic, ghostly hint of fundamental pitch barely survives inside a dense, overwhelming cloud of high-frequency white noise. The acoustic output is primarily breath and air leakage, with only a tiny, highly fragile fraction of energy successfully converted into tonal resonance. It sounds like a heavy, desperate whisper struggling to find its voice. |
| 2 | The audio is distinctly breathy, featuring a clear fundamental pitch that is continuously wrapped in a thick, highly audible layer of air turbulence. Known psychoacoustically as 'bedroom voice,' the vocal folds fail to close completely, allowing a steady stream of hissing noise to mix with the tone. The resulting sound is highly soft, deeply diffused, and highly aerated. |
| 3 | The signal achieves the standard, healthy ratio of periodic tone to aperiodic noise characteristic of normal human speaking voices. The fundamental pitch is solid and highly clear, acting as the primary acoustic carrier while only a negligible, natural amount of breath noise escapes. The vocal fold closure is biologically balanced and perfectly functional. |
| 4 | The voice is highly resonant and exceptionally focused, featuring a massive fundamental pitch and a remarkably low acoustic noise floor. The tonal energy is highly efficient and distinctly ringing, characteristic of a highly trained public speaker maximizing their vocal fold closure. The audio wave is deeply concentrated and musically tunable. |
| 5 | The acoustic output is exceptionally tonal and brilliantly bell-like, successfully converting virtually all breath energy into a pure, ringing fundamental frequency. There is absolutely no audible air leakage or turbulent hiss, resulting in an incredibly clean, almost musical vocal tone. The harmonic stack is incredibly rich, highly stable, and mathematically beautiful. |
| 6 | The harmonicity is so absolute and mathematically perfect that the voice sounds entirely computer-generated or digitally synthesized. It is completely devoid of the organic 'fuzz', breath, or aperiodic noise inherently attached to human biology. The resulting tone is a sterile, digitally pure acoustic ideal that feels entirely artificial. |

### `FULL` -- Fullness

_Describes the size or volume of the sound's spectral footprint_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is overwhelmingly thin and heavily paper-like, completely lacking any spectral body, warmth, or lower-frequency foundation. The acoustic footprint is a flat, one-dimensional sliver of sound that resembles audio forced through a tiny, severely broken watch speaker. It entirely fails to occupy any significant portion of the sonic frequency spectrum. |
| 1 | The spectral bandwidth is heavily restricted, artificially rolling off both the low-end bodily thump and the high-end airy frequencies. The resulting audio is highly narrow, severely small, and distinctly boxed-in, perfectly mirroring the heavy frequency compression of a low-quality telephone call. The signal lacks both warmth and brilliance, feeling highly confined. |
| 2 | The vocal profile is highly intelligible and clear but remains distinctly slender and physically lightweight in its overall spectral footprint. It lacks the rich low-mid body required to actively command the acoustic space, feeling functionally present but physically very small. The voice does not push significant air or project heavy acoustic mass. |
| 3 | The audio features a solid, naturally healthy spectral bandwidth that feels organically complete and appropriately sized for a human speaker. It perfectly balances enough low-mid energy to sound grounded with enough high-end to sound distinctly realistic and present. The acoustic volume occupies a standard, highly comfortable amount of space. |
| 4 | The vocal signal is incredibly rich and highly wide, spreading out across a broad frequency spectrum to create a highly expensive acoustic texture. It possesses a strong, thick physical body that feels highly present and structurally robust in the audio mix. The resulting sound wave occupies a massive amount of highly pleasing sonic territory. |
| 5 | The audio footprint is massive, physically thick, and highly commanding, featuring a heavily reinforced spectral body that projects immense power. The voice sounds perfectly captured by a high-end studio microphone, fully utilizing a deep proximity effect to generate a heavy, weighty acoustic mass. It effortlessly and entirely dominates the low and mid-frequency ranges. |
| 6 | The vocal fullness reaches an overwhelming, IMAX-level cinematic scale, completely saturating the entire acoustic field with unbelievable depth and width. The sound is astronomically big, creating a highly exaggerated, rumbling voiceover effect that vibrates deep within the listener's physical chest. It is a hyper-saturated, omnipresent wall of spectral density. |

### `WARM` -- Warmth

_Refers to the emphasis on low-mid frequencies (around 200Hz-500Hz)_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic profile is entirely devoid of low-mid frequency resonance, resulting in a distinctly cold, metallic, and surgically sterile sound. The voice completely lacks any organic 'cuddle' or bodily depth, sounding like a harsh dental drill or a highly detached, emotionless machine. It is an actively uninviting, heavily rigid sonic texture. |
| 1 | The audio is distinctly tinny and highly top-heavy, severely lacking the low-mid bodily warmth required to sound natural or comforting. This heavily skewed frequency response creates a cheap, highly hollow, and distinctly irritating acoustic signature. The voice feels entirely disconnected from the physical warmth of the human chest. |
| 2 | The vocal resonance is highly functional but distinctly neutral-cool, operating with a very dry acoustic profile that prioritizes hard information over comfort. The low-mid frequencies are present but distinctly suppressed, preventing the voice from sounding naturally inviting or physically relaxed. It is a highly utilitarian, slightly stiff acoustic delivery. |
| 3 | The resonance perfectly strikes the standard human baseline, offering a highly balanced, universally comfortable acoustic profile. There is exactly enough low-mid frequency presence to sound biologically grounded and human, but not so much that the audio becomes muddy or muffled. The natural warmth of the chest and throat is perfectly proportioned. |
| 4 | The voice exhibits a highly pleasant, distinctly woody, and cozy resonance heavily supported by a smooth, gentle low-mid frequency emphasis. This targeted boost in the lower harmonics creates a deeply soothing, physically close acoustic sensation that feels instantly comfortable to the listener. The sonic texture is round, inviting, and inherently relaxing. |
| 5 | High levels of rich, low-mid resonance heavily saturate the audio, creating a deeply velvety, highly attractive 'late-night radio DJ' acoustic profile. The vocal folds produce a heavy, exceptionally warm hum that perfectly rounds off all harsh transients and fills the sonic space with comfortable depth. It is a highly engineered, deeply attractive resonant glow. |
| 6 | The warmth is so overwhelmingly massive that the voice essentially melts into a gigantic, enveloping physical hug of pure low-frequency resonance. It is a completely bass-heavy, hyper-cozy, 'chocolate-like' acoustic environment that completely swallows the listener's ears. The sound wave entirely sacrifices sharp intelligibility in favor of absolute, overwhelming sonic comfort. |

### `METL` -- Metallic Character

_Describes the presence of clanging or ringing inharmonic frequencies_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is purely organic and impossibly soft, completely lacking any hard acoustic edges or inharmonic ringing frequencies. The resonance is distinctly wooden or wool-like, perfectly absorbing all high-frequency transients and preventing any harsh reflections within the vocal tract. The audio profile is the absolute, unyielding antithesis of a metallic machine. |
| 1 | The acoustic output is deeply grounded in a soft, earthy resonance with only the absolute minimal, microscopic trace of any metallic ping. The sound wave is highly natural, prioritizing a warm, fleshy biological dampening over hard, reflective acoustic amplification. The voice feels intimately human and completely devoid of artificial stiffness. |
| 2 | The audio features a standard, highly typical human resonance containing only a negligible, entirely organic amount of high-frequency acoustic 'ping.' The biological vocal tract functions normally, beautifully balancing soft tissue absorption with the hard reflection of the hard palate. The overall sonic footprint remains firmly and functionally natural. |
| 3 | A distinct firmness begins to visibly infiltrate the acoustic profile, giving the voice a stiff, slightly 'hard' physical quality. The speaker sounds as though they are operating with a tightly clenched jaw, causing higher frequencies to reflect harshly off the internal vocal tract. This introduces a subtle but completely un-ignorable metallic sheen. |
| 4 | The audio is deeply characterized by a clear, highly audible metallic resonance that sounds exactly like a voice bouncing directly off a steel plate. A pronounced, highly inharmonic 'ring' or 'ping' actively interferes with the fundamental tone, highly typical of extreme vocal tension or aggressive projection. The sound wave feels intensely rigid and unyielding. |
| 5 | The vocal output is fiercely clanging and distinctly piercing, heavily saturated with sharp, iron-like inharmonic frequencies that actively assault the ear. The resonance is incredibly hard and physically dangerous, completely stripping away any illusion of soft, organic biological tissue. The acoustic edge is highly cold, severely abrasive, and deeply metallic. |
| 6 | The voice is entirely consumed by a pure, robotic metallic resonance, sounding exactly like a synthetic machine or a heavily armored entity. It is a violently hard, continuously clanging acoustic signal that lacks even a single trace of organic, human softness. The sonic texture is a perfect, terrifying manifestation of cold, vibrating steel. |

### `ESTH` -- Esthetics

_A highly consistent dimension measuring the overall attractiveness or pleasantness of the sound_

| Level | Description |
|:-----:|:------------|
| 0 | The audio is physically intolerable and viscerally repulsive, instantly triggering an immediate, desperate desire to stop the playback. It consists of highly painful acoustic elements like violent distortion, extreme high-pitched screeching, or deeply disturbing biological noises. The sonic experience is an absolute, unmitigated assault on the listener's auditory system. |
| 1 | The vocal quality is deeply unpleasant and highly grating, actively forcing the listener to endure harsh, whining, or exceptionally 'ugly' acoustic textures. The signal is deeply irritating, feeling audibly 'dirty' or aggressively fatiguing to process over any length of time. Listening to this audio requires significant effort and active tolerance of severe sonic flaws. |
| 2 | The audio profile is entirely mundane and fundamentally boring, heavily lacking any discernible acoustic appeal or engaging characteristics. While technically free of painful distortion, the voice sounds highly cheap, exceptionally dry, and thoroughly uninteresting to the human ear. It is a highly forgettable, perfectly mediocre, and strictly utilitarian sonic experience. |
| 3 | The recording presents a highly acceptable, perfectly functional standard of audio quality typical of average human voices or standard consumer microphones. The signal is cleanly intelligible and entirely unobtrusive, though it completely lacks any professional polish or distinct vocal charisma. It is the absolute baseline for safe, everyday auditory consumption. |
| 4 | The acoustic experience is highly pleasant, featuring a genuinely good human voice flawlessly captured by a clean, professional recording chain. The audio is incredibly easy on the ears, highly well-balanced, and completely devoid of any distracting technical or biological flaws. It provides a highly satisfying, premium-quality listening environment. |
| 5 | The vocal delivery radiates massive 'star power,' flawlessly combining an exceptionally charismatic, rich voice with a beautifully polished studio recording. The acoustic texture is deeply soothing, highly engaging, and undeniably attractive to the listener's ear. It is a premium, highly engineered auditory product that actively and effortlessly commands positive attention. |
| 6 | The audio achieves a sublime, almost holy level of absolute acoustic beauty, representing the absolute pinnacle of perfect vocal and technical synergy. The voice is an ecstatic, physically thrilling listening experience that flawlessly balances deep warmth, striking clarity, and massive emotional resonance. It is a flawless, deeply transcendent sonic masterpiece. |

---

## TEMPORAL DYNAMICS

### `VFLX` -- Velocity Flux

_Measures the change in speed over the duration of the clip_

| Level | Description |
|:-----:|:------------|
| 0 | The vocal trajectory begins at a noticeably rapid, high-energy pace before undergoing a massive, continuous deceleration, progressively grinding down to an extremely slow, labored crawl by the end. The speech starts rushed and compressed, then steadily loses all forward momentum as syllables stretch wider and wider apart. The speed differential from start to finish is extreme, covering the full range from 1.5x to 0.5x normal pace. |
| 1 | The speech begins at a clearly elevated pace and undergoes a heavy, sustained deceleration across the full duration. The initial brisk delivery gradually and obviously slows down, with the speaker audibly losing rhythmic momentum as the phrase progresses. The tempo drops from roughly 1.3x to 0.7x normal speed, creating a pronounced but not extreme winding-down effect. |
| 2 | The temporal dynamics exhibit a subtle, natural easing of the tempo, starting slightly faster than normal and gently braking to a slightly slower pace by the end. This mild deceleration functions as a deliberate, comfortable cadence, carefully reducing momentum from about 1.1x to 0.9x normal speed. The slowdown is controlled and rhythmically satisfying, like a speaker naturally winding down a thought. |
| 3 | The velocity of the speech remains absolutely locked and perfectly steady from the very first syllable to the last. There is exactly zero detectable acceleration or deceleration, resulting in a strictly linear, metronomic temporal delivery. The acoustic momentum is held at a completely constant speed throughout the entire clip. |
| 4 | The speech trajectory features a subtle, natural forward lean, beginning slightly slower than normal and gently pushing the tempo faster as the phrase concludes. This mild acceleration from about 0.9x to 1.1x normal speed suggests a rising tide of engagement or growing confidence. The acoustic momentum subtly builds, creating a gently forward-moving vector. |
| 5 | The audio begins at a noticeably slow, deliberate pace and captures a clear, sustained acceleration across the full duration, with the speaker audibly speeding up from roughly 0.7x to 1.3x normal speed. The acoustic rhythm compresses progressively as the phrase progresses, driven by an escalating surge of energy or urgency. The temporal vector is distinctly forward-thrusting with an obvious speed buildup. |
| 6 | The tempo vector begins extremely slowly, with syllables heavily stretched and drawn out, before undergoing a massive, continuous acceleration that ends at a rapid, high-energy pace. The speech transforms from a crawl at 0.5x to a rush at 1.5x normal speed, with the acoustic momentum building relentlessly throughout. The speed differential from start to finish is extreme, the mirror image of bucket 0. |

### `DARC` -- Dynamic Arc

_Measures the trajectory of loudness across the clip_

| Level | Description |
|:-----:|:------------|
| 0 | The acoustic volume begins at a highly audible level before suffering a total, catastrophic collapse, fading away into absolute dead silence. The audio completely dissolves and dies out, perfectly tracking a highly linear trajectory toward complete acoustic nothingness. The volume envelope is a pure, absolutely unrecoverable fade to black. |
| 1 | The speech vector is characterized by a clear, incredibly steady, and highly pronounced diminuendo that drops the volume drastically across the clip. The speaker sounds exactly as if they are rapidly losing all conviction or physically backing away from the microphone mid-sentence. The dynamic envelope heavily deflates, bleeding massive acoustic energy continuously. |
| 2 | The volume trajectory features a gentle, highly natural softening of acoustic power at the very end of the linguistic phrase. This minor dip in decibels is a standard, deeply organic human cadence used to gently signal the conclusion of a thought. The dynamic envelope 'lands' smoothly and softly without entirely losing its structural integrity. |
| 3 | The decibel level remains absolutely flatlined, exhibiting exactly zero change in audio volume from the beginning to the very end of the clip. The dynamic range is perfectly compressed and highly static, flawlessly maintaining an unwavering, mathematically constant acoustic output. The volume envelope is entirely frozen in place. |
| 4 | The dynamic envelope perfectly traces a highly controlled, deeply satisfying bell curve of rising and falling acoustic volume. The speech begins quietly, smoothly swells into a louder, highly energetic peak in the middle, and then gracefully softens back down at the end. It is a deeply musical, highly planned trajectory of deliberate vocal power. |
| 5 | The volume vector tracks a clear, highly aggressive crescendo, building massive acoustic loudness and physical insistence across the duration of the clip. The speaker is actively 'powering up,' visibly driving the decibel level significantly higher with every passing word. The dynamic envelope is highly ascendant, continuously gathering forceful, undeniable momentum. |
| 6 | The dynamic arc covers an impossibly extreme range, starting as a tiny, barely audible whisper and exploding directly into a massive, floor-shaking scream. This violent trajectory acts as a sheer vertical ramp of acoustic power, completely shattering the initial quiet with an overwhelming surge of volume. The decibel differential is absolutely massive and highly shocking. |

### `ARSH` -- Arousal Shift

_Measures the change in physiological activation over the duration of the clip_

| Level | Description |
|:-----:|:------------|
| 0 | The physiological trajectory begins in a state of high activation before entirely collapsing into a near-comatose, heavily sedated acoustic state. The speaker's biological engine suffers a total loss of vitality, perfectly mirroring a person passing out or entirely surrendering to absolute exhaustion. The energy vector is a catastrophic, completely unrecoverable downward plunge. |
| 1 | The audio tracks a highly significant de-escalation of autonomic nervous system energy, moving rapidly from high alert down to a state of deep, relative calm. This deep drop in physical intensity perfectly mimics a massive 'sigh of relief' or the immediate, breathless aftermath of a resolved conflict. The acoustic engine spins down aggressively and audibly. |
| 2 | The physiological vector exhibits a subtle, highly comforting settling of bodily tension as the audio smoothly progresses to its end. The vocal energy softens and relaxes slightly towards the final words, clearly suggesting the speaker is becoming physically more comfortable or gently tired. The acoustic momentum gently decelerates into a highly relaxed bodily posture. |
| 3 | The autonomic activation level remains entirely static and completely constant, demonstrating exactly zero change in baseline physiological arousal. The exact energy level at the start of the clip is perfectly mirrored at the end, firmly locking the speaker into an unyielding physical state. The biological engine neither revs up nor physically spins down. |
| 4 | The speech trajectory tracks a subtle, highly engaging 'perking up' of physiological alertness and rising bodily energy. The speaker sounds perceptibly more awake, highly interested, and physically engaged by the end of the phrase than they did at the exact beginning. The acoustic engine gently revs up, applying a light, highly active forward-moving pressure. |
| 5 | The energy vector traces a clear, highly driving, and intensely aggressive escalation of physical intensity and massive autonomic arousal. The vocal folds are pushed significantly harder, louder, and faster as the speaker actively works themselves up into a heightened state of severe excitement or anger. The biological engine rapidly accelerates toward an explosive boiling point. |
| 6 | The physiological trajectory is a completely violent, 0-to-100 explosion of extreme autonomic mobilization that fundamentally shatters the acoustic baseline. The clip begins in a state of calm and literally detonates into a full scream, shout, or full-blown panic attack within seconds. The energy vector is a highly terrifying, totally combustible biological acceleration. |

### `VALS` -- Valence Shift

_Tracks the change in emotional polarity across the segment_

| Level | Description |
|:-----:|:------------|
| 0 | The emotional trajectory suffers a catastrophic, heavily tragic turn, plunging instantly from a positive or neutral state into deep despair or absolute, sheer horror. The acoustic 'smile' is violently stripped away mid-sentence, instantly replaced by the heavy, trembling tones of absolute psychological agony or shock. It is the highly distressing sonic equivalent of receiving completely devastating, life-altering news. |
| 1 | The mood vector tracks a highly significant souring and intense darkening of the psychological state, rapidly deteriorating into distinct, heavy negativity. Initial optimism or polite interaction audibly curdles into deep doubt, aggressive hostility, or profound sadness by the end of the clip. The underlying 'light' completely and noticeably drains out of the acoustic signal. |
| 2 | The emotional arc features a subtle, highly perceptible clouding of the initial positive or perfectly neutral affective baseline. A distinct hint of worry, minor disappointment, or mild unease slowly creeps into the final acoustic phrases, gently pulling the mood slightly downward. The psychological vector leans gently toward a more negative space. |
| 3 | The emotional trajectory remains absolutely constant, exhibiting exactly zero change in affective state from the beginning to the very end of the acoustic signal. The mood is perfectly frozen, demonstrating an unwavering commitment to a single, unchanging psychological posture. The emotional vector is completely flatlined. |
| 4 | The mood trajectory features a subtle, highly uplifting, and positive turn that gradually brightens as the audio progresses. The emotional state gently warms and opens up, suggesting a rising tide of confidence, relief, or genuine pleasure. The psychological vector leans softly upward into a more positive, visibly lighter emotional space. |
| 5 | The emotional trajectory features a significant brightening and pronounced uplifting, with the affective state steadily improving and becoming increasingly positive. Initial neutrality or guarded reservation audibly transforms into genuine warmth, hope, or manifest joy by the end of the clip. The underlying 'light' steadily and perceptibly floods into the acoustic signal. |
| 6 | The emotional arc undergoes a shocking, highly transformative elevation, surging instantly from a neutral or negative state into absolute euphoric joy or ecstatic triumph. The acoustic profile explodes with positivity and delight, completely shattering an initially subdued or gloomy baseline. It is the highly uplifting sonic equivalent of receiving absolutely life-changing, wonderful news. |

---

## LANGUAGE & RECORDING

### `RCQL` -- Recording Quality

_Measures the technical fidelity and cleanliness of the audio recording_

| Level | Description |
|:-----:|:------------|
| 0 | The recording is completely unusable and technically catastrophic, dominated by overwhelming background noise, severe distortion, clipping, or extreme technical corruption. The audio is incomprehensible due to technical failure, resembling audio retrieved from a severely damaged cassette tape or a deeply corrupted digital file. The signal-to-noise ratio is irretrievably destroyed. |
| 1 | The recording quality is severely degraded, featuring heavy background noise, noticeable hum, significant distortion, or severe technical artifacts. While potentially containing some salvageable content, the technical flaws are so pronounced that listening requires significant tolerance. The audio feels like a heavily compressed, extremely low-bitrate file. |
| 2 | The recording exhibits noticeable technical flaws including moderate background noise, slight distortion, or audible compression artifacts. While perfectly intelligible, the audio clearly lacks professional-grade quality and feels somewhat budget or casually recorded. The technical fidelity is functional but distinctly imperfect. |
| 3 | The recording is clean and technically solid, captured with standard consumer-grade microphone equipment or basic professional tools. The background noise is minimal, there is no perceptible distortion, and the signal is cleanly transmitted. It is a highly functional, perfectly acceptable standard recording that represents the baseline of technical competence. |
| 4 | The audio is clearly captured with professional-grade microphone equipment and studio-quality recording techniques. The background noise is essentially eliminated, the signal is clean and undistorted, and the overall technical execution is polished. It is a high-quality, clearly professional recording that avoids all technical pitfalls. |
| 5 | The recording is exceptionally clean and expertly captured using high-end studio microphones and professional acoustic treatment. The signal is pristine, completely free of any background noise or technical artifacts, and demonstrates meticulous technical mastery. It is a premium-quality, audiophile-grade recording that represents the absolute pinnacle of technical excellence. |
| 6 | The recording achieves an almost impossibly perfect technical standard, captured using the absolute highest-end professional studio equipment in an acoustically perfect environment. The signal is so clean and artifact-free that it borders on the synthetic, completely devoid of any trace of real-world recording challenges. It is a hyper-engineered, perfectionist audio production. |

### `BKGN` -- Background Noise

_Measures the prominence and intrusiveness of ambient environmental sounds_

| Level | Description |
|:-----:|:------------|
| 0 | The background noise is absolutely overwhelming and completely dominates the acoustic signal, rendering the foreground speech nearly inaudible. The ambient sound is so loud and intrusive that it actively obscures the primary vocal content, making comprehension extremely difficult. The listener must work incredibly hard to extract any speech from the dense wall of background sound. |
| 1 | The background noise is heavily prominent and quite intrusive, actively competing with the primary voice for acoustic space. The ambient sound is clearly audible and distinctly distracting, making it necessary for the listener to concentrate intensely to follow the speech. The noise floor is high and persistently undermines acoustic legibility. |
| 2 | The background noise is moderately noticeable and mildly distracting, taking up a meaningful portion of the acoustic spectrum. The foreground speech remains perfectly intelligible, but the ambient sound never entirely fades into the background. The listener can easily identify and mentally filter the noise, but it remains a persistent, acknowledged presence. |
| 3 | The background noise is minimal and entirely non-distracting, consisting only of negligible room tone or the faintest trace of ambient environment. The acoustic signal is essentially clean, with the foreground speech completely dominating the mix. The noise floor is barely conscious and does not interfere with listening pleasure. |
| 4 | The background is acoustically dead and completely silent, with absolutely zero ambient environmental sound visible in the recording. The acoustic signal is pristine and isolated, perfectly capturing only the foreground voice without even a whisper of room tone. The noise floor is essentially non-existent. |

### `EXPL` -- Content Appropriateness (3-point Scale)

_New redesigned scale measuring content appropriateness for different audiences_

| Level | Description |
|:-----:|:------------|
| 0 | **Safe for School (SFS)**: Content is completely harmless and suitable for a K-12 classroom. Normal conversation, educational content, or everyday topics. No intense violence, no sexuality beyond age-appropriate mention. Mild language at worst (PG-13). A teacher could play this in class without any concern. |
| 1 | **Safe for Work (SFW)**: Content acceptable for adults in a professional setting during a break. May contain moderate violence, flirting, suggestive content (e.g., describing a makeout scene, discussing revealing clothing), or moderately strong language. An adult could listen with colleagues without serious discomfort, but it would be somewhat inappropriate for a school classroom. |
| 2 | **Not Safe for Work (NSFW)**: Content containing extreme gore, pornographic or overtly sexual material, graphic violence, or anything that would obviously violate social norms if played in any professional or formal setting. Content that would be embarrassing, offensive, or cringe-inducing to share in a workplace. |

---

## RESONANCE PLACEMENT

### `R_CHST` -- Chest Resonance

_Measures the degree of lower-body resonance originating from the chest cavity_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is entirely stripped of chest resonance, producing only high-frequency head resonance without any low-frequency body. The acoustic output is impossibly light and thin, completely devoid of the physical 'weight' and warmth associated with chest vibration. The sound is purely ethereal, floating entirely above the body. |
| 1 | Chest resonance is severely suppressed or nearly absent, with the voice operating almost exclusively from the head and throat cavities. The audio lacks any significant low-frequency depth or physical body, sounding thin and somewhat disconnected from the lower body. The acoustic footprint is heavily weighted toward the upper registers. |
| 2 | Chest resonance is minimally present, with only a subtle layer of lower-frequency body supporting the predominantly head-based resonance. The voice retains some physical grounding but lacks the full, deep chest 'boom' characteristic of maximally resonant delivery. The low-end presence is gently supportive rather than dominant. |
| 3 | Chest resonance is present at a standard, balanced level, providing a healthy foundation of lower-frequency body without overwhelming the overall acoustic profile. The voice naturally integrates chest and head resonance in a comfortable, biologically typical proportion. It is a functional, unremarkable balance of vertical resonance. |
| 4 | Chest resonance is distinctly prominent and notably audible, providing a substantial layer of deep, rich low-frequency body to the voice. The lower register is actively engaged and clearly present, adding significant acoustic weight and warmth to the overall profile. The audio demonstrates strong chest cavity involvement. |
| 5 | Chest resonance is heavily emphasized and deeply prominent, completely dominating the acoustic output with a thick, rumbling low-frequency presence. The voice sounds as though it is being projected directly from the deepest regions of the chest cavity. The acoustic weight is substantial, powerful, and unmistakably body-centered. |
| 6 | Chest resonance is maximally developed and overwhelmingly dominant, completely submerging the voice in a cavernous, booming low-frequency rumble. The acoustic output is almost entirely derived from chest vibration, creating an impossibly deep, physically overwhelming bass presence. It is a gargantuan, body-centered resonance that fills the entire acoustic space. |

### `R_THRT` -- Throat Resonance

_Measures the degree of resonance originating from the throat and pharyngeal cavity_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is entirely stripped of throat resonance, sounding hollow and disconnected from the pharyngeal cavity. The acoustic output lacks the natural warmth and body associated with throat vibration, producing a thin, disembodied sound. The pharynx is acoustically 'shut down' and entirely uninvolved in resonance. |
| 1 | Throat resonance is severely suppressed or nearly absent, with the voice operating from the mouth and nasal cavities but without utilizing the pharyngeal space. The audio lacks the natural vocal warmth characteristic of throat-based resonance, sounding somewhat disconnected and lacking depth. The pharyngeal cavity is acoustically invisible. |
| 2 | Throat resonance is minimally present, with only a subtle layer of pharyngeal body supporting the overall acoustic profile. The voice retains some natural vocal color but lacks the full, warm throat resonance characteristic of maximally engaged delivery. The pharyngeal presence is gently supportive rather than prominent. |
| 3 | Throat resonance is present at a standard, balanced level, providing a healthy foundation of vocal warmth and body without overwhelming the overall acoustic profile. The voice naturally engages the pharyngeal cavity in a comfortable, biologically typical proportion. It is a functional, natural level of throat involvement. |
| 4 | Throat resonance is distinctly prominent and notably audible, providing a substantial layer of warm, rich pharyngeal body to the voice. The throat cavity is actively engaged and clearly present, adding significant acoustic color and depth to the overall profile. The audio demonstrates strong pharyngeal involvement. |
| 5 | Throat resonance is heavily emphasized and deeply prominent, powerfully engaging the pharyngeal cavity with a thick, warm acoustic presence. The voice sounds as though it is being projected through a highly resonant throat cavity. The pharyngeal warmth is substantial and unmistakably dominant. |
| 6 | Throat resonance is maximally developed and overwhelmingly dominant, completely saturating the voice with deep, cavernous pharyngeal resonance. The acoustic output is almost entirely derived from throat vibration, creating a heavily warmed, deeply throaty vocal quality. It is a maximally engaged pharyngeal resonance that completely dominates the acoustic signature. |

### `R_ORAL` -- Oral Resonance

_Measures the degree of resonance originating from the oral cavity_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is entirely stripped of oral resonance, producing only non-oral cavity vibration without any mouth-based resonance. The acoustic output lacks the natural articulation and brightness associated with oral cavity involvement. The mouth cavity is acoustically 'shut down' and entirely uninvolved in resonance. |
| 1 | Oral resonance is severely suppressed or nearly absent, with the voice operating from the throat and nasal cavities but without utilizing the mouth space. The audio lacks the natural clarity and articulation characteristic of oral-resonant delivery, sounding somewhat muted. The oral cavity is acoustically invisible. |
| 2 | Oral resonance is minimally present, with only a subtle layer of mouth-based brightness supporting the overall acoustic profile. The voice retains some natural clarity but lacks the full, bright oral resonance characteristic of maximally engaged delivery. The oral presence is gently supportive rather than prominent. |
| 3 | Oral resonance is present at a standard, balanced level, providing healthy clarity and articulation without overwhelming the overall acoustic profile. The mouth cavity naturally engages in a comfortable, biologically typical proportion. It is a functional, natural level of oral involvement. |
| 4 | Oral resonance is distinctly prominent and notably audible, providing a substantial layer of bright, clear mouth-based acoustic presence. The oral cavity is actively engaged and clearly present, adding significant clarity and articulation to the overall profile. The audio demonstrates strong oral involvement. |
| 5 | Oral resonance is heavily emphasized and deeply prominent, powerfully engaging the mouth cavity with a bright, clear acoustic presence. The voice sounds as though it is being projected through a highly resonant oral space. The oral brightness is substantial and unmistakably dominant. |
| 6 | Oral resonance is maximally developed and overwhelmingly dominant, completely saturating the voice with bright, forward oral cavity resonance. The acoustic output is almost entirely derived from oral vibration, creating an impossibly bright, articulate vocal quality. It is a maximally engaged oral resonance that completely dominates the acoustic signature. |

### `R_MASK` -- Mask Resonance

_Measures the degree of resonance originating from the facial mask/sinus area_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is entirely stripped of mask resonance, producing only non-facial cavity vibration without any sinus-based resonance. The acoustic output lacks the brightening and shimmer associated with mask involvement. The facial mask is acoustically 'shut down' and entirely uninvolved in resonance. |
| 1 | Mask resonance is severely suppressed or nearly absent, with the voice operating without utilizing the facial mask cavity. The audio lacks the natural brightness and sheen characteristic of mask-resonant delivery. The sinus cavity is acoustically invisible. |
| 2 | Mask resonance is minimally present, with only a subtle layer of facial brightness supporting the overall acoustic profile. The voice retains some natural sparkle but lacks the full, bright mask resonance characteristic of maximally engaged delivery. The mask presence is gently supportive rather than prominent. |
| 3 | Mask resonance is present at a standard, balanced level, providing healthy brightness and shimmer without overwhelming the overall acoustic profile. The facial mask naturally engages in a comfortable, biologically typical proportion. It is a functional, natural level of mask involvement. |
| 4 | Mask resonance is distinctly prominent and notably audible, providing a substantial layer of bright, shimmering facial cavity presence. The mask is actively engaged and clearly present, adding significant brightness and forward projection to the overall profile. The audio demonstrates strong mask involvement. |
| 5 | Mask resonance is heavily emphasized and deeply prominent, powerfully engaging the facial mask with bright, brilliant acoustic presence. The voice sounds as though it is being projected through a highly resonant facial mask. The brightness is substantial and unmistakably dominant. |
| 6 | Mask resonance is maximally developed and overwhelmingly dominant, completely saturating the voice with brilliant, forward mask cavity resonance. The acoustic output is almost entirely derived from facial mask vibration, creating an impossibly bright, forward-projected vocal quality. It is a maximally engaged mask resonance that completely dominates the acoustic signature. |

### `R_NASL` -- Nasal Resonance

_Measures the degree of resonance originating from the nasal cavity_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is entirely stripped of nasal resonance, producing only non-nasal cavity vibration. The acoustic output lacks any nasal coloration, sounding completely oral and devoid of nasal buzz. The nasal cavity is acoustically 'shut down' and entirely uninvolved in resonance. |
| 1 | Nasal resonance is severely suppressed or nearly absent, with the voice operating without significant nasal cavity involvement. The audio lacks the characteristic nasal tone, sounding clear and entirely de-nasalized. The nasal cavity is acoustically invisible. |
| 2 | Nasal resonance is minimally present, with only a subtle layer of nasal coloration supporting the overall acoustic profile. The voice retains slight nasal characteristics but lacks pronounced nasal resonance. The nasal presence is gently supportive rather than prominent. |
| 3 | Nasal resonance is present at a standard, balanced level, providing healthy nasal coloration on nasal consonants without overwhelming the overall acoustic profile. The nasal cavity naturally engages in a comfortable, biologically typical proportion. It is a functional, natural level of nasal involvement. |
| 4 | Nasal resonance is distinctly prominent and notably audible, providing a substantial layer of nasal cavity resonance. The nasal cavity is actively engaged and clearly present, adding noticeable nasal coloration to the overall profile. The audio demonstrates strong nasal involvement. |
| 5 | Nasal resonance is heavily emphasized and deeply prominent, powerfully engaging the nasal cavity with substantial nasal acoustic presence. The voice sounds distinctly nasal and pinched. The nasal character is unmistakably dominant. |
| 6 | Nasal resonance is maximally developed and overwhelmingly dominant, completely saturating the voice with heavy, pronounced nasal resonance. The acoustic output is almost entirely derived from nasal vibration, creating an extremely nasalized, pinched vocal quality. It is a maximally engaged nasal resonance that completely dominates the acoustic signature. |

### `R_HEAD` -- Head Resonance

_Measures the degree of resonance originating from the head cavity_

| Level | Description |
|:-----:|:------------|
| 0 | The voice is entirely stripped of head resonance, producing only non-head-based vibration without any upper resonance. The acoustic output lacks the brightness and ring associated with head resonance. The head cavity is acoustically 'shut down' and entirely uninvolved in resonance. |
| 1 | Head resonance is severely suppressed or nearly absent, with the voice operating without significant upper-cavity involvement. The audio lacks the brightness and shimmer characteristic of head-resonant delivery. The head cavity is acoustically invisible. |
| 2 | Head resonance is minimally present, with only a subtle layer of upper brightness supporting the overall acoustic profile. The voice retains some natural sparkle but lacks the full, bright head resonance characteristic of maximally engaged delivery. The head presence is gently supportive rather than prominent. |
| 3 | Head resonance is present at a standard, balanced level, providing healthy brightness and ring without overwhelming the overall acoustic profile. The head cavity naturally engages in a comfortable, biologically typical proportion. It is a functional, natural level of head involvement. |
| 4 | Head resonance is distinctly prominent and notably audible, providing a substantial layer of bright, ringing upper resonance. The head cavity is actively engaged and clearly present, adding significant brightness and sparkle to the overall profile. The audio demonstrates strong head involvement. |
| 5 | Head resonance is heavily emphasized and deeply prominent, powerfully engaging the head cavity with bright, brilliant acoustic presence. The voice sounds as though it is being projected through a highly resonant head space. The brightness is substantial and unmistakably dominant. |
| 6 | Head resonance is maximally developed and overwhelmingly dominant, completely saturating the voice with brilliant, ringing head cavity resonance. The acoustic output is almost entirely derived from head vibration, creating an impossibly bright, ethereal vocal quality. It is a maximally engaged head resonance that completely dominates the acoustic signature. |

### `R_MIXD` -- Mixed Resonance

_Measures the integration and balance of resonance across all cavities_

| Level | Description |
|:-----:|:------------|
| 0 | The resonance is entirely one-dimensional and heavily isolated to a single cavity, creating a highly imbalanced and acoustically disconnected sound. One resonance source completely dominates while all others are suppressed, resulting in a sonically lopsided profile that lacks integration. The various cavities are acoustically disconnected. |
| 1 | The resonance is heavily dominated by a single cavity with only minimal contribution from secondary resonance sources. The overall profile is imbalanced and skewed, lacking the cohesion and integration of well-mixed resonance. The voice sounds compartmentalized rather than unified. |
| 2 | The resonance shows some degree of integration, with a primary cavity supported by a secondary resonance source. While functional, the overall profile lacks the balanced, cohesive quality of truly mixed resonance. The acoustic integration is incomplete. |
| 3 | The resonance is nicely balanced and naturally integrated across multiple cavities, creating a cohesive, unified vocal sound. All cavity contributions are proportionately weighted, producing a complete, well-rounded acoustic profile. The mixed resonance is harmonious and biologically typical. |
| 4 | The resonance is highly integrated and expertly balanced, with all cavities contributing proportionately to create a unified, powerful acoustic whole. The mixed resonance is seamless and demonstrates sophisticated acoustic integration. The voice sounds completely unified and complex. |
| 5 | The resonance is exceptionally well-mixed and highly sophisticated, with all cavities contributing in perfectly balanced proportion to create an exceptionally rich, unified acoustic presence. The integrated resonance is seamless, complex, and deeply satisfying. The voice sounds completely integrated and multifaceted. |
| 6 | The resonance is flawlessly integrated and perfectly balanced across all cavities, creating an impossibly unified, complex, and complete acoustic whole. Every cavity contributes exactly the right amount to create a perfect acoustic symphony. The mixed resonance is sublime and represents the absolute pinnacle of resonance integration. |

---

## SPEAKING STYLE

### `S_CASU` -- Casual Style

_Measures the degree of informal, relaxed conversational delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely formal, highly structured, and entirely devoid of any casual element. It sounds like a strictly codified, robotic recitation with zero personality or conversational ease. The acoustic profile is maximally rigid and completely resistant to any informal tone. |
| 1 | The delivery is extremely formal and highly buttoned-up, maintaining strict professional boundaries and completely avoiding any casual colloquialism. The acoustic output sounds stiff, rehearsed, and distinctly lacking in warmth or accessibility. The voice is completely sealed off from conversational ease. |
| 2 | The delivery is primarily formal with only subtle hints of casual element, maintaining strict professional distance while allowing the faintest trace of conversational ease. The acoustic profile is still heavily structured but slightly more approachable than pure formality. The casual content is minimal and highly restrained. |
| 3 | The delivery balances formality and informality in a neutral, standard conversational tone. The speaker is appropriately professional but comfortably approachable, striking a functional middle ground between stiff formality and extreme casualness. The acoustic profile is neither distinctly formal nor distinctly casual. |
| 4 | The delivery is notably casual and relaxed, featuring comfortable colloquialisms and a friendly, informal tone. The speaker sounds genuinely at ease, utilizing conversational phrasing and a warm, approachable acoustic profile. The formality is clearly stepped back in favor of approachability. |
| 5 | The delivery is highly casual and extremely relaxed, sounding like an intimate conversation with a close friend. The speaker is completely comfortable with informal language, abundant colloquialisms, and a deeply warm, utterly approachable acoustic presence. The formality is almost entirely abandoned. |
| 6 | The delivery is ultra-casual and extremely informal, sounding exactly like spontaneous, unscripted conversation with zero attempt at professionalism. The speaker uses heavy slang, frequent colloquialisms, and an entirely unprepared acoustic profile that completely rejects any formality. It is the absolute antithesis of structured speech. |

### `S_CONV` -- Conversational Style

_Measures the degree of two-way, dialogue-oriented delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely monological, completely rejecting any conversational element and sounding exactly like a one-way lecture. The acoustic profile shows zero awareness of or invitation toward dialogue, presenting as a pure soliloquy. The delivery is entirely speaker-centric and completely resistant to interaction. |
| 1 | The delivery is extremely monological with only the faintest hint of conversational awareness. While technically audible, the voice gives almost no impression of considering or acknowledging a listener. The acoustic profile is heavily lecture-based and distinctly resistant to back-and-forth exchange. |
| 2 | The delivery is primarily monological with subtle conversational elements, occasionally acknowledging a listener's potential presence but maintaining a mostly one-way flow. The acoustic profile is still heavily weighted toward lecture mode but includes minor signals of conversational awareness. |
| 3 | The delivery strikes a balanced middle ground between monologue and dialogue, featuring standard conversational phrasing that acknowledges a listener without heavily soliciting active participation. The acoustic profile is naturally and functionally conversational. |
| 4 | The delivery is distinctly conversational, featuring natural dialogue-like phrasing that actively invites listener participation and response. The speaker sounds engaged in a genuine two-way exchange, utilizing rhetorical questions and conversational turn-taking patterns. The acoustic profile is highly interactive. |
| 5 | The delivery is highly conversational and deeply interactive, sounding exactly like a natural, dynamic two-way dialogue. The speaker is highly attuned to potential responses, utilizing abundant rhetorical questions and active listening cues. The acoustic profile is maximally dialogic. |
| 6 | The delivery is ultra-conversational and intensely interactive, sounding like an absorbed participant in an intimate, back-and-forth conversation. The speaker demonstrates extreme responsiveness to anticipated listener reactions and heavily solicits participation. The acoustic profile is maximally dialogue-centered. |

### `S_FORM` -- Formal Style

_Measures the degree of structured, professional, high-register delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely casual and completely informal, entirely rejecting any structured, formal element. The acoustic profile shows zero attempt at professionalism or formal registration, sounding spontaneous and utterly unprepared. The delivery is maximally resistant to formal conventions. |
| 1 | The delivery is extremely casual and highly informal, with only the faintest hint of formal structure. The acoustic profile is dominated by conversational ease and completely avoids formal language or professional distance. The formal content is almost entirely absent. |
| 2 | The delivery is primarily casual with subtle formal elements, occasionally adopting professional phrasing but maintaining a predominantly informal tone. The acoustic profile is still heavily weighted toward casual speech but includes minor signals of formal awareness. |
| 3 | The delivery strikes a balanced middle ground between formal and casual, featuring standard professional-conversational phrasing that is neither stiffly formal nor extremely informal. The acoustic profile is naturally and functionally formal without excess. |
| 4 | The delivery is distinctly formal and professionally structured, featuring clear formal phrasing, professional distance, and a clearly elevated register. The speaker sounds competent, serious, and committed to professional standards. The acoustic profile is clearly formal but not excessively rigid. |
| 5 | The delivery is highly formal and exceptionally professional, featuring impressive formal phrasing, clear professional distance, and a distinctly elevated, polished register. The speaker sounds authoritative, cultured, and deeply committed to formal conventions. The acoustic profile is maximally formal. |
| 6 | The delivery is ultra-formal and intensely rigid, sounding exactly like a stiffly codified, excessively pompous formal oration. The speaker maintains extreme formal distance, uses overly elevated language, and sounds almost robotic in their commitment to rigid formality. The acoustic profile is maximally formal to the point of caricature. |

### `S_DRAM` -- Dramatic Style

_Measures the degree of theatrical, emotionally heightened, expressive delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely understated and flat, completely rejecting any theatrical or dramatically heightened element. The acoustic profile shows zero emotional exaggeration, sounding deadpan and utterly emotionless. The delivery is maximally resistant to dramatic conventions. |
| 1 | The delivery is extremely restrained and highly muted, with only the faintest hint of dramatic element. The acoustic profile is dominated by emotional flatness and completely avoids heightened expression or theatrical phrasing. The dramatic content is almost entirely absent. |
| 2 | The delivery is primarily understated with subtle dramatic elements, occasionally adopting theatrical phrasing but maintaining a predominantly flat, muted tone. The acoustic profile is still heavily weighted toward emotional restraint but includes minor signals of dramatic awareness. |
| 3 | The delivery strikes a balanced middle ground between dramatic and understated, featuring standard expressive phrasing that is neither stiffly flat nor excessively theatrical. The acoustic profile is naturally and functionally dramatic without excess. |
| 4 | The delivery is distinctly dramatic and theatrically heightened, featuring clear emotional expression, theatrical phrasing, and a clearly elevated emotional register. The speaker sounds engaged, expressive, and committed to emotional communication. The acoustic profile is clearly dramatic but not excessively overacted. |
| 5 | The delivery is highly dramatic and exceptionally theatrical, featuring impressive emotional expression, heightened phrasing, and a distinctly elevated, passionate register. The speaker sounds deeply engaged, emotionally invested, and committed to theatrical conventions. The acoustic profile is maximally dramatic. |
| 6 | The delivery is ultra-dramatic and intensely theatrical, sounding exactly like an over-the-top, excessively hammy theatrical performance. The speaker maintains extreme emotional heightening, uses overly theatrical phrasing and pitch, and sounds almost comically over-expressive. The acoustic profile is maximally dramatic to the point of caricature. |

### `S_NARR` -- Narrator Style

_Measures the degree of storybook, narrative-focused delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely anti-narrative and completely non-storybook, entirely rejecting any narrative or storytelling element. The acoustic profile shows zero narrative pacing or story structure, sounding technical and utterly non-narrative. The delivery is maximally resistant to narrative conventions. |
| 1 | The delivery is extremely non-narrative and highly non-storybook, with only the faintest hint of narrative element. The acoustic profile is dominated by technical, expository phrasing and completely avoids storybook cadence or narrative pacing. The narrative content is almost entirely absent. |
| 2 | The delivery is primarily non-narrative with subtle narrative elements, occasionally adopting storybook phrasing but maintaining a predominantly technical, non-narrative tone. The acoustic profile is still heavily weighted toward non-narrative speech but includes minor signals of narrative awareness. |
| 3 | The delivery strikes a balanced middle ground between narrative and non-narrative, featuring standard phrasing that is neither stiffly technical nor distinctly storybook-like. The acoustic profile is naturally and functionally narrative without excess. |
| 4 | The delivery is distinctly narrative and storybook-oriented, featuring clear narrative pacing, storybook cadence, and a clearly story-focused structure. The speaker sounds engaged with storytelling, utilizing narrative conventions naturally. The acoustic profile is clearly narrative but not excessively theatrical. |
| 5 | The delivery is highly narrative and exceptionally storybook-oriented, featuring impressive narrative structure, rich storybook pacing, and a distinctly story-focused, absorbing register. The speaker sounds deeply engaged with storytelling, committed to narrative conventions. The acoustic profile is maximally narrative. |
| 6 | The delivery is ultra-narrative and intensely storybook-like, sounding exactly like a professional audiobook narrator with maximum storybook affectation. The speaker maintains extreme narrative pacing, uses rich storybook phrasing and cadence throughout. The acoustic profile is maximally narrative to the point of excess. |

### `S_NEWS` -- Newsreader Style

_Measures the degree of broadcast news, authoritative, report-focused delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely non-journalistic and completely non-broadcast, entirely rejecting any newsreader or report-focused element. The acoustic profile shows zero broadcast authority or journalistic structure, sounding conversational and utterly non-professional. The delivery is maximally resistant to newsreader conventions. |
| 1 | The delivery is extremely non-journalistic and highly non-broadcast, with only the faintest hint of newsreader element. The acoustic profile is dominated by casual, conversational phrasing and completely avoids authoritative broadcast cadence or journalistic structure. The newsreader content is almost entirely absent. |
| 2 | The delivery is primarily non-journalistic with subtle newsreader elements, occasionally adopting broadcast phrasing but maintaining a predominantly conversational, non-journalistic tone. The acoustic profile is still heavily weighted toward casual speech but includes minor signals of broadcast awareness. |
| 3 | The delivery strikes a balanced middle ground between journalistic and non-journalistic, featuring standard phrasing that is neither stiffly authoritative nor entirely conversational. The acoustic profile is naturally and functionally newsreader without excess. |
| 4 | The delivery is distinctly journalistic and broadcast-oriented, featuring clear newsreader authority, professional structure, and a clearly serious, report-focused tone. The speaker sounds competent, credible, and committed to professional broadcast standards. The acoustic profile is clearly newsreader but not excessively rigid. |
| 5 | The delivery is highly journalistic and exceptionally broadcast-oriented, featuring impressive newsreader authority, polished professional structure, and a distinctly authoritative, credible register. The speaker sounds deeply professional, highly credible, and committed to broadcast excellence. The acoustic profile is maximally newsreader. |
| 6 | The delivery is ultra-journalistic and intensely broadcast-like, sounding exactly like a classic network news anchor with maximum authoritative affectation. The speaker maintains extreme newsreader pacing and authority, uses polished, elevated broadcast phrasing throughout. The acoustic profile is maximally newsreader to the point of caricature. |

### `S_TECH` -- Teacher/Didactic Style

_Measures the degree of instructional, educational, teaching-focused delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely non-educational and completely non-didactic, entirely rejecting any teaching or instructional element. The acoustic profile shows zero pedagogical structure or teaching clarity, sounding spontaneous and utterly non-instructional. The delivery is maximally resistant to teaching conventions. |
| 1 | The delivery is extremely non-educational and highly non-didactic, with only the faintest hint of teaching element. The acoustic profile is dominated by casual phrasing and completely avoids pedagogical clarity or instructional structure. The teaching content is almost entirely absent. |
| 2 | The delivery is primarily non-educational with subtle teaching elements, occasionally adopting instructional phrasing but maintaining a predominantly non-didactic tone. The acoustic profile is still heavily weighted toward casual speech but includes minor signals of teaching awareness. |
| 3 | The delivery strikes a balanced middle ground between didactic and non-didactic, featuring standard phrasing that is neither stiffly instructional nor entirely casual. The acoustic profile is naturally and functionally didactic without excess. |
| 4 | The delivery is distinctly pedagogical and teaching-oriented, featuring clear instructional structure, teaching clarity, and a clearly educational, explanatory tone. The speaker sounds competent, patient, and committed to clear explanation. The acoustic profile is clearly didactic but not excessively rigid. |
| 5 | The delivery is highly pedagogical and exceptionally teaching-oriented, featuring impressive instructional structure, excellent teaching clarity, and a distinctly patient, educational register. The speaker sounds deeply professional at teaching, highly adept at explanation. The acoustic profile is maximally didactic. |
| 6 | The delivery is ultra-pedagogical and intensely teaching-like, sounding exactly like a master teacher with maximum instructional affectation. The speaker maintains extreme pedagogical pacing and clarity, uses highly structured, educational phrasing throughout. The acoustic profile is maximally didactic to the point of over-explanation. |

### `S_AUTH` -- Authoritative Style

_Measures the degree of commanding, powerful, leadership-focused delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely non-authoritative and completely submissive, entirely rejecting any commanding or leadership element. The acoustic profile shows zero authority or commanding presence, sounding meek and utterly non-powerful. The delivery is maximally resistant to authoritative conventions. |
| 1 | The delivery is extremely non-authoritative and highly submissive, with only the faintest hint of authority. The acoustic profile is dominated by deferential phrasing and completely avoids commanding presence or leadership tone. The authoritative content is almost entirely absent. |
| 2 | The delivery is primarily non-authoritative with subtle authoritative elements, occasionally adopting commanding phrasing but maintaining a predominantly deferential tone. The acoustic profile is still heavily weighted toward submission but includes minor signals of authority. |
| 3 | The delivery strikes a balanced middle ground between authoritative and non-authoritative, featuring standard phrasing that is neither stiffly commanding nor entirely submissive. The acoustic profile is naturally and functionally authoritative without excess. |
| 4 | The delivery is distinctly authoritative and commanding, featuring clear leadership presence, powerful tone, and a clearly confident, commanding register. The speaker sounds competent, powerful, and naturally in command. The acoustic profile is clearly authoritative but not excessively aggressive. |
| 5 | The delivery is highly authoritative and exceptionally commanding, featuring impressive leadership presence, powerful phrasing, and a distinctly dominant, authoritative register. The speaker sounds deeply confident, powerfully in control, committed to clear authority. The acoustic profile is maximally authoritative. |
| 6 | The delivery is ultra-authoritative and intensely commanding, sounding exactly like a supreme leader with maximum commanding affectation. The speaker maintains extreme authority and power, uses highly dominant, imperative phrasing throughout. The acoustic profile is maximally authoritative to the point of tyranny. |

### `S_PLAY` -- Playful Style

_Measures the degree of humorous, lighthearted, fun-focused delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely serious and completely humorless, entirely rejecting any playful or humorous element. The acoustic profile shows zero playfulness or lightheartedness, sounding grave and utterly joyless. The delivery is maximally resistant to playful conventions. |
| 1 | The delivery is extremely serious and highly grave, with only the faintest hint of playfulness. The acoustic profile is dominated by serious phrasing and completely avoids humor or lightheartedness. The playful content is almost entirely absent. |
| 2 | The delivery is primarily serious with subtle playful elements, occasionally adopting humorous phrasing but maintaining a predominantly grave tone. The acoustic profile is still heavily weighted toward seriousness but includes minor signals of playfulness. |
| 3 | The delivery strikes a balanced middle ground between playful and serious, featuring standard phrasing that is neither stiffly grave nor excessively silly. The acoustic profile is naturally and functionally playful without excess. |
| 4 | The delivery is distinctly playful and humorous, featuring clear lightheartedness, fun-focused tone, and a clearly joyful, entertaining register. The speaker sounds engaged with humor, naturally entertaining. The acoustic profile is clearly playful but not excessively silly. |
| 5 | The delivery is highly playful and exceptionally humorous, featuring impressive humor, rich lightheartedness, and a distinctly fun, entertaining register. The speaker sounds deeply engaged with playfulness, committed to entertainment. The acoustic profile is maximally playful. |
| 6 | The delivery is ultra-playful and intensely humorous, sounding exactly like a professional comedian with maximum playfulness affectation. The speaker maintains extreme playfulness throughout, uses rich, comedic phrasing. The acoustic profile is maximally playful to the point of zaniness. |

### `S_CART` -- Cartoonish Style

_Measures the degree of exaggerated, character-voiced, animated delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely naturalistic and completely non-cartoonish, entirely rejecting any exaggerated or character-voiced element. The acoustic profile shows zero cartoon affectation or exaggeration, sounding entirely grounded and realistic. The delivery is maximally resistant to cartoonish conventions. |
| 1 | The delivery is extremely naturalistic and highly grounded, with only the faintest hint of cartoonish element. The acoustic profile is dominated by natural phrasing and completely avoids exaggeration or character voices. The cartoonish content is almost entirely absent. |
| 2 | The delivery is primarily naturalistic with subtle cartoonish elements, occasionally adopting exaggerated phrasing but maintaining a predominantly grounded tone. The acoustic profile is still heavily weighted toward naturalism but includes minor signals of cartoon awareness. |
| 3 | The delivery strikes a balanced middle ground between cartoonish and naturalistic, featuring standard phrasing that is neither stiffly realistic nor excessively exaggerated. The acoustic profile is naturally and functionally cartoonish without excess. |
| 4 | The delivery is distinctly cartoonish and exaggerated, featuring clear character voices, playful exaggeration, and a clearly entertainment-focused, animated register. The speaker sounds engaged with character work, naturally animated. The acoustic profile is clearly cartoonish but not excessively over-the-top. |
| 5 | The delivery is highly cartoonish and exceptionally exaggerated, featuring impressive character voices, rich exaggeration, and a distinctly animated, character-driven register. The speaker sounds deeply engaged with character work, committed to animation. The acoustic profile is maximally cartoonish. |
| 6 | The delivery is ultra-cartoonish and intensely exaggerated, sounding exactly like a professional voice actor with maximum cartoonish affectation. The speaker maintains extreme character voices and exaggeration throughout. The acoustic profile is maximally cartoonish to the point of absurdity. |

### `S_ASMR` -- ASMR Style

_Measures the degree of whispered, intimate, sensory-focused delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely loud and completely non-ASMR, entirely rejecting any whispered or intimate element. The acoustic profile shows zero ASMR characteristics or sensory focus, sounding harsh and utterly non-soothing. The delivery is maximally resistant to ASMR conventions. |
| 1 | The delivery is extremely loud and highly non-intimate, with only the faintest hint of ASMR element. The acoustic profile is dominated by normal to loud phrasing and completely avoids whispered delivery or sensory-focused content. The ASMR content is almost entirely absent. |
| 2 | The delivery is primarily non-ASMR with subtle intimate elements, occasionally adopting quiet phrasing but maintaining a predominantly normal speaking volume. The acoustic profile is still heavily weighted toward standard speech but includes minor signals of ASMR awareness. |
| 3 | The delivery strikes a balanced middle ground between ASMR and non-ASMR, featuring standard phrasing that is neither stiffly loud nor excessively whispered. The acoustic profile is naturally and functionally intimate without excess. |
| 4 | The delivery is distinctly ASMR-oriented and intimate, featuring clear whispered delivery, sensory focus, and a clearly soothing, relaxing register. The speaker sounds engaged with ASMR aesthetics, naturally soothing. The acoustic profile is clearly ASMR but not excessively extreme. |
| 5 | The delivery is highly ASMR-oriented and exceptionally intimate, featuring impressive whispered delivery, rich sensory focus, and a distinctly calming, intimate register. The speaker sounds deeply engaged with ASMR work, committed to relaxation. The acoustic profile is maximally ASMR. |
| 6 | The delivery is ultra-ASMR and intensely whispered, sounding exactly like a professional ASMR artist with maximum sensory focus. The speaker maintains extreme whispered delivery and sensory detail throughout. The acoustic profile is maximally ASMR to the point of meditation. |

### `S_WHIS` -- Whisper-Talk Style

_Measures the degree of whispered, breath-based, stage-whisper delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely loud and completely non-whispered, entirely rejecting any whispered or breath-based element. The acoustic profile shows zero whisper characteristics or breathy quality, sounding harsh and utterly non-whispered. The delivery is maximally resistant to whisper conventions. |
| 1 | The delivery is extremely loud and highly non-breathy, with only the faintest hint of whispered element. The acoustic profile is dominated by normal to loud phrasing and completely avoids whispered delivery or breath sounds. The whispered content is almost entirely absent. |
| 2 | The delivery is primarily non-whispered with subtle whispered elements, occasionally adopting quiet breathy phrasing but maintaining a predominantly normal speaking volume. The acoustic profile is still heavily weighted toward standard speech but includes minor signals of whisper awareness. |
| 3 | The delivery strikes a balanced middle ground between whispered and non-whispered, featuring standard phrasing that is neither stiffly loud nor excessively breathy. The acoustic profile is naturally and functionally whispered without excess. |
| 4 | The delivery is distinctly whispered and breathy, featuring clear whispered delivery, audible breath, and a clearly stage-whisper quality. The speaker sounds engaged with whisper-talk aesthetics, naturally breathy. The acoustic profile is clearly whispered but not excessively extreme. |
| 5 | The delivery is highly whispered and exceptionally breathy, featuring impressive whispered delivery, rich breath presence, and a distinctly intimate, stage-whisper register. The speaker sounds deeply engaged with whisper-talk work. The acoustic profile is maximally whispered. |
| 6 | The delivery is ultra-whispered and intensely breathy, sounding exactly like an extreme stage-whisper with maximum breath presence. The speaker maintains extreme whispered delivery and breath audibility throughout. The acoustic profile is maximally whispered to the point of unintelligibility. |

### `S_RANT` -- Ranting/Angry Style

_Measures the degree of aggressive, emotionally charged, ranting delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely calm and completely non-ranting, entirely rejecting any aggressive or angry element. The acoustic profile shows zero ranting characteristics or emotional charge, sounding serene and utterly non-hostile. The delivery is maximally resistant to ranting conventions. |
| 1 | The delivery is extremely calm and highly non-aggressive, with only the faintest hint of ranting element. The acoustic profile is dominated by serene phrasing and completely avoids angry delivery or emotional aggression. The ranting content is almost entirely absent. |
| 2 | The delivery is primarily calm with subtle ranting elements, occasionally adopting mildly aggressive phrasing but maintaining a predominantly serene tone. The acoustic profile is still heavily weighted toward calm speech but includes minor signals of anger. |
| 3 | The delivery strikes a balanced middle ground between ranting and calm, featuring standard phrasing that is neither stiffly serene nor excessively angry. The acoustic profile is naturally and functionally angry without excess. |
| 4 | The delivery is distinctly ranting and emotionally charged, featuring clear aggressive phrasing, emotional intensity, and a clearly angry, passionate register. The speaker sounds engaged with frustration, naturally emotional. The acoustic profile is clearly ranting but not excessively extreme. |
| 5 | The delivery is highly ranting and exceptionally angry, featuring impressive aggressive delivery, rich emotional charge, and a distinctly furious, passionate register. The speaker sounds deeply engaged with anger, committed to emotional expression. The acoustic profile is maximally ranting. |
| 6 | The delivery is ultra-ranting and intensely angry, sounding exactly like an extreme rant with maximum aggression and emotional charge. The speaker maintains extreme anger and aggression throughout. The acoustic profile is maximally ranting to the point of incoherence. |

### `S_STRY` -- Storytelling Style

_Measures the degree of narrative-focused, plot-driven, entertaining delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely non-narrative and completely non-storytelling, entirely rejecting any plot-driven or narrative element. The acoustic profile shows zero storytelling characteristics or narrative structure, sounding technical and utterly non-narrative. The delivery is maximally resistant to storytelling conventions. |
| 1 | The delivery is extremely non-narrative and highly non-storytelling, with only the faintest hint of narrative element. The acoustic profile is dominated by technical, expository phrasing and completely avoids storytelling cadence or plot structure. The storytelling content is almost entirely absent. |
| 2 | The delivery is primarily non-narrative with subtle storytelling elements, occasionally adopting narrative phrasing but maintaining a predominantly technical, non-narrative tone. The acoustic profile is still heavily weighted toward non-narrative speech but includes minor signals of storytelling awareness. |
| 3 | The delivery strikes a balanced middle ground between narrative and non-narrative, featuring standard phrasing that is neither stiffly technical nor distinctly story-driven. The acoustic profile is naturally and functionally narrative without excess. |
| 4 | The delivery is distinctly narrative and storytelling-oriented, featuring clear plot structure, storytelling cadence, and a clearly narrative-focused, entertaining register. The speaker sounds engaged with storytelling, naturally engaging. The acoustic profile is clearly narrative but not excessively theatrical. |
| 5 | The delivery is highly narrative and exceptionally storytelling-oriented, featuring impressive plot structure, rich storytelling pacing, and a distinctly narrative, engaging register. The speaker sounds deeply engaged with storytelling, committed to entertainment. The acoustic profile is maximally narrative. |
| 6 | The delivery is ultra-narrative and intensely storytelling-like, sounding exactly like a master storyteller with maximum narrative affectation. The speaker maintains extreme storytelling pacing and structure throughout. The acoustic profile is maximally narrative to the point of excess. |

### `S_MONO` -- Monologue Style

_Measures the degree of solo, introspective, one-person-show delivery_

| Level | Description |
|:-----:|:------------|
| 0 | The delivery is intensely dialogic and completely non-monologic, entirely rejecting any solo or introspective element. The acoustic profile shows zero monologue characteristics or introspective quality, sounding interactive and utterly non-monologic. The delivery is maximally resistant to monologue conventions. |
| 1 | The delivery is extremely dialogic and highly non-monologic, with only the faintest hint of monologue element. The acoustic profile is dominated by interactive phrasing and completely avoids solo delivery or introspective structure. The monologue content is almost entirely absent. |
| 2 | The delivery is primarily dialogic with subtle monologic elements, occasionally adopting introspective phrasing but maintaining a predominantly interactive tone. The acoustic profile is still heavily weighted toward dialogue but includes minor signals of monologue awareness. |
| 3 | The delivery strikes a balanced middle ground between monologic and dialogic, featuring standard phrasing that is neither stiffly solo nor excessively interactive. The acoustic profile is naturally and functionally monologic without excess. |
| 4 | The delivery is distinctly monologic and introspective, featuring clear solo structure, introspective phrasing, and a clearly one-person-show register. The speaker sounds engaged with monologue aesthetics, naturally introspective. The acoustic profile is clearly monologic but not excessively self-focused. |
| 5 | The delivery is highly monologic and exceptionally introspective, featuring impressive solo structure, rich introspective phrasing, and a distinctly solitary, introspective register. The speaker sounds deeply engaged with monologue work. The acoustic profile is maximally monologic. |
| 6 | The delivery is ultra-monologic and intensely introspective, sounding exactly like an extreme one-person-show with maximum introspective focus. The speaker maintains extreme monologic delivery and introspection throughout. The acoustic profile is maximally monologic to the point of solipsism. |

---

*Generated from `voicenet_ext_taxonomy.html`.*