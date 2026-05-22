"""Pipeline orchestrator for all DramaBox operating modes.

Mode 1: Generate prompts only           -> CSV files
Mode 2: Synthesize audio                -> WAV files from existing CSV
Mode 3: End-to-end                      -> CSV + WAV files
Mode 4: Reference audio pipeline (C)    -> prompts + audio using reference
Mode 5: MOSS Audio thinking (D)         -> prompts + audio via MOSS reasoning
Mode 6: Demo grid                       -> HTML comparison grids
"""
import glob
import json
import logging
import os
import random
import time
from pathlib import Path

log = logging.getLogger(__name__)


def run_mode1(config: dict) -> Path:
    """Mode 1: Generate DramaBox prompts to CSV chunks.

    Returns the output directory containing CSV files.
    """
    from .prompt_generator import generate_all_prompts
    return generate_all_prompts(config)


def run_mode2(config: dict, csv_path: str) -> Path:
    """Mode 2: Synthesize audio from an existing CSV file.

    Args:
        config: Full configuration dict.
        csv_path: Path to CSV file with dramabox_prompt column.

    Returns the output directory containing WAV files.
    """
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    from .tts_synthesizer import synthesize_from_csv
    return synthesize_from_csv(csv_path, config)


def run_mode3(config: dict) -> Path:
    """Mode 3: End-to-end pipeline — generate prompts then synthesize audio.

    1. Generates prompts to CSV chunks (same as Mode 1).
    2. Synthesizes audio from each chunk (same as Mode 2).

    Returns the output directory.
    """
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # Step 1: Generate prompts
    outdir = run_mode1(config)

    # Step 2: Synthesize audio from all generated chunks
    from .tts_synthesizer import synthesize_from_csv

    csv_pattern = str(outdir / "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"No CSV files found in {outdir}", flush=True)
        return outdir

    print(f"\n{'='*72}", flush=True)
    print(f"  Starting TTS synthesis for {len(csv_files)} CSV chunks...", flush=True)
    print(f"{'='*72}", flush=True)

    for csv_file in csv_files:
        print(f"\nSynthesizing: {csv_file}", flush=True)
        synthesize_from_csv(csv_file, config)

    print(f"\nEnd-to-end pipeline complete. Output: {outdir}", flush=True)
    return outdir


def run_reference_pipeline(config: dict, ref_dir: str, total: int) -> Path:
    """Mode 4: Path C — Reference audio pipeline.

    For each reference audio:
    1. Load metadata (timbre caption, etc.)
    2. Generate timbre caption on-the-fly if missing
    3. Sample situation-dependent VoiceNet dims + emotions
    4. Generate DramaBox prompt via LLM
    5. Synthesize audio with DramaBox TTS using reference
    6. Optionally refine with Resemble Enhance + Chatterbox VC
    7. Optionally score and rank with Best-of-N

    Args:
        config: Full configuration dict.
        ref_dir: Directory with .json + .mp3 reference pairs.
        total: Number of samples to generate.

    Returns the output directory.
    """
    from .reference_sampling import (
        load_reference_metadata, get_situation_dependent_dims,
        sample_reference_path, build_reference_full_prompt,
        SYSTEM_INSTRUCTION_PATHC,
    )
    from .taxonomy import parse_voicenet_html, load_emonet
    from .timbre_whisper import ensure_timbre_caption

    dp = config.get("data_paths", {})
    output_cfg = config.get("output", {})
    ref_cfg = config.get("reference_audio", {})
    outdir = Path(output_cfg.get("output_dir", "./output")) / "reference"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("  PATH C: REFERENCE AUDIO PIPELINE", flush=True)
    print(f"  Reference dir: {ref_dir}", flush=True)
    print(f"  Total samples: {total}", flush=True)
    print(f"  Output dir   : {outdir}", flush=True)
    print("=" * 72, flush=True)

    # Load taxonomies
    all_dims = parse_voicenet_html(Path(dp["voicenet_html"]))
    situation_dims = get_situation_dependent_dims(all_dims)
    emonet = load_emonet(Path(dp["emonet_json"]))
    emotion_categories = list(emonet.keys())

    print(f"  Situation-dependent dims: {len(situation_dims)}", flush=True)
    print(f"  Emotions: {len(emotion_categories)}", flush=True)

    # Find reference files
    ref_dir = Path(ref_dir)
    ref_jsons = sorted(ref_dir.glob("*.json"))
    if not ref_jsons:
        print(f"No .json files found in {ref_dir}", flush=True)
        return outdir

    print(f"  Available references: {len(ref_jsons)}", flush=True)

    # Sample references for this run
    if total < len(ref_jsons):
        selected_refs = random.sample(ref_jsons, total)
    else:
        selected_refs = ref_jsons[:total]

    # Process each reference
    results = []
    for i, ref_json in enumerate(selected_refs):
        ref_id = ref_json.stem
        ref_mp3 = ref_json.with_suffix(".mp3")
        if not ref_mp3.exists():
            print(f"  [{i+1}/{len(selected_refs)}] Skipping {ref_id}: no .mp3 found", flush=True)
            continue

        print(f"  [{i+1}/{len(selected_refs)}] Processing {ref_id}...", flush=True)

        # Load metadata
        metadata = load_reference_metadata(ref_json)

        # Get or generate timbre caption
        timbre_caption = ensure_timbre_caption(
            metadata, ref_mp3,
            device=f"cuda:{config.get('demo', {}).get('gpus', [0])[0]}",
        )
        print(f"    Timbre: {timbre_caption[:80]}...", flush=True)

        # Sample attributes
        sample = sample_reference_path(
            timbre_caption=timbre_caption,
            situation_dims=situation_dims,
            emotion_categories=emotion_categories,
            config=config,
            reference_audio_path=str(ref_mp3),
            reference_metadata=metadata,
        )

        # Build prompt
        full_prompt = build_reference_full_prompt(sample)

        # Save sample metadata
        sample_dir = outdir / f"ref_{ref_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_meta = {
            "ref_id": ref_id,
            "ref_audio": str(ref_mp3),
            "timbre_caption": timbre_caption,
            "sampling_path": "reference",
            "language": sample["language"],
            "accent": sample["accent"],
            "emotions": sample["emotions"],
            "attributes_clean": sample["attributes_clean"],
            "word_count_target": sample["word_count_target"],
            "system_prompt": SYSTEM_INSTRUCTION_PATHC,
            "user_prompt": full_prompt,
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=2, ensure_ascii=False)

        results.append(sample_meta)
        print(f"    Language: {sample['language']}, Emotions: {sample['emotions']}", flush=True)

    # Save all results summary
    with open(outdir / "reference_samples.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nPath C pipeline complete. {len(results)} samples saved to {outdir}", flush=True)
    return outdir


def run_moss_pipeline(config: dict, ref_dir: str, total: int) -> Path:
    """Mode 5: Path D — MOSS Audio Thinking pipeline.

    For each reference audio:
    1. MOSS Audio listens and reasons about the speaker
    2. Generates a DramaBox prompt with reasoning trace
    3. Saves reasoning + prompt to JSON

    Args:
        config: Full configuration dict.
        ref_dir: Directory with reference audio files.
        total: Number of samples to generate.

    Returns the output directory.
    """
    from .moss_pipeline import generate_dramabox_prompt, save_moss_result

    moss_cfg = config.get("moss_audio", {})
    output_cfg = config.get("output", {})
    outdir = Path(output_cfg.get("output_dir", "./output")) / "moss"
    outdir.mkdir(parents=True, exist_ok=True)

    gpus = config.get("demo", {}).get("gpus", [0])
    device = f"cuda:{gpus[0]}"
    moss_dir = moss_cfg.get("moss_dir", "")

    print("=" * 72, flush=True)
    print("  PATH D: MOSS AUDIO THINKING PIPELINE", flush=True)
    print(f"  Reference dir: {ref_dir}", flush=True)
    print(f"  Total samples: {total}", flush=True)
    print(f"  Device       : {device}", flush=True)
    print(f"  Output dir   : {outdir}", flush=True)
    print("=" * 72, flush=True)

    ref_dir = Path(ref_dir)
    audio_files = sorted(ref_dir.glob("*.mp3")) + sorted(ref_dir.glob("*.wav"))
    if not audio_files:
        print(f"No audio files found in {ref_dir}", flush=True)
        return outdir

    selected = audio_files[:total] if total <= len(audio_files) else audio_files

    emotions_pool = [
        "Contemplation (clearly present)",
        "Amusement (slightly present), Warmth (clearly present)",
        "Tension (extremely present), Determination (clearly present)",
        "Sadness (clearly present), Nostalgia (slightly present)",
        "Excitement (very intensely present)",
    ]

    for i, audio_file in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] Analyzing {audio_file.name}...", flush=True)

        emotions = random.choice(emotions_pool)
        language = random.choice(config.get("_active_languages", ["English"]))
        word_count = random.randint(10, 50)

        try:
            result = generate_dramabox_prompt(
                audio_path=audio_file,
                language=language,
                word_count=word_count,
                emotions=emotions,
                device=device,
                moss_dir=moss_dir,
                max_new_tokens=moss_cfg.get("max_new_tokens_dramabox", 2048),
            )
            save_moss_result(result, outdir / f"{audio_file.stem}_moss.json")
            print(f"    Done: {len(result.get('reasoning_trace', ''))} chars reasoning, "
                  f"{len(result.get('dramabox_prompt', ''))} chars prompt", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)

    print(f"\nPath D pipeline complete. Output: {outdir}", flush=True)
    return outdir


def run_demo_pipeline(config: dict) -> Path:
    """Mode 6: Generate demo grids.

    1. Pick N Emolia references
    2. For each reference, generate samples across M configurations:
       - Path C (reference + sampled dims) with different emotion/attribute combos
       - Path D (MOSS Audio thinking)
       - With/without Chatterbox VC
       - With/without Resemble Enhance
    3. Score all samples with Best-of-N ranking
    4. Generate HTML comparison grid

    Returns the demo output directory.
    """
    from .reference_sampling import (
        load_reference_metadata, get_situation_dependent_dims,
        sample_reference_path, build_reference_full_prompt,
        SYSTEM_INSTRUCTION_PATHC,
    )
    from .taxonomy import parse_voicenet_html, load_emonet
    from .timbre_whisper import ensure_timbre_caption
    from .demo_grid import generate_demo_html

    demo_cfg = config.get("demo", {})
    dp = config.get("data_paths", {})

    emolia_dir = Path(demo_cfg.get("emolia_dir",
        "/run/user/1001/speaker_encoder_dataset/emolia_references/cluster_best"))
    n_refs = demo_cfg.get("n_references", 5)
    n_configs = demo_cfg.get("n_configs_per_ref", 10)
    outdir = Path(demo_cfg.get("output_dir", "/tmp/dramabox_demo"))
    gpus = demo_cfg.get("gpus", [6, 7])
    device = f"cuda:{gpus[0]}"

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("  DEMO GRID GENERATION", flush=True)
    print(f"  Emolia dir   : {emolia_dir}", flush=True)
    print(f"  References   : {n_refs}", flush=True)
    print(f"  Configs/ref  : {n_configs}", flush=True)
    print(f"  GPUs         : {gpus}", flush=True)
    print(f"  Output dir   : {outdir}", flush=True)
    print("=" * 72, flush=True)

    # Load taxonomies
    all_dims = parse_voicenet_html(Path(dp["voicenet_html"]))
    situation_dims = get_situation_dependent_dims(all_dims)
    emonet = load_emonet(Path(dp["emonet_json"]))
    emotion_categories = list(emonet.keys())

    # Select references
    ref_jsons = sorted(emolia_dir.glob("*.json"))
    if len(ref_jsons) > n_refs:
        # Pick diverse references (spread across the list)
        step = len(ref_jsons) // n_refs
        selected_refs = [ref_jsons[i * step] for i in range(n_refs)]
    else:
        selected_refs = ref_jsons[:n_refs]

    print(f"  Selected {len(selected_refs)} references", flush=True)

    # Configuration variants for demo
    config_names = [
        "path_c_default",
        "path_c_high_tempo",
        "path_c_low_tempo",
        "path_c_high_emotion",
        "path_c_scattered_flow",
        "path_c_flowing",
        "path_c_german",
        "path_c_french",
        "path_c_spanish",
        "path_c_multi_emotion",
    ][:n_configs]

    grid_data = []

    for ref_idx, ref_json in enumerate(selected_refs):
        ref_id = ref_json.stem
        ref_mp3 = ref_json.with_suffix(".mp3")
        if not ref_mp3.exists():
            print(f"  Skipping {ref_id}: no .mp3", flush=True)
            continue

        metadata = load_reference_metadata(ref_json)
        timbre_caption = ensure_timbre_caption(metadata, ref_mp3, device=device)

        ref_row = {
            "reference": {
                "audio": str(ref_mp3),
                "label": f"Ref {ref_id}",
                "metadata": {
                    "id": metadata.get("id", ref_id),
                    "speaker": metadata.get("speaker", ""),
                    "duration": metadata.get("duration", 0),
                    "timbre": timbre_caption[:100],
                },
            },
            "variants": [],
        }

        ref_outdir = outdir / f"ref_{ref_idx:03d}"
        ref_outdir.mkdir(parents=True, exist_ok=True)

        for cfg_idx, cfg_name in enumerate(config_names):
            print(f"  [{ref_idx+1}/{len(selected_refs)}] Config: {cfg_name}...", flush=True)

            # Vary the config based on variant name
            demo_config = dict(config)
            sample_overrides = _get_demo_config_overrides(cfg_name)

            sample = sample_reference_path(
                timbre_caption=timbre_caption,
                situation_dims=situation_dims,
                emotion_categories=emotion_categories,
                config={**config, **sample_overrides},
                reference_audio_path=str(ref_mp3),
                reference_metadata=metadata,
            )

            prompt = build_reference_full_prompt(sample)

            # Save config metadata
            cfg_dir = ref_outdir / cfg_name
            cfg_dir.mkdir(parents=True, exist_ok=True)

            cfg_meta = {
                "config_name": cfg_name,
                "ref_id": ref_id,
                "language": sample["language"],
                "accent": sample["accent"],
                "emotions": sample["emotions"],
                "attributes_clean": sample["attributes_clean"],
                "word_count_target": sample["word_count_target"],
                "dramabox_prompt": prompt,
                "system_prompt": SYSTEM_INSTRUCTION_PATHC,
            }
            with open(cfg_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg_meta, f, indent=2, ensure_ascii=False)

            ref_row["variants"].append({
                "audio": "",  # Will be filled after TTS
                "label": cfg_name,
                "config": cfg_name,
                "metadata": {
                    "language": sample["language"],
                    "emotions": sample["emotions"][:60],
                },
            })

        grid_data.append(ref_row)

    # Generate HTML grid (without audio for now — prompts only)
    html_path = generate_demo_html(
        grid_data,
        outdir / "demo_grid.html",
        title=f"DramaBox Demo: {n_refs} References x {n_configs} Configs",
    )

    # Save grid data for later TTS processing
    with open(outdir / "grid_data.json", "w", encoding="utf-8") as f:
        json.dump(grid_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDemo grid generated: {html_path}", flush=True)
    print(f"Grid data saved: {outdir / 'grid_data.json'}", flush=True)
    print(f"Total configs: {sum(len(r['variants']) for r in grid_data)}", flush=True)
    return outdir


def _get_demo_config_overrides(config_name: str) -> dict:
    """Get sampling config overrides for a demo configuration variant."""
    overrides = {}

    if config_name == "path_c_high_tempo":
        overrides["sampling"] = {"tempo_bias_threshold": 1, "tempo_bias_weight": 5.0}
    elif config_name == "path_c_low_tempo":
        overrides["sampling"] = {"tempo_bias_threshold": 6, "tempo_bias_weight": 0.2}
    elif config_name == "path_c_high_emotion":
        overrides["sampling"] = {"emotions_min": 3, "emotions_max": 3}
    elif config_name == "path_c_scattered_flow":
        overrides["sampling"] = {"flow_style_distribution": {"scattered": 1.0, "flowing": 0.0, "mixed": 0.0}}
    elif config_name == "path_c_flowing":
        overrides["sampling"] = {"flow_style_distribution": {"scattered": 0.0, "flowing": 1.0, "mixed": 0.0}}
    elif config_name == "path_c_german":
        overrides["_active_languages"] = ["German"]
        overrides["_language_accents"] = {}
    elif config_name == "path_c_french":
        overrides["_active_languages"] = ["French"]
        overrides["_language_accents"] = {}
    elif config_name == "path_c_spanish":
        overrides["_active_languages"] = ["Spanish"]
        overrides["_language_accents"] = {}
    elif config_name == "path_c_multi_emotion":
        overrides["sampling"] = {"emotions_min": 2, "emotions_max": 3}

    return overrides
