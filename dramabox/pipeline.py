"""Pipeline orchestrator for all DramaBox operating modes.

Mode 1: Generate prompts only           -> CSV files
Mode 2: Synthesize audio                -> WAV files from existing CSV
Mode 3: End-to-end                      -> CSV + WAV files
Mode 4: Reference audio pipeline (D)    -> prompts + audio using reference
Mode 5: Demo grid                       -> HTML comparison grids
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
    """Mode 4: Path D — Reference audio pipeline.

    For each reference audio:
    1. Load metadata (timbre caption, etc.)
    2. Generate timbre caption on-the-fly if missing
    3. Sample situation-dependent VoiceNet dims + emotions
    4. Generate DramaBox prompt via LLM
    5. Synthesize audio with DramaBox TTS (text-only, NO voice_ref)
    6. Voice-convert generated audio to match reference via Chatterbox VC
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
    print("  PATH D: REFERENCE AUDIO PIPELINE", flush=True)
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

    print(f"\nPath D pipeline complete. {len(results)} samples saved to {outdir}", flush=True)
    return outdir


def run_demo_pipeline(config: dict) -> Path:
    """Mode 6: Generate demo grids.

    1. Pick N Emolia references
    2. For each reference, generate samples across M configurations:
       - Path D (reference + sampled dims) with different emotion/attribute combos
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
        "path_d_default",
        "path_d_high_tempo",
        "path_d_low_tempo",
        "path_d_high_emotion",
        "path_d_scattered_flow",
        "path_d_flowing",
        "path_d_german",
        "path_d_french",
        "path_d_spanish",
        "path_d_multi_emotion",
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

    if config_name == "path_d_high_tempo":
        overrides["sampling"] = {"tempo_bias_threshold": 1, "tempo_bias_weight": 5.0}
    elif config_name == "path_d_low_tempo":
        overrides["sampling"] = {"tempo_bias_threshold": 6, "tempo_bias_weight": 0.2}
    elif config_name == "path_d_high_emotion":
        overrides["sampling"] = {"emotions_min": 3, "emotions_max": 3}
    elif config_name == "path_d_scattered_flow":
        overrides["sampling"] = {"flow_style_distribution": {"scattered": 1.0, "flowing": 0.0, "mixed": 0.0}}
    elif config_name == "path_d_flowing":
        overrides["sampling"] = {"flow_style_distribution": {"scattered": 0.0, "flowing": 1.0, "mixed": 0.0}}
    elif config_name == "path_d_german":
        overrides["_active_languages"] = ["German"]
        overrides["_language_accents"] = {}
    elif config_name == "path_d_french":
        overrides["_active_languages"] = ["French"]
        overrides["_language_accents"] = {}
    elif config_name == "path_d_spanish":
        overrides["_active_languages"] = ["Spanish"]
        overrides["_language_accents"] = {}
    elif config_name == "path_d_multi_emotion":
        overrides["sampling"] = {"emotions_min": 2, "emotions_max": 3}

    return overrides


# ─── Full 4-Path Demo Pipeline ───────────────────────────────────────────────

def run_full_demo(config: dict) -> Path:
    """Generate a comprehensive demo grid for all 4 paths (A, B, C, D).

    5-phase pipeline with sequential GPU loading to stay within VRAM:

    Phase 1: Generate all prompts
      GPU 6: LLM (Gemma) for Path A/B/C prompts
      GPU 6: Timbre Whisper for Path D reference analysis
      -> Unload LLM, Timbre Whisper

    Phase 2: TTS synthesis (all paths, text-only)
      GPU 6: DramaBox TTS for all prompts x N candidates
      Path A/B/C also get self-VC pass
      -> Unload TTS

    Phase 3: Voice conversion (Path D only)
      GPU 7: Chatterbox VC to match reference voice
      -> Unload VC

    Phase 4: Scoring + Best-of-N ranking
      GPU 7: Parakeet ASR + Empathic Insight Plus
      -> Unload scoring models

    Phase 5: Generate self-contained HTML
      Convert best audio to base64 MP3, build standalone HTML

    Returns the demo output directory.
    """
    from .prompt_generator import load_llm, generate_single_prompt, unload_llm
    from .tts_synthesizer import load_tts_server, synthesize_prompt, unload_tts_server
    from .audio_refine import voice_convert, voice_convert_batch, unload_all as unload_refine
    from .scoring import score_audio, best_of_n, unload_all as unload_scoring
    from .sampling import sample_voicenet, sample_archetype
    from .reference_sampling import (
        load_reference_metadata, get_situation_dependent_dims,
        sample_reference_path, build_reference_full_prompt,
        SYSTEM_INSTRUCTION_PATHC,
    )
    from .prompts import SYSTEM_INSTRUCTION, build_full_prompt
    from .taxonomy import (
        parse_voicenet_html, load_vocal_bursts, load_archetypes,
        format_vocal_bursts_block, load_emonet,
    )
    from .timbre_whisper import ensure_timbre_caption
    from .wordlists import get_word_list
    from .demo_grid import generate_full_demo_html

    demo_cfg = config.get("demo", {})
    dp = config.get("data_paths", {})
    sampling_cfg = config.get("sampling", {})
    bon_cfg = config.get("best_of_n", {})

    n_prompts = demo_cfg.get("n_prompts_per_path", 10)
    best_of_n_count = demo_cfg.get("best_of_n", bon_cfg.get("n_candidates", 3))
    gpus = demo_cfg.get("gpus", [6, 7])
    gpu_llm = f"cuda:{gpus[0]}"
    gpu_moss = f"cuda:{gpus[1]}" if len(gpus) > 1 else gpu_llm
    emolia_dir = Path(demo_cfg.get("emolia_dir",
        "/run/user/1001/speaker_encoder_dataset/emolia_references/cluster_best"))
    outdir = Path(demo_cfg.get("output_dir", "/tmp/dramabox_demo"))
    outdir.mkdir(parents=True, exist_ok=True)

    seed = config.get("prompt_generation", {}).get("seed", 42)
    random.seed(seed)

    print("=" * 72, flush=True)
    print("  FULL 4-PATH DEMO GRID", flush=True)
    print(f"  Prompts per path : {n_prompts}", flush=True)
    print(f"  Best-of-N        : {best_of_n_count}", flush=True)
    print(f"  GPUs             : {gpus}", flush=True)
    print(f"  Emolia dir       : {emolia_dir}", flush=True)
    print(f"  Output dir       : {outdir}", flush=True)
    print("=" * 72, flush=True)

    # Load taxonomies
    all_dims = parse_voicenet_html(Path(dp["voicenet_html"]))
    mandatory_dim_codes = set(sampling_cfg.get("mandatory_dims", ["TEMP", "GEND", "AGEV"]))
    mandatory_dims = [d for d in all_dims if d["code"] in mandatory_dim_codes]
    optional_dims = [d for d in all_dims if d["code"] not in mandatory_dim_codes]
    situation_dims = get_situation_dependent_dims(all_dims)
    temp_dim = next(d for d in all_dims if d["code"] == "TEMP")
    arou_dim = next(d for d in all_dims if d["code"] == "AROU")

    vb_taxonomy = load_vocal_bursts(Path(dp["vocal_bursts_json"]))
    vb_block = format_vocal_bursts_block(vb_taxonomy)
    archetypes = load_archetypes(Path(dp["archetypes_json"]))
    emonet = load_emonet(Path(dp["emonet_json"]))
    emotion_categories = list(emonet.keys())
    wordlists_dir = Path(dp.get("wordlists_dir", "data/wordlists"))

    def _wordlist_fn(language):
        return get_word_list(language, wordlists_dir)

    # Select reference audio files for Path D
    ref_jsons = sorted(emolia_dir.glob("*.json"))
    ref_selected = random.sample(ref_jsons, min(n_prompts, len(ref_jsons)))

    # Storage for all paths
    path_results = {
        "A": [],  # VoiceNet sampling
        "B": [],  # Archetype sampling
        "C": [],  # Archetype sampling (named)
        "D": [],  # Reference audio + VC
    }

    # ═══ Phase 1: Generate all prompts ═══════════════════════════════════════
    print(f"\n{'='*72}", flush=True)
    print("  PHASE 1: PROMPT GENERATION", flush=True)
    print(f"{'='*72}", flush=True)

    t_phase1 = time.time()

    # Path A: VoiceNet sampling + LLM
    print(f"\n  Path A: Generating {n_prompts} VoiceNet prompts on {gpu_llm}...", flush=True)
    for i in range(n_prompts):
        sample = sample_voicenet(mandatory_dims, optional_dims,
                                 emotion_categories, config,
                                 wordlist_fn=_wordlist_fn)
        user_prompt = build_full_prompt(sample, vb_block)
        dramabox_text = generate_single_prompt(
            SYSTEM_INSTRUCTION, user_prompt, config, device=gpu_llm,
        )
        path_results["A"].append({
            "idx": i,
            "sample": sample,
            "system_prompt": SYSTEM_INSTRUCTION,
            "user_prompt": user_prompt,
            "dramabox_prompt": dramabox_text,
        })
        print(f"    [{i+1}/{n_prompts}] Path A: {sample['language']}, "
              f"{len(dramabox_text)} chars", flush=True)

    # Path B: Archetype sampling + LLM
    print(f"\n  Path B: Generating {n_prompts} Archetype prompts on {gpu_llm}...", flush=True)
    for i in range(n_prompts):
        sample = sample_archetype(archetypes, temp_dim, arou_dim,
                                  emotion_categories, config)
        user_prompt = build_full_prompt(sample, vb_block)
        dramabox_text = generate_single_prompt(
            SYSTEM_INSTRUCTION, user_prompt, config, device=gpu_llm,
        )
        path_results["B"].append({
            "idx": i,
            "sample": sample,
            "system_prompt": SYSTEM_INSTRUCTION,
            "user_prompt": user_prompt,
            "dramabox_prompt": dramabox_text,
        })
        print(f"    [{i+1}/{n_prompts}] Path B: {sample.get('archetype_info', '')[:40]}, "
              f"{len(dramabox_text)} chars", flush=True)

    # Path D: Reference audio + Timbre Whisper + LLM
    print(f"\n  Path D: Generating {n_prompts} Reference prompts on {gpu_llm}...", flush=True)
    for i, ref_json in enumerate(ref_selected[:n_prompts]):
        ref_mp3 = ref_json.with_suffix(".mp3")
        if not ref_mp3.exists():
            ref_mp3 = ref_json.with_suffix(".wav")
        if not ref_mp3.exists():
            print(f"    [{i+1}/{n_prompts}] Skipping {ref_json.stem}: no audio", flush=True)
            continue

        metadata = load_reference_metadata(ref_json)
        timbre_caption = ensure_timbre_caption(metadata, ref_mp3, device=gpu_llm)

        sample = sample_reference_path(
            timbre_caption=timbre_caption,
            situation_dims=situation_dims,
            emotion_categories=emotion_categories,
            config=config,
            reference_audio_path=str(ref_mp3),
            reference_metadata=metadata,
        )
        user_prompt = build_reference_full_prompt(sample)
        dramabox_text = generate_single_prompt(
            SYSTEM_INSTRUCTION_PATHC, user_prompt, config, device=gpu_llm,
        )

        path_results["D"].append({
            "idx": i,
            "ref_id": ref_json.stem,
            "ref_audio": str(ref_mp3),
            "timbre_caption": timbre_caption,
            "sample": sample,
            "system_prompt": SYSTEM_INSTRUCTION_PATHC,
            "user_prompt": user_prompt,
            "dramabox_prompt": dramabox_text,
        })
        print(f"    [{i+1}/{n_prompts}] Path D: ref={ref_json.stem}, "
              f"{len(dramabox_text)} chars", flush=True)

    # Unload LLM + Timbre Whisper from GPU 6
    print("  Unloading LLM from GPU...", flush=True)
    unload_llm()

    print(f"\n  Phase 1 complete in {time.time()-t_phase1:.1f}s", flush=True)
    print(f"  Path A: {len(path_results['A'])} prompts", flush=True)
    print(f"  Path B: {len(path_results['B'])} prompts", flush=True)
    print(f"  Path C: {len(path_results['C'])} prompts", flush=True)
    print(f"  Path D: {len(path_results['D'])} prompts", flush=True)

    # Save prompts checkpoint
    _save_checkpoint(path_results, outdir / "phase1_prompts.json")

    # ═══ Phase 2: TTS synthesis ══════════════════════════════════════════════
    print(f"\n{'='*72}", flush=True)
    print("  PHASE 2: TTS SYNTHESIS", flush=True)
    print(f"{'='*72}", flush=True)

    t_phase2 = time.time()
    tts_device = gpu_llm  # Reuse GPU 6 for TTS
    server = load_tts_server(config, device=tts_device)

    audio_dir = outdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for path_key in ["A", "B", "C", "D"]:
        items = path_results[path_key]
        n_items = len(items)
        needs_self_vc = path_key in ("A", "B", "C")

        print(f"\n  Path {path_key}: {n_items} prompts x {best_of_n_count} candidates"
              f"{' + self-VC' if needs_self_vc else ''}...", flush=True)

        for item_idx, item in enumerate(items):
            prompt = item["dramabox_prompt"]
            if not prompt:
                continue

            candidates = []
            for n in range(best_of_n_count):
                raw_path = audio_dir / f"path{path_key}_{item_idx:03d}_n{n}_raw.wav"
                gen_seed = seed + item_idx * 100 + n

                try:
                    synthesize_prompt(
                        prompt=prompt,
                        output_path=raw_path,
                        server=server,
                        config=config,
                        seed=gen_seed,
                        voice_ref=None,  # Text-only for all paths
                    )

                    candidate = {"raw": str(raw_path)}

                    # Self-VC for Path A/B only
                    if needs_self_vc:
                        vc_path = audio_dir / f"path{path_key}_{item_idx:03d}_n{n}_selfvc.wav"
                        synthesize_prompt(
                            prompt=prompt,
                            output_path=vc_path,
                            server=server,
                            config=config,
                            seed=gen_seed,
                            voice_ref=str(raw_path),
                        )
                        candidate["selfvc"] = str(vc_path)
                        candidate["audio_for_scoring"] = str(vc_path)
                    else:
                        candidate["audio_for_scoring"] = str(raw_path)

                    candidates.append(candidate)
                except Exception as e:
                    print(f"    Path {path_key}[{item_idx}] n={n} TTS ERROR: {e}", flush=True)

            item["candidates"] = candidates
            print(f"    [{item_idx+1}/{n_items}] Path {path_key}: "
                  f"{len(candidates)} candidates generated", flush=True)

    # Unload TTS
    print("  Unloading TTS from GPU...", flush=True)
    unload_tts_server()

    print(f"\n  Phase 2 complete in {time.time()-t_phase2:.1f}s", flush=True)
    _save_checkpoint(path_results, outdir / "phase2_tts.json")

    # ═══ Phase 3: Voice conversion (Path D only) ════════════════════════════
    print(f"\n{'='*72}", flush=True)
    print("  PHASE 3: VOICE CONVERSION (Path D)", flush=True)
    print(f"{'='*72}", flush=True)

    t_phase3 = time.time()
    vc_device = gpu_moss  # Use GPU 7 for VC
    vc_dir = outdir / "audio_vc"
    vc_dir.mkdir(parents=True, exist_ok=True)

    for path_key in ["D"]:
        items = path_results[path_key]
        print(f"\n  Path {path_key}: {len(items)} prompts, VC to reference voice...", flush=True)

        for item_idx, item in enumerate(items):
            ref_audio = item.get("ref_audio", "")
            if not ref_audio or not Path(ref_audio).exists():
                print(f"    [{item_idx+1}] No reference audio, skipping VC", flush=True)
                continue

            for cand in item.get("candidates", []):
                raw_path = cand.get("raw", "")
                if not raw_path or not Path(raw_path).exists():
                    continue

                vc_out = vc_dir / f"path{path_key}_{item_idx:03d}_{Path(raw_path).stem}_vc.wav"
                try:
                    voice_convert(
                        source_audio=raw_path,
                        reference_audio=ref_audio,
                        output_path=vc_out,
                        device=vc_device,
                    )
                    cand["vc"] = str(vc_out)
                    cand["audio_for_scoring"] = str(vc_out)
                except Exception as e:
                    print(f"    VC ERROR for {Path(raw_path).name}: {e}", flush=True)

            print(f"    [{item_idx+1}/{len(items)}] Path {path_key}: VC done", flush=True)

    # Unload VC
    print("  Unloading VC from GPU...", flush=True)
    unload_refine()

    print(f"\n  Phase 3 complete in {time.time()-t_phase3:.1f}s", flush=True)
    _save_checkpoint(path_results, outdir / "phase3_vc.json")

    # ═══ Phase 4: Scoring + Best-of-N ranking ════════════════════════════════
    print(f"\n{'='*72}", flush=True)
    print("  PHASE 4: SCORING + BEST-OF-N RANKING", flush=True)
    print(f"{'='*72}", flush=True)

    t_phase4 = time.time()
    score_device = gpu_moss  # Use GPU 7 for scoring

    for path_key in ["A", "B", "C", "D"]:
        items = path_results[path_key]
        print(f"\n  Path {path_key}: Scoring {len(items)} prompts...", flush=True)

        for item_idx, item in enumerate(items):
            prompt = item.get("dramabox_prompt", "")
            candidates = item.get("candidates", [])

            # Collect audio paths for scoring
            audio_paths = []
            for cand in candidates:
                score_path = cand.get("audio_for_scoring", "")
                if score_path and Path(score_path).exists():
                    audio_paths.append(score_path)

            if not audio_paths:
                item["best"] = None
                continue

            try:
                bon_result = best_of_n(audio_paths, prompt, device=score_device)
                item["best"] = bon_result
                item["best_audio"] = bon_result["best_path"]
                item["all_scores"] = bon_result["all_scores"]

                best_score = bon_result["best_score"]
                print(f"    [{item_idx+1}/{len(items)}] Path {path_key}: "
                      f"best={bon_result['best_idx']}, "
                      f"reward={best_score.get('reward', 0):.3f}, "
                      f"wer={best_score.get('wer', 0):.3f}, "
                      f"enjoy={best_score.get('content_enjoyment', 0):.3f}",
                      flush=True)
            except Exception as e:
                print(f"    [{item_idx+1}/{len(items)}] Scoring ERROR: {e}", flush=True)
                item["best"] = None

    # Unload scoring models
    print("  Unloading scoring models from GPU...", flush=True)
    unload_scoring()

    print(f"\n  Phase 4 complete in {time.time()-t_phase4:.1f}s", flush=True)
    _save_checkpoint(path_results, outdir / "phase4_scored.json")

    # ═══ Phase 5: Generate self-contained HTML ═══════════════════════════════
    print(f"\n{'='*72}", flush=True)
    print("  PHASE 5: HTML GENERATION", flush=True)
    print(f"{'='*72}", flush=True)

    html_path = outdir / "demo_grid_full.html"
    generate_full_demo_html(
        path_results=path_results,
        output_path=html_path,
        title=f"DramaBox Demo Grid — All 4 Paths",
    )

    # Save final results JSON
    _save_checkpoint(path_results, outdir / "full_demo_results.json")

    print(f"\n{'='*72}", flush=True)
    print("  FULL DEMO COMPLETE", flush=True)
    print(f"  HTML          : {html_path}", flush=True)
    print(f"  Results JSON  : {outdir / 'full_demo_results.json'}", flush=True)
    print(f"  Audio dir     : {audio_dir}", flush=True)
    n_total = sum(len(v) for v in path_results.values())
    n_best = sum(1 for v in path_results.values()
                 for item in v if item.get("best_audio"))
    print(f"  Total prompts : {n_total}", flush=True)
    print(f"  Best-of-N wins: {n_best}", flush=True)
    print(f"{'='*72}", flush=True)

    return outdir


def _save_checkpoint(path_results: dict, path: Path):
    """Save path_results as a JSON checkpoint (skip non-serializable fields)."""
    import copy

    serializable = {}
    for key, items in path_results.items():
        serializable[key] = []
        for item in items:
            clean = {}
            for k, v in item.items():
                if k == "sample":
                    # sample dicts may have non-serializable items, stringify them
                    clean[k] = {sk: str(sv) if not isinstance(sv, (str, int, float, bool, list, dict, type(None))) else sv
                                for sk, sv in v.items()}
                else:
                    clean[k] = v
            serializable[key].append(clean)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Checkpoint saved: {path}", flush=True)
