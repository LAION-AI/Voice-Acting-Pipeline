"""Command-line interface for the DramaBox pipeline."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="dramabox",
        description="DramaBox voice prompt generation and audio synthesis pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Mode 1: generate-prompts ─────────────────────────────────────────
    p1 = subparsers.add_parser(
        "generate-prompts",
        help="Generate DramaBox prompts to CSV files.",
    )
    p1.add_argument("--config", default="config.json",
                    help="Path to config.json (default: config.json)")
    p1.add_argument("--total", type=int,
                    help="Override total number of prompts to generate")
    p1.add_argument("--gpus", type=str,
                    help="Override GPU list (comma-separated, e.g. 0,1,2,3)")
    p1.add_argument("--seed", type=int,
                    help="Override random seed")
    p1.add_argument("--output-dir", type=str,
                    help="Override output directory")

    # ── Mode 2: synthesize ───────────────────────────────────────────────
    p2 = subparsers.add_parser(
        "synthesize",
        help="Synthesize audio from a CSV file containing DramaBox prompts.",
    )
    p2.add_argument("--csv", required=True,
                    help="Path to CSV file with dramabox_prompt column")
    p2.add_argument("--config", default="config.json",
                    help="Path to config.json (default: config.json)")
    p2.add_argument("--gpus", type=str,
                    help="Override GPU list (comma-separated)")
    p2.add_argument("--output-dir", type=str,
                    help="Override audio output directory")

    # ── Mode 3: run (end-to-end) ─────────────────────────────────────────
    p3 = subparsers.add_parser(
        "run",
        help="End-to-end: generate prompts then synthesize audio.",
    )
    p3.add_argument("--config", default="config.json",
                    help="Path to config.json (default: config.json)")
    p3.add_argument("--total", type=int,
                    help="Override total number of prompts")
    p3.add_argument("--gpus", type=str,
                    help="Override GPU list (comma-separated)")
    p3.add_argument("--seed", type=int,
                    help="Override random seed")
    p3.add_argument("--output-dir", type=str,
                    help="Override output directory")

    # ── Mode 4: reference pipeline (Path C) ──────────────────────────────
    p4 = subparsers.add_parser(
        "reference",
        help="Path C: Generate prompts + audio using reference audio with timbre annotations.",
    )
    p4.add_argument("--config", default="config.json",
                    help="Path to config.json")
    p4.add_argument("--ref-dir", type=str, required=True,
                    help="Directory containing reference .json + .mp3 pairs")
    p4.add_argument("--total", type=int, default=10,
                    help="Number of samples to generate")
    p4.add_argument("--gpus", type=str,
                    help="Override GPU list (comma-separated)")
    p4.add_argument("--output-dir", type=str,
                    help="Override output directory")

    # ── Mode 5: MOSS Audio thinking (Path D) ─────────────────────────────
    p5 = subparsers.add_parser(
        "moss",
        help="Path D: Generate DramaBox prompts from audio using MOSS Audio Thinking.",
    )
    p5.add_argument("--config", default="config.json",
                    help="Path to config.json")
    p5.add_argument("--ref-dir", type=str, required=True,
                    help="Directory containing reference audio files")
    p5.add_argument("--total", type=int, default=10,
                    help="Number of samples to generate")
    p5.add_argument("--gpus", type=str,
                    help="Override GPU list (comma-separated)")
    p5.add_argument("--output-dir", type=str,
                    help="Override output directory")

    # ── Mode 6: demo grid ────────────────────────────────────────────────
    p6 = subparsers.add_parser(
        "demo",
        help="Generate demo grid: 5 Emolia references x N configs with HTML output.",
    )
    p6.add_argument("--config", default="config.json",
                    help="Path to config.json")
    p6.add_argument("--gpus", type=str,
                    help="Override GPU list (comma-separated)")
    p6.add_argument("--output-dir", type=str,
                    help="Override demo output directory")

    # ── Score an existing audio file ─────────────────────────────────────
    p7 = subparsers.add_parser(
        "score",
        help="Score an audio file using ASR WER + content enjoyment.",
    )
    p7.add_argument("--audio", required=True,
                    help="Path to audio file to score")
    p7.add_argument("--prompt", required=True,
                    help="DramaBox prompt text (or path to file containing it)")
    p7.add_argument("--gpu", type=int, default=0,
                    help="GPU ID to use (default: 0)")

    args = parser.parse_args()

    from .config_loader import load_config

    if args.command == "score":
        # Score command doesn't need full config
        _run_score(args)
        return

    config = load_config(args.config, cli_overrides=args)

    if args.command == "generate-prompts":
        from .pipeline import run_mode1
        run_mode1(config)

    elif args.command == "synthesize":
        from .pipeline import run_mode2
        run_mode2(config, args.csv)

    elif args.command == "run":
        from .pipeline import run_mode3
        run_mode3(config)

    elif args.command == "reference":
        from .pipeline import run_reference_pipeline
        run_reference_pipeline(config, args.ref_dir, args.total)

    elif args.command == "moss":
        from .pipeline import run_moss_pipeline
        run_moss_pipeline(config, args.ref_dir, args.total)

    elif args.command == "demo":
        from .pipeline import run_demo_pipeline
        run_demo_pipeline(config)


def _run_score(args):
    """Run the scoring command for a single audio file."""
    import os
    from pathlib import Path

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from .scoring import score_audio

    # Read prompt from file or use directly
    prompt_text = args.prompt
    if Path(prompt_text).is_file():
        prompt_text = Path(prompt_text).read_text(encoding="utf-8")

    result = score_audio(args.audio, prompt_text, device="cuda")

    print(f"Audio: {result['audio_path']}")
    print(f"Transcription: {result['transcription']}")
    print(f"Expected: {result['expected_text'][:100]}...")
    print(f"WER: {result['wer']:.4f}")
    print(f"Content Enjoyment: {result['content_enjoyment']:.4f}")
    print(f"Reward: {result['reward']:.4f}")


if __name__ == "__main__":
    main()
