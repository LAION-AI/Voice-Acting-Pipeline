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

    # ── Mode 4: reference pipeline (Path D) ──────────────────────────────
    p4 = subparsers.add_parser(
        "reference",
        help="Path D: Generate prompts + audio using reference audio with timbre annotations.",
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

    # ── Mode 5: demo grid ──────────────────────────────────────────────
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
    p6.add_argument("--n-prompts", type=int,
                    help="Number of prompts per path (default: 10)")
    p6.add_argument("--best-of-n", type=int,
                    help="Best-of-N candidates (default: 3)")
    p6.add_argument("--serve", action="store_true",
                    help="After generation, serve HTML via HTTP + Cloudflare tunnel")
    p6.add_argument("--port", type=int, default=8080,
                    help="HTTP server port (default: 8080)")
    p6.add_argument("--full", action="store_true",
                    help="Run full 4-path demo (A+B+C+D)")

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

    elif args.command == "demo":
        # Apply demo-specific CLI overrides
        if hasattr(args, 'n_prompts') and args.n_prompts:
            config.setdefault("demo", {})["n_prompts_per_path"] = args.n_prompts
        if hasattr(args, 'best_of_n') and args.best_of_n:
            config.setdefault("demo", {})["best_of_n"] = args.best_of_n

        if hasattr(args, 'full') and args.full:
            from .pipeline import run_full_demo
            outdir = run_full_demo(config)
        else:
            from .pipeline import run_demo_pipeline
            outdir = run_demo_pipeline(config)

        if hasattr(args, 'serve') and args.serve:
            port = getattr(args, 'port', 8080) or 8080
            _serve_demo(outdir, port)


def _serve_demo(outdir, port: int = 8080):
    """Serve demo HTML via HTTP server + Cloudflare tunnel."""
    import http.server
    import subprocess
    import threading
    from pathlib import Path

    outdir = Path(outdir)

    # Find HTML file to serve
    html_files = list(outdir.glob("*.html"))
    if not html_files:
        print(f"No HTML files found in {outdir}")
        return

    print(f"\nServing demo from {outdir} on port {port}...")

    # Start HTTP server in background
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(outdir), **kw)

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"  HTTP server: http://localhost:{port}/")

    # Try to start Cloudflare tunnel
    try:
        cf_proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        print("  Starting Cloudflare tunnel...")
        # Read output to find the tunnel URL
        for line in cf_proc.stdout:
            line = line.strip()
            if "trycloudflare.com" in line or ".cloudflare" in line:
                # Extract URL
                import re
                urls = re.findall(r'https?://[^\s]+', line)
                if urls:
                    print(f"  Cloudflare URL: {urls[0]}")
            if "Registered tunnel" in line or "connector" in line.lower():
                print(f"  {line}")

        print("\n  Press Ctrl+C to stop serving.")
        cf_proc.wait()
    except FileNotFoundError:
        print("  cloudflared not found — serving locally only.")
        print(f"  Open http://localhost:{port}/{html_files[0].name}")
        print("\n  Press Ctrl+C to stop serving.")
        try:
            server_thread.join()
        except KeyboardInterrupt:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


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
