"""Demo grid generator: HTML audio comparison grids.

Generates HTML pages with audio player grids for comparing:
- Reference audio vs generated variants
- Different pipeline configurations (Path C, Path D, with/without VC, etc.)
- Best-of-N candidate rankings

Layout: Each reference speaker is a section/card. Variants are displayed
as a responsive flex-wrap grid within each card (5 per row max).
Audio files are copied into the output directory so the HTML can use
relative paths for serving over HTTP.
"""
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)


def generate_demo_html(
    grid_data: list[dict],
    output_path: str | Path,
    title: str = "DramaBox Demo Grid",
    copy_audio: bool = True,
) -> Path:
    """Generate an HTML demo grid page.

    Args:
        grid_data: List of row dicts, each with:
            - "reference": {"audio": path, "label": str, "metadata": dict}
            - "variants": list of {"audio": path, "label": str, "config": str,
                                   "scores": dict_or_None}
        output_path: Where to write the HTML file.
        title: Page title.
        copy_audio: If True, copy audio files to output dir and use relative paths.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = output_path.parent / "audio"

    if copy_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    # Process grid data: copy audio files and rewrite paths to relative
    processed_data = []
    for row_idx, row in enumerate(grid_data):
        p_row = dict(row)
        ref = dict(row.get("reference", {}))

        # Copy reference audio
        ref_audio = ref.get("audio", "")
        if ref_audio and copy_audio:
            if Path(ref_audio).is_file():
                dest_name = f"ref_{row_idx:03d}{Path(ref_audio).suffix}"
                dest = audio_dir / dest_name
                if not dest.exists():
                    shutil.copy2(ref_audio, dest)
                ref["audio"] = f"audio/{dest_name}"
            else:
                ref["audio"] = ""  # Mark as unavailable
        # When copy_audio=False, keep existing paths as-is (may be relative)

        p_row["reference"] = ref

        # Copy variant audio
        p_variants = []
        for v_idx, v in enumerate(row.get("variants", [])):
            pv = dict(v)
            v_audio = v.get("audio", "")
            if v_audio and copy_audio:
                if Path(v_audio).is_file():
                    v_label = v.get("label", v.get("config", f"var_{v_idx}"))
                    safe_label = v_label.replace("/", "_").replace(" ", "_")
                    dest_name = f"ref_{row_idx:03d}_{safe_label}{Path(v_audio).suffix}"
                    dest = audio_dir / dest_name
                    if not dest.exists():
                        shutil.copy2(v_audio, dest)
                    pv["audio"] = f"audio/{dest_name}"
                else:
                    pv["audio"] = ""
            # When copy_audio=False, keep existing paths as-is
            p_variants.append(pv)

        p_row["variants"] = p_variants
        processed_data.append(p_row)

    html = _build_html(processed_data, title)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_variants = sum(len(r.get("variants", [])) for r in processed_data)
    log.info("Demo grid written to %s (%d rows, %d variants total)",
             output_path, len(processed_data), n_variants)
    return output_path


def _build_html(grid_data: list[dict], title: str) -> str:
    """Build the full HTML string for the demo grid using card layout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections_html = []
    for row_idx, row in enumerate(grid_data):
        ref = row.get("reference", {})
        ref_audio = ref.get("audio", "")
        ref_label = ref.get("label", f"Speaker {row_idx + 1}")
        ref_meta = ref.get("metadata", {})

        # Reference card
        ref_player = _audio_player_html(ref_audio)
        ref_meta_html = _meta_html(ref_meta)

        # Variant cards
        variant_cards = []
        for v in row.get("variants", []):
            v_audio = v.get("audio", "")
            v_label = v.get("label", v.get("config", ""))
            v_scores = v.get("scores", {})
            v_meta = {**v.get("metadata", {}), **v_scores}

            v_player = _audio_player_html(v_audio)
            v_meta_html = _meta_html(v_meta)

            variant_cards.append(f"""
            <div class="variant-card">
                <div class="card-label">{_escape(v_label)}</div>
                {v_player}
                <div class="meta">{v_meta_html}</div>
            </div>""")

        section = f"""
        <div class="speaker-section">
            <div class="speaker-header">
                <div class="ref-badge">REF {row_idx + 1}</div>
                <span class="speaker-title">{_escape(ref_label)}</span>
            </div>
            <div class="speaker-body">
                <div class="ref-card">
                    <div class="card-label">Reference Audio</div>
                    {ref_player}
                    <div class="meta">{ref_meta_html}</div>
                </div>
                <div class="variants-grid">
                    {''.join(variant_cards)}
                </div>
            </div>
        </div>"""
        sections_html.append(section)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(title)}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #1a1a2e;
        color: #eee;
        padding: 20px;
        max-width: 1600px;
        margin: 0 auto;
    }}
    h1 {{
        text-align: center;
        margin-bottom: 8px;
        color: #e94560;
        font-size: 1.8em;
    }}
    .subtitle {{
        text-align: center;
        color: #888;
        margin-bottom: 24px;
        font-size: 0.9em;
    }}

    /* Speaker section */
    .speaker-section {{
        margin-bottom: 28px;
        background: #16213e;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #0f3460;
    }}
    .speaker-header {{
        background: #0f3460;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .ref-badge {{
        background: #e94560;
        color: #fff;
        font-size: 0.7em;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }}
    .speaker-title {{
        font-weight: 600;
        font-size: 0.95em;
    }}
    .speaker-body {{
        padding: 14px;
    }}

    /* Reference card */
    .ref-card {{
        background: #1a1a3e;
        border: 2px solid #e94560;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 14px;
        max-width: 350px;
    }}

    /* Variants grid */
    .variants-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .variant-card {{
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 10px;
        width: calc(20% - 8px);
        min-width: 220px;
        flex-shrink: 0;
    }}

    /* Card internals */
    .card-label {{
        font-weight: 600;
        font-size: 0.78em;
        color: #e94560;
        margin-bottom: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    audio {{
        width: 100%;
        height: 40px;
        margin: 6px 0;
        border-radius: 4px;
    }}
    .no-audio {{
        width: 100%;
        height: 40px;
        margin: 6px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #111;
        border-radius: 4px;
        color: #555;
        font-size: 0.75em;
        font-style: italic;
    }}
    .meta {{
        font-size: 0.72em;
        color: #aaa;
        line-height: 1.5;
        word-break: break-word;
    }}
    .meta .score {{
        color: #53d769;
        font-weight: 600;
    }}
    .meta .score.low {{
        color: #ff6b6b;
    }}

    /* Legend */
    .legend {{
        margin-top: 24px;
        padding: 16px;
        background: #16213e;
        border-radius: 8px;
        font-size: 0.82em;
        color: #aaa;
        line-height: 1.6;
    }}
    .legend h3 {{
        color: #e94560;
        margin-bottom: 8px;
    }}

    /* Responsive */
    @media (max-width: 1200px) {{
        .variant-card {{ width: calc(25% - 8px); }}
    }}
    @media (max-width: 900px) {{
        .variant-card {{ width: calc(33.33% - 7px); }}
    }}
    @media (max-width: 600px) {{
        .variant-card {{ width: calc(50% - 5px); min-width: 160px; }}
    }}
</style>
</head>
<body>
<h1>{_escape(title)}</h1>
<p class="subtitle">Generated {timestamp} | DramaBox Pipeline</p>

{''.join(sections_html)}

<div class="legend">
    <h3>Legend</h3>
    <p><strong>Reference</strong>: Original audio from Emolia dataset</p>
    <p><strong>Variants</strong>: Different pipeline configurations for the same reference speaker</p>
    <p><strong>WER</strong>: Word Error Rate (lower is better, 0 = perfect transcription)</p>
    <p><strong>Enjoyment</strong>: Content enjoyment score from Empathic Insight Plus (higher is better)</p>
    <p><strong>Reward</strong>: Composite score = (1 - WER) &times; Enjoyment</p>
    <p style="margin-top:8px;color:#666;">Cells showing "No audio yet" have prompts generated but audio not yet synthesized.</p>
</div>

</body>
</html>"""


def _audio_player_html(audio_path: str) -> str:
    """Build HTML for an audio player, or a placeholder if no audio."""
    if audio_path:
        return f'<audio controls preload="none"><source src="{_escape(audio_path)}"></audio>'
    return '<div class="no-audio">No audio yet</div>'


def _meta_html(metadata: dict) -> str:
    """Build metadata HTML snippet."""
    if not metadata:
        return ""
    parts = []
    for k, v in metadata.items():
        if k.startswith("_"):
            continue
        if k in ("wer", "reward", "content_enjoyment"):
            css = "score low" if k == "wer" and isinstance(v, (int, float)) and v > 0.5 else "score"
            if isinstance(v, float):
                parts.append(f'<span class="{css}">{k}: {v:.3f}</span>')
            else:
                parts.append(f'<span class="{css}">{k}: {v}</span>')
        elif isinstance(v, float):
            parts.append(f"{_escape(str(k))}: {v:.2f}")
        elif isinstance(v, str) and len(v) > 80:
            parts.append(f"{_escape(str(k))}: {_escape(v[:77])}...")
        else:
            parts.append(f"{_escape(str(k))}: {_escape(str(v))}")
    return "<br>".join(parts)


def _escape(text: str) -> str:
    """Basic HTML escaping."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def build_demo_grid_data(
    results_dir: str | Path,
    references: list[dict],
    configs: list[str],
) -> list[dict]:
    """Build grid_data from a results directory structure.

    Expected directory structure:
        results_dir/
            ref_000/
                reference.mp3
                config_name/
                    raw.wav
                    vc.wav
                    enhanced.wav
                    scores.json

    Args:
        results_dir: Path to results directory.
        references: List of reference dicts with "audio", "label", "metadata".
        configs: List of configuration names.

    Returns:
        grid_data suitable for generate_demo_html().
    """
    results_dir = Path(results_dir)
    grid_data = []

    for ref_idx, ref in enumerate(references):
        ref_dir = results_dir / f"ref_{ref_idx:03d}"
        row = {
            "reference": ref,
            "variants": [],
        }

        for config_name in configs:
            config_dir = ref_dir / config_name
            if not config_dir.exists():
                continue

            # Find best audio file
            for audio_name in ["best.wav", "vc.wav", "enhanced.wav", "raw.wav"]:
                audio_file = config_dir / audio_name
                if audio_file.exists():
                    break
            else:
                audio_file = None

            # Load scores if available
            scores = {}
            scores_file = config_dir / "scores.json"
            if scores_file.exists():
                with open(scores_file) as f:
                    scores = json.load(f)

            row["variants"].append({
                "audio": str(audio_file) if audio_file else "",
                "label": config_name,
                "config": config_name,
                "scores": scores,
            })

        grid_data.append(row)

    return grid_data
