"""Demo grid generator: HTML audio comparison grids.

Generates HTML pages with audio player grids for comparing:
- Reference audio vs generated variants
- Different pipeline configurations (Path A/B/C/D, with/without VC, etc.)
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


## ─── Full 4-Path Demo HTML ───────────────────────────────────────────────────

def _audio_to_base64_mp3(audio_path: str | Path, bitrate: str = "128k") -> str:
    """Convert a WAV file to base64-encoded MP3 via ffmpeg.

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        bitrate: MP3 encoding bitrate (e.g. "128k", "64k", "48k").

    Returns:
        Base64-encoded MP3 string, or empty string on failure.
    """
    import base64
    import subprocess
    import tempfile

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return ""

    # If already MP3 and using default bitrate, just base64 it
    if audio_path.suffix.lower() == ".mp3" and bitrate == "128k":
        return base64.b64encode(audio_path.read_bytes()).decode("ascii")

    # Convert to MP3 with ffmpeg
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-codec:a", "libmp3lame",
             "-b:a", bitrate, "-ar", "24000", "-ac", "1", tmp_path],
            capture_output=True, check=True, timeout=30,
        )
        data = Path(tmp_path).read_bytes()
        import os
        os.unlink(tmp_path)
        return base64.b64encode(data).decode("ascii")
    except Exception as e:
        log.error("Failed to convert %s to MP3: %s", audio_path, e)
        return ""


def _base64_audio_player(b64_data: str, label: str = "") -> str:
    """Build HTML for an inline base64 audio player."""
    if not b64_data:
        return '<div class="no-audio">No audio</div>'
    return (f'<audio controls preload="none">'
            f'<source src="data:audio/mp3;base64,{b64_data}" type="audio/mp3">'
            f'</audio>')


def generate_full_demo_html(
    path_results: dict,
    output_path: str | Path,
    title: str = "DramaBox Demo Grid — All 4 Paths",
    mp3_bitrate: str = "128k",
) -> Path:
    """Generate a self-contained HTML file for the full 4-path demo.

    Each path (A, B, C, D) gets its own section with N prompt cards.
    Each card shows: audio player (base64 MP3), scores table (all candidates
    with winner highlighted), and a collapsible DramaBox script.
    Path D cards also include reference audio player.

    Args:
        path_results: Dict with keys "A", "B", "C", "D", each containing
            a list of item dicts with dramabox_prompt, best_audio,
            all_scores, candidates, and optionally ref_audio.
        output_path: Where to write the HTML file.
        title: Page title.
        mp3_bitrate: Bitrate for base64 MP3 encoding (e.g. "48k" for smaller files).

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    path_labels = {
        "A": "PATH A — VoiceNet Sampling",
        "B": "PATH B — Archetype Sampling",
        "C": "PATH C — Archetype Sampling (named)",
        "D": "PATH D — Reference Audio + VC",
    }
    path_descriptions = {
        "A": "VoiceNet (57 dims) sampling → LLM prompt gen → DramaBox TTS → RE-USE",
        "B": "Archetype (920 chars) sampling → LLM prompt gen → DramaBox TTS → RE-USE",
        "C": "Archetype sampling → LLM explicitly names archetype role in DramaBox script → DramaBox TTS → RE-USE",
        "D": "Timbre Whisper + situation dims → LLM prompt gen → DramaBox TTS (text-only) → Chatterbox VC",
    }

    sections_html = []
    total_prompts = 0
    total_audio = 0

    for path_key in ["A", "B", "C", "D"]:
        items = path_results.get(path_key, [])
        if not items:
            continue

        cards_html = []
        for item in items:
            idx = item.get("idx", 0)
            prompt = item.get("dramabox_prompt", "")
            all_scores = item.get("all_scores", [])
            ref_audio = item.get("ref_audio", "")
            best_result = item.get("best", {})
            best_idx = best_result.get("best_idx", -1) if best_result else -1
            candidates = item.get("candidates", [])

            # Reference audio for Path D (reference audio + VC)
            ref_player_html = ""
            if path_key == "D" and ref_audio:
                ref_b64 = _audio_to_base64_mp3(ref_audio, mp3_bitrate)
                if ref_b64:
                    ref_player_html = f"""
                    <div class="ref-audio-inline">
                        <span class="ref-label">Reference:</span>
                        {_base64_audio_player(ref_b64)}
                    </div>"""

            # Build candidate rows with audio players
            candidates_html = ""
            if candidates or all_scores:
                rows = []
                n_candidates = max(len(candidates), len(all_scores))
                for c_idx in range(n_candidates):
                    is_best = (c_idx == best_idx)
                    cls = ' class="winner"' if is_best else ''
                    marker = " ★" if is_best else ""

                    # Score data
                    score = all_scores[c_idx] if c_idx < len(all_scores) else {}
                    wer = score.get("wer", 0)
                    enjoy = score.get("content_enjoyment", 0)
                    reward = score.get("reward", 0)

                    # Audio for this candidate — raw TTS + processed version
                    cand_audio = ""
                    raw_audio = ""
                    if c_idx < len(candidates):
                        cand = candidates[c_idx]
                        cand_audio = cand.get("audio_for_scoring", "") or cand.get("vc", "") or cand.get("selfvc", "") or cand.get("raw", "")
                        raw_audio = cand.get("raw", "")

                    cand_b64 = _audio_to_base64_mp3(cand_audio, mp3_bitrate) if cand_audio else ""
                    if cand_b64:
                        total_audio += 1

                    # Show side-by-side: Raw TTS vs processed
                    # A/B: Raw TTS | After RE-USE   C/D: Raw TTS | After VC
                    if raw_audio:
                        raw_b64 = _audio_to_base64_mp3(raw_audio, mp3_bitrate)
                        if raw_b64:
                            total_audio += 1
                        after_label = "After VC" if path_key == "D" else "After RE-USE"
                        audio_html = (
                            f'<div class="raw-vs-vc">'
                            f'<div class="rv-col"><span class="rv-label">Raw TTS</span>{_base64_audio_player(raw_b64)}</div>'
                            f'<div class="rv-col"><span class="rv-label">{after_label}</span>{_base64_audio_player(cand_b64)}</div>'
                            f'</div>'
                        )
                    else:
                        audio_html = _base64_audio_player(cand_b64)

                    rows.append(
                        f'<div class="candidate-row{" winner" if is_best else ""}">'
                        f'<div class="candidate-header">'
                        f'<span class="candidate-label">#{c_idx+1}{marker}</span>'
                        f'<span class="candidate-scores">'
                        f'WER {wer:.3f} · Enjoy {enjoy:.2f} · <strong>R {reward:.2f}</strong>'
                        f'</span></div>'
                        f'{audio_html}'
                        f'</div>'
                    )
                candidates_html = f'<div class="candidates-list">{"".join(rows)}</div>'

            # Collapsible script
            escaped_prompt = _escape(prompt)
            script_id = f"script_{path_key}_{idx}"
            script_html = f"""
            <details class="script-details">
                <summary>DramaBox Script</summary>
                <pre class="script-text" id="{script_id}">{escaped_prompt}</pre>
            </details>"""

            # Sample info
            sample = item.get("sample", {})
            lang = ""
            emotions = ""
            if isinstance(sample, dict):
                lang = sample.get("language", item.get("language", ""))
                emotions = sample.get("emotions", item.get("emotions", ""))
            else:
                lang = item.get("language", "")
                emotions = item.get("emotions", "")

            meta_parts = []
            if lang:
                meta_parts.append(f"Lang: {_escape(str(lang))}")
            if emotions:
                emo_str = str(emotions)[:60]
                meta_parts.append(f"Emotions: {_escape(emo_str)}")
            meta_html = "<br>".join(meta_parts)

            card = f"""
            <div class="prompt-card">
                <div class="card-header">Prompt {idx+1}</div>
                {ref_player_html}
                {candidates_html}
                <div class="meta">{meta_html}</div>
                {script_html}
            </div>"""
            cards_html.append(card)
            total_prompts += 1

        section = f"""
        <div class="path-section">
            <div class="path-header">
                <span class="path-badge">{path_key}</span>
                <span class="path-title">{_escape(path_labels[path_key])}</span>
            </div>
            <div class="path-description">{_escape(path_descriptions[path_key])}</div>
            <div class="cards-grid">
                {''.join(cards_html)}
            </div>
        </div>"""
        sections_html.append(section)

    # Build config guide
    config_guide = """
    <details class="config-guide" open>
        <summary>Configuration Guide</summary>
        <div class="guide-content">
            <p><strong>Path A (VoiceNet Sampling):</strong> Sample 57 voice dimensions (tempo, gender, age, timbre, etc.) + emotions &rarr; LLM generates DramaBox script &rarr; DramaBox TTS &rarr; RE-USE enhancement</p>
            <p><strong>Path B (Archetype Sampling):</strong> Sample from 920 character archetypes across 92 genres &rarr; LLM generates DramaBox script &rarr; DramaBox TTS &rarr; RE-USE enhancement</p>
            <p><strong>Path C (Archetype Sampling — named):</strong> Same as B but the LLM is explicitly instructed to name the archetype role in the DramaBox script (e.g. "a battle-hardened noble knight") so DramaBox knows WHAT character to perform as &rarr; DramaBox TTS &rarr; RE-USE enhancement</p>
            <p><strong>Path D (Reference Audio + VC):</strong> Timbre Whisper analyzes reference voice + sample situation-dependent dims &rarr; LLM generates script with matching speaker description &rarr; DramaBox TTS (text-only, NO voice_ref) &rarr; Chatterbox VC converts to reference voice. <em style="color:#999">Most promising path for voice cloning: the timbre caption guides Gemma 4 E4B-it to produce a speaker-consistent DramaBox script, while Chatterbox VC handles the actual voice transfer.</em></p>
            <p><strong>Scoring:</strong> reward = (1 &minus; WER) &times; content_enjoyment | WER via Parakeet v3 ASR | Enjoyment via Empathic Insight Plus</p>
        </div>
    </details>"""

    html = f"""<!DOCTYPE html>
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
        max-width: 1800px;
        margin: 0 auto;
    }}
    h1 {{
        text-align: center;
        margin-bottom: 6px;
        color: #e94560;
        font-size: 2em;
    }}
    .subtitle {{
        text-align: center;
        color: #888;
        margin-bottom: 20px;
        font-size: 0.9em;
    }}

    /* Config guide */
    .config-guide {{
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        margin-bottom: 24px;
        padding: 0;
    }}
    .config-guide summary {{
        padding: 12px 16px;
        cursor: pointer;
        font-weight: 700;
        color: #e94560;
        font-size: 1em;
    }}
    .guide-content {{
        padding: 6px 16px 14px;
        font-size: 0.85em;
        color: #bbb;
        line-height: 1.7;
    }}
    .guide-content p {{ margin-bottom: 4px; }}
    .guide-content strong {{ color: #e94560; }}

    /* Path sections */
    .path-section {{
        margin-bottom: 28px;
        background: #16213e;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #0f3460;
    }}
    .path-header {{
        background: #0f3460;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .path-badge {{
        background: #e94560;
        color: #fff;
        font-size: 0.8em;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 4px;
        letter-spacing: 1px;
    }}
    .path-title {{
        font-weight: 700;
        font-size: 1.05em;
    }}
    .path-description {{
        padding: 8px 16px;
        font-size: 0.8em;
        color: #999;
        border-bottom: 1px solid #0f3460;
    }}

    /* Cards grid */
    .cards-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        padding: 14px;
    }}
    .prompt-card {{
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 12px;
        width: calc(20% - 10px);
        min-width: 260px;
        flex-shrink: 0;
    }}
    .card-header {{
        font-weight: 700;
        font-size: 0.85em;
        color: #e94560;
        margin-bottom: 8px;
    }}

    /* Audio */
    audio {{
        width: 100%;
        height: 36px;
        margin: 4px 0;
        border-radius: 4px;
    }}
    .no-audio {{
        width: 100%;
        height: 36px;
        margin: 4px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #111;
        border-radius: 4px;
        color: #555;
        font-size: 0.75em;
    }}
    .ref-label {{
        font-size: 0.72em;
        color: #888;
        font-weight: 600;
    }}
    .ref-audio-inline {{
        margin-bottom: 6px;
        padding: 4px 6px;
        background: #141428;
        border-radius: 4px;
        border-left: 3px solid #e94560;
    }}

    /* Candidates list */
    .candidates-list {{
        margin: 6px 0;
    }}
    .candidate-row {{
        padding: 5px 6px;
        border-radius: 4px;
        margin-bottom: 4px;
        background: #141428;
        border-left: 3px solid #0f3460;
    }}
    .candidate-row.winner {{
        background: #142a1e;
        border-left: 3px solid #53d769;
    }}
    .candidate-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2px;
    }}
    .candidate-label {{
        font-size: 0.75em;
        font-weight: 700;
        color: #aaa;
    }}
    .candidate-row.winner .candidate-label {{
        color: #53d769;
    }}
    .candidate-scores {{
        font-size: 0.68em;
        color: #888;
    }}
    .candidate-row.winner .candidate-scores {{
        color: #53d769;
    }}
    .candidate-scores strong {{
        color: inherit;
    }}

    /* Raw vs VC comparison */
    .raw-vs-vc {{
        display: flex;
        gap: 6px;
    }}
    .rv-col {{
        flex: 1;
    }}
    .rv-label {{
        font-size: 0.68em;
        font-weight: 600;
        color: #888;
        display: block;
        margin-bottom: 1px;
    }}
    .candidate-row .rv-col:last-child .rv-label {{
        color: #53d769;
    }}
    @media (max-width: 600px) {{
        .raw-vs-vc {{ flex-direction: column; gap: 4px; }}
    }}

    /* Meta */
    .meta {{
        font-size: 0.72em;
        color: #888;
        line-height: 1.5;
        margin: 4px 0;
    }}

    /* Script details */
    .script-details {{
        margin-top: 6px;
    }}
    .script-details summary {{
        font-size: 0.72em;
        color: #666;
        cursor: pointer;
    }}
    .script-text {{
        font-size: 0.68em;
        color: #999;
        background: #111;
        padding: 8px;
        border-radius: 4px;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
        margin-top: 4px;
    }}

    /* Responsive */
    @media (max-width: 1400px) {{
        .prompt-card {{ width: calc(25% - 9px); }}
    }}
    @media (max-width: 1000px) {{
        .prompt-card {{ width: calc(33.33% - 8px); }}
    }}
    @media (max-width: 700px) {{
        .prompt-card {{ width: calc(50% - 6px); min-width: 200px; }}
    }}
</style>
</head>
<body>
<h1>{_escape(title)}</h1>
<p class="subtitle">{total_prompts} prompts &times; Best-of-N | Generated {timestamp} | {total_audio} audio files embedded</p>

{config_guide}

{''.join(sections_html)}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info("Full demo HTML written to %s (%d prompts, %d audio)",
             output_path, total_prompts, total_audio)
    print(f"  HTML written: {output_path} ({total_prompts} prompts, "
          f"{total_audio} audio embedded)", flush=True)
    return output_path


# ─── Legacy grid data builder ────────────────────────────────────────────────

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
