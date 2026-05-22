"""CSV chunk writer and reader for DramaBox prompts."""
import csv
from pathlib import Path

CSV_FIELDNAMES = [
    "global_idx",
    "sampling_path",
    "archetype_info",
    "language",
    "accent",
    "emotions",
    "word_count_target",
    "must_include_words",
    "flow_style",
    "flow_forced_by_voicenet",
    "emotion_alignment",
    "direction_style",
    "vocal_bursts_enabled",
    "attributes_raw",
    "dramabox_prompt",
    # Path C/D additional fields
    "reference_audio",
    "timbre_caption",
]


def save_chunk(rows: list[dict], chunk_idx: int, outdir: Path,
               prefix: str = "dramabox") -> Path:
    """Save a list of prompt rows as a CSV chunk file."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{prefix}_chunk_{chunk_idx:03d}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_prompts_from_csv(csv_path: str) -> list[dict]:
    """Load prompts from a CSV file. Returns list of row dicts."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
