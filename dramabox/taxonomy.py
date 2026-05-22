"""Taxonomy loaders for VoiceNet, EmoNet, vocal bursts, and archetypes."""
import json
import re
from pathlib import Path


def parse_voicenet_html(path: Path) -> list[dict]:
    """Parse the VoiceNet HTML taxonomy file and extract all dimensions.

    Returns a list of dicts, each with keys:
        code   (str)  : e.g. "TEMP"
        name   (str)  : e.g. "Tempo"
        desc   (str)  : short description
        levels (list)  : [{"val": 0, "desc": "..."}, ...]
    """
    html = path.read_text(encoding="utf-8")
    dimensions = []

    dim_blocks = re.split(r'<div class="dim" id="dim-(\d+)">', html)

    for i in range(1, len(dim_blocks), 2):
        block = dim_blocks[i + 1] if i + 1 < len(dim_blocks) else ""

        code_m = re.search(r'<span class="code">([^<]+)</span>', block)
        code = code_m.group(1).strip() if code_m else f"DIM{dim_blocks[i]}"

        name_m = re.search(
            r'<span class="code">[^<]+</span>\s*(.+?)\s*<span class="arrow">',
            block,
        )
        name = name_m.group(1).strip() if name_m else code

        desc_m = re.search(r'<div class="dim-desc">([^<]+)</div>', block)
        desc = desc_m.group(1).strip() if desc_m else ""

        levels = []
        for lv_m in re.finditer(
            r'<div class="level-val">(\d+)</div>\s*<div class="level-desc">(.*?)</div>',
            block,
            re.DOTALL,
        ):
            val = int(lv_m.group(1))
            lv_desc = re.sub(r"<[^>]+>", "", lv_m.group(2)).strip()
            lv_desc = lv_desc.replace("&#x27;", "'").replace("&amp;", "&")
            levels.append({"val": val, "desc": lv_desc})

        dimensions.append({"code": code, "name": name, "desc": desc, "levels": levels})

    return dimensions


def load_emonet(path: Path) -> dict[str, list[str]]:
    """Load EmoNet emotion taxonomy from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_vocal_bursts(path: Path) -> dict[str, str]:
    """Load vocal bursts taxonomy from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_archetypes(path: Path) -> dict[str, list[str]]:
    """Load voice archetypes taxonomy from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_vocal_bursts_block(taxonomy: dict[str, str]) -> str:
    """Build the instruction block for vocal bursts to include in the LLM prompt."""
    lines = [f"  - {name}: {desc}" for name, desc in taxonomy.items()]
    taxonomy_text = "\n".join(lines)
    return f"""\
VOCAL BURSTS — REFERENCE TAXONOMY:
{taxonomy_text}

VOCAL BURST INSTRUCTIONS:
- Weave vocal bursts into the stage directions wherever they naturally fit the emotional arc.
- Vocal bursts go OUTSIDE of quotes as stage directions — non-lexical sounds produced organically.
- They must merge seamlessly with the flow — a sob that bleeds into the next sentence, a chuckle
  that trails off as the speaker continues, a gasp that launches the next line.
- Pick bursts appropriate for the emotions and physical state. Do not force mismatches.
- Use exact names from the taxonomy (e.g. "A Stifled Sob escapes as she continues").
- Aim for 2–5 vocal bursts distributed naturally — not clustered, not mechanical.
"""
