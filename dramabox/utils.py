"""Small utility helpers."""
import random
import re


def weighted_choice(options: list[tuple[str, float]]) -> str:
    """Pick one option from a list of (value, weight) tuples."""
    r = random.random()
    cumulative = 0.0
    for value, weight in options:
        cumulative += weight
        if r < cumulative:
            return value
    return options[-1][0]


def clean_level_desc(desc: str) -> str:
    """Remove taxonomy codes and level markers from a description string."""
    cleaned = re.sub(r"\s*\[[A-Z_]+\]\s*", " ", desc)
    cleaned = re.sub(r"\s*\(level\s+\d+\)\s*:\s*", ": ", cleaned)
    return re.sub(r"  +", " ", cleaned).strip()
