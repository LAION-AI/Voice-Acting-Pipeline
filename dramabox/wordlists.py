"""Per-language word lists for mandatory word injection."""
import json
from pathlib import Path

_WORD_CACHE: dict[str, list[str]] = {}


def _load_english_words() -> list[str]:
    """Load English words from NLTK corpus."""
    try:
        from nltk.corpus import words as en_words
        return list({w.lower() for w in en_words.words() if 3 <= len(w) <= 12 and w.isalpha()})
    except LookupError:
        import nltk
        nltk.download("words", quiet=True)
        from nltk.corpus import words as en_words
        return list({w.lower() for w in en_words.words() if 3 <= len(w) <= 12 and w.isalpha()})


def _load_spanish_words(data_dir: Path) -> list[str]:
    """Load Spanish words from NLTK corpus, fallback to bundled list."""
    try:
        from nltk.corpus import cess_esp
        ws = {w.lower().strip() for s in cess_esp.sents() for w in s
              if w.strip().isalpha() and 3 <= len(w.strip()) <= 14}
        if len(ws) > 200:
            return list(ws)
    except Exception:
        pass
    # Fallback to bundled word list
    wl_path = data_dir / "spanish.json"
    if wl_path.exists():
        with open(wl_path, encoding="utf-8") as f:
            return json.load(f)
    return ["aventura", "camino", "esperanza", "silencio", "corazón"]


def get_word_list(language: str, data_dir: Path) -> list[str]:
    """Return a word list for the given language (cached after first load).

    Args:
        language: Language name (e.g. "English", "German").
        data_dir: Path to the data/wordlists/ directory.
    """
    if language in _WORD_CACHE:
        return _WORD_CACHE[language]

    lang_lower = language.lower()
    if lang_lower == "english":
        wl = _load_english_words()
    elif lang_lower == "spanish":
        wl = _load_spanish_words(data_dir)
    else:
        # Try loading from bundled JSON file
        wl_path = data_dir / f"{lang_lower}.json"
        if wl_path.exists():
            with open(wl_path, encoding="utf-8") as f:
                wl = json.load(f)
        else:
            # Fallback to English
            wl = _load_english_words()

    _WORD_CACHE[language] = wl
    return wl
