"""Load, validate, and merge configuration."""
import json
from pathlib import Path
from typing import Any


def load_config(config_path: str, cli_overrides: Any = None) -> dict:
    """Load config.json and merge CLI overrides.

    Args:
        config_path: Path to config.json.
        cli_overrides: argparse Namespace with optional override fields.

    Returns:
        Merged configuration dict with resolved paths.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Resolve data paths relative to config file location
    base_dir = config_path.parent
    if "data_paths" in config:
        for key, val in config["data_paths"].items():
            if key.startswith("_"):
                continue
            config["data_paths"][key] = str((base_dir / val).resolve())

    # Resolve output dir
    if "output" in config:
        out_dir = config["output"].get("output_dir", "./output")
        config["output"]["output_dir"] = str((base_dir / out_dir).resolve())

    # Build active languages and accents from config
    languages = []
    language_accents = {}
    for lang, lang_cfg in config.get("languages", {}).items():
        if not isinstance(lang_cfg, dict):
            continue
        if lang_cfg.get("_enabled", False):
            languages.append(lang)
            accents = lang_cfg.get("accents", [])
            if accents:
                language_accents[lang] = accents
    config["_active_languages"] = languages
    config["_language_accents"] = language_accents

    # Apply CLI overrides
    if cli_overrides is not None:
        if hasattr(cli_overrides, "total") and cli_overrides.total is not None:
            config.setdefault("prompt_generation", {})["total_prompts"] = cli_overrides.total
        if hasattr(cli_overrides, "seed") and cli_overrides.seed is not None:
            config.setdefault("prompt_generation", {})["seed"] = cli_overrides.seed
        if hasattr(cli_overrides, "gpus") and cli_overrides.gpus is not None:
            gpu_list = [int(g) for g in cli_overrides.gpus.split(",")]
            config.setdefault("prompt_generation", {})["gpus"] = gpu_list
            config.setdefault("tts", {})["gpus"] = gpu_list
        if hasattr(cli_overrides, "output_dir") and cli_overrides.output_dir is not None:
            config.setdefault("output", {})["output_dir"] = str(Path(cli_overrides.output_dir).resolve())

    return config
