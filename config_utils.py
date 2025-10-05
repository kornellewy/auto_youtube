# config_utils.py
import yaml
from pathlib import Path


def get_config_path(paper_name: str, base_dir: Path) -> Path:
    """Generate a unique config filename for each paper."""
    safe_name = "".join(c if c.isalnum() else "_" for c in paper_name)
    return base_dir / f"{safe_name}.yaml"


def load_or_create_config(config_path: Path, default: dict) -> dict:
    if config_path.exists():
        with config_path.open("rt", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    else:
        with config_path.open("wt", encoding="utf-8") as fh:
            yaml.dump(default, fh, sort_keys=False, allow_unicode=True)
        return default


def save_config(config_path: Path, cfg: dict) -> None:
    with config_path.open("wt", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, sort_keys=False, allow_unicode=True)
