# main.py
from pathlib import Path
from config_utils import get_config_path, load_or_create_config, save_config
from source_and_topic_gather import main_single as source_and_topic_gather_function

CONFIGS_PATH = Path(__file__).parent / "configs"
CONFIGS_PATH.mkdir(exist_ok=True, parents=True)
# Base defaults (used only if config doesn't exist)
default_cfg = {
    "joke": "ADD",
    "thumbnails": ["ADD", "ADD", "ADD"],
    "outro_video": "ADD",
    "default_photo": "alan_turing1",
    "default_anim": "zoomin",
    "peper_work_dir": "",
}

all_papers = [
    "https://arxiv.org/html/2507.20028v1",
    "https://arxiv.org/html/2505.16839v3",
]

papers_dir_paths = []
for paper in all_papers:
    folder_path = Path(source_and_topic_gather_function(paper))
    folder_path.mkdir(parents=True, exist_ok=True)

    # Per-paper config
    config_path = get_config_path(folder_path.stem, CONFIGS_PATH)
    config = load_or_create_config(config_path, default_cfg)

    # Example: allow override per paper
    config["paper_url"] = paper
    config["peper_work_dir"] = str(folder_path)
    save_config(config_path, config)

    papers_dir_paths.append(folder_path)

# pomysl:
# co tydzien pusczam to odaje wlasne outra
# wyberam minature
# dobieram zart
