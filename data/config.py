import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "datasets.json"
DIFFUSION_FILE_SUFFIX = ".IM"


def load_dataset_config(
        dataset: str,
        config_path: str = None
) -> Tuple[str, Dict[str, Any]]:
    """Load a dataset configuration and merge it with global defaults."""
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    canonical_name = config.get("aliases", {}).get(dataset, dataset)
    if canonical_name not in config["datasets"]:
        available = ", ".join(sorted(config["datasets"]))
        raise KeyError(f"未知数据集 {dataset!r}。可选数据集: {available}")

    merged = copy.deepcopy(config.get("defaults", {}))
    merged.update(copy.deepcopy(config["datasets"][canonical_name]))
    merged["name"] = canonical_name
    merged["config_path"] = str(path)
    return canonical_name, merged


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path
