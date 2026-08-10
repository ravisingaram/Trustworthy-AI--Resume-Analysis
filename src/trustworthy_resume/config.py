from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    output_root: str = "outputs"
    run_id: str = "qwen_v4"
    num_candidates: int = 10
    fairness_templates_per_attribute: int = 10
    repeatability_samples: int = 10
    repeatability_repeats: int = 3
    random_seed: int = 42
    stage: str = "all"
    device: str = "auto"
    dtype: str = "auto"
    shortlist_fraction: float = 0.25
    boundary_margin: float = 5.0
    batch_size: int = 4
    use_cache: bool = True
    backend: str = "qwen"
    csv_data_path: str = ""

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
