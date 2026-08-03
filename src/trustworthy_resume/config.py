from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    output_root: str = "outputs"
    run_id: str = "qwen_v2"
    num_candidates: int = 40
    fairness_templates_per_attribute: int = 5
    repeatability_samples: int = 5
    repeatability_repeats: int = 3
    random_seed: int = 42
    device: str = "auto"
    dtype: str = "auto"
    shortlist_fraction: float = 0.25
    boundary_margin: float = 5.0
    use_cache: bool = True
    backend: str = "qwen"

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
