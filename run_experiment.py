#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trustworthy_resume import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trustworthy AI resume-screening benchmark.")
    parser.add_argument("--backend", choices=["qwen", "test"], default="qwen", help="Use qwen for real metrics; test is a fast deterministic smoke backend.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--run-id", default="qwen_v4")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--fairness-templates", type=int, default=10)
    parser.add_argument("--reliability-samples", type=int, default=10)
    parser.add_argument("--repeatability-repeats", type=int, default=3)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts generated together. Reduce this if the GPU runs out of memory.")
    parser.add_argument("--csv-data-path", default="", help="Optional CSV file path to load real resume candidates instead of synthetic data.")
    parser.add_argument("--stage", choices=["all", "prepare", "baseline", "defended", "clean_attacked_defended", "final"], default="all", help="Run one stage: prepare, baseline, defended, clean_attacked_defended, final, or all.")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ExperimentConfig(
        backend=args.backend,
        device=args.device,
        run_id=args.run_id,
        output_root=args.output_root,
        num_candidates=args.num_candidates,
        fairness_templates_per_attribute=args.fairness_templates,
        repeatability_samples=args.reliability_samples,
        repeatability_repeats=args.repeatability_repeats,
        model_name=args.model_name,
        batch_size=args.batch_size,
        csv_data_path=args.csv_data_path,
        stage=args.stage,
        use_cache=not args.no_cache,
    )
    print(json.dumps(run_experiment(config), indent=2))
