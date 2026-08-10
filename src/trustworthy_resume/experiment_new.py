from __future__ import annotations

import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .audits import (
    add_selection_decision,
    bootstrap_pair_gap_ci,
    build_governance_outputs,
    explanation_audit,
    fairness_metrics,
    qualification_metrics,
    reliability_metrics,
    repeatability_metrics,
)
from .config import ExperimentConfig
from .data import generate_attacks, generate_clean_resumes, generate_counterfactual_resumes, make_metamorphic_variants, make_repeatability_resumes
from .model import DeterministicTestClient, QwenClient, select_device
from .pipeline import run_baseline, run_defended
from .pipeline import BASELINE_PROMPT, DEFENDED_PROMPT, EXTRACTION_PROMPT


def _maybe_colab_download(path: Path) -> None:
    if path.suffix.lower() != ".csv":
        return
    if importlib.util.find_spec("google.colab") is None:
        return
    try:
        colab = importlib.import_module("google.colab")
        files = getattr(colab, "files", None)
        if files is None:
            return
        files.download(str(path))
        print(f"Colab download triggered: {path.name}")
    except Exception as exc:
        print(f"Could not download {path.name} in Colab: {exc}")


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved {path}")
    _maybe_colab_download(path)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared file: {path}")
    if path.suffix == ".json":
        return pd.read_json(path, orient="records")
    return pd.read_csv(path)


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing output file: {path}")
    frame = pd.read_json(path, lines=True)
    if "score" in frame.columns and "pipeline_status" in frame.columns and "rank" not in frame.columns:
        frame["rank"] = np.nan
        ok = frame["pipeline_status"] == "ok"
        frame.loc[ok, "rank"] = frame.loc[ok, "score"].rank(method="first", ascending=False)
    return frame


def _load_prepared_data(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = config.output_dir / "parsed_csv_resumes.csv"
    generated = config.output_dir / "generated_csv_resumes.csv"
    if parsed.exists():
        clean = pd.read_csv(parsed)
    elif generated.exists():
        clean = pd.read_csv(generated)
    else:
        raise FileNotFoundError(
            "No prepared resume data found. Run with --stage prepare or --stage all first."
        )
    attacked = _load_frame(config.output_dir / "attacked_resumes.json")
    fairness = _load_frame(config.output_dir / "fairness_counterfactual_resumes.json")
    metamorphic = _load_frame(config.output_dir / "metamorphic_resumes.json")
    repeated = _load_frame(config.output_dir / "repeatability_resumes.json")
    return clean, attacked, fairness, metamorphic, repeated


def _write_prepared_data(config: ExperimentConfig, clean: pd.DataFrame, attacked: pd.DataFrame, fairness: pd.DataFrame, metamorphic: pd.DataFrame, repeated: pd.DataFrame) -> None:
    if config.csv_data_path:
        _save_frame(clean, config.output_dir / "parsed_csv_resumes.csv")
    else:
        _save_frame(clean, config.output_dir / "generated_csv_resumes.csv")
    for name, frame in [
        ("clean_resumes", clean),
        ("attacked_resumes", attacked),
        ("fairness_counterfactual_resumes", fairness),
        ("metamorphic_resumes", metamorphic),
        ("repeatability_resumes", repeated),
    ]:
        _save_frame(frame, config.output_dir / f"{name}.json")


def _baseline_stage(client: Any, config: ExperimentConfig, clean: pd.DataFrame, attacked: pd.DataFrame, fairness: pd.DataFrame, metamorphic: pd.DataFrame, repeated: pd.DataFrame) -> Dict[str, Any]:
    run_baseline(client, clean, "resume_text", config.output_dir / "baseline_clean.jsonl", config.use_cache, config.batch_size)
    run_baseline(client, attacked, "attacked_resume_text", config.output_dir / "baseline_attacked.jsonl", config.use_cache, config.batch_size)
    run_baseline(client, fairness, "raw_resume_text", config.output_dir / "fairness_baseline_raw.jsonl", config.use_cache, config.batch_size)
    run_baseline(client, fairness, "masked_resume_text", config.output_dir / "fairness_baseline_masked.jsonl", config.use_cache, config.batch_size)
    run_baseline(client, metamorphic, "resume_text", config.output_dir / "metamorphic_baseline.jsonl", config.use_cache, config.batch_size)
    run_baseline(client, repeated, "resume_text", config.output_dir / "repeatability_baseline.jsonl", config.use_cache, config.batch_size)
    _write_manifest(config, "complete")
    return {
        "stage": "baseline",
        "output_dir": str(config.output_dir),
        "baseline_outputs": [
            "baseline_clean.jsonl",
            "baseline_attacked.jsonl",
            "fairness_baseline_raw.jsonl",
            "fairness_baseline_masked.jsonl",
            "metamorphic_baseline.jsonl",
            "repeatability_baseline.jsonl",
        ],
    }


def _defended_stage(client: Any, config: ExperimentConfig, clean: pd.DataFrame, attacked: pd.DataFrame, fairness: pd.DataFrame, metamorphic: pd.DataFrame, repeated: pd.DataFrame) -> Dict[str, Any]:
    run_defended(client, clean, "resume_text", config.output_dir, "clean", config.use_cache, config.batch_size)
    run_defended(client, attacked, "attacked_resume_text", config.output_dir, "attacked", config.use_cache, config.batch_size)
    run_defended(client, fairness, "raw_resume_text", config.output_dir, "fairness_raw", config.use_cache, config.batch_size)
    run_defended(client, fairness, "masked_resume_text", config.output_dir, "fairness_masked", config.use_cache, config.batch_size)
    run_defended(client, metamorphic, "resume_text", config.output_dir, "metamorphic", config.use_cache, config.batch_size)
    run_defended(client, repeated, "resume_text", config.output_dir, "repeatability", config.use_cache, config.batch_size)
    _write_manifest(config, "complete")
    return {
        "stage": "defended",
        "output_dir": str(config.output_dir),
        "defended_outputs": [
            "clean_extraction.jsonl",
            "clean_defended.jsonl",
            "attacked_extraction.jsonl",
            "attacked_defended.jsonl",
            "fairness_raw_extraction.jsonl",
            "fairness_raw_defended.jsonl",
            "fairness_masked_extraction.jsonl",
            "fairness_masked_defended.jsonl",
            "metamorphic_extraction.jsonl",
            "metamorphic_defended.jsonl",
            "repeatability_extraction.jsonl",
            "repeatability_defended.jsonl",
        ],
    }


def _load_baseline_outputs(config: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    return {
        "baseline_clean": _load_jsonl(config.output_dir / "baseline_clean.jsonl"),
        "baseline_attacked": _load_jsonl(config.output_dir / "baseline_attacked.jsonl"),
        "fairness_baseline_raw": _load_jsonl(config.output_dir / "fairness_baseline_raw.jsonl"),
        "fairness_baseline_masked": _load_jsonl(config.output_dir / "fairness_baseline_masked.jsonl"),
        "metamorphic_baseline": _load_jsonl(config.output_dir / "metamorphic_baseline.jsonl"),
        "repeatability_baseline": _load_jsonl(config.output_dir / "repeatability_baseline.jsonl"),
    }


def _load_defended_outputs(config: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    return {
        "defended_clean": _load_jsonl(config.output_dir / "clean_defended.jsonl"),
        "defended_attacked": _load_jsonl(config.output_dir / "attacked_defended.jsonl"),
        "fairness_defended_raw": _load_jsonl(config.output_dir / "fairness_raw_defended.jsonl"),
        "fairness_defended_masked": _load_jsonl(config.output_dir / "fairness_masked_defended.jsonl"),
        "metamorphic_defended": _load_jsonl(config.output_dir / "metamorphic_defended.jsonl"),
        "repeatability_defended": _load_jsonl(config.output_dir / "repeatability_defended.jsonl"),
    }


def _fairness_results(scored: pd.DataFrame, pipeline: str, input_variant: str, config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = scored.copy()
    scored["pipeline"] = pipeline
    scored["input_variant"] = input_variant
    groups, disparities, pairs = fairness_metrics(scored, config.shortlist_fraction)
    for frame in [groups, disparities, pairs]:
        frame["pipeline"] = pipeline
        frame["input_variant"] = input_variant
    return scored, groups, disparities, pairs


def _final_stage(config: ExperimentConfig, clean: pd.DataFrame, attacked: pd.DataFrame, fairness: pd.DataFrame, metamorphic: pd.DataFrame, repeated: pd.DataFrame) -> Dict[str, Any]:
    baseline = _load_baseline_outputs(config)
    defended = _load_defended_outputs(config)

    robust = pd.concat([
        _robustness_comparison(baseline["baseline_clean"], baseline["baseline_attacked"], "baseline"),
        _robustness_comparison(defended["defended_clean"], defended["defended_attacked"], "defended"),
    ], ignore_index=True)
    robust_summary = robust.groupby(["pipeline", "attack_type"]).agg(
        n=("candidate_id", "size"),
        attack_success_rate=("attack_success", "mean"),
        average_score_gain=("score_gain", "mean"),
        average_rank_gain=("rank_gain", "mean"),
    ).reset_index()
    _save_frame(robust, config.output_dir / "robustness_comparison.csv")
    _save_frame(robust_summary, config.output_dir / "robustness_summary.csv")

    _, groups_baseline_raw, disparities_baseline_raw, pairs_baseline_raw = _fairness_results(
        baseline["fairness_baseline_raw"], "baseline", "raw", config
    )
    _, groups_baseline_masked, disparities_baseline_masked, pairs_baseline_masked = _fairness_results(
        baseline["fairness_baseline_masked"], "baseline", "masked", config
    )
    _, groups_defended_raw, disparities_defended_raw, pairs_defended_raw = _fairness_results(
        defended["fairness_defended_raw"], "defended", "raw", config
    )
    _, groups_defended_masked, disparities_defended_masked, pairs_defended_masked = _fairness_results(
        defended["fairness_defended_masked"], "defended", "masked", config
    )

    fairness_groups = pd.concat([
        groups_baseline_raw,
        groups_baseline_masked,
        groups_defended_raw,
        groups_defended_masked,
    ], ignore_index=True)
    fairness_disparities = pd.concat([
        disparities_baseline_raw,
        disparities_baseline_masked,
        disparities_defended_raw,
        disparities_defended_masked,
    ], ignore_index=True)
    fairness_pairs = pd.concat([
        pairs_baseline_raw,
        pairs_baseline_masked,
        pairs_defended_raw,
        pairs_defended_masked,
    ], ignore_index=True)
    _save_frame(fairness_groups, config.output_dir / "fairness_group_metrics.csv")
    _save_frame(fairness_disparities, config.output_dir / "fairness_disparities.csv")
    _save_frame(fairness_pairs, config.output_dir / "fairness_counterfactual_pairs.csv")

    ci_frames = []
    for (pipeline_name, input_variant), group in fairness_pairs.groupby(["pipeline", "input_variant"]):
        ci = bootstrap_pair_gap_ci(group, config.random_seed)
        ci["pipeline"] = pipeline_name
        ci["input_variant"] = input_variant
        ci_frames.append(ci)
    fairness_ci = pd.concat(ci_frames, ignore_index=True) if ci_frames else pd.DataFrame()
    _save_frame(fairness_ci, config.output_dir / "fairness_bootstrap_ci.csv")

    explain = explanation_audit(defended["defended_clean"])
    _save_frame(explain, config.output_dir / "explainability_audit.csv")

    reliability = pd.concat([
        reliability_metrics(baseline["metamorphic_baseline"]).assign(pipeline="baseline"),
        reliability_metrics(defended["metamorphic_defended"]).assign(pipeline="defended"),
    ], ignore_index=True)
    _save_frame(reliability, config.output_dir / "metamorphic_test_results.csv")

    repeatability = pd.concat([
        repeatability_metrics(baseline["repeatability_baseline"]).assign(pipeline="baseline"),
        repeatability_metrics(defended["repeatability_defended"]).assign(pipeline="defended"),
    ], ignore_index=True)
    _save_frame(repeatability, config.output_dir / "repeatability_results.csv")

    qualification = pd.DataFrame([
        {"pipeline": "baseline", **qualification_metrics(baseline["baseline_clean"])},
        {"pipeline": "defended", **qualification_metrics(defended["defended_clean"])},
    ])
    _save_frame(qualification, config.output_dir / "qualification_reliability.csv")

    selected, threshold = add_selection_decision(defended["defended_clean"], config.shortlist_fraction)
    governance_queue, governance_summary = build_governance_outputs(defended["defended_clean"], explain, threshold, config.boundary_margin)
    _save_frame(governance_queue, config.output_dir / "human_review_queue.csv")
    _save_frame(governance_summary, config.output_dir / "governance_summary.csv")
    governance_queue.to_json(config.output_dir / "audit_log.jsonl", orient="records", lines=True, force_ascii=False)
    policy = {
        "shortlist_threshold": threshold,
        "shortlist_fraction": config.shortlist_fraction,
        "boundary_margin": config.boundary_margin,
        "automated_final_hiring_decision": False,
        "mandatory_review_triggers": [
            "pipeline error",
            "risk score >= 1",
            "near threshold",
            "evidence validity < 0.8",
            "protected attribute in explanation",
            "JSON retry",
        ],
    }
    (config.output_dir / "policy_config.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    headline = {
        "num_clean": len(clean),
        "num_attacked": len(attacked),
        "num_counterfactual_variants": len(fairness),
        "baseline_asr": float(robust.loc[robust["pipeline"] == "baseline", "attack_success"].mean()),
        "defended_asr": float(robust.loc[robust["pipeline"] == "defended", "attack_success"].mean()),
        "mean_counterfactual_gap_raw_defended": float(
            fairness_pairs.loc[
                (fairness_pairs["pipeline"] == "defended") & (fairness_pairs["input_variant"] == "raw"),
                "score_gap",
            ].mean()
        ),
        "mean_counterfactual_gap_masked_defended": float(
            fairness_pairs.loc[
                (fairness_pairs["pipeline"] == "defended") & (fairness_pairs["input_variant"] == "masked"),
                "score_gap",
            ].mean()
        ),
        "evidence_validity_rate": float(explain["evidence_validity_rate"].mean()),
        "metamorphic_pass_rate": float(reliability["passed"].mean()),
        "repeatability_exact_agreement_rate": float(repeatability["exact_score_agreement"].mean()),
        "human_review_required_rate": float(governance_summary.iloc[0]["human_review_required_rate"]),
    }
    (config.output_dir / "headline_metrics.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    _write_manifest(config, "complete")
    return headline


def _manifest(config: ExperimentConfig, status: str, error: str = "") -> Dict[str, Any]:
    import hashlib
    try:
        import torch
        import transformers
        versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    except Exception:
        versions = {}
    return {
        **config.to_dict(),
        "status": status,
        "error": error,
        "device_resolved": select_device(config.device),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "prompt_sha256": {
            "baseline": hashlib.sha256(BASELINE_PROMPT.encode()).hexdigest(),
            "extraction": hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest(),
            "defended": hashlib.sha256(DEFENDED_PROMPT.encode()).hexdigest(),
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Synthetic research benchmark only; not a production hiring system.",
    }


def _write_manifest(config: ExperimentConfig, status: str, error: str = "") -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "run_manifest.json").write_text(json.dumps(_manifest(config, status, error), ensure_ascii=False, indent=2), encoding="utf-8")

# rest of original file below...
