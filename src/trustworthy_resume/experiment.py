from __future__ import annotations

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


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    else:
        df.to_csv(path, index=False)


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


def _robustness_comparison(clean: pd.DataFrame, attacked: pd.DataFrame, label: str) -> pd.DataFrame:
    clean_ok = clean[clean["pipeline_status"] == "ok"][["candidate_id", "score", "rank"]].rename(columns={"score": "clean_score", "rank": "clean_rank"})
    attacked_ok = attacked[attacked["pipeline_status"] == "ok"].copy()
    result = attacked_ok.merge(clean_ok, on="candidate_id", how="left")
    result["score_gain"] = result["score"] - result["clean_score"]
    clean_pool = clean[clean["pipeline_status"] == "ok"]
    result["attacked_rank_in_clean_pool"] = result.apply(lambda row: 1 + int(((clean_pool["candidate_id"] != row["candidate_id"]) & (clean_pool["score"] > row["score"])).sum()), axis=1)
    result["rank_gain"] = result["clean_rank"] - result["attacked_rank_in_clean_pool"]
    result["attack_success"] = result["rank_gain"] > 0
    result["pipeline"] = label
    return result


def _fairness_suite(client: Any, fairness_df: pd.DataFrame, config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_scores, all_groups, all_disparities, all_pairs = [], [], [], []
    conditions = [
        ("baseline", "raw", "raw_resume_text"),
        ("baseline", "masked", "masked_resume_text"),
        ("defended", "raw", "raw_resume_text"),
        ("defended", "masked", "masked_resume_text"),
    ]
    for pipeline_name, input_variant, text_col in conditions:
        prefix = f"fairness_{pipeline_name}_{input_variant}"
        if pipeline_name == "baseline":
            scored = run_baseline(client, fairness_df, text_col, config.output_dir / f"{prefix}.jsonl", config.use_cache)
        else:
            scored = run_defended(client, fairness_df, text_col, config.output_dir, prefix, config.use_cache)
        scored["pipeline"] = pipeline_name
        scored["input_variant"] = input_variant
        groups, disparities, pairs = fairness_metrics(scored, config.shortlist_fraction)
        for frame in [groups, disparities, pairs]:
            frame["pipeline"] = pipeline_name
            frame["input_variant"] = input_variant
        all_scores.append(scored)
        all_groups.append(groups)
        all_disparities.append(disparities)
        all_pairs.append(pairs)
    return tuple(pd.concat(frames, ignore_index=True) for frames in [all_scores, all_groups, all_disparities, all_pairs])  # type: ignore[return-value]


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the complete benchmark and write report-ready artifacts."""
    _write_manifest(config, "running")
    try:
        client = DeterministicTestClient() if config.backend == "test" else QwenClient(config.model_name, config.device, config.dtype)
        clean = generate_clean_resumes(config.num_candidates, config.random_seed)
        attacked = generate_attacks(clean, config.random_seed)
        fairness = generate_counterfactual_resumes(clean, config.fairness_templates_per_attribute, config.random_seed)
        metamorphic = make_metamorphic_variants(clean, config.repeatability_samples, config.random_seed)
        repeated = make_repeatability_resumes(clean, config.repeatability_samples, config.repeatability_repeats, config.random_seed)
        for name, frame in [("clean_resumes", clean), ("attacked_resumes", attacked), ("fairness_counterfactual_resumes", fairness), ("metamorphic_resumes", metamorphic), ("repeatability_resumes", repeated)]:
            _save_frame(frame, config.output_dir / f"{name}.json")

        baseline_clean = run_baseline(client, clean, "resume_text", config.output_dir / "baseline_clean.jsonl", config.use_cache)
        baseline_attacked = run_baseline(client, attacked, "attacked_resume_text", config.output_dir / "baseline_attacked.jsonl", config.use_cache)
        defended_clean = run_defended(client, clean, "resume_text", config.output_dir, "clean", config.use_cache)
        defended_attacked = run_defended(client, attacked, "attacked_resume_text", config.output_dir, "attacked", config.use_cache)

        robust = pd.concat([
            _robustness_comparison(baseline_clean, baseline_attacked, "baseline"),
            _robustness_comparison(defended_clean, defended_attacked, "defended"),
        ], ignore_index=True)
        robust_summary = robust.groupby(["pipeline", "attack_type"]).agg(n=("candidate_id", "size"), attack_success_rate=("attack_success", "mean"), average_score_gain=("score_gain", "mean"), average_rank_gain=("rank_gain", "mean")).reset_index()
        _save_frame(robust, config.output_dir / "robustness_comparison.csv")
        _save_frame(robust_summary, config.output_dir / "robustness_summary.csv")

        fairness_scores, fairness_groups, fairness_disparities, fairness_pairs = _fairness_suite(client, fairness, config)
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

        explain = explanation_audit(defended_clean)
        _save_frame(explain, config.output_dir / "explainability_audit.csv")

        baseline_meta = run_baseline(client, metamorphic, "resume_text", config.output_dir / "metamorphic_baseline.jsonl", config.use_cache)
        defended_meta = run_defended(client, metamorphic, "resume_text", config.output_dir, "metamorphic", config.use_cache)
        reliability = pd.concat([
            reliability_metrics(baseline_meta).assign(pipeline="baseline"),
            reliability_metrics(defended_meta).assign(pipeline="defended"),
        ], ignore_index=True)
        _save_frame(reliability, config.output_dir / "metamorphic_test_results.csv")

        baseline_repeated = run_baseline(client, repeated, "resume_text", config.output_dir / "repeatability_baseline.jsonl", config.use_cache)
        defended_repeated = run_defended(client, repeated, "resume_text", config.output_dir, "repeatability", config.use_cache)
        repeatability = pd.concat([
            repeatability_metrics(baseline_repeated).assign(pipeline="baseline"),
            repeatability_metrics(defended_repeated).assign(pipeline="defended"),
        ], ignore_index=True)
        _save_frame(repeatability, config.output_dir / "repeatability_results.csv")

        qualification = pd.DataFrame([
            {"pipeline": "baseline", **qualification_metrics(baseline_clean)},
            {"pipeline": "defended", **qualification_metrics(defended_clean)},
        ])
        _save_frame(qualification, config.output_dir / "qualification_reliability.csv")

        selected, threshold = add_selection_decision(defended_clean, config.shortlist_fraction)
        governance_queue, governance_summary = build_governance_outputs(defended_clean, explain, threshold, config.boundary_margin)
        _save_frame(governance_queue, config.output_dir / "human_review_queue.csv")
        _save_frame(governance_summary, config.output_dir / "governance_summary.csv")
        governance_queue.to_json(config.output_dir / "audit_log.jsonl", orient="records", lines=True, force_ascii=False)
        policy = {
            "shortlist_threshold": threshold,
            "shortlist_fraction": config.shortlist_fraction,
            "boundary_margin": config.boundary_margin,
            "automated_final_hiring_decision": False,
            "mandatory_review_triggers": ["pipeline error", "risk score >= 1", "near threshold", "evidence validity < 0.8", "protected attribute in explanation", "JSON retry"],
        }
        (config.output_dir / "policy_config.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

        headline = {
            "num_clean": len(clean),
            "num_attacked": len(attacked),
            "num_counterfactual_variants": len(fairness),
            "baseline_asr": float(robust.loc[robust["pipeline"] == "baseline", "attack_success"].mean()),
            "defended_asr": float(robust.loc[robust["pipeline"] == "defended", "attack_success"].mean()),
            "mean_counterfactual_gap_raw_defended": float(fairness_pairs.loc[(fairness_pairs["pipeline"] == "defended") & (fairness_pairs["input_variant"] == "raw"), "score_gap"].mean()),
            "mean_counterfactual_gap_masked_defended": float(fairness_pairs.loc[(fairness_pairs["pipeline"] == "defended") & (fairness_pairs["input_variant"] == "masked"), "score_gap"].mean()),
            "evidence_validity_rate": float(explain["evidence_validity_rate"].mean()),
            "metamorphic_pass_rate": float(reliability["passed"].mean()),
            "repeatability_exact_agreement_rate": float(repeatability["exact_score_agreement"].mean()),
            "human_review_required_rate": float(governance_summary.iloc[0]["human_review_required_rate"]),
        }
        (config.output_dir / "headline_metrics.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
        _write_manifest(config, "complete")
        return headline
    except Exception as exc:
        _write_manifest(config, "failed", f"{type(exc).__name__}: {exc}")
        raise
