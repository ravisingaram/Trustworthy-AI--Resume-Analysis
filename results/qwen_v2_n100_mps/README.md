# Qwen v2 — 100-Candidate MPS Results

This directory contains the curated, report-ready outputs from the completed formal experiment identified by `run_id=qwen_v2_n100_mps`.

## Provenance

- Model: `Qwen/Qwen3-0.6B`
- Backend: `qwen`
- Device: Apple Silicon MPS
- Base candidates: 100
- Adversarial variants: 110
- Counterfactual fairness variants: 100
- Fairness templates per protected attribute: 10
- Reliability candidates: 10
- Repeats per repeatability candidate: 3
- Random seed: 42
- Run status: complete

See `run_manifest.json` for the full environment and prompt hashes.

## Files

- `headline_metrics.json`: main report metrics.
- `run_manifest.json`: experiment configuration, environment, hashes, and completion status.
- `policy_config.json`: human-review and shortlist policy.
- `robustness_summary.csv`: attack success rate by attack type and pipeline.
- `robustness_comparison.csv`: per-attack score and rank changes.
- `fairness_bootstrap_ci.csv`: counterfactual score-gap confidence intervals.
- `fairness_group_metrics.csv`: group-level score and selection metrics.
- `fairness_disparities.csv`: group disparity summaries.
- `explainability_audit.csv`: evidence validity and rubric-coverage audit.
- `metamorphic_test_results.csv`: neutral-padding and section-reordering checks.
- `repeatability_results.csv`: repeated-run score and risk agreement.
- `qualification_reliability.csv`: ranking correlation, NDCG, failures, and retries.
- `governance_summary.csv`: human-review and evidence-policy outcomes.

## Scope

The resumes and protected attributes are synthetic. These results describe a research benchmark and must not be interpreted as evidence that the system is suitable for autonomous hiring. Raw model-call checkpoints remain under the ignored `outputs/` directory and are not versioned here.
