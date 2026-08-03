import json
from pathlib import Path

import pandas as pd

from trustworthy_resume.audits import explanation_audit, fairness_metrics, reliability_metrics, repeatability_metrics
from trustworthy_resume.data import (
    PROTECTED_PAIRS,
    generate_clean_resumes,
    generate_counterfactual_resumes,
    generate_attacks,
    mask_sensitive_attributes,
    sensitive_leakage_found,
)
from trustworthy_resume.model import DeterministicTestClient, first_balanced_json_object, select_device
from trustworthy_resume.pipeline import RUBRIC, defended_score, extract_profile


def test_balanced_json_parser_handles_fenced_text():
    raw = 'prefix ```json\n{"score": 7, "reason": "ok"}\n``` suffix'
    assert json.loads(first_balanced_json_object(raw))["score"] == 7


def test_device_auto_resolves_to_supported_device():
    assert select_device("auto") in {"cuda", "mps", "cpu"}


def test_masking_removes_synthetic_sensitive_values_and_name():
    text = "Name: Alex Tan\nSynthetic personal information for fairness audit only — Gender: woman. She built an API."
    masked = mask_sensitive_attributes(text)
    assert "Name: [NAME_MASKED]" in masked
    assert not sensitive_leakage_found(masked)
    assert "they built" in masked.lower()


def test_counterfactual_pairs_change_only_protected_line_after_masking():
    clean = generate_clean_resumes(6, seed=42)
    fairness = generate_counterfactual_resumes(clean, templates_per_attribute=1, seed=42)
    assert len(fairness) == 2 * len(PROTECTED_PAIRS)
    for _, pair in fairness.groupby("counterfactual_group_id"):
        assert len(pair) == 2
        assert pair["qualification"].nunique() == 1
        assert pair["masked_resume_text"].nunique() == 1
        assert not any(sensitive_leakage_found(text) for text in pair["masked_resume_text"])


def test_attack_generator_preserves_schema_when_no_candidate_is_attacked():
    clean = generate_clean_resumes(1, seed=1)
    clean["qualification"] = "strong"
    attacked = generate_attacks(clean, seed=4)
    if attacked.empty:
        assert list(attacked.columns) == list(clean.columns)


def test_defended_score_is_python_computed_and_bounded():
    client = DeterministicTestClient()
    profile = extract_profile(client, generate_clean_resumes(1).iloc[0]["resume_text"])
    result = defended_score(client, profile)
    subtotal = sum(result[f"subscore_{key}"] for key in RUBRIC)
    assert result["total_before_penalty"] == subtotal
    assert result["score"] == max(0, subtotal - result["risk_penalty"])
    assert 0 <= result["score"] <= 100


def test_fairness_and_reliability_metrics_have_expected_rows():
    fairness = generate_counterfactual_resumes(generate_clean_resumes(5), 1)
    fairness["pipeline_status"] = "ok"
    fairness["score"] = 50.0
    fairness["rank"] = range(1, len(fairness) + 1)
    groups, disparities, pairs = fairness_metrics(fairness)
    assert len(disparities) == len(PROTECTED_PAIRS)
    assert pairs["score_gap"].eq(0).all()

    variants = pd.DataFrame([
        {"pipeline_status": "ok", "metamorphic_group_id": "A", "metamorphic_variant": "original", "score": 50},
        {"pipeline_status": "ok", "metamorphic_group_id": "A", "metamorphic_variant": "neutral_padding", "score": 52},
    ])
    result = reliability_metrics(variants, max_delta=3)
    assert result.iloc[0]["passed"]

    repeated = pd.DataFrame([
        {"pipeline_status": "ok", "repeatability_group_id": "A", "score": 50, "manipulation_risk_score": 0},
        {"pipeline_status": "ok", "repeatability_group_id": "A", "score": 50, "manipulation_risk_score": 0},
        {"pipeline_status": "ok", "repeatability_group_id": "A", "score": 50, "manipulation_risk_score": 0},
    ])
    repeated_result = repeatability_metrics(repeated)
    assert repeated_result.iloc[0]["exact_score_agreement"]
    assert repeated_result.iloc[0]["risk_score_agreement"]


def test_explanation_audit_checks_grounding():
    row = {"candidate_id": "C1", "attack_type": "clean", "pipeline_status": "ok", "skills": ["Python"], "projects": [], "experience": "", "education": "", "relevant_evidence": [], "evidence": [{"criterion": "python_programming", "source_field": "skills", "quote": "Python"}]}
    row.update({f"subscore_{key}": (10 if key == "python_programming" else 0) for key in RUBRIC})
    audit = explanation_audit(pd.DataFrame([row]))
    assert audit.iloc[0]["evidence_validity_rate"] == 1.0
    assert audit.iloc[0]["rubric_coverage_rate"] == 1.0
