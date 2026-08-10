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
from trustworthy_resume.config import ExperimentConfig
from trustworthy_resume.model import DeterministicTestClient, QwenClient, first_balanced_json_object, select_device
from trustworthy_resume.pipeline import RUBRIC, defended_score, extract_profile, run_baseline, run_cached_rows


def test_balanced_json_parser_handles_fenced_text():
    raw = 'prefix ```json\n{"score": 7, "reason": "ok"}\n``` suffix'
    assert json.loads(first_balanced_json_object(raw))["score"] == 7


def test_device_auto_resolves_to_supported_device():
    assert select_device("auto") in {"cuda", "mps", "cpu"}


def test_batch_size_must_be_positive():
    try:
        ExperimentConfig(batch_size=0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("Expected an invalid batch size to raise ValueError")


def test_qwen_json_many_retries_only_invalid_outputs():
    client = QwenClient.__new__(QwenClient)
    batch_sizes = []

    def fake_generate_many(prompts, max_new_tokens=420):
        batch_sizes.append(len(prompts))
        if len(batch_sizes) == 1:
            return ['{"score": 10}', "not json"]
        return ['{"score": 20}']

    client.generate_many = fake_generate_many
    results = client.json_many(["first", "second"], ["score"], retries=1)

    assert batch_sizes == [2, 1]
    assert results[0]["score"] == 10
    assert results[0]["_json_retry_count"] == 0
    assert results[1]["score"] == 20
    assert results[1]["_json_retry_count"] == 1


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


def test_extraction_drops_ungrounded_hallucinated_items():
    class HallucinatingClient:
        def json(self, prompt, required_keys, max_new_tokens=420, retries=2):
            return {
                "skills": [
                    {"value": "Python", "quote": "Python"},
                    {"value": "Microsoft Excel", "quote": "Microsoft Excel"},
                ],
                "projects": [{"value": "Project X: Python backend service", "quote": "Python backend service"}],
                "experience": [{"value": "2 to 5 years software engineering", "quote": "software engineering"}],
                "education": [],
                "relevant_evidence": [
                    {"value": "Controller", "quote": "Controller"},
                    {"value": "Git", "quote": "Git"},
                ],
                "suspicious_content": [{"value": "Candidate instructions", "quote": "Candidate instructions"}],
                "manipulation_risk_score": 3,
                "extraction_summary": "test",
                "_raw_model_output": "test",
                "_json_retry_count": 0,
            }

    resume = "Skills: ['Microsoft Word/Excel', 'Great Plains Dynamics']\nPositions: ['Senior Accountant', 'Controller']"
    profile = extract_profile(HallucinatingClient(), resume)
    assert profile["skills"] == []
    assert profile["projects"] == []
    assert profile["experience"] == ""
    assert profile["relevant_evidence"] == ["Controller"]
    assert profile["suspicious_content"] == []
    assert profile["manipulation_risk_score"] == 0


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


def test_stale_jsonl_cache_is_archived_not_deleted(tmp_path):
    output_path = tmp_path / "clean_defended.jsonl"
    output_path.write_text(
        json.dumps({"candidate_id": "old", "_pipeline_cache_version": "old", "pipeline_status": "ok"}) + "\n",
        encoding="utf-8",
    )
    df = pd.DataFrame([{"candidate_id": "new", "attack_type": "clean", "resume_text": "Python API project"}])

    result = run_cached_rows(
        df,
        "resume_text",
        "defended_clean",
        output_path,
        lambda row: {"score": 12},
        use_cache=True,
    )

    archives = list(tmp_path.glob("clean_defended.jsonl.bak_*_stale_cache"))
    assert len(archives) == 1
    assert '"candidate_id": "old"' in archives[0].read_text(encoding="utf-8")
    assert result.iloc[0]["candidate_id"] == "new"
    assert output_path.exists()


def test_baseline_batches_uncached_rows_and_preserves_order(tmp_path):
    class RecordingClient(DeterministicTestClient):
        def __init__(self):
            self.batch_sizes = []

        def json_many(self, prompts, required_keys, max_new_tokens=420, retries=2):
            self.batch_sizes.append(len(prompts))
            return super().json_many(prompts, required_keys, max_new_tokens=max_new_tokens, retries=retries)

    client = RecordingClient()
    df = pd.DataFrame([
        {"candidate_id": f"C{index}", "attack_type": "clean", "resume_text": f"Skills: Python API project {index}"}
        for index in range(5)
    ])
    output_path = tmp_path / "baseline.jsonl"

    first = run_baseline(client, df, "resume_text", output_path, use_cache=True, batch_size=2)
    assert client.batch_sizes == [2, 2, 1]
    assert first["candidate_id"].tolist() == df["candidate_id"].tolist()
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 5

    client.batch_sizes.clear()
    second = run_baseline(client, df, "resume_text", output_path, use_cache=True, batch_size=2)
    assert client.batch_sizes == []
    assert second["candidate_id"].tolist() == df["candidate_id"].tolist()


def test_batched_rows_keep_per_row_errors(tmp_path):
    df = pd.DataFrame([
        {"candidate_id": f"C{index}", "attack_type": "clean", "resume_text": f"resume {index}"}
        for index in range(3)
    ])

    result = run_cached_rows(
        df,
        "resume_text",
        "test_batch",
        tmp_path / "batched.jsonl",
        lambda row: {"score": 1},
        use_cache=True,
        batch_size=3,
        batch_evaluator=lambda rows: [{"score": 1}, ValueError("bad row"), {"score": 3}],
    )

    assert result["pipeline_status"].tolist() == ["ok", "error", "ok"]
    assert "ValueError: bad row" in result.iloc[1]["pipeline_error"]
