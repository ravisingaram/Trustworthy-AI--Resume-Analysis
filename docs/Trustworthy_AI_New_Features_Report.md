# Trustworthy AI Extensions for Resume Analysis

## Executive Summary

This project extends a language-model-based resume screener with four assurance layers: **fairness**, **explainability**, **verification and reliability**, and **governance**. The implementation does more than calculate a hiring score. It creates counterfactual resumes, tests prompt-injection attacks, validates explanation evidence against source resumes, checks output stability, records audit artifacts, and routes uncertain cases to human review.

The formal experiment used Qwen3-0.6B on Apple Silicon MPS with 100 synthetic candidates, 110 adversarial resume variants, 100 protected-attribute counterfactual variants, and repeated reliability tests. The defended pipeline substantially reduced the overall attack success rate from **99.09% to 38.18%**, a reduction of **60.91 percentage points**. Masking protected attributes reduced the mean defended counterfactual score gap from **1.08 points to 0.00**. Repeatability was perfect under deterministic decoding, but the overall metamorphic pass rate was only **60%**, and only **28.72%** of cited explanation evidence was validated against the source text. These results support using the system as an auditable decision-support prototype, not as an autonomous hiring tool.

## 1. System Overview

The original workflow directly prompted a language model to score resumes. The new workflow separates untrusted resume text from decision logic:

1. Generate or ingest a resume.
2. Detect and extract structured qualifications from untrusted text.
3. Assign manipulation risk and retain suspicious content separately.
4. Score the structured profile against a fixed rubric.
5. Attach source-grounded evidence, concerns, and uncertainty reasons.
6. Apply policy rules and create a mandatory human-review queue.
7. Export checkpointed audit artifacts for later inspection.

The implementation supports three execution devices through one code path: NVIDIA CUDA for Google Colab, Apple Silicon MPS for macOS, and CPU as a fallback. JSONL checkpoints allow interrupted runs to resume without repeating completed model calls.

## 2. New Trustworthiness Features

### 2.1 Fairness

The fairness suite uses matched-pair counterfactual testing. Each pair keeps qualifications constant while changing one protected attribute: age group, ethnicity, gender, marital status, or religion. This design tests whether irrelevant identity information changes the model score.

Two input variants are evaluated:

- **Raw:** the protected attribute remains visible to the model.
- **Masked:** protected-attribute phrases are removed before scoring.

For every attribute and pipeline, the experiment reports the mean absolute score gap and a bootstrap 95% confidence interval. This is more informative than comparing unrelated demographic groups because the matched-pair design controls the qualification content.

### 2.2 Explainability

The defended scorer returns criterion-level sub-scores, a risk penalty, supporting evidence, concerns, and uncertainty reasons. The explanation audit then checks whether each cited evidence item can be found in the original resume.

The exported metrics include:

- evidence count and validated evidence count;
- evidence validity rate;
- rubric coverage rate;
- unsupported positive criteria;
- references to protected attributes in explanations.

This distinction is important: a fluent explanation is not necessarily a faithful explanation. The audit treats source-grounding as a measurable property rather than assuming that generated rationales are correct.

### 2.3 Verification and Reliability

The verification suite contains four complementary checks:

- **Adversarial robustness:** direct prompt injection, role-play injection, keyword stuffing, and resume inflation.
- **Qualification reliability:** Spearman correlation, Kendall correlation, pairwise ordering accuracy, and NDCG against known synthetic qualification levels.
- **Metamorphic testing:** neutral padding and section reordering should not materially change a candidate score. The tolerance is three points.
- **Repeatability:** identical inputs are scored three times to measure exact agreement, score standard deviation, and score range.

The pipeline also reports model-call failure rates and JSON retry rates. These operational metrics distinguish semantic instability from simple parsing failures.

### 2.4 Governance

The governance layer explicitly prevents automated hiring or rejection. Scores are advisory and subject to human review. Mandatory review triggers include:

- a pipeline error;
- manipulation risk of at least one;
- a score near the shortlist threshold;
- evidence validity below 0.8;
- a protected attribute appearing in an explanation;
- a JSON parsing retry.

The run exports a policy configuration, run manifest, review queue, audit log, prompt hashes, model and package versions, random seed, and device metadata. These artifacts support traceability and reproducibility while making the system's limitations visible to reviewers.

## 3. Experimental Design

| Component | Formal configuration |
|---|---:|
| Model | Qwen/Qwen3-0.6B |
| Device | Apple Silicon MPS |
| Base candidates | 100 |
| Adversarial variants | 110 |
| Counterfactual fairness variants | 100 |
| Protected attributes | 5 |
| Matched pairs per attribute | 10 |
| Reliability candidates | 10 |
| Repeatability runs per candidate | 3 |
| Random seed | 42 |
| Decoding | Deterministic greedy decoding |

The dataset is synthetic. Qualification levels and expected ordering are therefore known, which enables controlled reliability measurements without using real applicant data. Results describe this benchmark and should not be generalized directly to real hiring populations.

## 4. Results

### 4.1 Adversarial Robustness

| Attack type | Baseline ASR | Defended ASR | Interpretation |
|---|---:|---:|---|
| Direct prompt injection | 96.77% | 0.00% | The extraction boundary neutralized explicit instruction override. |
| Keyword stuffing | 100.00% | 82.76% | Repetition remained a major weakness. |
| Resume inflation | 100.00% | 21.05% | Evidence extraction and penalties reduced unsupported claims. |
| Role-play injection | 100.00% | 83.33% | Persona-based manipulation remained difficult to detect. |
| **Overall** | **99.09%** | **38.18%** | Defense reduced ASR by **60.91 percentage points**, but did not solve all attacks. |

The defense works particularly well against explicit prompt injection and inflated claims. However, keyword stuffing and role-play attacks still frequently succeed. The result suggests that isolating untrusted text is necessary but insufficient; stronger repetition detection, evidence entailment, and adversarial training are needed.

### 4.2 Fairness

The defended raw-input pipeline had a mean counterfactual gap of **1.08 points** across attributes. Attribute-specific defended raw gaps were 1.4 for age group, 1.0 for ethnicity, 1.3 for gender, 0.6 for marital status, and 1.1 for religion. After masking, all five defended mean gaps were **0.0** in this benchmark.

The baseline was especially sensitive to visible age information, with a raw mean gap of **7.5 points** and a wide bootstrap interval. Masking eliminated the observed gaps for both pipelines. This supports protected-attribute masking as an effective benchmark control, but zero observed gap does not prove fairness in deployment. The sample contains only ten matched pairs per attribute, uses synthetic resumes, and does not test intersectional groups or real-world selection rates.

### 4.3 Explainability

| Metric | Result |
|---|---:|
| Mean evidence items per candidate | 4.86 |
| Mean validated evidence items | 1.46 |
| Evidence validity rate | 28.72% |
| Rubric coverage rate | 56.23% |
| Mean unsupported positive criteria | 2.81 |
| Mean protected-attribute references | 0.04 |

The explanation audit exposes a significant limitation. The model produced multiple evidence items, but fewer than one third could be validated against source text using the implemented evidence matcher. Some failures may be caused by paraphrases that the lexical validator cannot recognize, but unsupported rationale generation is also plausible. Explanations should therefore be shown with direct source spans and treated as review aids, not as proof of model correctness.

### 4.4 Verification and Reliability

The defended pipeline improved qualification ranking quality:

| Metric | Baseline | Defended |
|---|---:|---:|
| Spearman correlation | 0.777 | 0.894 |
| Kendall correlation | 0.724 | 0.781 |
| Strong-over-weak pairwise accuracy | 1.000 | 1.000 |
| NDCG | 0.931 | 0.989 |
| Pipeline failure rate | 0.00% | 0.00% |
| JSON retry rate | 0.00% | 1.00% |

Repeatability exact agreement was **100%** for both pipelines, consistent with deterministic greedy decoding. This does not establish robustness to prompt wording or document structure.

The overall metamorphic pass rate was **60%**. Baseline neutral-padding and section-reordering pass rates were 60% and 90%, respectively. For the defended pipeline, neutral-padding pass rate was 60%, while section-reordering pass rate fell to **30%**, with a mean absolute score change of **10.1 points**. The defended pipeline is repeatable for identical text but remains sensitive to semantically irrelevant layout changes. Structure-aware parsing and order-invariant aggregation should be prioritized.

### 4.5 Governance

The policy routed **92%** of clean candidates to human review, mainly because **84%** had low evidence validity. This high review rate is a deliberate safety outcome under the current thresholds: the system declines to treat weakly supported scores as final decisions.

At the same time, a 92% review rate limits operational usefulness. The correct response is not simply to lower the threshold. Evidence grounding should first improve; the review policy can then be recalibrated using measured false-positive and false-negative costs with qualified human reviewers.

## 5. Overall Interpretation

The experiment demonstrates that trustworthiness is multidimensional. The defended system improved adversarial robustness, qualification ranking, fairness under masking, and exact repeatability. These improvements do not imply that the model is ready for autonomous hiring.

Three findings are especially important:

1. **Security improved, but attack coverage remains uneven.** Explicit prompt injection was eliminated in this run, while keyword stuffing and role-play attacks remained highly effective.
2. **Fairness controls are effective within the synthetic benchmark, not proven in deployment.** Masking removed observed counterfactual gaps, but broader demographic, intersectional, and outcome-based audits are still required.
3. **Reliability and explainability are the main deployment blockers.** Identical inputs are deterministic, yet section order changes can move scores substantially, and most explanation evidence is not source-validated.

The governance layer correctly converts these technical weaknesses into operational safeguards by requiring human review and preserving audit artifacts.

## 6. Recommended Next Steps

1. Add semantic evidence entailment and require every positive criterion to cite an exact resume span.
2. Make profile extraction order-invariant by normalizing sections before model inference.
3. Add dedicated detection and penalties for keyword repetition and role-play instructions.
4. Expand fairness testing to intersectional attributes, more templates, selection-rate metrics, and real-world validation reviewed for privacy and ethics.
5. Calibrate score thresholds and review triggers on a separate validation set.
6. Run the same frozen configuration on Colab T4 CUDA and compare results without mixing device-specific checkpoints.
7. Preserve the human-in-the-loop policy until evidence validity and metamorphic stability meet predefined acceptance thresholds.

## 7. Reproducibility Artifacts

The implementation provides the following report-ready artifacts under `outputs/qwen_v2_n100_mps/`:

- `run_manifest.json` and `policy_config.json`;
- `robustness_summary.csv` and `robustness_comparison.csv`;
- `fairness_bootstrap_ci.csv`, group metrics, and counterfactual-pair data;
- `explainability_audit.csv`;
- `metamorphic_test_results.csv`, `repeatability_results.csv`, and `qualification_reliability.csv`;
- `governance_summary.csv`, `human_review_queue.csv`, and `audit_log.jsonl`;
- resumable JSONL checkpoints for every model-evaluation stage.

The executed notebook `AI_Resume_Screener_Trustworthy_AI_v2_Colab.ipynb` embeds the formal MPS summary tables and is portable across MPS, CUDA, and CPU environments.
