# Trustworthy AI Resume Screener v2

This implementation evaluates a Qwen-based resume screener for robustness, fairness, explainability, reliability, and governance. All resumes and protected attributes are synthetic. The system is a research benchmark, not a production hiring tool.

## What is implemented

- Raw-resume baseline and extraction-then-ranking defense.
- Direct injection, role-play, keyword-stuffing, and resume-inflation attacks.
- Matched counterfactual pairs for gender, age, ethnicity, religion, and marital status.
- Sensitive-attribute masking and leakage assertions.
- Demographic parity, subgroup, counterfactual score-gap, rank-gap, and decision-flip metrics.
- Rubric-level evidence output and automatic evidence-grounding audit.
- Metamorphic tests for neutral padding and section reordering, plus three-run repeatability checks.
- Qualification-ranking metrics, pipeline failure rates, and JSON retry rates.
- Human-review policy, review queue, audit-oriented outputs, and run manifest.
- JSONL checkpoints so interrupted Colab runs can resume.
- Batched model inference with per-row JSON retries and checkpointed results.

## Local Mac run

Python 3.9+ is supported. Apple Silicon automatically uses MPS; Intel Macs fall back to CPU.

```bash
cd Trustworthy-AI--Resume-Analysis
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest -q
python3 run_experiment.py --backend test --run-id smoke_test --num-candidates 8 --fairness-templates 2 --reliability-samples 2
python3 run_experiment.py --backend qwen --run-id qwen_n100_mac --batch-size 4
```

The `test` backend validates the complete pipeline but must never be used for report metrics. Use `qwen` for the real experiment.
The default batch size is 4. On a CUDA GPU, try `--batch-size 8` for higher throughput; reduce it to 2 or 1 if generation runs out of memory. Cached rows are excluded before batches are formed.

## Google Colab T4 run

Open `AI_Resume_Screener_Trustworthy_AI_v2_Colab.ipynb` and run all cells. It automatically selects CUDA, MPS, or CPU. In Colab, select **Runtime > Change runtime type > T4 GPU** first. Device-specific run IDs prevent Mac results from being mistaken for CUDA results.

GitHub cloning is optional. On your Mac, compress the whole project folder into one ZIP; it must contain `run_experiment.py`, `requirements.txt`, and `src/`. Upload the notebook to Colab and run its first code cell. When the upload picker appears, select that ZIP. The notebook extracts it and locates the project automatically. If an extracted project is already under `/content`, it is reused without prompting.

For a first GPU check, use 4 candidates, 1 fairness template, and 1 reliability sample. The report configuration uses 100 candidates, 10 matched-pair templates per protected attribute, and 10 reliability samples. Reusing the same `run_id` resumes completed JSONL rows.

## Important output files

- `run_manifest.json`: environment, model, device, run status, and configuration.
- `robustness_summary.csv`: attack success and rank/score changes.
- `fairness_group_metrics.csv`: disaggregated selection and score metrics.
- `fairness_counterfactual_pairs.csv`: matched-pair gaps and decision flips.
- `fairness_bootstrap_ci.csv`: 95% bootstrap intervals.
- `explainability_audit.csv`: evidence validity and rubric coverage.
- `metamorphic_test_results.csv`: reliability invariance checks.
- `repeatability_results.csv`: repeated-run score and risk agreement.
- `audit_log.jsonl`: machine-readable governance audit records.
- `human_review_queue.csv`: cases requiring mandatory human review.
- `governance_summary.csv` and `policy_config.json`: governance policy artifacts.
- `headline_metrics.json`: report-ready headline metrics.

Do not mix these outputs with the earlier `trustworthy_resume_outputs/` surrogate results.
The original `AI_Resume_Screener_Trustworthy_AI_Assignment (4).ipynb` is retained as the team's executed baseline notebook; v2 does not overwrite its embedded T4 results.
