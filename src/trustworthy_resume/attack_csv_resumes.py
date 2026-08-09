#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from trustworthy_resume.data import ATTACK_INSERTS
from trustworthy_resume.model import DeterministicTestClient, QwenClient
from trustworthy_resume.pipeline import run_baseline, run_defended

DEFAULT_FIELDS = [
    "candidate_id",
    "career_objective",
    "skills",
    "degree_names",
    "professional_company_names",
    "positions",
    "responsibilities",
    "educationaL_requirements",
    "experiencere_requirement",
    "matched_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score resumes from a CSV and apply prompt-injection attacks.")
    parser.add_argument("--csv", required=True, help="Path to the resume CSV file.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "csv_attack"), help="Directory where JSONL/csv outputs are written.")
    parser.add_argument("--backend", choices=["test", "qwen"], default="test", help="Use qwen for real LLM scoring or test for a fast deterministic offline backend.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B", help="Model name for Qwen backend.")
    parser.add_argument("--attack-type", choices=["direct_prompt_injection", "keyword_stuffing", "resume_inflation", "role_play_injection", "random"], default="random")
    parser.add_argument("--sample-size", type=int, default=0, help="Randomly sample this many resumes from the CSV before scoring. Use 0 to keep all rows.")
    parser.add_argument("--use-cache", action="store_true", help="Use existing JSONL cache files when available.")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Do not use existing JSONL cache files.")
    parser.add_argument("--resume-text-column", default="resume_text", help="If the CSV already contains a resume text column, use it instead of building one.")
    return parser.parse_args()


def build_resume_text(row: pd.Series) -> str:
    fields = []
    if "candidate_id" in row and pd.notna(row["candidate_id"]):
        fields.append(f"Candidate ID: {row['candidate_id']}")
    if "career_objective" in row and pd.notna(row["career_objective"]):
        fields.append(f"Career objective: {row['career_objective']}")
    if "skills" in row and pd.notna(row["skills"]):
        fields.append(f"Skills: {row['skills']}")
    if "degree_names" in row and pd.notna(row["degree_names"]):
        fields.append(f"Education: {row['degree_names']}")
    if "professional_company_names" in row and pd.notna(row["professional_company_names"]):
        fields.append(f"Company: {row['professional_company_names']}")
    if "positions" in row and pd.notna(row["positions"]):
        fields.append(f"Position: {row['positions']}")
    if "responsibilities" in row and pd.notna(row["responsibilities"]):
        fields.append(f"Responsibilities: {row['responsibilities']}")
    if "educationaL_requirements" in row and pd.notna(row["educationaL_requirements"]):
        fields.append(f"Education requirements: {row['educationaL_requirements']}")
    if "experiencere_requirement" in row and pd.notna(row["experiencere_requirement"]):
        fields.append(f"Experience requirements: {row['experiencere_requirement']}")
    if not fields:
        raise ValueError("Unable to build resume text: CSV row contains no known resume fields.")
    return "\n".join(fields)


def choose_attack_type(row: pd.Series, attack_type: str) -> str:
    if attack_type == "random":
        return row.get("attack_type") if pd.notna(row.get("attack_type")) else pd.Series(list(ATTACK_INSERTS)).sample(1).iloc[0]
    return attack_type


def safe_cast_candidate_id(value: Any, idx: int) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return f"CSV_{idx:04d}"
    return str(value)


def summarize_results(clean: pd.DataFrame, attacked: pd.DataFrame, prefix: str, output_dir: Path) -> pd.DataFrame:
    clean_ok = clean[clean["pipeline_status"] == "ok"][["candidate_id", "score", "rank"]].rename(columns={"score": "clean_score", "rank": "clean_rank"})
    attacked_ok = attacked[attacked["pipeline_status"] == "ok"].copy()
    result = attacked_ok.merge(clean_ok, on="candidate_id", how="left")
    result["score_gain"] = result["score"] - result["clean_score"]
    clean_pool = clean[clean["pipeline_status"] == "ok"]
    result["attacked_rank_in_clean_pool"] = result.apply(
        lambda row: 1 + int(((clean_pool["candidate_id"] != row["candidate_id"]) & (clean_pool["score"] > row["score"])).sum()),
        axis=1,
    )
    result["rank_gain"] = result["clean_rank"] - result["attacked_rank_in_clean_pool"]
    result["attack_success"] = result["rank_gain"] > 0
    output_path = output_dir / f"{prefix}_summary.csv"
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.sample_size > 0 and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {args.sample_size} resumes from the input CSV.")

    if "candidate_id" not in df.columns:
        df["candidate_id"] = [safe_cast_candidate_id(None, i) for i in range(len(df))]
    else:
        df["candidate_id"] = [safe_cast_candidate_id(val, i) for i, val in enumerate(df["candidate_id"])]

    if args.resume_text_column in df.columns and args.resume_text_column != "resume_text":
        df["resume_text"] = df[args.resume_text_column].astype(str)
    elif "resume_text" not in df.columns:
        df["resume_text"] = df.apply(build_resume_text, axis=1)

    if args.attack_type == "random":
        df["attack_type"] = df.apply(lambda _: pd.Series(list(ATTACK_INSERTS)).sample(1).iloc[0], axis=1)
    else:
        df["attack_type"] = args.attack_type

    df["attacked_resume_text"] = df.apply(
        lambda row: f"{row['resume_text']}\n\n{ATTACK_INSERTS[row['attack_type']]}" if row['attack_type'] in ATTACK_INSERTS else row['resume_text'],
        axis=1,
    )

    attack_inputs_path = output_dir / "csv_attack_inputs.csv"
    df[["candidate_id", "attack_type", "resume_text", "attacked_resume_text"]].to_csv(attack_inputs_path, index=False)
    print(f"Saved attacked resume inputs to {attack_inputs_path}")
    print("\nFirst attacked resume examples:")
    for _, row in df.head(3).iterrows():
        print("\n---")
        print(f"candidate_id: {row['candidate_id']}")
        print(f"attack_type: {row['attack_type']}")
        print("resume_text:")
        print(row['resume_text'])
        print("\nattacked_resume_text:")
        print(row['attacked_resume_text'])

    client = DeterministicTestClient() if args.backend == "test" else QwenClient(args.model_name, args.device)

    baseline_clean = run_baseline(client, df, "resume_text", output_dir / "baseline_clean.jsonl", args.use_cache)
    baseline_attacked = run_baseline(client, df, "attacked_resume_text", output_dir / "baseline_attacked.jsonl", args.use_cache)
    defended_clean = run_defended(client, df, "resume_text", output_dir, "csv_clean", args.use_cache)
    defended_attacked = run_defended(client, df, "attacked_resume_text", output_dir, "csv_attacked", args.use_cache)

    summarize_results(baseline_clean, baseline_attacked, "baseline", output_dir)
    summarize_results(defended_clean, defended_attacked, "defended", output_dir)

    print("Done.")
    print(f"Baseline summary: {output_dir / 'baseline_summary.csv'}")
    print(f"Defended summary: {output_dir / 'defended_summary.csv'}")


if __name__ == "__main__":
    main()
