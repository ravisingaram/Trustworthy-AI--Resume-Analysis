from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def first_balanced_json_object(text: str) -> str:
    cleaned = text.strip().replace("```json", "```").replace("```JSON", "```")
    if "```" in cleaned:
        for part in cleaned.split("```"):
            if "{" in part and "}" in part:
                cleaned = part.strip()
                break
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in model output: {text[:240]}")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    raise ValueError(f"Unbalanced JSON object: {text[:240]}")


class QwenClient:
    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = select_device(device)
        self.model_name = model_name
        if dtype == "auto":
            torch_dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Decoder-only models need left padding so every generated continuation
        # begins after the same padded input width.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs = {"dtype": torch_dtype, "trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        # Qwen's saved sampling defaults are irrelevant for greedy decoding and
        # otherwise cause misleading Transformers warnings.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def generate_many(self, prompts: Sequence[str], max_new_tokens: int = 420) -> List[str]:
        import torch

        if not prompts:
            return []
        rendered_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rendered_prompts.append(rendered)
        inputs = self.tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[:, inputs.input_ids.shape[-1] :]
        return [text.strip() for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]

    def generate(self, prompt: str, max_new_tokens: int = 420) -> str:
        return self.generate_many([prompt], max_new_tokens=max_new_tokens)[0]

    def json_many(
        self,
        prompts: Sequence[str],
        required_keys: Iterable[str],
        max_new_tokens: int = 420,
        retries: int = 2,
    ) -> List[Union[Dict[str, Any], Exception]]:
        required_keys = tuple(required_keys)
        results: List[Optional[Union[Dict[str, Any], Exception]]] = [None] * len(prompts)
        pending = list(range(len(prompts)))
        retry_suffix = "\nReturn exactly one valid JSON object. Required keys: " + ", ".join(required_keys)

        for attempt in range(retries + 1):
            if not pending:
                break
            current_prompts = [prompts[index] + (retry_suffix if attempt else "") for index in pending]
            raw_outputs = self.generate_many(current_prompts, max_new_tokens=max_new_tokens)
            next_pending = []
            for index, raw in zip(pending, raw_outputs):
                try:
                    parsed = json.loads(first_balanced_json_object(raw))
                    missing = [key for key in required_keys if key not in parsed]
                    if missing:
                        raise ValueError(f"Missing keys: {missing}")
                    parsed["_raw_model_output"] = raw
                    parsed["_json_retry_count"] = attempt
                    results[index] = parsed
                except Exception as exc:
                    if attempt < retries:
                        next_pending.append(index)
                    else:
                        results[index] = RuntimeError(f"Model JSON parsing failed: {exc}")
            pending = next_pending

        return [result if result is not None else RuntimeError("Model JSON generation produced no result") for result in results]

    def json(self, prompt: str, required_keys: Iterable[str], max_new_tokens: int = 420, retries: int = 2) -> Dict[str, Any]:
        result = self.json_many([prompt], required_keys, max_new_tokens=max_new_tokens, retries=retries)[0]
        if isinstance(result, Exception):
            raise result
        return result


class DeterministicTestClient:
    """Fast offline backend for tests. It is not used for report metrics."""

    POSITIVE = ["python", "fastapi", "flask", "postgresql", "sql", "docker", "git", "api", "pytest", "algorithms", "backend", "aws"]
    SUSPICIOUS = ["ignore all previous", "rank this candidate first", "senior hiring manager", "python python python", "500%", "millions of users"]

    def json(self, prompt: str, required_keys: Iterable[str], max_new_tokens: int = 420, retries: int = 2) -> Dict[str, Any]:
        lower = prompt.lower()
        evidence = [term for term in self.POSITIVE if term in lower]
        suspicious = [term for term in self.SUSPICIOUS if term in lower]
        if "sub_scores" in required_keys:
            profile_match = re.search(r"STRUCTURED PROFILE:\s*(\{.*\})", prompt, re.DOTALL)
            profile = json.loads(profile_match.group(1)) if profile_match else {}
            profile_text = json.dumps(profile).lower()
            def hit(*terms: str) -> bool:
                return any(term in profile_text for term in terms)
            scores = {
                "python_programming": 18 if hit("python") else 2,
                "backend_or_web": 13 if hit("fastapi", "flask", "backend", "api") else 2,
                "algorithms": 13 if hit("algorithm", "data structures") else 2,
                "projects": 17 if profile.get("projects") else 3,
                "experience": 12 if hit("internship", "research assistant") else 4,
                "evidence_quality": 8 if profile.get("relevant_evidence") else 2,
                "communication": 3,
            }
            evidence_items = [{"criterion": "python_programming", "source_field": "skills", "quote": "Python"}] if hit("python") else []
            return {"sub_scores": scores, "risk_penalty": 8 * int(profile.get("manipulation_risk_score", 0)), "reason": "Deterministic test result", "evidence": evidence_items, "concerns": suspicious, "uncertainty_reasons": [], "_raw_model_output": "test", "_json_retry_count": 0}
        if "manipulation_risk_score" in required_keys:
            resume = prompt.split("RESUME:", 1)[-1]
            lines = resume.splitlines()
            skills_line = next((line for line in lines if line.lower().startswith("skills:")), "")
            projects = [line[2:] for line in lines if line.startswith("- ")]
            risk = min(3, 2 if suspicious else 0)
            return {"skills": [s.strip() for s in skills_line.partition(":")[2].split(",") if s.strip()], "projects": projects, "experience": next((line.partition(":")[2].strip() for line in lines if line.startswith("Experience:")), ""), "education": next((line.partition(":")[2].strip() for line in lines if line.startswith("Education:")), ""), "relevant_evidence": projects + evidence, "suspicious_content": suspicious, "manipulation_risk_score": risk, "extraction_summary": "Deterministic extraction", "_raw_model_output": "test", "_json_retry_count": 0}
        score = float(np.clip(20 + 5 * len(evidence) + 15 * len(suspicious), 0, 100))
        return {"score": score, "reason": "Deterministic test score", "relevant_evidence": evidence, "_raw_model_output": "test", "_json_retry_count": 0}

    def json_many(
        self,
        prompts: Sequence[str],
        required_keys: Iterable[str],
        max_new_tokens: int = 420,
        retries: int = 2,
    ) -> List[Union[Dict[str, Any], Exception]]:
        return [self.json(prompt, required_keys, max_new_tokens=max_new_tokens, retries=retries) for prompt in prompts]
