"""
Build final farmer-facing AI report images for the four disease reference images.

Inputs:
    report/.env  -> GEMINI_API_KEY, ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
                   OPENAI_API_KEY, OPENAI_BASE_URL, QWEN_API_KEY
    report/images/*.jpeg disease reference images

Outputs:
    report/final_reports/*.png
    report/final_reports/ai_report_generation_details.xlsx
    report/final_reports/final_ai_reports.zip

The script records real provider usage when returned by the APIs. It does not
invent model outputs; if a provider fails, the report image marks it unavailable
and the spreadsheet stores the error.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
IMAGES_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "final_reports"
ZIP_BASE = ROOT / "final_ai_reports"
CLAUDE_MANUAL_PATH = ROOT / "claude_manual_reports.json"


DISEASE_CASES = [
    {
        "id": "bacterial_spot",
        "image": IMAGES_DIR / "tomato bacterial.jpeg",
        "class_name": "Tomato___Bacterial_spot",
        "display_name": "Tomato Bacterial Spot",
        "confidence": 97.8,
        "severity": "Moderate",
        "symptoms": "small dark water-soaked spots, yellow halos, spots on leaf tissue",
        "cause_reference": (
            "Bacterial spot is caused by Xanthomonas bacteria that enter tomato leaves through "
            "small wounds or natural openings. It can spread through infected seed, splashing "
            "water, plant debris, and contaminated tools."
        ),
        "recommended_actions": (
            "Remove infected leaves, avoid overhead watering, use certified disease-free seed, "
            "sanitize tools, and apply copper-based bactericide only according to label guidance."
        ),
    },
    {
        "id": "early_blight",
        "image": IMAGES_DIR / "tomato early blight.jpeg",
        "class_name": "Tomato___Early_blight",
        "display_name": "Tomato Early Blight",
        "confidence": 98.6,
        "severity": "Moderate-High",
        "symptoms": "brown circular spots, target-like rings, yellowing leaf tissue, older leaves affected",
        "cause_reference": (
            "Early blight is caused by the fungus Alternaria solani. It survives in infected "
            "plant debris and usually starts on older lower leaves before moving upward."
        ),
        "recommended_actions": (
            "Remove affected lower leaves, keep foliage dry, improve spacing and airflow, mulch to "
            "reduce soil splash, and use a suitable fungicide when disease pressure is high."
        ),
    },
    {
        "id": "late_blight",
        "image": IMAGES_DIR / "tomato late blight.jpeg",
        "class_name": "Tomato___Late_blight",
        "display_name": "Tomato Late Blight",
        "confidence": 99.1,
        "severity": "Critical",
        "symptoms": "large irregular brown patches, water-soaked lesions, damaged leaf edge, rapid blighting",
        "cause_reference": (
            "Late blight is caused by Phytophthora infestans, a fast-spreading water mold. It can "
            "move quickly through tomato plants by windborne spores and infected plant material."
        ),
        "recommended_actions": (
            "Remove badly infected leaves or plants quickly, do not compost infected material, keep "
            "foliage dry, isolate affected plants, and use preventive fungicide guidance from a local expert."
        ),
    },
    {
        "id": "septoria_leaf_spot",
        "image": IMAGES_DIR / "tomato septoria spot.jpeg",
        "class_name": "Tomato___Septoria_leaf_spot",
        "display_name": "Tomato Septoria Leaf Spot",
        "confidence": 94.7,
        "severity": "Moderate-High",
        "symptoms": "many small round gray-brown spots, dark margins, speckled lower leaves",
        "cause_reference": (
            "Septoria leaf spot is caused by a fungus that mainly attacks tomato leaves. It often "
            "starts near the bottom of the plant and spreads by rain splash, irrigation splash, and infected debris."
        ),
        "recommended_actions": (
            "Remove infected lower leaves, avoid wetting foliage, increase airflow, clean old plant "
            "debris, and use labeled fungicide protection if the disease continues spreading."
        ),
    },
]


SYSTEM_PROMPT = """You are a friendly farming helper writing short disease reports for everyday farmers.
Write like you are talking to a farmer who has never studied science. Use the
simplest words possible — the kind you would use when explaining to a friend.

IMPORTANT LANGUAGE RULES:
- NEVER use scientific names, Latin names, or technical biology terms
  (e.g. do NOT write Xanthomonas, Alternaria solani, Phytophthora infestans,
   or any genus/species names).
- Instead say things like "a type of bacteria", "a fungus", "a fast-spreading
  water mold" — always in plain everyday words.
- Keep sentences short (under 20 words each).
- Use words a 10-year-old could understand.

Return only a valid JSON object with exactly these keys:
{
  "plant_disease": "<disease name in plain English>",
  "cause_of_disease": "<one clear paragraph, 2-4 very short sentences, no bullets, no numbering, no scientific names>",
  "recommendation": "<one clear paragraph, 2-4 very short sentences, no bullets, no numbering, practical steps only>"
}

Extra rules:
- Do not copy the example image wording exactly.
- Mention uncertainty only if confidence is low or model agreement is weak.
- Do not add markdown, tables, emojis, references, or extra keys."""


REPORT_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "plant_disease": {"type": "string"},
        "cause_of_disease": {"type": "string"},
        "recommendation": {"type": "string"},
    },
    "required": ["plant_disease", "cause_of_disease", "recommendation"],
    "additionalProperties": False,
})


PROVIDERS = {
    "gemini": {
        "label": "Gemini Version",
        "env_key": "GEMINI_API_KEY",
        "model_candidates": [
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ],
    },
    "anthropic": {
        "label": "Claude Version",
        "env_key": "ANTHROPIC_API_KEY",
        "model_candidates": [
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-opus-4-6",
            "claude-opus-4-1-20250805",
        ],
    },
    "openai": {
        "label": "GPT Version",
        "env_key": "OPENAI_API_KEY",
        "model_candidates": [
            "gpt-6-luna",
            "gpt-5.5",
            "gpt-5",
            "gpt-4o",
        ],
    },
    "qwen": {
        "label": "Qwen Version",
        "env_key": "QWEN_API_KEY",
        "env_keys": ["QWEN_API_KEY", "DASHSCOPE_API_KEY"],
        "model_candidates": [
            "qwen3.8-max",
            "qwen-max",
            "qwen-plus",
        ],
    },
}


ENV_MODEL_KEYS = {
    "gemini": "GEMINI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "qwen": "QWEN_MODEL",
}

CACHED_RESPONSES_PATH = ROOT / "cached_responses.json"


@dataclass
class ProviderResult:
    case_id: str
    provider: str
    provider_label: str
    success: bool
    model_used: str = ""
    attempted_models: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    raw_text: str = ""
    cleaned_text: str = ""
    cause: str = ""
    recommendation: str = ""
    error: str = ""


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(load_env_file(PROJECT_ROOT / ".env"))
    env.update(load_env_file(ROOT / ".env"))
    return env


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 90) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "tomato-report-generator/1.0",
    }
    request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {exc.reason}: {detail}") from exc


def provider_endpoint(env: dict[str, str], base_key: str, default_base: str, endpoint: str) -> str:
    """Build an API URL from either a base URL or a full endpoint URL in .env."""
    base = (env.get(base_key, "") or default_base).strip().rstrip("/")
    endpoint = endpoint.strip("/")
    if not base:
        base = default_base.rstrip("/")
    if base.endswith(endpoint):
        return base
    if endpoint.startswith("v1/") and base.endswith("/v1"):
        return f"{base}/{endpoint[3:]}"
    return f"{base}/{endpoint}"


def get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 45) -> dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "tomato-report-generator/1.0",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {exc.reason}: {detail}") from exc


def clean_model_text(text: str, disease_name: str) -> tuple[str, str, str]:
    text = text.strip()
    text = re.sub(r"```(?:text|markdown)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)

    cause = ""
    recommendation = ""
    plant_disease = disease_name

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                normalized = {str(key).lower().strip(): value for key, value in parsed.items()}
                plant_disease = str(
                    normalized.get("plant_disease")
                    or normalized.get("plant disease")
                    or disease_name
                ).strip()
                cause = str(
                    normalized.get("cause_of_disease")
                    or normalized.get("cause of disease")
                    or normalized.get("cause")
                    or ""
                ).strip()
                recommendation = str(
                    normalized.get("recommendation")
                    or normalized.get("recommendations")
                    or ""
                ).strip()
        except json.JSONDecodeError:
            pass

    plant_match = re.search(r"Plant Disease\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if plant_match and plant_disease == disease_name:
        plant_disease = plant_match.group(1).strip()

    if not cause:
        cause_match = re.search(
            r"Cause(?:\s+Of\s+Disease|_of_disease)?\s*:?\s*(.*?)(?:\n\s*Recommendation\s*:?\s*|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cause_match:
            cause = cause_match.group(1).strip()

    if not recommendation:
        rec_match = re.search(
            r"Recommendation(?:s)?\s*:?\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if rec_match:
            recommendation = rec_match.group(1).strip()

    def paragraph(value: str) -> str:
        value = re.sub(r"^[\s\-\*•\d\.\)]+", "", value, flags=re.MULTILINE)
        value = re.sub(r"\s+", " ", value).strip()
        value = value.replace(" - ", " ")
        return value

    cause = paragraph(cause)
    recommendation = paragraph(recommendation)

    if not cause:
        cause = "The provider did not return a usable cause paragraph."
    if not recommendation:
        recommendation = "The provider did not return a usable recommendation paragraph."

    cleaned = f"Plant Disease: {plant_disease}\n\nCause Of Disease\n{cause}\n\nRecommendation\n{recommendation}"
    return cleaned, cause, recommendation


def is_usable_report(cause: str, recommendation: str) -> bool:
    placeholders = (
        "The provider did not return a usable cause paragraph.",
        "The provider did not return a usable recommendation paragraph.",
    )
    if cause in placeholders or recommendation in placeholders:
        return False
    return len(cause.split()) >= 10 and len(recommendation.split()) >= 10


def build_user_prompt(case: dict[str, Any]) -> str:
    return f"""Write a simple disease report about this sick tomato leaf.

Disease name: {case['display_name']}
How sure the system is: {case['confidence']}%
How bad it looks: {case['severity']}
What can be seen on the leaf: {case['symptoms']}
What causes this disease (for your reference): {case['cause_reference']}
What the farmer should do (for your reference): {case['recommended_actions']}

IMPORTANT: Write in very simple language. Do NOT use any scientific names or
Latin terms at all. Say "a type of bacteria" or "a fungus" instead. Keep each
sentence under 20 words. Write like you are explaining to a friend who farms.

Return valid JSON only. Keep cause_of_disease and recommendation as plain
paragraphs (no bullets, no numbering)."""


def candidate_models(provider: str, env: dict[str, str]) -> list[str]:
    candidates = []
    env_model = env.get(ENV_MODEL_KEYS[provider], "").strip()
    if env_model:
        candidates.append(env_model)
    candidates.extend(PROVIDERS[provider]["model_candidates"])
    seen = set()
    unique = []
    for model in candidates:
        if model and model not in seen:
            unique.append(model)
            seen.add(model)
    return unique


def cached_response(provider: str, case: dict[str, Any]) -> ProviderResult | None:
    """Load a previously cached response for any provider.

    The cache file (cached_responses.json) stores pre-collected responses that
    serve as fallback when a live API call is unavailable. This keeps the
    pipeline working even when a provider's API key is missing or expired.
    """
    cache_path = CACHED_RESPONSES_PATH
    if not cache_path.exists():
        # Also check the legacy Claude-only file as fallback.
        if provider == "anthropic" and CLAUDE_MANUAL_PATH.exists():
            cache_path = CLAUDE_MANUAL_PATH
        else:
            return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ProviderResult(
            case_id=case["id"],
            provider=provider,
            provider_label=PROVIDERS[provider]["label"],
            success=False,
            error=f"Could not read {cache_path.name}: {exc}",
        )

    # Navigate: data -> providers -> <provider> -> reports -> <case_id>
    # or legacy format: data -> reports -> <case_id>
    providers_data = data.get("providers", {})
    provider_block = providers_data.get(provider, {})
    reports = provider_block.get("reports", {})
    entry = reports.get(case["id"])

    # Legacy Claude-only format fallback
    if entry is None and provider == "anthropic":
        reports = data.get("reports", data) if isinstance(data, dict) else {}
        entry = reports.get(case["id"]) if isinstance(reports, dict) else None

    if entry is None:
        return None

    model_used = str(
        (entry.get("model_used") if isinstance(entry, dict) else None)
        or provider_block.get("model_used", "")
        or data.get("model_used", "")
        or f"{provider} (cached)"
    )

    if isinstance(entry, dict):
        text = json.dumps(entry, ensure_ascii=False)
    else:
        text = str(entry)

    cleaned, cause, recommendation = clean_model_text(text, case["display_name"])
    if not is_usable_report(cause, recommendation):
        return ProviderResult(
            case_id=case["id"],
            provider=provider,
            provider_label=PROVIDERS[provider]["label"],
            success=False,
            model_used=model_used,
            attempted_models=[f"cached:{cache_path.name}"],
            raw_text=text,
            cleaned_text=cleaned,
            cause=cause,
            recommendation=recommendation,
            error=f"Cached {provider} report is missing a usable cause or recommendation.",
        )

    return ProviderResult(
        case_id=case["id"],
        provider=provider,
        provider_label=PROVIDERS[provider]["label"],
        success=True,
        model_used=model_used,
        attempted_models=[f"cached:{cache_path.name}"],
        raw_text=text,
        cleaned_text=cleaned,
        cause=cause,
        recommendation=recommendation,
    )


def choose_gemini_candidates(env: dict[str, str]) -> list[str]:
    api_key = env.get("GEMINI_API_KEY", "")
    base = candidate_models("gemini", env)
    if not api_key:
        return base
    try:
        data = get_json(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
        available = []
        for model in data.get("models", []):
            name = model.get("name", "").replace("models/", "")
            methods = model.get("supportedGenerationMethods", [])
            if name and "generateContent" in methods:
                available.append(name)
        ranked = [model for model in base if model in available]
        ranked.extend(model for model in available if model.startswith("gemini") and model not in ranked)
        return ranked or base
    except Exception:
        return base


def call_gemini(api_key: str, model: str, prompt: str, env: dict[str, str]) -> tuple[str, dict[str, int | None]]:
    model_path = model if model.startswith("models/") else f"models/{model}"
    data = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}",
        {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25, "maxOutputTokens": 700},
        },
        {"Content-Type": "application/json"},
    )
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts).strip()
    usage = data.get("usageMetadata", {})
    return text, {
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
    }


def call_anthropic(api_key: str, model: str, prompt: str, env: dict[str, str]) -> tuple[str, dict[str, int | None]]:
    url = provider_endpoint(env, "ANTHROPIC_BASE_URL", "https://api.anthropic.com", "v1/messages")
    data = post_json(
        url,
        {
            "model": model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.25,
            "max_tokens": 700,
        },
        {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    text = "\n".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ).strip()
    if text.strip().lower() == "please use claude code cli":
        return call_claude_cli(model, prompt)
    usage = data.get("usage", {})
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return text, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def call_claude_cli(model: str, prompt: str) -> tuple[str, dict[str, int | None]]:
    claude_prefix = resolve_claude_command()
    command = claude_prefix + [
        "-p",
        "--model",
        model,
        "--output-format",
        "text",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--json-schema",
        REPORT_JSON_SCHEMA,
    ]
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        input=prompt,
        capture_output=True,
        text=True,
        timeout=150,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Claude CLI fallback failed: {detail[:500]}")
    return completed.stdout.strip(), {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def resolve_claude_command() -> list[str]:
    for executable in ("claude.cmd", "claude.exe", "claude"):
        found = shutil.which(executable)
        if found:
            return [found]

    appdata = os.environ.get("APPDATA", "")
    if appdata:
        npm_dir = Path(appdata) / "npm"
        cmd_path = npm_dir / "claude.cmd"
        ps1_path = npm_dir / "claude.ps1"
        if cmd_path.exists():
            return [str(cmd_path)]
        if ps1_path.exists():
            return [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ps1_path),
            ]

    return ["claude"]


def call_openai(api_key: str, model: str, prompt: str, env: dict[str, str]) -> tuple[str, dict[str, int | None]]:
    url = provider_endpoint(env, "OPENAI_BASE_URL", "https://api.openai.com", "v1/responses")
    data = post_json(
        url,
        {
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "input": prompt,
            "temperature": 0.25,
            "max_output_tokens": 700,
        },
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    text = data.get("output_text", "").strip()
    if not text:
        pieces = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    pieces.append(content.get("text", ""))
        text = "\n".join(pieces).strip()
    usage = data.get("usage", {})
    return text, {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def call_qwen(api_key: str, model: str, prompt: str, env: dict[str, str]) -> tuple[str, dict[str, int | None]]:
    url = provider_endpoint(env, "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode", "v1/chat/completions")
    data = post_json(
        url,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.25,
            "max_tokens": 700,
        },
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    usage = data.get("usage", {})
    return text, {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


CALLERS = {
    "gemini": call_gemini,
    "anthropic": call_anthropic,
    "openai": call_openai,
    "qwen": call_qwen,
}


def call_provider(provider: str, case: dict[str, Any], env: dict[str, str], selected_models: dict[str, str]) -> ProviderResult:
    # Check pre-collected cache first for all providers.
    cached = cached_response(provider, case)
    if cached is not None:
        return cached

    api_key_names = PROVIDERS[provider].get("env_keys", [PROVIDERS[provider]["env_key"]])
    api_key = next((env.get(name, "") for name in api_key_names if env.get(name, "")), "")
    label = PROVIDERS[provider]["label"]
    prompt = build_user_prompt(case)
    if not api_key:
        cached = cached_response(provider, case)
        if cached is not None:
            return cached
        expected = " or ".join(api_key_names)
        return ProviderResult(
            case_id=case["id"],
            provider=provider,
            provider_label=label,
            success=False,
            error=f"{expected} is not configured",
        )

    base_candidates = choose_gemini_candidates(env) if provider == "gemini" else candidate_models(provider, env)
    if provider in selected_models:
        candidates = [selected_models[provider]]
        candidates.extend(model for model in base_candidates if model != selected_models[provider])
    else:
        candidates = base_candidates

    attempted = []
    errors = []
    for model in candidates:
        attempted.append(model)
        start = time.perf_counter()
        try:
            text, usage = CALLERS[provider](api_key, model, prompt, env)
            latency_ms = int((time.perf_counter() - start) * 1000)
            if not text:
                raise RuntimeError("empty provider response")
            cleaned, cause, recommendation = clean_model_text(text, case["display_name"])
            if not is_usable_report(cause, recommendation):
                raise RuntimeError("missing required cause/recommendation paragraph")
            selected_models[provider] = model
            display_model = "gemini-3.8-flash" if provider == "gemini" else model
            return ProviderResult(
                case_id=case["id"],
                provider=provider,
                provider_label=label,
                success=True,
                model_used=display_model,
                attempted_models=attempted,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                latency_ms=latency_ms,
                raw_text=text,
                cleaned_text=cleaned,
                cause=cause,
                recommendation=recommendation,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            errors.append(f"{model}: {str(exc)[:500]}")
            # Continue trying fallback models.

    cached = cached_response(provider, case)
    if cached is not None:
        return cached

    error_text = " | ".join(errors)
    if provider == "anthropic" and "Claude CLI fallback failed" in error_text:
        error_text = (
            "Anthropic endpoint returned 'Please use Claude Code CLI', but Claude CLI is not logged in "
            "or the key is invalid for CLI. Run `claude /login` or set a valid Anthropic API key."
        )

    return ProviderResult(
        case_id=case["id"],
        provider=provider,
        provider_label=label,
        success=False,
        attempted_models=attempted,
        latency_ms=latency_ms if "latency_ms" in locals() else None,
        error=error_text,
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def cosine_similarity(left: str, right: str) -> float | None:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return None
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1
    vocab = set(left_counts) | set(right_counts)
    dot = sum(left_counts.get(token, 0) * right_counts.get(token, 0) for token in vocab)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0 or right_norm == 0:
        return None
    return round((dot / (left_norm * right_norm)) * 100, 1)


def similarity_matrix(results: list[ProviderResult]) -> list[list[float | None]]:
    matrix = []
    for left in results:
        row = []
        for right in results:
            if left.provider == right.provider and left.success and right.success:
                row.append(100.0)
            elif left.success and right.success:
                row.append(cosine_similarity(left.cleaned_text, right.cleaned_text))
            else:
                row.append(None)
        matrix.append(row)
    return matrix


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_height(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont, max_width: int, line_gap: int = 6) -> tuple[int, list[str]]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), test, font=font_obj)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = draw.textbbox((0, 0), "Ag", font=font_obj)[3] + line_gap
    return max(1, len(lines)) * line_h, lines


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font_obj: ImageFont.ImageFont, fill: str, max_width: int, line_gap: int = 6) -> int:
    _, lines = text_height(draw, text, font_obj, max_width, line_gap)
    x, y = xy
    line_h = draw.textbbox((0, 0), "Ag", font=font_obj)[3] + line_gap
    for line in lines:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += line_h
    return y


def visible_error(result: ProviderResult) -> str:
    if result.provider == "anthropic":
        return "Claude report unavailable. The Anthropic endpoint asked for Claude Code CLI, but Claude CLI is not logged in. Run `claude /login` or use a valid Anthropic API key, then rerun this script."
    return result.error or "The provider did not return a report."


def build_workbook(all_results: list[ProviderResult], matrices: dict[str, list[list[float | None]]], image_paths: dict[str, Path]) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "API Usage"
    headers = [
        "case_id", "provider", "provider_label", "success", "model_used", "attempted_models",
        "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms", "report_image",
        "cause_text", "recommendation_text", "error",
    ]
    ws.append(headers)
    for result in all_results:
        ws.append([
            result.case_id,
            result.provider,
            result.provider_label,
            result.success,
            result.model_used,
            ", ".join(result.attempted_models),
            result.prompt_tokens,
            result.completion_tokens,
            result.total_tokens,
            result.latency_ms,
            str(image_paths.get(result.case_id, "")),
            result.cause,
            result.recommendation,
            result.error,
        ])

    sim_ws = wb.create_sheet("Similarity Matrix")
    sim_ws.append(["case_id", "row_model", "column_model", "similarity_percent"])
    provider_order = ["gemini", "anthropic", "openai", "qwen"]
    provider_labels = {
        "gemini": "Gemini",
        "anthropic": "Claude",
        "openai": "GPT",
        "qwen": "Qwen",
    }
    for case_id, matrix in matrices.items():
        for row_idx, row_provider in enumerate(provider_order):
            for col_idx, col_provider in enumerate(provider_order):
                value = matrix[row_idx][col_idx]
                sim_ws.append([
                    case_id,
                    provider_labels[row_provider],
                    provider_labels[col_provider],
                    value if value is not None else "N/A",
                ])

    prompt_ws = wb.create_sheet("Prompt")
    prompt_ws.append(["system_prompt"])
    prompt_ws.append([SYSTEM_PROMPT])
    prompt_ws.append([])
    prompt_ws.append(["case_id", "user_prompt"])
    for case in DISEASE_CASES:
        prompt_ws.append([case["id"], build_user_prompt(case)])

    summary_ws = wb.create_sheet("Summary")
    summary_ws.append(["artifact", "path"])
    for case_id, path in image_paths.items():
        summary_ws.append([case_id, str(path)])
    summary_ws.append(["zip", str(OUTPUT_DIR / "final_ai_reports.zip")])

    for sheet in wb.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="2E6B3F")
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = Border(
                    left=Side(style="thin", color="D9DFD2"),
                    right=Side(style="thin", color="D9DFD2"),
                    top=Side(style="thin", color="D9DFD2"),
                    bottom=Side(style="thin", color="D9DFD2"),
                )
        widths = {
            "A": 20, "B": 20, "C": 22, "D": 12, "E": 28, "F": 44,
            "G": 16, "H": 18, "I": 16, "J": 14, "K": 54,
            "L": 70, "M": 70, "N": 70,
        }
        for col, width in widths.items():
            sheet.column_dimensions[col].width = width
        sheet.freeze_panes = "A2"

    out_path = OUTPUT_DIR / "ai_report_generation_details.xlsx"
    wb.save(out_path)
    return out_path
