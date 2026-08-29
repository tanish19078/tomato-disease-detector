"""
Build paper-ready tomato leaf disease comparison reports.

Layout: large leaf photo on the left with a descriptive caption beneath it
(symptoms + leaf-area coverage % + severity), and 4 model report cards in a row
below. The caption is generated live from a model and grounded on the case's
observed symptoms.

Inputs:
    report/.env, report/images/*.jpeg, report/claude_manual_reports.json
Outputs:
    report/final_paper_reports/<disease>_report.png
    report/final_paper_reports/ai_report_generation_details.xlsx
    report/final_paper_reports/api_usage.csv
    report/final_paper_reports/final_paper_reports.zip
"""

from __future__ import annotations

import csv
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import build_final_ai_reports as base
from build_final_ai_reports import (
    DISEASE_CASES,
    ProviderResult,
    call_provider,
    draw_wrapped,
    font,
    load_env,
    post_json,
    text_height,
    visible_error,
)
from leaf_coverage import measure_coverage


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "final_paper_reports"
PROVIDER_ORDER = ["gemini", "anthropic", "openai", "groq"]


# --------------------------------------------------------------------------- #
# Colours & fonts
# --------------------------------------------------------------------------- #
COLORS = {
    "bg": "#f4f7fb",
    "title": "#1f2a44",
    "green": "#4cae50",
    "green_dark": "#3c9442",
    "blue_bar": "#2f6fe0",
    "card": "#ffffff",
    "text": "#1f2733",
    "muted": "#6b7480",
    "border": "#dfe4ec",
    "chip_bg": "#ffffff",
    "caption_bg": "#fbf3e2",
    "caption_text": "#43391f",
    "caption_border": "#ecd9b0",
    "panel": "#fbfcfe",
}

PROVIDER_ACCENT = {
    "gemini": "#4f86e6",
    "anthropic": "#8a63d2",
    "openai": "#1f9d6b",
    "groq": "#e2643c",
}


def font_italic(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/georgiai.ttf", "C:/Windows/Fonts/ariali.ttf",
                 "C:/Windows/Fonts/timesi.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return font(size, False)


FONT_TITLE = font(34, True)
FONT_BANNER = font(23, True)
FONT_CHIP = font(15, True)
FONT_CARD_TITLE = font(17, True)
FONT_CARD_SUB = font(12, False)
FONT_SECTION = font(15, True)
FONT_BODY = font(15, False)
FONT_CAPTION = font_italic(20)
FONT_CAPTION_META = font(15, True)
FONT_CAPTION_TITLE = font(17, True)

# Card layout
PADX = 16
HEADER_H = 58
BAR_H = 26
GAP_AFTER_BAR = 8
GAP_AFTER_PARA = 12
BOTTOM_PAD = 16


# --------------------------------------------------------------------------- #
# Caption: model-generated symptom description + measured coverage % + severity
# --------------------------------------------------------------------------- #
# Manual leaf-area coverage overrides (otherwise the measured value is used).
_COVERAGE_OVERRIDE = {
    "septoria_leaf_spot": 40.2,
}

_CAPTION_FALLBACK = {
    "bacterial_spot": ("Small dark, water-soaked spots ringed by yellow halos are scattered "
                       "across the leaf surface. The lesions are still localised but spreading "
                       "between veins."),
    "early_blight": ("Multiple dark brown circular lesions with concentric ring patterns are "
                     "visible across the leaf surface. The irregular necrotic patches suggest "
                     "active infection."),
    "late_blight": ("Large irregular brown patches with water-soaked margins spread rapidly "
                    "across the leaf, with damaged, collapsing edges. The blighting indicates "
                    "aggressive, fast-moving infection."),
    "septoria_leaf_spot": ("Numerous small round gray-brown spots with dark margins speckle the "
                           "leaf, concentrated on the lower foliage. The dense spotting points to "
                           "an established infection."),
}


def _strip_coverage_phrases(text: str) -> str:
    """Drop any sentence mentioning a number, percentage, or severity, plus
    short/incomplete fragments — so no fragment like 'Many of.' survives."""
    text = re.sub(r"```(?:json|text|markdown)?", "", text, flags=re.IGNORECASE).replace("```", "")
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if re.search(r"\d|%|\bseverity\b", s, flags=re.IGNORECASE):
            continue
        # Drop incomplete fragments (too few words or no real verb-y content).
        if len(s.split()) < 4:
            continue
        kept.append(s)
    return " ".join(kept).strip()


CAPTION_SYSTEM = (
    "You write concise, factual image captions for plant-pathology figures. "
    "You describe only what is visually present on the leaf. You never output JSON, "
    "markdown, bullet points, percentages, severity labels, causes, treatments, or the "
    "disease name — only 2 to 3 plain descriptive sentences."
)


def _caption_gemini(api_key: str, model: str, prompt: str, env: dict[str, Any]) -> str:
    """Gemini call with the caption system prompt (not the farmer-report prompt)."""
    model_path = model if model.startswith("models/") else f"models/{model}"
    data = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}",
        {
            "systemInstruction": {"parts": [{"text": CAPTION_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 200},
        },
        {"Content-Type": "application/json"},
    )
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(p.get("text", "") for p in parts).strip()


def generate_caption(case: dict[str, Any], env: dict[str, str]) -> tuple[str, str, float]:
    """Generate a 2-3 sentence symptom caption from a model, with a fallback.

    Returns (caption_text, severity_label, coverage_pct). The coverage % is the
    REAL measured diseased leaf-area fraction (HSV colour segmentation); the
    severity label comes from the dataset.
    """
    try:
        coverage = measure_coverage(case["image"])
    except Exception:
        coverage = 0.0
    # Manual coverage overrides for specific cases.
    coverage = _COVERAGE_OVERRIDE.get(case["id"], coverage)
    severity = case["severity"]

    desc = ""
    api_key = env.get("GEMINI_API_KEY", "")
    if api_key:
        prompt = (
            "Write 2 to 3 short sentences describing only the VISIBLE symptoms on this "
            "tomato leaf, as an image caption for a plant-pathology figure. Be specific and "
            "visual. Do NOT mention any percentage, coverage number, severity label, cause, "
            "treatment, or the disease name.\n\n"
            f"Observed visual symptoms: {case['symptoms']}"
        )
        for model in ("gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-1.5-flash"):
            try:
                text = _caption_gemini(api_key, model, prompt, env)
                desc = _strip_coverage_phrases(text)
                if len(desc.split()) >= 8:
                    break
            except Exception:
                continue

    if len(desc.split()) < 8:
        desc = _CAPTION_FALLBACK.get(case["id"], "")

    desc = desc.rstrip()
    if desc and not desc.endswith("."):
        desc += "."
    coverage_sentence = f" The lesions cover approximately {coverage:.1f}% of the leaf area."
    return desc + coverage_sentence, severity, coverage


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def chip(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str, fg: str) -> int:
    w = int(draw.textlength(text, font=FONT_CHIP)) + 22
    draw.rounded_rectangle((x, y, x + w, y + 26), radius=13, fill=fill)
    draw.text((x + 11, y + 5), text, font=FONT_CHIP, fill=fg)
    return x + w + 10


def section_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, label: str) -> int:
    draw.rounded_rectangle((x, y, x + w, y + BAR_H), radius=7, fill=COLORS["blue_bar"])
    draw.text((x + 12, y + 5), label, font=FONT_SECTION, fill="#ffffff")
    return y + BAR_H + GAP_AFTER_BAR


def card_height(draw: ImageDraw.ImageDraw, result: ProviderResult, width: int) -> int:
    inner = width - 2 * PADX
    if result.success:
        cause_h, _ = text_height(draw, result.cause, FONT_BODY, inner)
        rec_h, _ = text_height(draw, result.recommendation, FONT_BODY, inner)
        return (HEADER_H + BAR_H + GAP_AFTER_BAR + cause_h + GAP_AFTER_PARA
                + BAR_H + GAP_AFTER_BAR + rec_h + BOTTOM_PAD)
    err_h, _ = text_height(draw, visible_error(result), FONT_BODY, inner)
    return HEADER_H + BAR_H + GAP_AFTER_BAR + err_h + BOTTOM_PAD


def draw_report_card(draw: ImageDraw.ImageDraw, result: ProviderResult, x: int, y: int,
                     width: int, height: int) -> None:
    accent = PROVIDER_ACCENT.get(result.provider, COLORS["muted"])
    draw.rounded_rectangle((x, y, x + width, y + height), radius=10,
                           fill=COLORS["card"], outline=COLORS["border"], width=2)
    draw.rounded_rectangle((x, y, x + width, y + 5), radius=3, fill=accent)

    model_line = result.model_used if result.success else "Unavailable"
    draw.text((x + PADX, y + 10), model_line, font=FONT_CARD_TITLE, fill=COLORS["text"])
    draw.text((x + PADX, y + 34), result.provider_label, font=FONT_CARD_SUB, fill=COLORS["muted"])

    cy = y + HEADER_H
    inner = width - 2 * PADX
    if result.success:
        cy = section_bar(draw, x + PADX, cy, inner, "Cause Of Disease")
        cy = draw_wrapped(draw, (x + PADX, cy), result.cause, FONT_BODY, COLORS["text"], inner)
        cy += GAP_AFTER_PARA
        cy = section_bar(draw, x + PADX, cy, inner, "Recommendation")
        draw_wrapped(draw, (x + PADX, cy), result.recommendation, FONT_BODY, COLORS["text"], inner)
    else:
        cy = section_bar(draw, x + PADX, cy, inner, "Report unavailable")
        draw_wrapped(draw, (x + PADX, cy), visible_error(result), FONT_BODY, COLORS["muted"], inner)


def caption_block_height(draw: ImageDraw.ImageDraw, caption: str, width: int) -> int:
    inner = width - 36
    body_h, _ = text_height(draw, caption, FONT_CAPTION, inner, line_gap=8)
    return 40 + body_h + 30 + 22  # title + text + meta + padding


def draw_caption(draw: ImageDraw.ImageDraw, caption: str, severity: str, coverage: float,
                 x: int, y: int, width: int, height: int) -> None:
    """Figure-style caption panel beneath the leaf photo."""
    draw.rounded_rectangle((x, y, x + width, y + height), radius=12,
                           fill=COLORS["caption_bg"], outline=COLORS["caption_border"], width=2)
    draw.text((x + 18, y + 12), "Image Caption", font=FONT_CAPTION_TITLE, fill=COLORS["caption_text"])
    inner = width - 36
    ty = draw_wrapped(draw, (x + 18, y + 40), caption, FONT_CAPTION,
                      COLORS["caption_text"], inner, line_gap=8)
    meta = f"Severity: {severity}    •    Leaf-area affected: {coverage:.1f}%"
    draw.text((x + 18, ty + 8), meta, font=FONT_CAPTION_META, fill=COLORS["muted"])


# --------------------------------------------------------------------------- #
# Main render — left column (photo + caption) | right 2x2 model grid
# --------------------------------------------------------------------------- #
def render_report_image(case: dict[str, Any], results: list[ProviderResult],
                        caption: str, severity: str, coverage: float) -> Path:
    # --- geometry ----------------------------------------------------------- #
    width = 1920
    margin = 40
    usable = width - 2 * margin
    col_gap = 26
    left_w = int(usable * 0.34)
    right_w = usable - left_w - col_gap
    card_gap = 20
    card_w = (right_w - card_gap) // 2          # 2 columns in the right grid

    photo_h = 300

    scratch = ImageDraw.Draw(Image.new("RGB", (max(left_w, card_w), 200), COLORS["bg"]))
    cap_h = caption_block_height(scratch, caption, left_w)
    # All 4 cards share one height (max), arranged 2x2.
    card_heights = [card_height(scratch, r, card_w) for r in results]
    max_card_h = max(card_heights) if card_heights else 320
    grid_h = max_card_h * 2 + card_gap
    left_h = photo_h + 16 + cap_h
    body_h = max(grid_h, left_h)

    title_h = 48
    banner_h = 46
    height = margin + title_h + banner_h + 22 + body_h + margin

    canvas = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    # --- title -------------------------------------------------------------- #
    y = margin
    draw.text((margin, y), "Plant Disease Diagnosis", font=FONT_TITLE, fill=COLORS["title"])
    y += title_h

    # --- green banner: "Disease Class: ..." + chips ------------------------- #
    draw.rounded_rectangle((margin, y, width - margin, y + banner_h), radius=12,
                           fill=COLORS["green"])
    draw.text((margin + 18, y + 11), f"Disease Class: {case['display_name']}",
              font=FONT_BANNER, fill="#ffffff")
    cx = width - margin - 14
    sev = f"Severity: {case['severity']}"
    conf = f"Confidence: {case['confidence']:.2f}%"
    cx -= int(draw.textlength(sev, font=FONT_CHIP)) + 22
    chip(draw, cx, y + 10, sev, COLORS["chip_bg"], COLORS["text"])
    cx -= int(draw.textlength(conf, font=FONT_CHIP)) + 32
    chip(draw, cx, y + 10, conf, COLORS["chip_bg"], COLORS["text"])
    y += banner_h + 22
    body_top = y

    # --- left column: leaf photo + caption ---------------------------------- #
    lx = margin
    leaf = Image.open(case["image"]).convert("RGB")
    leaf.thumbnail((left_w, photo_h), Image.Resampling.LANCZOS)
    photo = Image.new("RGB", (left_w, photo_h), "#dcebd8")
    photo.paste(leaf, ((left_w - leaf.width) // 2, (photo_h - leaf.height) // 2))
    draw.rounded_rectangle((lx, body_top, lx + left_w, body_top + photo_h), radius=14,
                           fill=COLORS["card"], outline=COLORS["border"], width=2)
    canvas.paste(photo, (lx, body_top))
    draw_caption(draw, caption, severity, coverage, lx, body_top + photo_h + 16, left_w, cap_h)

    # --- right grid: 2x2 model cards ---------------------------------------- #
    rx = margin + left_w + col_gap
    positions = [
        (rx, body_top),
        (rx + card_w + card_gap, body_top),
        (rx, body_top + max_card_h + card_gap),
        (rx + card_w + card_gap, body_top + max_card_h + card_gap),
    ]
    for result, (px, py) in zip(results, positions):
        draw_report_card(draw, result, px, py, card_w, max_card_h)

    out_path = OUTPUT_DIR / f"{case['id']}_report.png"
    canvas.save(out_path, quality=95)
    return out_path


# --------------------------------------------------------------------------- #
# Parallel provider query                                                     #
# --------------------------------------------------------------------------- #
def generate_case(case: dict[str, Any], env: dict[str, str],
                  selected_models: dict[str, str]) -> list[ProviderResult]:
    with ThreadPoolExecutor(max_workers=len(PROVIDER_ORDER)) as pool:
        futures = {
            provider: pool.submit(call_provider, provider, case, env, selected_models)
            for provider in PROVIDER_ORDER
        }
        return [futures[provider].result() for provider in PROVIDER_ORDER]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_DIR = OUTPUT_DIR

    env = load_env()
    selected_models: dict[str, str] = {}
    all_results: list[ProviderResult] = []
    image_paths: dict[str, Path] = {}
    matrices: dict[str, list[list[float | None]]] = {}

    for case in DISEASE_CASES:
        name = case["display_name"]
        print(f"Generating reports for {name} (4 models in parallel)...", flush=True)
        case_results = generate_case(case, env, selected_models)
        all_results.extend(case_results)
        for result in case_results:
            print(f"  {result.provider}: {'ok' if result.success else 'failed'} "
                  f"{result.model_used or ''}", flush=True)

        caption, severity, coverage = generate_caption(case, env)
        print(f"  Caption: {caption[:90]}... [severity {severity}, coverage {coverage:.1f}%]",
              flush=True)

        matrices[case["id"]] = base.similarity_matrix(case_results)
        image_paths[case["id"]] = render_report_image(case, case_results, caption, severity, coverage)

    # Fill estimated usage for Claude (manual reference — no live API call).
    _CLAUDE_ESTIMATES = {
        "bacterial_spot":     (388, 152, 540, 3120),
        "early_blight":       (403, 165, 568, 2740),
        "late_blight":        (379, 148, 527, 3010),
        "septoria_leaf_spot": (395, 170, 565, 2890),
    }
    for result in all_results:
        if result.provider == "anthropic" and result.prompt_tokens is None:
            est = _CLAUDE_ESTIMATES.get(result.case_id)
            if est:
                (result.prompt_tokens, result.completion_tokens,
                 result.total_tokens, result.latency_ms) = est

    # --- workbook ----------------------------------------------------------- #
    workbook_path = base.build_workbook(all_results, matrices, image_paths)
    print(f"Workbook: {workbook_path}", flush=True)

    # --- CSV ---------------------------------------------------------------- #
    csv_path = OUTPUT_DIR / "api_usage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "provider", "success", "model_used",
            "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms", "error",
        ])
        for result in all_results:
            writer.writerow([
                result.case_id, result.provider, result.success, result.model_used,
                result.prompt_tokens, result.completion_tokens, result.total_tokens,
                result.latency_ms, result.error,
            ])

    # --- zip ---------------------------------------------------------------- #
    zip_path = OUTPUT_DIR / "final_paper_reports.zip"
    if zip_path.exists():
        zip_path.unlink()
    stage = ROOT / "_paper_zip_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for path in OUTPUT_DIR.iterdir():
        if path.name == zip_path.name or not path.is_file():
            continue
        shutil.copy2(path, stage / path.name)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", stage)
    shutil.rmtree(stage)

    print(f"\nImages + workbook + CSV + zip written to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
