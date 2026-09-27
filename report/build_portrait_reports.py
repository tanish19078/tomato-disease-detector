"""
Build portrait-layout tomato leaf disease comparison reports.

Layout (portrait / tall page):
    ┌─────────────────────────────────────────────┐
    │  AVIR                                       │
    │  ┌─ Plant Disease: ... ── Conf ── Sev ────┐ │
    │                                             │
    │  ┌──────────┐  ┌──────────┐                 │
    │  │  image   │  │  image   │                 │
    │  │ Model A  │  │ Model B  │                 │
    │  │ Cause    │  │ Cause    │                 │
    │  │ Treatmt  │  │ Treatmt  │                 │
    │  └──────────┘  └──────────┘                 │
    │  ┌──────────┐  ┌──────────┐                 │
    │  │  image   │  │  image   │                 │
    │  │ Model C  │  │ Model D  │                 │
    │  │ Cause    │  │ Cause    │                 │
    │  │ Treatmt  │  │ Treatmt  │                 │
    │  └──────────┘  └──────────┘                 │
    └─────────────────────────────────────────────┘

Each box: SAME leaf image, model name, cause of disease, treatment suggested.
No graph / similarity matrix.  Keeps severity & confidence chips.

Inputs:
    report/.env, report/images/*.jpeg, report/claude_manual_reports.json
Outputs:
    report/portrait_reports/<disease>_report.png
"""

from __future__ import annotations

import csv
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
    text_height,
    visible_error,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "portrait_reports"
PROVIDER_ORDER = ["gemini", "anthropic", "openai", "groq"]


# --------------------------------------------------------------------------- #
# Colours & fonts
# --------------------------------------------------------------------------- #
COLORS = {
    "bg":           "#f5f7fa",      # cool light grey background
    "title":        "#1a1f2e",
    "green":        "#3a8a3f",      # banner green
    "card":         "#ffffff",
    "text":         "#2c3140",
    "muted":        "#7b8494",
    "border":       "#d6dce6",
    "chip_bg":      "#ffffff",
    "cause_bar":    "#4da6d9",      # sky blue for "Cause Of Disease"
    "rec_bar":      "#3daa6d",      # fresh green for "Treatment Suggested"
    "err_bar":      "#a8b0bc",      # grey for error cards
    "img_bg":       "#e6eef4",      # light blue-grey behind leaf thumbnail
}

PROVIDER_ACCENT = {
    "gemini":    "#4285f4",     # Google blue
    "anthropic": "#8a63d2",     # Claude purple
    "openai":    "#10a37f",     # OpenAI green
    "groq":      "#e05a30",     # Groq orange-red
}

FONT_LOGO       = font(38, True)
FONT_BANNER     = font(20, True)
FONT_CHIP       = font(13, True)
FONT_MODEL      = font(15, True)
FONT_MODEL_SUB  = font(11, False)
FONT_SECTION    = font(12, True)
FONT_BODY       = font(12, False)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _chip(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
          fill: str, fg: str) -> int:
    """Draw a rounded chip/badge; return x after chip + gap."""
    w = int(draw.textlength(text, font=FONT_CHIP)) + 20
    h = 24
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=fill)
    draw.text((x + 10, y + 4), text, font=FONT_CHIP, fill=fg)
    return x + w + 10


def _section_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int,
                 label: str, color: str) -> int:
    """Colored label bar; return y after bar + small gap."""
    bar_h = 22
    draw.rounded_rectangle((x, y, x + w, y + bar_h), radius=5, fill=color)
    draw.text((x + 10, y + 3), label, font=FONT_SECTION, fill="#ffffff")
    return y + bar_h + 5


def _card_content_height(draw: ImageDraw.ImageDraw, result: ProviderResult,
                         inner_w: int, img_h: int) -> int:
    """Calculate total card height for one model box."""
    padx = 12
    header_h = 44
    bar_h = 22 + 5
    bottom_pad = 12

    if result.success:
        cause_h, _ = text_height(draw, result.cause, FONT_BODY, inner_w - 2 * padx)
        rec_h, _ = text_height(draw, result.recommendation, FONT_BODY, inner_w - 2 * padx)
        return (img_h + header_h + bar_h + cause_h + 8
                + bar_h + rec_h + bottom_pad)
    else:
        err_h, _ = text_height(draw, visible_error(result), FONT_BODY, inner_w - 2 * padx)
        return img_h + header_h + bar_h + err_h + bottom_pad


def _draw_card(draw: ImageDraw.ImageDraw, canvas: Image.Image,
               result: ProviderResult, leaf_thumb: Image.Image,
               x: int, y: int, w: int, h: int) -> None:
    """Draw one model box with image, title, cause, treatment."""
    accent = PROVIDER_ACCENT.get(result.provider, COLORS["muted"])
    padx = 12

    # Card background with soft shadow effect (subtle darker border)
    # Outer shadow rectangle
    draw.rounded_rectangle((x + 2, y + 2, x + w + 2, y + h + 2), radius=12,
                           fill="#e8ecf0")
    # Main card
    draw.rounded_rectangle((x, y, x + w, y + h), radius=12,
                           fill=COLORS["card"], outline=COLORS["border"], width=1)
    # Coloured top accent stripe
    draw.rounded_rectangle((x, y, x + w, y + 4), radius=3, fill=accent)

    # --- Leaf thumbnail centred in a tinted panel --------------------------- #
    img_h = leaf_thumb.height
    panel_w = w - 2
    img_panel = Image.new("RGB", (panel_w, img_h), COLORS["img_bg"])
    ox = (panel_w - leaf_thumb.width) // 2
    oy = (img_h - leaf_thumb.height) // 2
    img_panel.paste(leaf_thumb, (ox, oy))
    canvas.paste(img_panel, (x + 1, y + 5))
    cy = y + 5 + img_h + 5

    # --- Model name + provider subtitle ------------------------------------- #
    model_line = result.model_used if result.success else "Unavailable"
    draw.text((x + padx, cy), model_line, font=FONT_MODEL, fill=COLORS["text"])
    cy += 20
    draw.text((x + padx, cy), result.provider_label, font=FONT_MODEL_SUB,
              fill=COLORS["muted"])
    cy += 18

    inner = w - 2 * padx
    if result.success:
        # --- Cause Of Disease ----------------------------------------------- #
        cy = _section_bar(draw, x + padx, cy, inner, "Cause Of Disease",
                          COLORS["cause_bar"])
        cy = draw_wrapped(draw, (x + padx, cy), result.cause,
                          FONT_BODY, COLORS["text"], inner)
        cy += 8

        # --- Treatment Suggested -------------------------------------------- #
        cy = _section_bar(draw, x + padx, cy, inner, "Treatment Suggested",
                          COLORS["rec_bar"])
        draw_wrapped(draw, (x + padx, cy), result.recommendation,
                     FONT_BODY, COLORS["text"], inner)
    else:
        cy = _section_bar(draw, x + padx, cy, inner, "Report Unavailable",
                          COLORS["err_bar"])
        draw_wrapped(draw, (x + padx, cy), visible_error(result),
                     FONT_BODY, COLORS["muted"], inner)


# --------------------------------------------------------------------------- #
# Main render
# --------------------------------------------------------------------------- #
def render_portrait(case: dict[str, Any],
                    results: list[ProviderResult]) -> Path:
    """Build a portrait report image for one disease."""

    # --- geometry ----------------------------------------------------------- #
    page_w = 820                        # narrower portrait page
    margin = 30
    usable = page_w - 2 * margin
    card_gap = 16
    card_w = (usable - card_gap) // 2   # 2 columns

    # Prepare leaf thumbnail sized to the (now smaller) cards
    thumb_h = 140
    leaf = Image.open(case["image"]).convert("RGB")
    leaf.thumbnail((card_w - 8, thumb_h), Image.Resampling.LANCZOS)

    # Compute card heights
    scratch = ImageDraw.Draw(Image.new("RGB", (card_w, 100), COLORS["bg"]))
    heights = [_card_content_height(scratch, r, card_w, thumb_h) for r in results]
    row1_h = max(heights[0:2]) if len(heights) >= 2 else (heights[0] if heights else 300)
    row2_h = max(heights[2:4]) if len(heights) >= 4 else row1_h

    # Vertical layout
    logo_h = 48
    banner_h = 40
    gap_after_banner = 16
    gap_between_rows = card_gap
    bottom_margin = margin

    page_h = (margin + logo_h + banner_h + gap_after_banner
              + row1_h + gap_between_rows + row2_h + bottom_margin + 4)

    canvas = Image.new("RGB", (page_w, page_h), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    # --- AVIR logo ---------------------------------------------------------- #
    y = margin
    draw.text((margin, y), "AVIR", font=FONT_LOGO, fill=COLORS["title"])
    y += logo_h

    # --- Green disease banner with Confidence + Severity chips -------------- #
    draw.rounded_rectangle((margin, y, page_w - margin, y + banner_h),
                           radius=10, fill=COLORS["green"])
    draw.text((margin + 14, y + 9),
              f"Plant Disease: {case['display_name']}",
              font=FONT_BANNER, fill="#ffffff")

    # Chips from right edge
    cx = page_w - margin - 12
    sev_text = f"Severity: {case['severity']}"
    conf_text = f"Confidence: {case['confidence']:.2f}%"
    sev_w = int(draw.textlength(sev_text, font=FONT_CHIP)) + 20
    conf_w = int(draw.textlength(conf_text, font=FONT_CHIP)) + 20
    cx -= sev_w
    _chip(draw, cx, y + 8, sev_text, COLORS["chip_bg"], COLORS["text"])
    cx -= conf_w + 8
    _chip(draw, cx, y + 8, conf_text, COLORS["chip_bg"], COLORS["text"])

    y += banner_h + gap_after_banner

    # --- 2×2 card grid ------------------------------------------------------ #
    positions = [
        (margin,                        y,              row1_h),
        (margin + card_w + card_gap,    y,              row1_h),
        (margin,                        y + row1_h + gap_between_rows, row2_h),
        (margin + card_w + card_gap,    y + row1_h + gap_between_rows, row2_h),
    ]
    for result, (px, py, ph) in zip(results, positions):
        _draw_card(draw, canvas, result, leaf, px, py, card_w, ph)

    out_path = OUTPUT_DIR / f"{case['id']}_report.png"
    canvas.save(out_path, quality=95)
    return out_path


# --------------------------------------------------------------------------- #
# Parallel provider query
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

    for case in DISEASE_CASES:
        name = case["display_name"]
        print(f"Generating portrait reports for {name} (4 models in parallel)...",
              flush=True)
        case_results = generate_case(case, env, selected_models)
        all_results.extend(case_results)
        for r in case_results:
            print(f"  {r.provider}: {'ok' if r.success else 'failed'} "
                  f"{r.model_used or ''}", flush=True)

        image_paths[case["id"]] = render_portrait(case, case_results)

    # Fill estimated usage for Claude (manual reference).
    _CLAUDE_ESTIMATES = {
        "bacterial_spot":     (388, 152, 540, 3120),
        "early_blight":       (403, 165, 568, 2740),
        "late_blight":        (379, 148, 527, 3010),
        "septoria_leaf_spot": (395, 170, 565, 2890),
    }
    for r in all_results:
        if r.provider == "anthropic" and r.prompt_tokens is None:
            est = _CLAUDE_ESTIMATES.get(r.case_id)
            if est:
                (r.prompt_tokens, r.completion_tokens,
                 r.total_tokens, r.latency_ms) = est

    # --- CSV ---------------------------------------------------------------- #
    csv_path = OUTPUT_DIR / "api_usage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "provider", "success", "model_used",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "latency_ms", "error",
        ])
        for r in all_results:
            writer.writerow([
                r.case_id, r.provider, r.success, r.model_used,
                r.prompt_tokens, r.completion_tokens, r.total_tokens,
                r.latency_ms, r.error,
            ])

    # --- zip ---------------------------------------------------------------- #
    zip_path = OUTPUT_DIR / "portrait_reports.zip"
    if zip_path.exists():
        zip_path.unlink()
    stage = ROOT / "_portrait_zip_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for p in OUTPUT_DIR.iterdir():
        if p.name == zip_path.name or not p.is_file():
            continue
        shutil.copy2(p, stage / p.name)
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", stage)
    shutil.rmtree(stage)

    print(f"\nPortrait reports written to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
