"""
Measure diseased leaf-area coverage (lesions / spots) from a leaf photo.

Improved, transparent pipeline (no deep learning):

  1. Leaf segmentation: GrabCut seeded from the image centre isolates the main
     in-focus leaf from background foliage / soil. A colour rule is the fallback.

  2. Healthy-tissue model: inside the leaf, the dominant healthy green is found
     from the robust median of the CIE-Lab a* (green-red) and b* (blue-yellow)
     channels. Healthy tomato leaf is strongly green (low a*).

  3. Lesion detection combines complementary cues, all restricted to the leaf:
       - chromatic shift : a* well above the healthy median  -> brown/red lesion
       - chlorosis       : b* high + hue yellow               -> yellowing
       - necrosis        : low lightness (dark dead tissue)
       - local spots     : black-hat / tophat highlights small dark or bright
                           lesions that differ from their local neighbourhood,
                           catching tiny spots a global threshold misses
     The union (minus healthy green) is the diseased mask.

  4. coverage_pct = diseased_pixels / leaf_pixels * 100, with light morphological
     cleanup so isolated single-pixel noise is not counted.

`measure_coverage(path)` returns the percentage (0-100, 1 dp).
`segment(path)` returns (bgr, leaf_mask, diseased_mask) for debugging/overlays.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


MAX_SIDE = 640


def _load(image_path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"could not read {image_path}")
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labels == largest


def _leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of the main (central, in-focus) leaf."""
    h, w = bgr.shape[:2]

    gc = np.zeros((h, w), np.uint8)
    mx, my = int(w * 0.05), int(h * 0.05)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=3)
        leaf = ndimage.binary_fill_holes(_largest_blob(fg))
        if 0.08 <= leaf.mean() <= 0.97:
            return leaf
    except Exception:
        pass

    # Fallback: colour rule (tissue = saturated / greenish / warm, not background).
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, g, r = bgr[..., 0].astype(int), bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    eg = g - (r + b) // 2
    leafish = (((hh >= 25) & (hh <= 95) & (s >= 40)) | ((s >= 55) & (v >= 55)) | (eg > 12)) & (v > 35)
    mask = leafish.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=3)
    return ndimage.binary_fill_holes(_largest_blob(mask))


def _diseased_mask(bgr: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    """Boolean mask of diseased tissue (lesions / spots) within the leaf."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]      # a*: green(-)->red(+), b*: blue(-)->yellow(+)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    if leaf.sum() < 50:
        return np.zeros_like(leaf)

    # Robust healthy-green reference from the leaf interior.
    a_med = float(np.median(A[leaf]))
    b_med = float(np.median(B[leaf]))
    L_med = float(np.median(L[leaf]))

    # 1) Chromatic shift toward red/brown: a* clearly above healthy median.
    brown = (A - a_med) > 14
    # 2) Chlorosis (yellowing): b* well above median AND turning (a* rising), bright.
    chlorosis = ((B - b_med) > 18) & ((A - a_med) > 3) & (val >= 110)
    # 3) Necrosis: distinctly darker than healthy tissue, not deep-green shadow.
    necrosis = (L < L_med - 40) & ~(((hue >= 35) & (hue <= 88)) & (sat >= 70))

    # 4) Local-contrast spots: black-hat surfaces small dark lesions that differ
    #    from their local neighbourhood (great for bacterial / septoria specks).
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    spots = (blackhat > 22) & ((A - a_med) > 4)          # dark + slightly warm = lesion, not vein shadow

    healthy = ((hue >= 35) & (hue <= 88) & (sat >= 45)) & ((A - a_med) <= 8)
    diseased = (brown | chlorosis | necrosis | spots) & ~healthy & leaf

    # Cleanup: drop single-pixel noise, lightly consolidate.
    diseased = cv2.morphologyEx(diseased.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    return diseased & leaf


def segment(image_path: str | Path):
    """Return (bgr, leaf_mask, diseased_mask) for the resized image."""
    bgr = _load(image_path)
    leaf = _leaf_mask(bgr)
    diseased = _diseased_mask(bgr, leaf)
    return bgr, leaf, diseased


def measure_coverage(image_path: str | Path) -> float:
    """Return diseased leaf-area coverage as a percentage (0-100, 1 dp)."""
    _, leaf, diseased = segment(image_path)
    leaf_px = int(leaf.sum())
    if leaf_px == 0:
        return 0.0
    pct = float(np.clip(diseased.sum() / leaf_px * 100.0, 0.0, 100.0))
    if 0.0 < pct < 1.0:
        pct = 1.0
    return round(pct, 1)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent / "images"
    cases = {
        "bacterial_spot": "tomato bacterial.jpeg",
        "early_blight": "tomato early blight.jpeg",
        "late_blight": "tomato late blight.jpeg",
        "septoria_leaf_spot": "tomato septoria spot.jpeg",
    }
    for cid, fname in cases.items():
        print(f"{cid:20s} {measure_coverage(here / fname):5.1f}%")
