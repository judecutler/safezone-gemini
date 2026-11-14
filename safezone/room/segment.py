"""Gemini detection boundary: one call finds every object type, masks become outlines, boxes dedupe."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from ..gemini import Detection, GeminiClient, GeminiUnavailable
from ..shapes import BBox, ImagePoint, Provenance, Segment, weakest
from ..transport import HttpError

PHRASES = [
    "table", "desk", "bookshelf", "cabinet", "wardrobe", "window", "glass door", "mirror",
    "ceiling light", "TV", "bed", "box", "door",
]

FLOOR_LINE = 0.62
"""Heuristic: a box whose bottom edge sits in the lower 38% of a level, eye-height photo reaches the floor."""

DEDUPE_IOU = 0.7
MASKS_NOT_DECODED = "Gemini returned no masks for some objects; showing boxes."
MASK_THRESHOLD = 127


@dataclass(frozen=True)
class Segmentation:
    segments: list[Segment]
    notes: list[str]
    provenance: Provenance


def segment_image(jpeg: bytes, image_size: tuple[int, int], client: GeminiClient) -> Segmentation:
    """One Gemini call for all phrases. Raises GeminiUnavailable when the call did not answer."""
    try:
        detections, provenance = client.detect(jpeg, PHRASES)
    except (HttpError, ValueError, KeyError, TypeError) as exc:
        raise GeminiUnavailable(f"Gemini detection failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
    segments, decoded_all = build_segments(detections, image_size)
    notes = [] if decoded_all else [MASKS_NOT_DECODED]
    return Segmentation(segments, notes, provenance)


def build_segments(detections: list[Detection], image_size: tuple[int, int]) -> tuple[list[Segment], bool]:
    """Dedupe overlapping boxes (highest confidence wins), number ids per label in PHRASES order."""
    order = {p: i for i, p in enumerate(PHRASES)}
    kept = _dedupe(detections, image_size)
    kept.sort(key=lambda d: order.get(d.label, len(order)))

    counts: dict[str, int] = {}
    segments = []
    decoded_all = True
    for det in kept:
        slug = re.sub(r"[^a-z0-9]+", "-", det.label.lower()).strip("-")
        n = counts.get(slug, 0)
        counts[slug] = n + 1
        seg, decoded = _segment(f"{slug}-{n}", det, image_size)
        decoded_all &= decoded
        segments.append(seg)
    return segments, decoded_all


def _dedupe(detections: list[Detection], image_size: tuple[int, int]) -> list[Detection]:
    ranked = sorted(detections, key=lambda d: -1.0 if d.confidence is None else d.confidence, reverse=True)
    kept: list[Detection] = []
    for det in ranked:
        box = _scaled_box(det, image_size)
        if all(iou(box, _scaled_box(k, image_size)) <= DEDUPE_IOU for k in kept):
            kept.append(det)
    return kept


def _scaled_box(det: Detection, image_size: tuple[int, int]) -> BBox:
    w, h = image_size
    l, t, r, b = det.box
    return (max(0.0, l * w), max(0.0, t * h), min(float(w), r * w), min(float(h), b * h))


def _segment(seg_id: str, det: Detection, image_size: tuple[int, int]) -> tuple[Segment, bool]:
    w, h = image_size
    box = _scaled_box(det, image_size)
    l, t, r, b = box
    outline, area, centroid = None, (r - l) * (b - t), ((l + r) / 2, (t + b) / 2)
    decoded = det.mask_png is None
    if det.mask_png is not None:
        mask_out = mask_to_outline(det.mask_png, box)
        if mask_out is not None:
            outline, area, centroid = mask_out
            decoded = True
    seg = Segment(id=seg_id, label=det.label, bbox=box, outline=outline, area_px=round(area, 1),
                  centroid=(round(centroid[0], 1), round(centroid[1], 1)),
                  touches_floor=b >= FLOOR_LINE * h, height_ratio=round((b - t) / h, 3),
                  confidence=det.confidence)
    return seg, decoded


def mask_to_outline(png: bytes, box: BBox, rows: int = 24) -> tuple[str, float, ImagePoint] | None:
    """Outline (SVG path in image pixels), area and centroid from a mask PNG stretched over `box`.

    The outline walks the mask's right edge down and left edge up on `rows` evenly spaced scan lines,
    which is enough to draw furniture and keeps the plan small. None when the mask is empty or unreadable.
    """
    try:
        from PIL import Image
        with Image.open(io.BytesIO(png)) as img:
            mask = img.convert("L")
            mw, mh = mask.size
            px = mask.load()
    except Exception:
        return None
    if mw == 0 or mh == 0:
        return None
    l, t, r, b = box
    sx, sy = (r - l) / mw, (b - t) / mh

    total = 0
    cx = cy = 0.0
    right: list[tuple[float, float]] = []
    left: list[tuple[float, float]] = []
    step = max(1, mh // rows)
    for y in range(mh):
        xs = [x for x in range(mw) if px[x, y] > MASK_THRESHOLD]
        if not xs:
            continue
        total += len(xs)
        cx += sum(xs) + 0.5 * len(xs)
        cy += (y + 0.5) * len(xs)
        if y % step == 0 or y == mh - 1:
            right.append((l + (xs[-1] + 1) * sx, t + (y + 0.5) * sy))
            left.append((l + xs[0] * sx, t + (y + 0.5) * sy))
    if total == 0 or len(right) < 2:
        return None
    points = right + left[::-1]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points) + " Z"
    centroid = (l + (cx / total) * sx, t + (cy / total) * sy)
    return path, float(total) * sx * sy, centroid


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
