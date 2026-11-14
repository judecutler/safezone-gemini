"""Gemini API client: one model does object detection with masks and structured vision reasoning.

Every call is a `generateContent` request over the project's Transport, so the
record/replay fixtures work exactly as they did for the previous providers.
Detection asks for `box_2d` in Gemini's native [ymin, xmin, ymax, xmax] on a
0..1000 grid plus an optional base64 PNG mask per object; reasoning calls
constrain the answer to a JSON schema with `responseJsonSchema`.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .shapes import Provenance
from .transport import Transport, get_json, post_json


class GeminiUnavailable(RuntimeError):
    """No API key, or the API did not answer. Callers degrade instead of failing the scan."""


@dataclass(frozen=True)
class Detection:
    label: str
    """One of the phrases the caller asked for."""
    box: tuple[float, float, float, float]
    """(left, top, right, bottom) normalized 0..1 of the image."""
    confidence: float | None
    mask_png: bytes | None
    """PNG of the object's mask, sized to the box, or None when the model gave no mask."""


DETECT_SYSTEM = """You are an object detector for home safety. Find every instance of the requested object
types in the photo. For each one return its label (exactly one of the requested types), box_2d as
[ymin, xmin, ymax, xmax] on a 0..1000 grid, a confidence 0..1, and mask: a base64 PNG segmentation mask
of the object cropped to its box (white = object), or an empty string if you cannot produce one.
Only report objects that are clearly visible. Do not report object types that were not requested."""

DETECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["objects"],
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "required": ["label", "box_2d", "confidence", "mask"],
        "properties": {
            "label": {"type": "string"},
            "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
            "confidence": {"type": "number"},
            "mask": {"type": "string"},
        }}}},
}


@dataclass(frozen=True)
class GeminiClient:
    transport: Transport
    settings: Settings

    def _headers(self) -> dict[str, str]:
        if not self.settings.gemini_api_key:
            return {}
        return {"x-goog-api-key": self.settings.gemini_api_key}

    def _url(self, model: str) -> str:
        return f"{self.settings.gemini_api_base}/models/{model}:generateContent"

    def list_models(self) -> list[str]:
        data, _ = get_json(self.transport, f"{self.settings.gemini_api_base}/models", headers=self._headers())
        return sorted(m["name"].removeprefix("models/") for m in data.get("models", []))

    def generate_json(self, jpeg: bytes, system: str, user: str, schema: dict[str, Any],
                      thinking: str = "low", timeout_s: float = 120) -> tuple[Any, Provenance]:
        """One image + prompt call constrained to `schema`. Returns the parsed JSON and its provenance."""
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
                {"text": user},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "thinkingConfig": {"thinkingLevel": thinking},
                "temperature": 0.2,
            },
        }
        data, provenance = post_json(self.transport, self._url(self.settings.vision_model), body,
                                     self._headers(), timeout_s=timeout_s)
        return json.loads(response_text(data)), provenance

    def vision_json(self, jpeg: bytes, system: str, user: str, schema_name: str,
                    schema: dict[str, Any], effort: str = "low") -> tuple[Any, Provenance]:
        """Same contract the classifier and go-bag check used before: image + prompt -> JSON under `schema`."""
        del schema_name  # Gemini names nothing; kept so callers read the same.
        return self.generate_json(jpeg, system, user, gemini_schema(schema), thinking=effort)

    def detect(self, jpeg: bytes, phrases: list[str]) -> tuple[list[Detection], Provenance]:
        """Every requested object type in one call. Objects with unknown labels or bad boxes are dropped."""
        user = "Requested object types: " + json.dumps(phrases)
        data, provenance = self.generate_json(jpeg, DETECT_SYSTEM, user, DETECT_SCHEMA, thinking="low",
                                              timeout_s=180)
        return parse_detections(data, phrases), provenance


def response_text(data: dict[str, Any]) -> str:
    """The text of the first candidate. A blocked or empty answer raises GeminiUnavailable."""
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise GeminiUnavailable(f"Gemini gave no answer: {reason}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise GeminiUnavailable(f"Gemini answer was empty: {candidates[0].get('finishReason', '?')}")
    return text


def parse_detections(data: Any, phrases: list[str]) -> list[Detection]:
    allowed = {p.lower(): p for p in phrases}
    out: list[Detection] = []
    for item in (data or {}).get("objects", []) or []:
        label = allowed.get(str(item.get("label", "")).strip().lower())
        box = item.get("box_2d")
        if label is None or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, float(v))) / 1000.0 for v in box)
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        confidence = item.get("confidence")
        conf = min(1.0, max(0.0, float(confidence))) if isinstance(confidence, (int, float)) else None
        out.append(Detection(label, (xmin, ymin, xmax, ymax), conf, decode_mask(item.get("mask"))))
    return out


def decode_mask(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value.strip():
        return None
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        raw = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError):
        return None
    return raw if raw.startswith(b"\x89PNG") else None


def gemini_schema(schema: Any) -> Any:
    """Strict OpenAI-style schemas carry keys Gemini's JSON-schema support does not want; strip them."""
    if isinstance(schema, dict):
        return {k: gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [gemini_schema(v) for v in schema]
    return schema
