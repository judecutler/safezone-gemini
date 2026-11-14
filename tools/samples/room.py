"""Hand-authored SAMPLE answers from Gemini for the demo photo only.

Boxes were drawn by looking at demo/living_room.jpg (1600x900) and are given in
Gemini's native box_2d form, [ymin, xmin, ymax, xmax] on a 0..1000 grid, with no
masks because masks cannot be authored by hand. The reasoning answer is what a
careful model should say under rules.CLASSIFIER_RULES: no table or desk, so no
cover. Record real responses with a key to replace these.
"""
from __future__ import annotations

import base64
import json

from safezone.config import ROOT
from safezone.transport import Request, Response

DEMO_IMAGE = ROOT / "demo" / "living_room.jpg"
W, H = 1600, 900


def _box(label: str, l: int, t: int, r: int, b: int, c: float) -> dict:
    return {"label": label, "confidence": c, "mask": "",
            "box_2d": [round(t * 1000 / H), round(l * 1000 / W), round(b * 1000 / H), round(r * 1000 / W)]}


LEFT_WARDROBE = (164, 188, 668, 812)
MIDDLE_WARDROBE = (708, 252, 868, 788)

OBJECTS: list[dict] = [
    _box("bookshelf", 1136, 238, 1378, 548, 0.84),
    _box("cabinet", 168, 192, 664, 806, 0.71),
    _box("cabinet", 710, 256, 866, 786, 0.68),
    _box("cabinet", 576, 140, 854, 256, 0.63),
    _box("wardrobe", *LEFT_WARDROBE, 0.91),
    _box("wardrobe", *MIDDLE_WARDROBE, 0.89),
    _box("ceiling light", 774, 4, 804, 44, 0.77),
    _box("bed", 0, 552, 544, 900, 0.93),
    _box("bed", 1040, 512, 1600, 900, 0.92),
    _box("box", 1220, 130, 1356, 310, 0.9),
    _box("box", 1182, 164, 1224, 308, 0.72),
    _box("box", 194, 166, 270, 216, 0.55),
    _box("door", 884, 232, 1108, 744, 0.88),
]

# Ids after dedupe: bookshelf-0, cabinet-0 (the open hutch above the wardrobes), wardrobe-0 (left, open),
# wardrobe-1 (middle, closed), ceiling-light-0, bed-0 (left), bed-1 (right), box-0, box-1, box-2 (basket), door-0.
SEGMENTS = [
    ("bookshelf-0", "hazard", "tall_furniture", "Tall wire shelf with cartons on top; no wall anchor is visible."),
    ("cabinet-0", "hazard", "heavy_up_high", "Open shelf above the wardrobes holds bottles and bags."),
    ("wardrobe-0", "hazard", "tall_furniture", "Tall wooden wardrobe, door open, no wall anchor visible."),
    ("wardrobe-1", "hazard", "tall_furniture", "Tall wooden wardrobe next to the door, no wall anchor visible."),
    ("ceiling-light-0", "neutral", None, "Small light mounted flush to the ceiling; not a falling hazard."),
    ("bed-0", "neutral", None, "A bed is not cover."),
    ("bed-1", "neutral", None, "A bed is not cover."),
    ("box-0", "hazard", "heavy_up_high", "Large carton stored on top of the wire shelf."),
    ("box-1", "hazard", "heavy_up_high", "Carton stored on top of the wire shelf."),
    ("box-2", "hazard", "heavy_up_high", "Basket with bottles on top of the left wardrobe."),
    ("door-0", "neutral", None, "The room's door; not a safe spot."),
]

RELATIONS = [
    ("in_tip_zone", "bed-0", "wardrobe-0", "The left bed is right in front of the open wardrobe; if it tips it lands on the bed."),
    ("in_tip_zone", "bed-1", "bookshelf-0", "The wire shelf stands at the head of the right bed and could fall onto it."),
    ("under_stored_items", "bed-1", "box-0", "The cartons on top of the wire shelf are above the head of the right bed."),
    ("blocks_exit", "door-0", "wardrobe-1", "The wardrobe beside the door could fall across the way out."),
]

ANSWER = {
    "segments": [{"id": i, "role": r, "hazard_kind": k, "reason": why} for i, r, k, why in SEGMENTS],
    "relations": [{"kind": k, "subject_id": s, "hazard_id": h, "note": n} for k, s, h, n in RELATIONS],
    # The only clear floor is the carpet between the beds. The code's tipping-zone guard rejects it,
    # which is the honest outcome for this room.
    "best_spot": {"x": 0.45, "y": 0.96, "description": "the open carpet between the two beds",
                  "reason": "Clear floor away from windows with nothing stored above it."},
    "exit_note": "The way to the door runs between the beds, past the middle wardrobe and the wire shelf; either could fall across it.",
}


def generate_content(content: dict) -> str:
    return json.dumps({"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(content)}]},
                                       "finishReason": "STOP"}],
                       "modelVersion": "sample"})


def respond(request: Request) -> Response | None:
    if request.method != "POST" or not request.body or ":generateContent" not in request.url:
        return None
    body = json.loads(request.body)
    demo = base64.b64encode(DEMO_IMAGE.read_bytes()).decode()
    parts = body["contents"][-1]["parts"]
    if not any(p.get("inlineData", {}).get("data") == demo for p in parts):
        return None
    text = next((p["text"] for p in parts if "text" in p), "")
    if text.startswith("Requested object types:"):
        return Response(200, generate_content({"objects": OBJECTS}), "sample")
    if '"segments"' in text:
        return Response(200, generate_content(ANSWER), "sample")
    return None
