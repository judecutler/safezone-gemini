"""Gemini detection: one call, boxes in Gemini's 0..1000 form, masks become outlines, overlaps dedupe."""
import base64
import io
import json

import pytest
from PIL import Image

from safezone.config import Settings
from safezone.gemini import GeminiClient, GeminiUnavailable, parse_detections
from safezone.room.segment import MASKS_NOT_DECODED, PHRASES, mask_to_outline, segment_image
from safezone.transport import Response
from tools.samples.room import generate_content

SETTINGS = Settings(gemini_api_key=None, gemini_api_base="https://gemini.test/v1beta", vision_model="g")


class Answer:
    def __init__(self, objects, status=200):
        self.objects, self.status, self.seen = objects, status, []

    def send(self, request):
        self.seen.append(request)
        if self.status >= 400:
            return Response(self.status, "boom", "live")
        return Response(200, generate_content({"objects": self.objects}), "recorded")


def obj(label, ymin, xmin, ymax, xmax, c=0.9, mask=""):
    return {"label": label, "box_2d": [ymin, xmin, ymax, xmax], "confidence": c, "mask": mask}


def scan(objects, size=(100, 100), status=200):
    return segment_image(b"jpeg", size, GeminiClient(Answer(objects, status), SETTINGS))


def mask_png(w, h, fill):
    """White rectangle `fill` = (l, t, r, b) in mask pixels on a black mask."""
    img = Image.new("L", (w, h), 0)
    for y in range(fill[1], fill[3]):
        for x in range(fill[0], fill[2]):
            img.putpixel((x, y), 255)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def test_box_2d_is_ymin_xmin_ymax_xmax_on_a_thousand_grid():
    result = scan([obj("bed", 700, 0, 1000, 500)])
    (bed,) = result.segments
    assert bed.id == "bed-0" and bed.bbox == (0.0, 70.0, 50.0, 100.0)
    assert bed.touches_floor and bed.height_ratio == 0.3 and bed.outline is None
    assert result.notes == [] and result.provenance == "recorded"


def test_one_request_carries_every_phrase():
    t = Answer([])
    segment_image(b"jpeg", (10, 10), GeminiClient(t, SETTINGS))
    assert len(t.seen) == 1
    body = json.loads(t.seen[0].body)
    assert json.loads(body["contents"][0]["parts"][1]["text"].split(": ", 1)[1]) == PHRASES
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_unknown_labels_and_bad_boxes_are_dropped():
    recs = parse_detections({"objects": [obj("sofa", 0, 0, 10, 10), obj("bed", 50, 50, 10, 10),
                                         {"label": "bed", "box_2d": [1, 2], "confidence": 1, "mask": ""},
                                         obj("Bed", 0, 0, 100, 100, c=1.4)]}, PHRASES)
    assert [(r.label, r.confidence) for r in recs] == [("bed", 1.0)]


def test_overlapping_boxes_keep_the_most_confident_and_ids_count_per_label():
    result = scan([obj("cabinet", 0, 0, 500, 500, c=0.6), obj("wardrobe", 0, 0, 520, 500, c=0.9),
                   obj("wardrobe", 500, 500, 1000, 1000, c=0.5)])
    assert [s.id for s in result.segments] == ["wardrobe-0", "wardrobe-1"]


def test_upstream_failure_is_unavailable_not_an_empty_room():
    with pytest.raises(GeminiUnavailable):
        scan([], status=500)


def test_mask_becomes_outline_area_and_centroid_in_image_pixels():
    png = base64.b64encode(mask_png(10, 10, (0, 0, 5, 10))).decode()
    result = scan([obj("table", 0, 0, 1000, 1000, mask=png)])
    (table,) = result.segments
    assert table.outline.startswith("M ") and table.outline.endswith(" Z")
    assert table.area_px == 5000.0 and table.centroid == (25.0, 50.0)
    assert result.notes == []


def test_unreadable_mask_falls_back_to_the_box_with_one_note():
    result = scan([obj("table", 0, 0, 1000, 1000, mask=base64.b64encode(b"\x89PNGnot really").decode())])
    (table,) = result.segments
    assert table.outline is None and table.area_px == 10000.0
    assert result.notes == [MASKS_NOT_DECODED]


def test_data_url_masks_and_empty_masks():
    png = mask_png(4, 4, (0, 0, 4, 4))
    assert mask_to_outline(png, (0, 0, 8, 8))[1] == 64.0
    assert mask_to_outline(mask_png(4, 4, (0, 0, 0, 0)), (0, 0, 8, 8)) is None
    recs = parse_detections({"objects": [obj("bed", 0, 0, 10, 10, mask="data:image/png;base64," + base64.b64encode(png).decode())]}, PHRASES)
    assert recs[0].mask_png == png
