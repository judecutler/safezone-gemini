"""Gemini inspection agent: function calling over the inference service, structured report + free-form Q&A."""
from __future__ import annotations

import json
import time

from gemini.client import client, unavailable, MODEL_FAST
from gemini.report_schema import InspectionReport, SelfAudit
from gemini.render import heatmap_png

SYSTEM = (
    "You are a senior non-destructive-testing engineer operating GoodVibes, a non-contact inspection system for sealed, "
    "opaque containers. A 10x10 grid of laser points measures the surface vibration of the container wall under "
    "acoustic excitation (50-1000 Hz chirp); a Vibration Transformer localizes hidden contents (position on an nx×ny grid "
    "behind the wall) and reports a confidence; low confidence means the container is flagged for manual QA. You have tools: "
    "run_inference, compare_with_empty_reference, get_point_spectrum, get_metrics, and get_ground_truth (only if the "
    "user explicitly asks to reveal). Always call run_inference and compare_with_empty_reference before concluding. "
    "Cite specific laser indices and frequency bands as evidence. Report the model's calibrated uncertainty honestly; "
    "never claim more than the tools show. Context: industrial / pharmaceutical quality inspection only."
)


class Agent:
    def __init__(self, inf):
        self.inf = inf

    # ---- tools (plain Python functions; the SDK derives the schema from signatures + docstrings) ----
    def _tools(self, allow_truth: bool):
        inf = self.inf

        def run_inference(sample_id: int) -> dict:
            """Run the Vibration Transformer on a recorded sample. Returns the estimated object position pos_xy (x=column
            0..nx-1, y=row 0..ny-1), a localization confidence (peak of the occupancy heatmap, <0.45 = flag for QA),
            and the 10x10 occupancy heatmap over the laser grid."""
            return inf.predict(sample_id)["pred"]

        def compare_with_empty_reference(sample_id: int) -> dict:
            """Per-band vibration energy change (dB) of each laser point vs the mean empty-container signature,
            plus the 8 strongest deviations. Negative = damped (mass/contact), positive = amplified."""
            return inf.compare_with_empty(sample_id)

        def get_point_spectrum(sample_id: int, laser_idx: int) -> dict:
            """Two-axis vibration spectrum (50-1000 Hz, log magnitude) of one laser point and the empty reference."""
            return inf.point_spectrum(sample_id, laser_idx)

        def get_metrics() -> dict:
            """Held-out evaluation metrics for all models and splits (S1 unseen excitation, S2 unseen location, S3 random)."""
            return inf.metrics()

        def get_ground_truth(sample_id: int) -> dict:
            """Reveal the recorded ground truth for a sample (only when the user explicitly asks to reveal)."""
            return inf.truth(sample_id)

        tools = [run_inference, compare_with_empty_reference, get_point_spectrum, get_metrics]
        if allow_truth:
            tools.append(get_ground_truth)
        return tools

    def report(self, sample_id: int, model: str = MODEL_FAST) -> dict:
        c = client()
        if c is None:
            return unavailable()
        from google.genai import types
        t0 = time.time()
        try:
            # pass 1: tool use, free text
            r1 = c.models.generate_content(
                model=model,
                contents=f"Inspect sample {sample_id}. Use the tools, then summarize your findings.",
                config=types.GenerateContentConfig(system_instruction=SYSTEM, tools=self._tools(False), temperature=0.2),
            )
            # pass 2: force the structured report from the findings
            r2 = c.models.generate_content(
                model=model,
                contents=f"Findings:\n{r1.text}\n\nProduce the InspectionReport JSON.",
                config=types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                                                   response_schema=InspectionReport, temperature=0.1),
            )
            rep = InspectionReport.model_validate_json(r2.text)
            return {"available": True, "model": model, "latency_s": round(time.time() - t0, 2),
                    "narrative": r1.text, "report": rep.model_dump()}
        except Exception as e:
            return unavailable(f"{type(e).__name__}: {e}")

    def chat(self, sample_id: int, history: list[dict], message: str, allow_truth: bool = False,
             model: str = MODEL_FAST) -> dict:
        """history: [{"role": "user"|"model", "text": str}, ...]"""
        c = client()
        if c is None:
            return unavailable()
        from google.genai import types
        contents = [types.Content(role=h["role"], parts=[types.Part.from_text(text=h["text"])]) for h in history]
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"(current sample_id={sample_id}) {message}")]))
        try:
            r = c.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM, tools=self._tools(allow_truth), temperature=0.3),
            )
            return {"available": True, "text": r.text}
        except Exception as e:
            return unavailable(f"{type(e).__name__}: {e}")

    def self_audit(self, sample_id: int, model: str = MODEL_FAST) -> dict:
        """Multimodal grading: predicted heatmap vs the overhead photo (or ground-truth occupancy if no photo)."""
        c = client()
        if c is None:
            return unavailable()
        from google.genai import types
        import numpy as np
        pred = self.inf.predict(sample_id)["pred"]
        truth = self.inf.truth(sample_id)
        parts = ["Image 1: predicted occupancy heatmap over the 10x10 laser grid (brighter = object).",
                 types.Part.from_bytes(data=heatmap_png(np.asarray(pred["occ"]), self.inf.grid_rc), mime_type="image/png")]
        if truth.get("overhead_path"):
            parts += ["Image 2: overhead photo of the open box (ground truth).",
                      types.Part.from_bytes(data=open(truth["overhead_path"], "rb").read(), mime_type="image/png")]
        else:
            parts += ["Image 2: ground-truth occupancy on the same grid.",
                      types.Part.from_bytes(data=heatmap_png(np.asarray(truth["occ_grid"]).ravel(), np.array([[r, c] for r in range(10) for c in range(10)]), "ground truth"), mime_type="image/png")]
        parts.append("Do the images agree on whether an object is present and roughly where? JSON.")
        try:
            r = c.models.generate_content(model=model, contents=parts,
                                          config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                             response_schema=SelfAudit, temperature=0.1))
            return {"available": True, **SelfAudit.model_validate_json(r.text).model_dump()}
        except Exception as e:
            return unavailable(f"{type(e).__name__}: {e}")
