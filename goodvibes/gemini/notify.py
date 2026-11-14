"""Push flagged inspections into the QA workflow: a Google Chat incoming webhook (or any webhook) and a CSV batch log.

Env: QA_WEBHOOK_URL (Google Chat space incoming-webhook URL; optional), QA_BATCH_LOG (CSV path, default results/batch_log.csv).
Always degrades gracefully — never raises into the request path.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import urllib.request

LOG_PATH = os.environ.get("QA_BATCH_LOG", "results/batch_log.csv")


def batch_log(sample_id: int, pred: dict, disposition: str, report: dict | None = None) -> dict:
    """Append one row to the batch record (21 CFR Part 11-style: timestamp, id, result, who/what decided)."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        new = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp_utc", "container_id", "disposition", "pos_x", "pos_y", "confidence",
                            "verdict", "recommended_action", "system"])
            w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), sample_id, disposition,
                        pred.get("pos_xy", [None, None])[0], pred.get("pos_xy", [None, None])[1], pred.get("confidence"),
                        (report or {}).get("verdict", ""), (report or {}).get("recommended_action", ""),
                        "GoodVibes VibrationTransformer + gemini-3.8-flash"])
        return {"logged": True, "path": LOG_PATH}
    except Exception as e:
        return {"logged": False, "reason": f"{type(e).__name__}: {e}"}


def notify_qa(sample_id: int, pred: dict, report: dict | None = None) -> dict:
    """Post a flagged container to the QA channel (Google Chat incoming webhook)."""
    url = os.environ.get("QA_WEBHOOK_URL")
    if not url:
        return {"sent": False, "reason": "QA_WEBHOOK_URL not set"}
    text = (f"*GoodVibes FLAG* container #{sample_id} — est. position ({pred.get('pos_xy', ['?', '?'])[0]}, "
            f"{pred.get('pos_xy', ['?', '?'])[1]}), confidence {pred.get('confidence')}.\n"
            + (f"Verdict: {report.get('verdict')} · Action: {report.get('recommended_action')}" if report else "Report pending."))
    try:
        req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return {"sent": r.status < 300, "status": r.status}
    except Exception as e:
        return {"sent": False, "reason": f"{type(e).__name__}: {e}"}
