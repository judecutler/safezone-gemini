"""Shared Gemini client. Every caller must tolerate `client() is None` (no key) and return {"available": False}."""
from __future__ import annotations

import os
from functools import lru_cache

MODEL_FAST = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_EMBED = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")


@lru_cache(maxsize=1)
def client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)


def unavailable(reason: str = "GEMINI_API_KEY not set") -> dict:
    return {"available": False, "reason": reason}
