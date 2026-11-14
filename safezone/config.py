"""Settings read once from the environment, with a `.env` file in the repo root as a fallback."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_api_base: str
    vision_model: str
    """One Gemini model detects, segments and reasons."""

    @property
    def segment_model(self) -> str:
        return self.vision_model


def load_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None,
        gemini_api_base=os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
        vision_model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
    )
