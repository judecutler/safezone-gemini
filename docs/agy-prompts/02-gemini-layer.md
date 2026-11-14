Read AGENTS.md. Slice 2: the Gemini client, the boundary it sits on, and the sample responders that speak its wire format.
Review `safezone/gemini.py` for correctness against the Gemini API (generateContent, inlineData, responseJsonSchema, thinkingConfig). If you find a real bug, fix it minimally and say what you changed. Then run `python -m pytest -q tests/test_foundation.py` and `git add`:
  safezone/__init__.py safezone/config.py safezone/transport.py safezone/shapes.py safezone/gemini.py safezone/rules.py safezone/geo.py tools/__init__.py tools/samples/
Commit: "Add GeminiClient: one generateContent boundary for detection, masks and structured JSON". Report the hash.
