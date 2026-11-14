# Antigravity commit plan

Run from the repo root, in order, one agy session per file (headless or interactive):

    agy "$(cat docs/agy-prompts/01-skeleton.md)"
    agy "$(cat docs/agy-prompts/02-gemini-layer.md)"
    ...

PowerShell: `agy (Get-Content docs/agy-prompts/01-skeleton.md -Raw)`

Slice 6 runs only after `python tools/record_fixtures.py` has recorded the real Gemini answers.
