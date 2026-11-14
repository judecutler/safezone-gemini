Read AGENTS.md. Slice 4: the server, the regions (unchanged logic, moved over), the web app with Gemini copy, and the rest of the tests.
Run `python -m pytest -q` and `node --test web/tests/*.test.mjs` (expect all green; one Python test skips until Gemini fixtures are recorded), then `git add`:
  safezone/app.py safezone/regions.py safezone/building/ safezone/japan/ safezone/route/ safezone/shelter/ web/ tests/ tools/ profile_address.py docs/ demo/ fixtures/
Commit: "Server, PWA and regional pipelines; UI credits Gemini for detection and reasoning". Report the hash.
