# GoodVibes — the first idea of this hackathon (built Fri 23:30 → Sat 07:30)

Speckle vibrometry for sealed-container inspection: a laser grid + camera read microscopic wall vibrations, a
Vibration Transformer infers what's inside. Trained on the Weizmann lab's public `laser-vibrations` dataset:
1.34-cell mean error on held-out recordings (baselines 1.42–1.73, random 3.8). Gemini 3.8 Flash few-shot inspector,
Gemini embedding kNN, and a function-calling inspection agent are in `gemini/`. Built with the Antigravity CLI.

At 07:40 we pivoted to SafeZone-Gemini (the parent folder) because judges can try it on their own room in seconds,
whereas this demo can only replay lab recordings. The code here is complete and runs; see README.md.
