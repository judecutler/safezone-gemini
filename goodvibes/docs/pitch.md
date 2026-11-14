# GoodVibes — video scripts

Say "laser", not "video", when describing today's sensor. Say "aggregation" only as roadmap. Numbers come from `results/metrics.json`.

## Video 1 — team + pitch (2:00)

**0:00–0:15 cold open.** "I'm Jude — CS and math at Berkeley. Quick question: are you feeling that?" *(beat, music)* "…those are some good vibes." Card: **GoodVibes — see inside sealed containers, without touching them.**

**0:15–0:40 the science.** "Shine a laser on any surface and a camera sees this grainy pattern — speckle. Move the surface a millionth of a metre and the speckle shifts. Every sealed object is quietly telling us about its contents: mass, stiffness, damping. These vibrations are in any video — MIT read sound off a chip bag with a plain camera in 2014. For a sealed container the signal is a thousand times smaller, so for this hackathon we amplify it with a five-dollar laser. The roadmap is to take the laser away."

**0:40–1:05 where the money is.** Three cards: *Consumer goods* — "the same vibrations say whether an avocado is ripe inside a sealed box; the founding paper names ripeness and spoilage as the open frontier." *Data centres* — "fans, drives and pumps announce failure in vibration weeks early; today that needs a contact sensor on every unit." *Pharmaceutical fill-finish* — "and this is where we start."

**1:05–1:40 why pharma.** Hold a vial. "Every sterile injectable is inspected 100%. The incumbent is laser headspace analysis — it reads the gas above the drug. It needs a few millimetres of gap to exist and is physically blind to the product itself: low-headspace fills, lyophilised cake, particulates, and the billion-dollar question in biologics — has the protein aggregated. Vibrometry measures the opposite thing: container and contents as one mechanical system. We don't replace their method; we're the orthogonal modality their own guidance says a complete programme needs. A biologic vial is worth thousands. A rejected lot, millions. And the demand is written into USP 1207 — mandated, not ROI-justified."

**1:40–2:00 close.** "Tonight, with Gemini and Antigravity as working members of the team, we built the inference layer nobody sells: an autonomous line that locates a hidden object through a sealed cardboard wall from vibration alone — and an AI inspector that explains every call. Here's the proof."

## Video 2 — prototype (1:00)

**0:00–0:10.** You with an orange pharmacy bottle + laser pointer: "Here's how GoodVibes works on one of these: laser, camera, a chirp of sound — and the wall tells us what's inside."

**0:10–0:20.** Line-monitor screen, line running. "Validated on the only public dataset of its kind: 551 sealed boxes, 100 laser points each, a hidden object at 130 different positions."

**0:20–0:45.** Click a flagged container → heatmap → **Reveal** → overhead photo matches → click the brightest laser → "that dip at 100–400 Hz is the object damping the wall." Scoreboard: "a fixed-line excitation, positions it has never seen in this combination: 1.5 cells mean error on an 11-by-12 grid, where guessing gives 3.8 — and it beats k-nearest-neighbours and gradient boosting on the same differential features."

**0:45–1:00.** Gemini panel filling: "Gemini 3.8 Flash is a second, zero-training inspector scored in the same table. The Gemini agent calls our model as a tool and writes the QC report with the lasers and bands that drove it, into the batch record. Built through Antigravity CLI, running on Cloud Run. Good vibes." End card: URL + repo.

## Three tiers (one slide / on-screen strip)
- **Proved tonight:** hidden contents located through a sealed opaque wall, non-contact.
- **Same measurement, next labels:** fill level (Weizmann ICCV'25), lyo cake, particulates, low-headspace vials.
- **Frontier:** viscosity/aggregation, ripeness & spoilage, equipment health, battery wetting.

## Q&A one-liners
- *Why not X-ray/ultrasound?* X-ray can't see protein state and costs/shields; ultrasound needs contact/couplant. We're non-contact and read mechanics.
- *Laser safety?* Class 2/3R pointer-level power per point in the dataset (~14 mW in the paper's rig); enclosed on a line.
- *Unseen excitation (S1)?* Hard — report it honestly; a line has a fixed excitation and calibrates against an empty reference, which is what S3 measures.
- *Why Gemini in the loop?* Zero-training baseline we can score, an explainable report that cites physics, and a multimodal self-audit. Not a chatbot.
