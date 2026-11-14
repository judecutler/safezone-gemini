#!/usr/bin/env bash
# Run on a laptop (Hugging Face is blocked from some sandboxes).
# Step 1: tiny test repo — learn the exact tree and npz keys.
# Step 2: only the small per-sample files for the 551 training samples.
set -euo pipefail
pip install -q -U "huggingface_hub[cli]"
mkdir -p data
hf download eturok-weizmann/laser-vibrations-test --repo-type dataset --local-dir data/test
echo "--- test tree:"; find data/test -maxdepth 4 | head -40
python - <<'EOF'
import glob, json, numpy as np
m = sorted(glob.glob("data/test/**/manifest.json", recursive=True))[0]
man = json.load(open(m)); print("manifest keys:", list(man))
print("processing_config:", json.dumps(man.get("processing_config"), indent=1)[:1500])
print("experiment_output:", json.dumps(man.get("experiment_output"), indent=1)[:800])
for f in glob.glob(m.replace("manifest.json", "*.npz")):
    z = np.load(f); print(f, {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
EOF
# Adjust the globs below if the tree printed above differs (e.g. an `experiment-16/` prefix).
hf download eturok-weizmann/laser-vibrations --repo-type dataset --local-dir data/train \
  --include "data/metadata.jsonl" "*/metadata.jsonl" \
            "*/manifest.json" "*/speckle_shifts_clean.npz" "*/overhead.png" \
            "*/mask.npz" "*/cropped_overhead.png"
echo "done. size:"; du -sh data/train
