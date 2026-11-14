"""Generate an HF-shaped synthetic `laser-vibrations` tree so the whole pipeline runs without the download.

Physics sketch (deliberately crude): the box side is a plate with a handful of modes; a hidden object
adds local mass/stiffness, which (a) shifts resonances and (b) damps the lasers over the object.
Speaker configuration applies a smooth random spectral coloring. Replace with real data ASAP.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

FS = 2000.0
T_SEC = 3.0
N_LASERS = 100
GRID = 10
POS_GRID = (4, 4)       # object position grid (x, y) like x_position/y_position
SPEAKERS = ["0001", "0010", "0100", "1000"]


def _chirp(t, f0=50.0, f1=1000.0):
    k = (f1 / f0) ** (1 / t[-1])
    return np.sin(2 * np.pi * f0 * (k ** t - 1) / np.log(k))


def _plate_response(rng, n_frames, obj_rc, speaker):
    t = np.arange(n_frames) / FS
    exc = _chirp(t)
    rows, cols = np.divmod(np.arange(N_LASERS), GRID)
    r = rows / (GRID - 1)
    c = cols / (GRID - 1)
    modes = [(1, 1, 120), (2, 1, 210), (1, 2, 260), (2, 2, 340), (3, 1, 430), (3, 2, 560), (1, 3, 610), (3, 3, 820)]
    sig = np.zeros((N_LASERS, n_frames, 2), np.float32)
    freqs_f = np.fft.rfftfreq(n_frames, 1 / FS)
    EXC = np.fft.rfft(exc)
    # speaker coloring + a speaker-position gain gradient across the plate
    col = np.exp(0.6 * np.sin(2 * np.pi * freqs_f / 900 * rng.uniform(0.5, 1.5) + rng.uniform(0, 6.28)))
    sp_idx = SPEAKERS.index(speaker)
    gain = 1 + 0.5 * (c if sp_idx % 2 else 1 - c) * (1 if sp_idx < 2 else -1) * 0.5
    for ax in range(2):
        H = np.zeros((N_LASERS, len(freqs_f)), np.complex64)
        for (m, n, f0) in modes:
            f = f0 * (1 + 0.03 * rng.standard_normal())
            if obj_rc is not None:
                f *= 1 + 0.04 * (m + n) / 6          # stiffening from the object
            shape = np.sin(np.pi * m * r) * np.sin(np.pi * n * c) * (1.0 if ax == 0 else 0.6)
            if obj_rc is not None:
                d = np.hypot(r - obj_rc[0], c - obj_rc[1])
                shape = shape * (1 - 0.7 * np.exp(-(d / 0.18) ** 2))   # local damping over the object
            zeta = 0.02
            H += shape[:, None] * (1 / (f**2 - freqs_f**2 + 1j * 2 * zeta * f * freqs_f))[None, :] * 1e3
        X = H * (EXC * col)[None, :] * gain[:, None]
        x = np.fft.irfft(X, n=n_frames, axis=1)
        sig[:, :, ax] = x + 0.02 * rng.standard_normal(x.shape)
    return sig / (np.abs(sig).max() + 1e-9) * 0.5


def main(out: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    data_dir = os.path.join(out, "data")
    img_dir = os.path.join(out, "image")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    n_frames = int(FS * T_SEC)
    rows = []
    for sid in range(1, n + 1):
        empty = rng.random() < 0.2
        spk = SPEAKERS[(sid - 1) % 4]
        if empty:
            x = y = -1
            obj_rc = None
        else:
            x, y = int(rng.integers(POS_GRID[0])), int(rng.integers(POS_GRID[1]))
            obj_rc = ((y + 0.5) / POS_GRID[1], (x + 0.5) / POS_GRID[0])
        sdir = os.path.join(data_dir, f"{sid:07d}")
        os.makedirs(sdir, exist_ok=True)
        sig = _plate_response(rng, n_frames, obj_rc, spk)
        np.savez_compressed(os.path.join(sdir, "speckle_shifts_clean.npz"), shifts=sig)
        rois = [[(i % GRID) * 12, (i // GRID) * 12, 6, 6] for i in range(N_LASERS)]
        man = {
            "sample_id": sid,
            "sample": {"object": "empty" if empty else "cube", "n_objects": 0 if empty else 1,
                       "x_position": x, "y_position": y, "speakers": spk, "box_material": "cardboard"},
            "experiment_config": {"audio": {"generation": {"f_start_hz": 50, "f_end_hz": 1000}, "duration_s": T_SEC},
                                  "laser_camera": {"capture": {"fps": FS}},
                                  "laser_grid": {"sensor_rois_xywh": rois}},
            "experiment_output": {"laser_grid": {"n_lasers": N_LASERS},
                                  "speckle_shifts": {"n_frames": n_frames, "duration_s": T_SEC}},
            "processing_config": {"speckle_shifts": {"fs_hz": FS},
                                  "speckle_shifts_clean": {"filter_type": "butterworth", "filter_mode": "bandpass",
                                                           "lowcut": 40, "highcut": 1000, "filter_order": 4,
                                                           "hann_applied": True}},
        }
        json.dump(man, open(os.path.join(sdir, "manifest.json"), "w"))
        # mask on a 64x64 overhead
        mask = np.zeros((64, 64), np.uint8)
        if not empty:
            cy, cx = int((y + 0.5) * 16), int((x + 0.5) * 16)
            mask[cy - 5:cy + 5, cx - 5:cx + 5] = 1
        idir = os.path.join(img_dir, f"{'empty' if empty else 'cube'}-{x:02d}x-{y:02d}y-{sid}")
        os.makedirs(idir, exist_ok=True)
        np.savez_compressed(os.path.join(idir, "mask.npz"), mask=mask)
        ys, xs = np.nonzero(mask)
        rows.append({
            "sample_id": sid, "experiment_id": f"synth-{sid:04d}", "speakers": spk,
            "x_position": x, "y_position": y,
            "x_com": float(xs.mean()) if len(xs) else -1.0, "y_com": float(ys.mean()) if len(ys) else -1.0,
            "n_objects": 0 if empty else 1, "object": "empty" if empty else "cube", "box_material": "cardboard",
            "mask_file_name": os.path.relpath(os.path.join(idir, "mask.npz"), out),
            "experiment_dir": "synth", "manifest": json.dumps(man),
        })
    with open(os.path.join(data_dir, "metadata.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {n} synthetic samples to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.n, a.seed)
