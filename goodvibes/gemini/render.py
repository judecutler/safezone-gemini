"""Render a sample's vibration signature as a PNG (for Gemini vision / embeddings) and as a short audio tour."""
from __future__ import annotations

import io

import numpy as np

from ml import features as F


def signature_png(spec: np.ndarray, grid_rc: np.ndarray, title: str = "") -> bytes:
    """Left: 10x10 band-energy maps (one per band). Right: mean spectrum per grid row. Returns PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band = F.band_energy(spec)                      # [P, B]
    nb = band.shape[1]
    fig, axes = plt.subplots(1, nb + 1, figsize=(2.2 * (nb + 1), 2.6), dpi=100)
    for b in range(nb):
        img = np.full((F.GRID, F.GRID), np.nan)
        for j, (r, c) in enumerate(grid_rc):
            img[r, c] = band[j, b]
        ax = axes[b]
        ax.imshow(img, cmap="magma", vmin=np.nanpercentile(band[:, b], 2), vmax=np.nanpercentile(band[:, b], 98))
        lo, hi = F.BANDS[b]
        ax.set_title(f"{lo}-{hi} Hz", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[-1]
    mean_ax = spec.mean(axis=1)                      # [P, N]
    for r in range(F.GRID):
        rows = [j for j, (rr, _) in enumerate(grid_rc) if rr == r]
        if rows:
            ax.plot(F.FREQS, mean_ax[rows].mean(0) + 0.6 * r, lw=0.6, label=f"row {r}")
    ax.set_xlabel("Hz", fontsize=7); ax.set_yticks([]); ax.tick_params(labelsize=6)
    ax.set_title("spectrum per grid row (offset)", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def heatmap_png(occ: np.ndarray, grid_rc: np.ndarray, title: str = "predicted occupancy") -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = np.zeros((F.GRID, F.GRID))
    for j, (r, c) in enumerate(grid_rc):
        img[r, c] = occ[j]
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=100)
    ax.imshow(img, cmap="inferno", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def audio_tour_wav(shifts: np.ndarray, fs_hz: float, seconds_per_row: float = 0.5, out_sr: int = 16000) -> bytes:
    """Sum the lasers of each grid row, play the rows one after another: 10 rows -> ~5 s WAV (mono, int16)."""
    import wave
    from scipy.signal import resample_poly
    from math import gcd
    x = F.canonical_shifts(shifts)[:, :, 0]          # dominant axis
    P, T = x.shape
    n = int(seconds_per_row * fs_hz)
    out = []
    for r in range(F.GRID):
        rows = [j for j in range(P) if j // F.GRID == r]
        seg = x[rows, :n].mean(0) if rows else np.zeros(n)
        seg = seg - seg.mean()
        out.append(seg / (np.abs(seg).max() + 1e-9))
    sig = np.concatenate(out)
    g = gcd(int(fs_hz), out_sr)
    sig = resample_poly(sig, out_sr // g, int(fs_hz) // g)
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(out_sr); w.writeframes(pcm.tobytes())
    return buf.getvalue()
