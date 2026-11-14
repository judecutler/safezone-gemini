"""Frozen feature spec: speckle shift signals -> spectral tensors.

Input per sample: shifts array shaped [n_lasers, T, 2] (XY sub-pixel shift per laser per frame)
and its sampling rate fs_hz. Output: spectrum tensor [n_lasers, 2, N_BINS] over F_LO..F_HI Hz at 1 Hz,
log-magnitude, z-scored per sample; plus a band-energy map [n_lasers, len(BANDS)] for baselines and
for Gemini-vision renderings.
"""
from __future__ import annotations

import numpy as np

F_LO, F_HI, F_STEP = 50.0, 1000.0, 1.0           # paper: 100-2500 Hz; dataset chirp is 50-1000 Hz
FREQS = np.arange(F_LO, F_HI, F_STEP)             # 950 bins
N_BINS = len(FREQS)
BANDS = [(50, 100), (100, 200), (200, 400), (400, 700), (700, 1000)]
GRID = 10                                          # 10x10 laser grid
SPEC = dict(f_lo=F_LO, f_hi=F_HI, f_step=F_STEP, n_bins=N_BINS, bands=BANDS, grid=GRID)


def canonical_shifts(arr: np.ndarray) -> np.ndarray:
    """Return shifts as [n_lasers, T, 2] regardless of the stored axis order."""
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"expected 3-D shifts array, got shape {a.shape}")
    xy_axis = [i for i, s in enumerate(a.shape) if s == 2]
    if not xy_axis:
        raise ValueError(f"no axis of size 2 (XY) in shape {a.shape}")
    xy = xy_axis[-1]
    rest = [i for i in range(3) if i != xy]
    t_axis = max(rest, key=lambda i: a.shape[i])
    l_axis = [i for i in rest if i != t_axis][0]
    return np.ascontiguousarray(np.transpose(a, (l_axis, t_axis, xy)).astype(np.float32))


def spectrum(shifts: np.ndarray, fs_hz: float) -> np.ndarray:
    """[n_lasers, T, 2] -> [n_lasers, 2, N_BINS] log-magnitude spectrum on the 1 Hz grid."""
    x = canonical_shifts(shifts)
    x = x - x.mean(axis=1, keepdims=True)
    n_l, T, _ = x.shape
    win = np.hanning(T).astype(np.float32)[None, :, None]
    mag = np.abs(np.fft.rfft(x * win, axis=1)) / T          # [n_l, F, 2]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs_hz)
    out = np.empty((n_l, 2, N_BINS), dtype=np.float32)
    for ax in range(2):
        for i in range(n_l):
            out[i, ax] = np.interp(FREQS, freqs, mag[i, :, ax])
    return np.log1p(out * 1e3)


def normalize(spec: np.ndarray) -> np.ndarray:
    """Per-sample z-score (removes gain differences between recordings)."""
    return ((spec - spec.mean()) / (spec.std() + 1e-6)).astype(np.float32)


def band_energy(spec: np.ndarray) -> np.ndarray:
    """[n_lasers, 2, N_BINS] -> [n_lasers, len(BANDS)] mean energy per band (both axes pooled)."""
    out = np.empty((spec.shape[0], len(BANDS)), dtype=np.float32)
    for b, (lo, hi) in enumerate(BANDS):
        m = (FREQS >= lo) & (FREQS < hi)
        out[:, b] = spec[:, :, m].mean(axis=(1, 2))
    return out


def grid_order(n_lasers: int, rois_xywh: list | None) -> np.ndarray:
    """Map laser index -> (row, col) on the GRID using sensor ROI positions if available."""
    if rois_xywh is not None and len(rois_xywh) == n_lasers:
        r = np.asarray(rois_xywh, dtype=float)
        cx, cy = r[:, 0] + r[:, 2] / 2, r[:, 1] + r[:, 3] / 2
        rows = np.argsort(np.argsort(cy)) * GRID // n_lasers
        # within each row, order by x
        rc = np.zeros((n_lasers, 2), dtype=np.int64)
        for row in range(GRID):
            idx = np.where(rows == row)[0]
            cols = np.argsort(np.argsort(cx[idx]))
            rc[idx, 0] = row
            rc[idx, 1] = cols * GRID // max(len(idx), 1)
        return rc
    i = np.arange(n_lasers)
    return np.stack([i // GRID, i % GRID], axis=1)
