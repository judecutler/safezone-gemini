"""Live / offline speckle capture from an ordinary camera (stretch goal).

Setup: laser pointer on a sealed box or opaque bottle; phone at 240 fps slow-mo (or webcam >=120 fps),
manual focus at NEAREST so the laser spot renders as a defocused speckle patch; excite with a 20-120 Hz chirp.

Usage:
  python -m ml.capture video.mp4 --fps 240 --out results/capture.npz      # offline (phone video)
  python -m ml.capture 0 --fps 120 --seconds 3                           # webcam index
Outputs shifts [1, T, 2] + fs_hz (upload-compatible with /api/upload after tiling to the grid size) and a spectrum PNG.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def find_speckle_roi(frame_gray: np.ndarray, size: int = 96) -> tuple[int, int]:
    """Brightest blob = the laser speckle patch. Returns top-left (x, y) of a size×size ROI."""
    blur = cv2.GaussianBlur(frame_gray, (31, 31), 0)
    _, _, _, (x, y) = cv2.minMaxLoc(blur)
    h, w = frame_gray.shape
    return int(np.clip(x - size // 2, 0, w - size)), int(np.clip(y - size // 2, 0, h - size))


def track(video, fps: float | None, seconds: float | None, roi_size: int = 96) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fs = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, f0 = cap.read()
    if not ok:
        raise SystemExit("empty video")
    g_prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    x0, y0 = find_speckle_roi(g_prev, roi_size)
    win = cv2.createHanningWindow((roi_size, roi_size), cv2.CV_32F)

    def prep(g):
        g = g[y0:y0 + roi_size, x0:x0 + roi_size].astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 6)        # high-pass: remove the stationary spot envelope
    prev = prep(g_prev)
    vel = []
    max_frames = int(seconds * fs) if seconds else None
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(vel) >= max_frames):
            break
        cur = prep(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)   # sub-pixel shift between consecutive frames
        vel.append((dx, dy) if resp > 0.05 else (0.0, 0.0))
        prev = cur
    cap.release()
    s = np.cumsum(np.asarray(vel, np.float32), axis=0)       # integrate velocity -> displacement
    s = s - cv2.blur(s, (1, max(3, int(fs / 2)) ))           # remove sub-0.5 Hz drift
    return s[None], fs                                   # [1, T, 2]


def spectrum_png(shifts: np.ndarray, fs: float, out: str, f_max: float | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = shifts[0] - shifts[0].mean(0)
    T = len(x)
    f = np.fft.rfftfreq(T, 1 / fs)
    mag = np.abs(np.fft.rfft(x * np.hanning(T)[:, None], axis=0))
    m = (f >= 2) & (f <= (f_max or fs / 2))
    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    ax.semilogy(f[m], mag[m, 0] + 1e-9, label="x"); ax.semilogy(f[m], mag[m, 1] + 1e-9, label="y")
    ax.set_xlabel("Hz"); ax.set_ylabel("|speckle shift|"); ax.legend(); ax.set_title(f"speckle vibration spectrum, fs={fs:.0f} Hz")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to video or webcam index")
    ap.add_argument("--fps", type=float, default=None, help="true capture rate (phone slo-mo files often report 30)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", default="results/capture.npz")
    ap.add_argument("--png", default="results/capture.png")
    a = ap.parse_args()
    src = int(a.video) if a.video.isdigit() else a.video
    sh, fs = track(src, a.fps, a.seconds)
    np.savez(a.out, shifts=sh, fs_hz=fs)
    spectrum_png(sh, fs, a.png)
    print(f"tracked {sh.shape[1]} frames @ {fs:.1f} Hz; rms shift x={sh[0,:,0].std():.3f}px y={sh[0,:,1].std():.3f}px -> {a.out}, {a.png}")
