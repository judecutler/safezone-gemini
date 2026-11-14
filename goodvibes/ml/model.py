"""Vibration Transformer (Kichler et al., ICCV 2025), scaled to ~500 samples.

Per laser point: spectrum [2, N_BINS] -> 2x50 patches -> tokens -> PointTransformer -> [pnt].
All points: [pnt] tokens + grid-position embedding + [cls] -> ShapeTransformer.
Heads: presence (2), cell (K), com (2), occ (1 per point).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ml.features import N_BINS, GRID

PATCH = 50
N_PATCH = N_BINS // PATCH   # 19


class _Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class VibrationTransformer(nn.Module):
    def __init__(self, n_cells: int, d: int = 256, heads: int = 4, depth_pt: int = 4, depth_st: int = 4,
                 drop: float = 0.1, token_drop: float = 0.5):
        super().__init__()
        self.token_drop = token_drop
        self.n_cells = n_cells
        # Point transformer
        self.patch = nn.Linear(2 * PATCH, d)
        self.pos_f = nn.Parameter(torch.zeros(1, N_PATCH, d))
        self.pnt = nn.Parameter(torch.zeros(1, 1, d))
        self.pt = nn.ModuleList([_Block(d, heads, drop) for _ in range(depth_pt)])
        self.ln_pt = nn.LayerNorm(d)
        # Shape transformer
        self.row_emb = nn.Embedding(GRID, d)
        self.col_emb = nn.Embedding(GRID, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.st = nn.ModuleList([_Block(d, heads, drop) for _ in range(depth_st)])
        self.ln_st = nn.LayerNorm(d)
        # heads
        def head(o):
            return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, o))
        self.h_presence = head(2)
        self.h_cell = head(max(n_cells, 1))
        self.h_pos = head(2)
        self.h_occ = head(1)
        for p in (self.pos_f, self.pnt, self.cls):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, spec: torch.Tensor, grid_rc: torch.Tensor, point_mask: torch.Tensor | None = None):
        """spec [B, P, 2, N_BINS]; grid_rc [P, 2] (row, col); point_mask [B, P] True = drop point."""
        B, P, _, _ = spec.shape
        x = spec.reshape(B * P, 2, N_PATCH, PATCH).permute(0, 2, 1, 3).reshape(B * P, N_PATCH, 2 * PATCH)
        x = self.patch(x) + self.pos_f
        kpm = None
        if self.training and self.token_drop > 0:
            kpm = torch.rand(B * P, N_PATCH, device=x.device) < self.token_drop
            kpm[:, 0] = False  # keep at least one
        x = torch.cat([self.pnt.expand(B * P, -1, -1), x], dim=1)
        if kpm is not None:
            kpm = torch.cat([torch.zeros(B * P, 1, dtype=torch.bool, device=x.device), kpm], dim=1)
        for blk in self.pt:
            x = blk(x, kpm)
        pnt = self.ln_pt(x[:, 0]).reshape(B, P, -1)
        pnt = pnt + self.row_emb(grid_rc[:, 0])[None] + self.col_emb(grid_rc[:, 1])[None]
        y = torch.cat([self.cls.expand(B, -1, -1), pnt], dim=1)
        kpm2 = None
        if point_mask is not None:
            kpm2 = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), point_mask], dim=1)
        for blk in self.st:
            y = blk(y, kpm2)
        y = self.ln_st(y)
        cls, pts = y[:, 0], y[:, 1:]
        return dict(
            presence=self.h_presence(cls),
            cell=self.h_cell(cls),
            pos=torch.sigmoid(self.h_pos(cls)),
            occ=self.h_occ(pts).squeeze(-1),      # [B, P] logits
        )


def loss_fn(out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    has_obj = batch["presence"] == 1
    l_pres = Fn.cross_entropy(out["presence"], batch["presence"])
    total = 0.1 * l_pres
    logs = {"pres": l_pres.item()}
    if has_obj.any():
        l_pos = Fn.smooth_l1_loss(out["pos"][has_obj], batch["pos"][has_obj], beta=0.05)
        total = total + 1.0 * l_pos
        logs["pos"] = l_pos.item()
        valid = has_obj & (batch["cell"] >= 0)
        if valid.any():
            l_cell = Fn.cross_entropy(out["cell"][valid], batch["cell"][valid])
            total = total + 0.2 * l_cell
            logs["cell"] = l_cell.item()
        l_occ = Fn.binary_cross_entropy_with_logits(out["occ"][has_obj], batch["occ"][has_obj])
        total = total + 0.5 * l_occ
        logs["occ"] = l_occ.item()
    return total, logs


def augment(spec: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
    """Speaker/room invariance: smooth random spectral filter, point dropout, small freq shift, noise."""
    B, P, A, N = spec.shape
    dev = spec.device
    f = torch.linspace(0, 1, N, device=dev)
    filt = torch.zeros(B, 1, 1, N, device=dev)
    for k in range(1, 5):
        amp = torch.randn(B, 1, 1, 1, device=dev) * 0.2 / k
        ph = torch.rand(B, 1, 1, 1, device=dev) * 6.283
        filt = filt + amp * torch.cos(2 * 3.1416 * k * f + ph)
    x = spec + filt                                       # additive in log-magnitude == multiplicative gain
    shift = int(torch.randint(-2, 3, (1,)).item())
    if shift:
        x = torch.roll(x, shift, dims=-1)
    keep = (torch.rand(B, P, 1, 1, device=dev) > 0.1).float()
    x = x * keep
    return x + 0.05 * torch.randn_like(x)
