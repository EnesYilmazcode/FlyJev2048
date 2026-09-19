"""2048 board -> input spike rates on the fly's photoreceptors.

Same idea as Flytris: the fly sees engineered board features, not pixels. Every line of the board
(4 rows, 4 columns) gives four numbers in 0..1: empty cells, available merges, how ordered the line is,
and how much tile mass it holds. The left eye's R1-R6 photoreceptors see the rows and the right eye's
see the columns: each eye is cut into four strips along its horizontal axis (one strip per line), and
each strip is split into four groups by body id (one group per feature). Photoreceptor positions come
from their strongest L1/L2 partner's optic-lobe hex column, exactly as in Flytris.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(os.environ.get("FLYTRIS_DATA", Path(__file__).resolve().parents[2] / "Flytris" / "data" / "malecns"))
MAP = Path(os.environ.get("FLY2048_EYE_MAP", DATA / "eye_map_2048.npz"))
N_LINES, N_FEATURES = 8, 4
N_CHANNELS = N_LINES * N_FEATURES
PHASE_SEED = 20260919


def _line_features(line):
    """[empty, merges, disorder, mass] for one line of exponents (raw, not yet scaled)."""
    empty = sum(1 for r in line if r == 0)
    merges, prev, counter = 0, 0, 0
    for r in line:
        if r == 0:
            continue
        if prev == r:
            counter += 1
        elif counter:
            merges += 1 + counter
            counter = 0
        prev = r
    if counter:
        merges += 1 + counter
    left = right = 0.0
    for i in range(1, 4):
        if line[i - 1] > line[i]:
            left += line[i - 1] ** 4 - line[i] ** 4
        else:
            right += line[i] ** 4 - line[i - 1] ** 4
    mass = sum(r ** 3.5 for r in line)
    return empty, merges, min(left, right), mass


# Every possible line, precomputed and scaled to 0..1.
_TABLE = np.zeros((65536, N_FEATURES), np.float32)
for _row in range(65536):
    e, m, d, s = _line_features([(_row >> (4 * i)) & 0xF for i in range(4)])
    _TABLE[_row] = (e / 4, min(m, 3) / 3, np.log1p(d) / np.log1p(3 * 15 ** 4), np.log1p(s) / np.log1p(4 * 15 ** 3.5))


def lines_of(board):
    """The 8 lines of a 64-bit board as 16-bit rows: rows 0-3 (left to right), then columns 0-3 (top down)."""
    b = int(board)
    rows = [(b >> (16 * r)) & 0xFFFF for r in range(4)]
    cols = []
    for c in range(4):
        col = 0
        for r in range(4):
            col |= ((b >> (4 * (4 * r + c))) & 0xF) << (4 * r)
        cols.append(col)
    return rows + cols


def board_channels(boards):
    """[B, 32] feature values in 0..1, ordered line by line: 8 lines x (empty, merges, disorder, mass)."""
    return np.stack([_TABLE[lines_of(b)].reshape(-1) for b in boards]).astype(np.float32)


def build_eye_map():
    nr = pd.read_parquet(DATA / "neurons.parquet", columns=["idx", "body_id", "type", "side"])
    ann = pd.read_feather(DATA / "raw/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                          columns=["bodyId", "assignedOlHex1", "assignedOlHex2"])
    hex_axis = pd.Series((ann.assignedOlHex1 - ann.assignedOlHex2).values, index=ann.bodyId)
    lam = nr[nr.type.isin(["L1", "L2"])]
    lam_pos = np.full(len(nr), np.nan)
    lam_pos[lam.idx] = hex_axis.reindex(lam.body_id).values

    r16 = nr[nr.type == "R1-R6"]
    is_r = np.zeros(len(nr), bool)
    is_r[r16.idx] = True
    z = np.load(DATA / "edges.npz")
    keep = np.nonzero(is_r[z["pre"]])[0]
    pre, post, syn = z["pre"][keep], z["post"][keep], z["syn"][keep]
    ok = ~np.isnan(lam_pos[post])
    pre, post, syn = pre[ok], post[ok], syn[ok]
    order = np.lexsort((-syn, pre))
    first = np.unique(pre[order], return_index=True)[1]
    r_pos = np.full(len(nr), np.nan)
    r_pos[pre[order][first]] = lam_pos[post[order][first]]

    in_idx, channel, scale, counts = [], [], [], {}
    for eye, side in enumerate("LR"):
        rs = r16[(r16.side == side) & ~np.isnan(r_pos[r16.idx])]
        rs = rs.assign(pos=r_pos[rs.idx]).sort_values(["pos", "body_id"])
        counts[side] = len(rs)
        strip = np.minimum(np.arange(len(rs)) * 4 // len(rs), 3)          # which line this photoreceptor sees
        feature = np.zeros(len(rs), int)
        for s in range(4):
            m = strip == s
            feature[m] = np.argsort(np.argsort(rs.body_id.values[m])) % N_FEATURES
        line = strip + 4 * eye                                              # left eye: rows, right eye: columns
        in_idx.append(rs.idx.values)
        channel.append(line * N_FEATURES + feature)
    n_max = max(counts.values())
    for side in "LR":
        scale.append(np.full(counts[side], n_max / counts[side], np.float32))
    np.savez(MAP, in_idx=np.concatenate(in_idx).astype(np.int64), channel=np.concatenate(channel).astype(np.int64),
             scale=np.concatenate(scale), n_left=counts["L"], n_right=counts["R"])
    return np.load(MAP)


class Eyes:
    def __init__(self, board_hz=500.0, device="cuda"):
        m = np.load(MAP) if MAP.exists() else build_eye_map()
        self.in_idx = torch.from_numpy(m["in_idx"]).to(device)
        self.channel = torch.from_numpy(m["channel"]).to(device)
        self.scale = torch.from_numpy(m["scale"]).to(device)
        rng = np.random.default_rng(PHASE_SEED)
        self.phase = torch.from_numpy(rng.random(len(m["in_idx"]), dtype=np.float32)).to(device)
        self.board_hz, self.device = board_hz, device

    def probs(self, boards):
        """Boards [B] -> input spikes per 1 ms step for every photoreceptor, [n_in, B]."""
        v = torch.as_tensor(board_channels(boards).T, device=self.device)
        p = v[self.channel] * self.scale[:, None] * (self.board_hz * 1e-3)
        return p.clamp_(0, 1)
