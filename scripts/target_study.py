"""No-brain study: which teacher target makes the best one-look readout on the fly's eye inputs?
Fits ridge readouts on the saved teacher positions and plays 16 seeded games with each (CPU only)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly2048.eyes import board_channels
from fly2048.game import Game
from fly2048.teacher import chance_value, heuristic

with np.load(ROOT / "runs" / "readout" / "positions.npz") as z:
    before, after, deep, group = z["before"], z["after"], z["target"], z["group"]


def per_position(v):
    out = np.empty_like(v)
    for g in np.unique(group):
        idx = group == g
        out[idx] = (v[idx] - v[idx].mean()) / max(v[idx].std(), 1e-6)
    return out


targets = {
    "deep expectimax (current)": deep,
    "one lookahead layer": per_position(np.array([chance_value(np.uint64(a), 1, 1.0) for a in after])),
    "board heuristic": per_position(np.array([heuristic(np.uint64(a)) for a in after])),
}
feats = {"before + after": lambda b, a: np.concatenate([board_channels(b), board_channels(a)], 1),
         "after only": lambda b, a: board_channels(a)}

for fname, f in feats.items():
    x = f(before, after).astype(np.float64)
    scale = x.std(0)
    scale[scale < 1e-6] = 1
    for tname, y in targets.items():
        z_ = x / scale
        w = np.linalg.solve(z_.T @ z_ + 1000 * np.eye(z_.shape[1]), z_.T @ y)
        scores, maxes = [], []
        for seed in range(5000, 5016):
            g = Game(seed)
            while not g.over:
                opts = g.afterstates()
                moves = list(opts)
                s = (f(np.full(len(moves), g.board, np.uint64), np.array([opts[m][0] for m in moves], np.uint64)) / scale) @ w
                g.step(moves[int(np.argmax(s))])
            scores.append(g.score)
            maxes.append(g.max_tile)
        t, c = np.unique(maxes, return_counts=True)
        print(f"{fname:15s} | {tname:26s} | mean score {np.mean(scores):6.0f} | {dict(zip(t.tolist(), c.tolist()))}", flush=True)
