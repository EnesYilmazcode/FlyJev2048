"""Baseline: the same ridge readout fit straight on the eye inputs (no connectome), playing seeded games.
Fits on the teacher positions in runs/readout/positions.npz. usage: python scripts/nobrain_games.py <first> <last>"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly2048.eyes import board_channels
from fly2048.game import Game

with np.load(ROOT / "runs" / "readout" / "positions.npz") as z:
    before, after, target = z["before"], z["after"], z["target"]
x = np.concatenate([board_channels(before), board_channels(after)], 1).astype(np.float64)
scale = x.std(0)
scale[scale < 1e-6] = 1
z = x / scale
w = np.linalg.solve(z.T @ z + 1000 * np.eye(z.shape[1]), z.T @ target)

maxes, scores = [], []
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    game = Game(seed)
    while not game.over:
        options = game.afterstates()
        moves = list(options)
        b = np.full(len(moves), game.board, np.uint64)
        a = np.array([options[m][0] for m in moves], np.uint64)
        s = (np.concatenate([board_channels(b), board_channels(a)], 1) / scale) @ w
        game.step(moves[int(np.argmax(s))])
    maxes.append(game.max_tile)
    scores.append(game.score)
    print(f"seed {seed}: max {game.max_tile}, score {game.score}, moves {game.moves}", flush=True)
print(f"mean score {np.mean(scores):.0f}")
