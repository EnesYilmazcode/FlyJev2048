"""Baseline: the build's no-brain readout (same data and target as the fly, fit straight on the 32 eye
inputs) playing seeded games. usage: python scripts/nobrain_games.py <first> <last>"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly2048.eyes import board_channels
from fly2048.game import Game

with np.load(ROOT / "runs" / "readout" / "readout_nobrain.npz") as z:
    scale, w = z["scale"], z["weights"]
out = ROOT / "runs" / "nobrain"
out.mkdir(parents=True, exist_ok=True)
scores = []
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    game, moves = Game(seed), []
    while not game.over:
        opts = game.afterstates()
        options = list(opts)
        s = (board_channels([opts[m][0] for m in options]) / scale) @ w
        moves.append(options[int(np.argmax(s))])
        game.step(moves[-1])
    scores.append(game.score)
    (out / f"{seed}.json").write_text(json.dumps({"seed": seed, "score": game.score, "max_tile": game.max_tile,
                                                   "n_moves": game.moves, "moves": moves}), encoding="utf8")
    print(f"seed {seed}: max {game.max_tile}, score {game.score}, moves {game.moves}", flush=True)
print(f"no brain: {len(scores)} games, mean score {np.mean(scores):.0f}")
