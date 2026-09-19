"""The fly plays seeded 2048 games in lockstep (all games' candidate boards share one GPU batch).
Saves runs/fly-<control>/<seed>.json. usage: python scripts/fly_games.py <first> <last> [control]"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly2048.game import Game
from fly2048.player import FlyPlayer

RUNS = Path(os.environ.get("FLY2048_RUNS", ROOT / "runs"))
first, last = int(sys.argv[1]), int(sys.argv[2])
control = sys.argv[3] if len(sys.argv) > 3 else "none"
out = RUNS / f"fly-{control}"
out.mkdir(parents=True, exist_ok=True)

player = FlyPlayer(RUNS / "readout" / "readout.npz", control=control)
games = {seed: Game(seed) for seed in range(first, last)}
moves = {seed: [] for seed in games}
started, step = time.time(), 0
while True:
    live = [s for s, g in games.items() if not g.over]
    if not live:
        break
    owners, options, boards = [], [], []
    for s in live:
        for m, (nb, _) in games[s].afterstates().items():
            owners.append(s)
            options.append(m)
            boards.append(nb)
    score = player.scores(np.array(boards, np.uint64))
    for s in live:
        idx = [i for i, o in enumerate(owners) if o == s]
        best = options[max(idx, key=lambda i: score[i])]
        games[s].step(best)
        moves[s].append(best)
    step += 1
    if step % 50 == 0:
        best_tile = max(g.max_tile for g in games.values())
        print(f"step {step}: {len(live)} games alive, best tile {best_tile}, {time.time() - started:.0f}s", flush=True)
for s, g in games.items():
    record = {"seed": s, "control": control, "score": g.score, "max_tile": g.max_tile, "n_moves": g.moves, "moves": moves[s]}
    (out / f"{s}.json").write_text(json.dumps(record), encoding="utf8")
    print(f"seed {s}: max {g.max_tile}, score {g.score}, moves {g.moves}", flush=True)
scores = [g.score for g in games.values()]
print(f"{control}: {len(scores)} games, mean score {np.mean(scores):.0f}, total {time.time() - started:.0f}s")
