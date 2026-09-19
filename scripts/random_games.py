"""Baseline: uniformly random legal swipes on the same seeds. usage: python scripts/random_games.py <first> <last>"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly2048.game import Game

maxes, scores = [], []
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    game, rng = Game(seed), np.random.default_rng(seed + 10**9)
    while not game.over:
        legal = list(game.afterstates())
        game.step(legal[int(rng.integers(len(legal)))])
    maxes.append(game.max_tile)
    scores.append(game.score)
tiles, counts = np.unique(maxes, return_counts=True)
print(f"{len(maxes)} games: mean score {np.mean(scores):.0f}, max tile " + ", ".join(f"{t}: {c}" for t, c in zip(tiles, counts)))
