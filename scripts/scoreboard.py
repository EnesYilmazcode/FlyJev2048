"""Compare every player on the seeds they all played. Reads runs/<player>/<seed>.json.
usage: python scripts/scoreboard.py [first seed] [last seed exclusive]"""
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
PLAYERS = [
    ("fly-none", "Fly (connectome + readout)"),
    ("nobrain", "Same readout, no brain"),
    ("jev-rules", "Jev, rules only"),
    ("jev-strategy", "Jev, rules + beginner strategy"),
    ("fly-shuffled_input", "Fly, eye wiring shuffled"),
    ("fly-shuffled_readout", "Fly, readout shuffled"),
    ("fly-silenced", "Fly, readout silenced"),
    ("random", "Random swipes"),
]
first = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
last = int(sys.argv[2]) if len(sys.argv) > 2 else 5016
seeds = range(first, last)


def load(folder):
    games = {}
    for seed in seeds:
        path = RUNS / folder / f"{seed}.json"
        if path.exists():
            games[seed] = json.loads(path.read_text(encoding="utf8"))
    return games


rows = []
for folder, label in PLAYERS:
    games = load(folder)
    if not games:
        continue
    scores = np.array([g["score"] for g in games.values()])
    tiles = [g["max_tile"] for g in games.values()]
    reached = lambda t: sum(x >= t for x in tiles)
    rows.append((label, len(games), scores.mean(), np.median(scores), max(tiles), reached(512), reached(1024), reached(2048)))

print(f"seeds {first}-{last - 1}\n")
print(f"| Player | Games | Mean score | Median | Best tile | Reached 512 | 1024 | 2048 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for label, n, mean, median, best, r512, r1024, r2048 in rows:
    print(f"| {label} | {n} | {mean:,.0f} | {median:,.0f} | {best} | {r512} | {r1024} | {r2048} |")

fly, jev = load("fly-none"), load("jev-rules")
both = sorted(set(fly) & set(jev))
if both:
    wins = sum(fly[s]["score"] > jev[s]["score"] for s in both)
    print(f"\nFly vs Jev on the same {len(both)} seeds: fly scores higher in {wins}, Jev in {len(both) - wins}.")
