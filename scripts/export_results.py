"""Collect every saved game into results/games/<player>.json (seed -> score, max tile, moves) and copy the
fly's readout and its gate report into results/, so the README's numbers can be checked from the repo."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "results"
PLAYERS = ["fly-none", "nobrain", "jev-rules", "jev-strategy", "random",
           "fly-silenced", "fly-shuffled_readout", "fly-shuffled_input"]
(OUT / "games").mkdir(parents=True, exist_ok=True)
for player in PLAYERS:
    games = {}
    for path in sorted((RUNS / player).glob("*.json"), key=lambda p: int(p.stem)):
        g = json.loads(path.read_text(encoding="utf8"))
        games[path.stem] = {k: g[k] for k in ("score", "max_tile", "n_moves", "moves") if k in g}
    (OUT / "games" / f"{player}.json").write_text(json.dumps(games), encoding="utf8")
    print(f"{player}: {len(games)} games")
for name in ("readout.npz", "gate.json"):
    shutil.copy(RUNS / "readout" / name, OUT / name)
