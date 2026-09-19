"""Jev plays seeded 2048 games, a few at a time, and saves each to runs/jev/<seed>.json.
usage: python scripts/jev_games.py <first seed> <last seed exclusive> [parallel games] [rules|strategy]"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly2048.game import Game
from fly2048.jev import play

MODE = sys.argv[4] if len(sys.argv) > 4 else "rules"
OUT = ROOT / "runs" / f"jev-{MODE}"
OUT.mkdir(parents=True, exist_ok=True)


def one(seed):
    game, started = Game(seed), time.time()
    record = play(game, log_every=0, strategy=MODE == "strategy")
    record.update(seed=seed, score=game.score, max_tile=game.max_tile, n_moves=game.moves,
                  seconds=round(time.time() - started))
    (OUT / f"{seed}.json").write_text(json.dumps(record), encoding="utf8")
    print(f"seed {seed}: max {game.max_tile}, score {game.score}, moves {game.moves}, "
          f"{record['tokens']} tokens, {record['seconds']}s", flush=True)
    return record


if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, range(first, last)))
