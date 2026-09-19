"""Play seeded games with the expectimax teacher and report max tile, score, moves, and speed.
usage: python scripts/teacher_games.py <first seed> <last seed exclusive>"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly2048.game import Game
from fly2048.teacher import teacher_move

for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    game, started = Game(seed), time.time()
    while not game.over:
        game.step(teacher_move(game.board))
    took = time.time() - started
    print(f"seed {seed}: max {game.max_tile}, score {game.score}, moves {game.moves}, "
          f"{took:.0f}s ({took / game.moves * 1000:.1f} ms/move)", flush=True)
