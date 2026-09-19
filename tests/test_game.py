import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly2048.game import DOWN, LEFT, RIGHT, UP, Game, from_grid, swipe, tiles, to_grid  # noqa: E402
from fly2048.teacher import teacher_move  # noqa: E402

BOARD = from_grid([[1, 1, 2, 0], [0, 0, 0, 0], [2, 0, 2, 2], [1, 0, 0, 1]])  # 2 2 4 . / . . . . / 4 . 4 4 / 2 . . 2


def after(move):
    board, gain = swipe(BOARD, move)
    return tiles(to_grid(board)).tolist(), gain


def test_swipes_merge_once_toward_the_wall():
    assert after(LEFT) == ([[4, 4, 0, 0], [0, 0, 0, 0], [8, 4, 0, 0], [4, 0, 0, 0]], 16)
    assert after(RIGHT) == ([[0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 4, 8], [0, 0, 0, 4]], 16)
    assert after(UP) == ([[2, 2, 8, 4], [4, 0, 0, 2], [2, 0, 0, 0], [0, 0, 0, 0]], 8)
    assert after(DOWN) == ([[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 4], [2, 2, 8, 2]], 8)


def test_grid_round_trip():
    grid = np.arange(16, dtype=np.int8).reshape(4, 4) % 12
    assert (to_grid(from_grid(grid)) == grid).all()


def test_same_seed_same_moves_same_game():
    a, b = Game(7), Game(7)
    assert a.board == b.board
    for _ in range(40):
        move = next(iter(a.afterstates()))
        a.step(move)
        b.step(move)
    assert a.board == b.board and a.score == b.score


def test_teacher_only_picks_legal_swipes():
    game = Game(3)
    for _ in range(30):
        move = teacher_move(game.board)
        assert move in game.afterstates()
        game.step(move)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
