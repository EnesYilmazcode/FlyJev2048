"""Replay a saved game move by move, tracking every tile so the video can slide and merge them.

Each move yields the tiles' start and end cells, which tiles merged, and the new tile that appeared.
The tracked board is checked against the game engine after every move, so the video can never show
a board the game did not reach.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly2048.game import Game, to_grid

VECTORS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}   # left, right, up, down as (dr, dc)


class Tile:
    _next = 0

    def __init__(self, exp, r, c):
        Tile._next += 1
        self.id, self.exp, self.r, self.c = Tile._next, exp, r, c


def slide(grid_tiles, move):
    """Move tiles like the game does. Returns (new tile dict, list of motions, list of merges)."""
    dr, dc = VECTORS[move]
    lines = []
    for i in range(4):
        cells = [(i, j) for j in range(4)] if dr == 0 else [(j, i) for j in range(4)]
        if dr > 0 or dc > 0:
            cells = cells[::-1]                      # walk from the wall the tiles slide toward
        lines.append(cells)
    new, motions, merges = {}, [], []
    for cells in lines:
        stack = [grid_tiles[rc] for rc in cells if rc in grid_tiles]
        k, slot = 0, 0
        while k < len(stack):
            target = cells[slot]
            a = stack[k]
            if k + 1 < len(stack) and stack[k + 1].exp == a.exp:
                b = stack[k + 1]
                motions += [(a, (a.r, a.c), target), (b, (b.r, b.c), target)]
                merged = Tile(a.exp + 1, *target)
                merges.append((merged, a, b))
                new[target] = merged
                k += 2
            else:
                motions.append((a, (a.r, a.c), target))
                new[target] = a
                k += 1
            slot += 1
    for tile, _, (r, c) in motions:
        tile.r, tile.c = r, c
    return new, motions, merges


def replay(seed, moves):
    """List of steps: dict(before, motions, merges, spawn, after, score) for every move."""
    game = Game(seed)
    grid = to_grid(game.board)
    tiles = {(r, c): Tile(int(grid[r, c]), r, c) for r in range(4) for c in range(4) if grid[r, c]}
    start = dict(tiles)
    steps = []
    for move in moves:
        before = dict(tiles)
        tiles, motions, merges = slide(tiles, move)
        game.step(move)
        grid = to_grid(game.board)
        spawn = None
        for r in range(4):
            for c in range(4):
                if grid[r, c] and (r, c) not in tiles:
                    spawn = Tile(int(grid[r, c]), r, c)
                    tiles[(r, c)] = spawn
        check = np.zeros((4, 4), np.int8)
        for (r, c), t in tiles.items():
            check[r, c] = t.exp
        assert (check == grid).all(), f"tracked board diverged at move {len(steps)} of seed {seed}"
        steps.append({"before": before, "motions": motions, "merges": merges, "spawn": spawn,
                      "after": dict(tiles), "score": game.score})
    return start, steps
