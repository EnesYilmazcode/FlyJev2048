"""2048 on a 64-bit board: 16 cells of 4 bits, each cell the tile's exponent (0 empty, 1 = 2, 11 = 2048).

Row moves come from lookup tables over all 65,536 possible rows, so both the game and the expectimax
teacher are fast under numba. A game is fully determined by its seed and the moves played: new tiles
come from the game's own numpy generator (a 2 with probability 0.9, else a 4, in a uniformly chosen
empty cell).
"""
import numpy as np
from numba import njit

LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3
MOVE_NAMES = ("left", "right", "up", "down")


def _build_row_tables():
    left = np.zeros(65536, np.uint16)
    right = np.zeros(65536, np.uint16)
    score = np.zeros(65536, np.float64)
    for row in range(65536):
        cells = [(row >> (4 * i)) & 0xF for i in range(4)]
        tiles = [c for c in cells if c]
        merged, gained, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < 15:
                merged.append(tiles[i] + 1)
                gained += 1 << (tiles[i] + 1)
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        merged += [0] * (4 - len(merged))
        out = 0
        for j, c in enumerate(merged):
            out |= c << (4 * j)
        left[row] = out
        score[row] = gained
        rev = ((row & 0xF) << 12) | (((row >> 4) & 0xF) << 8) | (((row >> 8) & 0xF) << 4) | (row >> 12)
        rev_out = ((out & 0xF) << 12) | (((out >> 4) & 0xF) << 8) | (((out >> 8) & 0xF) << 4) | (out >> 12)
        right[rev] = rev_out
    return left, right, score


ROW_LEFT, ROW_RIGHT, ROW_SCORE = _build_row_tables()


@njit(cache=True)
def transpose(b):
    a1 = b & np.uint64(0xF0F00F0FF0F00F0F)
    a2 = b & np.uint64(0x0000F0F00000F0F0)
    a3 = b & np.uint64(0x0F0F00000F0F0000)
    a = a1 | (a2 << np.uint64(12)) | (a3 >> np.uint64(12))
    b1 = a & np.uint64(0xFF00FF0000FF00FF)
    b2 = a & np.uint64(0x00FF00FF00000000)
    b3 = a & np.uint64(0x00000000FF00FF00)
    return b1 | (b2 >> np.uint64(24)) | (b3 << np.uint64(24))


@njit(cache=True)
def _rows(b, table):
    out = np.uint64(0)
    for i in range(4):
        row = (b >> np.uint64(16 * i)) & np.uint64(0xFFFF)
        out |= np.uint64(table[row]) << np.uint64(16 * i)
    return out


@njit(cache=True)
def _row_gain(b):
    g = 0.0
    for i in range(4):
        g += ROW_SCORE[(b >> np.uint64(16 * i)) & np.uint64(0xFFFF)]
    return g


@njit(cache=True)
def apply_move(b, move):
    """Board after a swipe (before the new tile) and the points it scores."""
    if move == 0:
        return _rows(b, ROW_LEFT), _row_gain(b)
    if move == 1:
        return _rows(b, ROW_RIGHT), _row_gain(b)
    t = transpose(b)
    if move == 2:
        return transpose(_rows(t, ROW_LEFT)), _row_gain(t)
    return transpose(_rows(t, ROW_RIGHT)), _row_gain(t)


@njit(cache=True)
def count_empty(b):
    n = 0
    for i in range(16):
        if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) == 0:
            n += 1
    return n


@njit(cache=True)
def max_exponent(b):
    m = 0
    for i in range(16):
        c = int((b >> np.uint64(4 * i)) & np.uint64(0xF))
        if c > m:
            m = c
    return m


def swipe(b, move):
    """Python-side apply_move: numba returns uint64 boards as plain ints, so wrap them back."""
    nb, gain = apply_move(np.uint64(b), move)
    return np.uint64(nb), int(gain)


def to_grid(b):
    """4x4 array of exponents; row 0 is the top row, column 0 the left column."""
    b = int(b)
    return np.array([[(b >> (4 * (4 * r + c))) & 0xF for c in range(4)] for r in range(4)], np.int8)


def from_grid(grid):
    b = 0
    for r in range(4):
        for c in range(4):
            b |= int(grid[r][c]) << (4 * (4 * r + c))
    return np.uint64(b)


def tiles(grid):
    """Exponents to tile values (0 for empty)."""
    grid = np.asarray(grid, np.int64)
    return np.where(grid > 0, 1 << grid, 0)


class Game:
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.board = np.uint64(0)
        self.score = 0
        self.moves = 0
        self._spawn()
        self._spawn()

    def _spawn(self):
        empty = [i for i in range(16) if (int(self.board) >> (4 * i)) & 0xF == 0]
        cell = empty[int(self.rng.integers(len(empty)))]
        value = 1 if self.rng.random() < 0.9 else 2
        self.board = np.uint64(int(self.board) | (value << (4 * cell)))

    def afterstates(self):
        """{move: (board after the swipe, points)} for every swipe that changes the board."""
        out = {}
        for m in range(4):
            nb, gain = swipe(self.board, m)
            if nb != self.board:
                out[m] = (nb, gain)
        return out

    def step(self, move):
        nb, gain = swipe(self.board, move)
        if nb == self.board:
            raise ValueError(f"illegal move {MOVE_NAMES[move]}")
        self.board = nb
        self.score += gain
        self.moves += 1
        self._spawn()

    @property
    def over(self):
        return not self.afterstates()

    @property
    def max_tile(self):
        return 1 << int(max_exponent(np.uint64(self.board)))
