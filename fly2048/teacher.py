"""Expectimax 2048 teacher: the fly's readout learns to imitate its judgment of each swipe.

The board heuristic is the widely used one from nneonneo/2048-ai (empty cells, merges, monotonic rows,
a penalty on large scattered tiles), scored over every row and column through a 65,536-entry table.
The search alternates player swipes and chance nodes (a 2 or a 4 in every empty cell) and stops at a
depth limit or when a branch becomes too unlikely to matter.
"""
import numpy as np
from numba import njit

from .game import apply_move, count_empty, swipe, transpose

LOST_PENALTY, MONO_POWER, MONO_WEIGHT = 200000.0, 4.0, 47.0
SUM_POWER, SUM_WEIGHT, MERGES_WEIGHT, EMPTY_WEIGHT = 3.5, 11.0, 700.0, 270.0
CPROB_THRESHOLD = 1e-4


def _build_heuristic_table():
    table = np.zeros(65536, np.float64)
    for row in range(65536):
        line = [(row >> (4 * i)) & 0xF for i in range(4)]
        total = empty = merges = 0.0
        prev = counter = 0
        for rank in line:
            total += rank ** SUM_POWER
            if rank == 0:
                empty += 1
            else:
                if prev == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = rank
        if counter > 0:
            merges += 1 + counter
        mono_left = mono_right = 0.0
        for i in range(1, 4):
            if line[i - 1] > line[i]:
                mono_left += line[i - 1] ** MONO_POWER - line[i] ** MONO_POWER
            else:
                mono_right += line[i] ** MONO_POWER - line[i - 1] ** MONO_POWER
        table[row] = (LOST_PENALTY + EMPTY_WEIGHT * empty + MERGES_WEIGHT * merges
                      - MONO_WEIGHT * min(mono_left, mono_right) - SUM_WEIGHT * total)
    return table


HEUR = _build_heuristic_table()


@njit(cache=True)
def heuristic(b):
    t = transpose(b)
    s = 0.0
    for i in range(4):
        s += HEUR[(b >> np.uint64(16 * i)) & np.uint64(0xFFFF)]
        s += HEUR[(t >> np.uint64(16 * i)) & np.uint64(0xFFFF)]
    return s


@njit(cache=True)
def chance_value(b, depth, cprob):
    """Expected value of an after-swipe board: average over new tiles of the best next swipe."""
    if depth <= 0 or cprob < CPROB_THRESHOLD:
        return heuristic(b)
    empties = count_empty(b)
    if empties == 0:
        return heuristic(b)
    total = 0.0
    for i in range(16):
        if ((b >> np.uint64(4 * i)) & np.uint64(0xF)) != 0:
            continue
        for k in range(2):
            value = 1 + k
            p = 0.9 if k == 0 else 0.1
            nb = b | (np.uint64(value) << np.uint64(4 * i))
            best = 0.0
            for m in range(4):
                mb, _ = apply_move(nb, m)
                if mb != nb:
                    v = chance_value(mb, depth - 1, cprob * p / empties)
                    if v > best:
                        best = v
            total += p * best
    return total / empties


@njit(cache=True)
def distinct_tiles(b):
    seen = 0
    n = 0
    for i in range(16):
        c = int((b >> np.uint64(4 * i)) & np.uint64(0xF))
        if c and not (seen >> c) & 1:
            seen |= 1 << c
            n += 1
    return n


def search_depth(board):
    """Deeper when the board is crowded (few branches) or has many distinct tiles."""
    empties = count_empty(board)
    if empties <= 3:
        return 4
    if empties <= 6 or distinct_tiles(board) >= 8:
        return 3
    return 2


def swipe_values(board):
    """{move: expectimax value of the board after that swipe} for every legal swipe."""
    board = np.uint64(board)
    depth = search_depth(board)
    out = {}
    for m in range(4):
        nb, _ = swipe(board, m)
        if nb != board:
            out[m] = float(chance_value(nb, depth, 1.0))
    return out


def teacher_move(board):
    values = swipe_values(board)
    return max(values, key=values.get) if values else None
