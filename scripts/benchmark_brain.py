"""Time the connectome on 2048 candidate swipes: board -> eyes -> 150 ms of spiking -> L1/L2 counts.
usage: python scripts/benchmark_brain.py [batch] [repeats]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly2048.brain import Brain, load_neurons
from fly2048.eyes import Eyes
from fly2048.game import Game
from fly2048.teacher import teacher_move

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 128
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# Real positions from a teacher game, each paired with one of its after-swipe boards.
before, after, game = [], [], Game(11)
while len(before) < batch and not game.over:
    for nb, _ in game.afterstates().values():
        before.append(game.board)
        after.append(nb)
    game.step(teacher_move(game.board))
before, after = before[:batch], after[:batch]

started = time.time()
brain = Brain(threshold=5, w_scale=1.0)
eyes = Eyes(board_hz=500)
neurons = load_neurons(["idx", "type"])
rec = torch.as_tensor(neurons.loc[neurons.type.isin(["L1", "L2"]), "idx"].sort_values().to_numpy(np.int64), device="cuda")
print(f"loaded {brain.n} neurons, {brain.nnz} synapse edges, {len(rec)} L1/L2 in {time.time() - started:.0f}s", flush=True)

p0, p1 = eyes.probs(before), eyes.probs(after)
for r in range(repeats):
    torch.cuda.synchronize(); t = time.time()
    counts, rate = brain.run(eyes.in_idx, p0, 150, rec, eyes.phase, p_in_after=p1, switch_step=75)
    torch.cuda.synchronize(); took = time.time() - t
    print(f"batch {batch}: {took:.2f}s ({took / batch * 1000:.1f} ms per candidate), mean L1/L2 spikes "
          f"{counts.mean().item():.2f}, whole-brain rate {rate.item():.4f}, peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
