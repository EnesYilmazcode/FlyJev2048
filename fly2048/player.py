"""The fly's 2048 player: each legal swipe's resulting board goes through the eyes into the connectome
for 150 ms, and a linear readout over the recorded L1/L2 neurons scores it. The best-scoring swipe is
played. Controls break one part of the pathway on purpose."""
import numpy as np
import torch

from .brain import Brain
from .eyes import Eyes


class FlyPlayer:
    def __init__(self, readout_file, control="none", device="cuda", max_batch=512):
        with np.load(readout_file) as z:
            self.rec = torch.as_tensor(z["neuron_idx"].astype(np.int64), device=device)
            self.scale = z["scale"].astype(np.float32)
            self.weights = z["weights"].astype(np.float32)
            self.steps = int(z["steps"])
        self.brain = Brain(threshold=5, w_scale=1.0, device=device)
        self.eyes = Eyes(board_hz=500, device=device)
        self.max_batch = max_batch
        rng = np.random.default_rng(99173)
        if control == "silenced":                 # readout gets nothing from the brain
            self.weights[:] = 0
        elif control == "shuffled_readout":       # right neurons, scrambled weights
            self.weights = self.weights[rng.permutation(len(self.weights))]
        elif control == "shuffled_input":         # the eyes' wiring into the brain scrambled
            self.eyes.in_idx = self.eyes.in_idx[torch.as_tensor(rng.permutation(len(self.eyes.in_idx)), device=device)]
        elif control != "none":
            raise ValueError(f"unknown control {control}")

    def scores(self, boards):
        """Readout score for each after-swipe board (higher is better)."""
        out = np.empty(len(boards), np.float32)
        for start in range(0, len(boards), self.max_batch):
            chunk = boards[start:start + self.max_batch]
            counts, _ = self.brain.run(self.eyes.in_idx, self.eyes.probs(chunk), self.steps, self.rec, self.eyes.phase)
            out[start:start + len(chunk)] = (counts.T.cpu().numpy() / self.scale) @ self.weights
        return out
