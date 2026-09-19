"""Build the fly's 2048 readout: teacher positions -> connectome spikes -> ridge readout.

1. Teacher games (seeds from 100,000) give positions. Every legal swipe from each one is scored with
   one layer of lookahead: the average, over every new tile that could appear, of the best next board
   under the teacher's heuristic. (A no-brain study showed this target makes the best one-look readout;
   see scripts/target_study.py.) Scores are centered and scaled within each position, since only the
   ranking of swipes matters.
2. Each swipe's resulting board is shown to the connectome for 150 ms through the 2048 eyes, and the
   3,555 L1/L2 neurons' spike counts are recorded.
3. On positions from training games, neurons are ranked by how well they track the teacher; the best
   `keep` feed a ridge readout. Held-out games give top-swipe agreement with the teacher. The same
   readout fit directly on the eye inputs (no brain) is reported as a baseline.

usage: python scripts/build_readout.py [positions] [keep]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fly2048.brain import Brain, load_neurons
from fly2048.eyes import Eyes, board_channels
from fly2048.game import Game
from fly2048.teacher import chance_value, swipe_values, teacher_move

N_POSITIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
KEEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
PER_GAME, STRIDE, STEPS, BATCH = 400, 3, 150, 256
OUT = Path(os.environ.get("FLY2048_RUNS", ROOT / "runs")) / "readout"
OUT.mkdir(parents=True, exist_ok=True)


def collect_positions():
    cache = OUT / "positions.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    before, after, target, group, game_of = [], [], [], [], []
    seed, parent = 100_000, 0
    while parent < N_POSITIONS:
        game, taken = Game(seed), 0
        while not game.over and taken < PER_GAME and parent < N_POSITIONS:
            if game.moves % STRIDE == 0:
                values = swipe_values(game.board)
                if len(values) > 1:
                    v = np.array(list(values.values()))
                    v = (v - v.mean()) / max(v.std(), 1e-6)
                    for (move, _), score in zip(values.items(), v):
                        before.append(int(game.board))
                        after.append(int(game.afterstates()[move][0]))
                        target.append(score)
                        group.append(parent)
                        game_of.append(seed)
                    parent += 1
                    taken += 1
            game.step(teacher_move(game.board))
        print(f"teacher game {seed}: {taken} positions (max tile {game.max_tile}), {parent}/{N_POSITIONS}", flush=True)
        seed += 1
    data = {k: np.asarray(v) for k, v in dict(before=before, after=after, target=target, group=group, game=game_of).items()}
    data["before"] = data["before"].astype(np.uint64)
    data["after"] = data["after"].astype(np.uint64)
    np.savez(cache, **data)
    return data


def lookahead_target(data):
    """One-lookahead-layer value of each after-swipe board, standardized within its position."""
    raw = np.array([chance_value(np.uint64(a), 1, 1.0) for a in data["after"]])
    out = np.empty_like(raw)
    for g in np.unique(data["group"]):
        idx = data["group"] == g
        out[idx] = (raw[idx] - raw[idx].mean()) / max(raw[idx].std(), 1e-6)
    return out


def simulate(data):
    cache = OUT / "spikes_after.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["counts"].astype(np.float32), z["neurons"]
    brain, eyes = Brain(threshold=5, w_scale=1.0), Eyes(board_hz=500)
    neurons = load_neurons(["idx", "type"])
    rec_np = neurons.loc[neurons.type.isin(["L1", "L2"]), "idx"].sort_values().to_numpy(np.int64)
    rec = torch.as_tensor(rec_np, device="cuda")
    n = len(data["target"])
    counts = np.empty((n, len(rec_np)), np.float16)
    started = time.time()
    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n)
        c, _ = brain.run(eyes.in_idx, eyes.probs(data["after"][start:stop]), STEPS, rec, eyes.phase)
        counts[start:stop] = c.T.cpu().numpy()
        print(f"simulated {stop}/{n} swipes, {time.time() - started:.0f}s", flush=True)
    np.savez(cache, counts=counts, neurons=rec_np)
    return counts.astype(np.float32), rec_np


def center_within(x, group):
    n_groups = group.max() + 1
    sums = np.zeros((n_groups, x.shape[1]))
    np.add.at(sums, group, x)
    return x - (sums / np.bincount(group, minlength=n_groups)[:, None])[group]


def fit_ridge(x, target, group, train, label):
    """Ridge over group-centered features; returns (agreement, lambda, weights, mean, scale)."""
    xc = center_within(x.astype(np.float64), group)
    mean, scale = xc[train].mean(0), xc[train].std(0)
    scale[scale < 1e-6] = 1
    z = (xc - mean) / scale
    test_groups = np.unique(group[~train])

    def agreement(score):
        hits = 0
        for g in test_groups:
            idx = np.flatnonzero(group == g)
            hits += idx[np.argmax(score[idx])] == idx[np.argmax(target[idx])]
        return hits / len(test_groups)

    best = None
    for lam in (0.1, 1, 10, 100, 1000, 10000):
        xt = z[train]
        w = np.linalg.solve(xt.T @ xt + lam * np.eye(z.shape[1]), xt.T @ target[train])
        acc = agreement(z @ w)
        print(f"  {label}: lambda {lam:g} -> held-out top-swipe agreement {acc:.3f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, lam, w, mean, scale)
    return best


def main():
    data = collect_positions()
    counts, rec_np = simulate(data)
    target, group = lookahead_target(data), data["group"].astype(np.int64)
    games = np.unique(data["game"])
    test_games = games[-max(2, len(games) // 5):]
    train = ~np.isin(data["game"], test_games)
    chance = np.mean([1 / np.sum(group == g) for g in np.unique(group[~train])])

    # Screen single neurons on training positions only.
    xc = center_within(counts.astype(np.float64), group)
    yc = target
    cov = ((xc[train] - xc[train].mean(0)) * (yc[train] - yc[train].mean())[:, None]).sum(0)
    corr = cov / np.sqrt(((xc[train] - xc[train].mean(0)) ** 2).sum(0) * ((yc[train] - yc[train].mean()) ** 2).sum() + 1e-12)
    selected = np.argsort(-np.abs(corr), kind="stable")[:KEEP]
    fly = fit_ridge(counts[:, selected], target, group, train, f"fly, {KEEP} L1/L2 neurons")

    eyes_only = fit_ridge(board_channels(data["after"]), target, group, train, "no brain, 32 eye inputs")

    np.savez(OUT / "readout.npz", neuron_idx=rec_np[selected], mean=fly[3], scale=fly[4], weights=fly[2],
             steps=STEPS, heldout_agreement=fly[0], ridge_lambda=fly[1], stimulus="after", target="one_lookahead_layer")
    np.savez(OUT / "readout_nobrain.npz", mean=eyes_only[3], scale=eyes_only[4], weights=eyes_only[2])
    report = {
        "positions": int(group.max() + 1), "swipes": int(len(target)), "teacher_games": len(games),
        "test_games": [int(g) for g in test_games], "chance_agreement": float(chance),
        "fly_agreement": float(fly[0]), "fly_lambda": fly[1], "neurons_kept": KEEP,
        "no_brain_agreement": float(eyes_only[0]), "no_brain_lambda": eyes_only[1],
        "mean_l1l2_spikes": float(counts.mean()),
    }
    (OUT / "gate.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
