"""Jev plays 2048: one choice question per move, called through the Vercel AI Gateway.

Jev sees the current board and, for each legal swipe, the board that swipe leaves before the new tile
appears. By default it gets only the rules and the goal. With strategy=True it also gets the standard
advice people give beginners, in words: the prompt counterpart of the fly's readout learning from
example moves. It never gets scores from a search. Needs AI_GATEWAY_API_KEY in the environment.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request

import numpy as np

from .game import MOVE_NAMES, tiles, to_grid

URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
GOAL = ("2048: swipe to slide every tile toward one wall; two equal tiles that meet merge into their sum. "
        "After each swipe a new 2 or 4 appears in a random empty cell. The game ends when the board is full "
        "and no swipe can merge anything. Goal: build the largest tile you can.")
INSTRUCTIONS = "Choose the swipe that gives the best chance of building bigger tiles without getting stuck."
STRATEGY = ("Strategy that works: keep your biggest tile in one corner and never move it out; keep the row along "
            "that corner full and ordered from big to small; prefer the two swipes toward that corner and use the "
            "third only when needed; avoid the swipe that pulls the biggest tile away; keep empty cells and "
            "equal neighbours next to each other so they can merge.")


def board_rows(board):
    return tiles(to_grid(board)).tolist()


def request_body(board, afterstates, strategy=False):
    criteria = {
        MOVE_NAMES[m]: f"Swipe {MOVE_NAMES[m]}. Board after the swipe, before the new tile: {json.dumps(board_rows(nb))}. Points scored: {gain}."
        for m, (nb, gain) in afterstates.items()
    }
    return {
        "state": {"game": GOAL, **({"strategy": STRATEGY} if strategy else {}),
                  "board": board_rows(board), "notation": "Rows top to bottom, 0 is an empty cell."},
        "questions": {"swipe": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
    }


def ask(body, key=None, attempts=12):
    """POST one evaluation; retries Jev's bursty 503s quickly with jitter."""
    key = key or os.environ["AI_GATEWAY_API_KEY"]
    data = json.dumps(body).encode()
    headers = {
        "Authorization": f"Bearer {key}", "content-type": "application/json",
        "ai-gateway-protocol-version": "0.0.1", "ai-evaluation-model-specification-version": "4",
        "ai-model-id": "typesafe-ai/jev",
    }
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(URL, data=data, headers=headers, method="POST")
            # Calls normally take under a second; a few hang, so give up fast and retry.
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            status = getattr(error, "code", None)
            if status not in (None, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            time.sleep(0.2 + random.random() * 0.3 + attempt * 0.3)


def jev_move(board, afterstates, key=None, strategy=False):
    """(move, probabilities by move name, input tokens) for one position."""
    result = ask(request_body(board, afterstates, strategy), key)
    answer = result["answers"]["swipe"]
    probs = answer.get("probabilities") or {answer["choice"]: 1.0}
    name = max(probs, key=probs.get)
    tokens = (result.get("usage") or {}).get("inputTokens", 0)
    return MOVE_NAMES.index(name), probs, tokens


def play(game, key=None, log_every=100, strategy=False):
    """Play a game to the end with Jev. Returns the move list and per-move top probability."""
    moves, confidence, tokens = [], [], 0
    while not game.over:
        after = game.afterstates()
        move, probs, used = jev_move(game.board, after, key, strategy)
        game.step(move)
        moves.append(move)
        confidence.append(float(probs.get(MOVE_NAMES[move], 0)))
        tokens += used
        if log_every and game.moves % log_every == 0:
            print(f"  seed {game.seed}: move {game.moves}, max {game.max_tile}, score {game.score}", flush=True)
    return {"moves": np.asarray(moves, np.uint8).tolist(), "confidence": confidence, "tokens": tokens}
