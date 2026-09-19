"""Fly vs Jev, side by side on the same seed, drawn with PIL at 2x and encoded with ffmpeg.

usage: python video/render2048.py <seed> [out.mp4] [fps]
Reads runs/fly-none/<seed>.json and runs/jev-rules/<seed>.json. Both games share one move clock: move k
starts at the same moment on both boards. The pace starts slow and speeds up, so a 1,000-move game fits
in about 20 seconds.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import replay

ROOT = Path(__file__).resolve().parents[1]
W, H, S = 1800, 1200, 2                                   # output size, supersampling factor
BG, BOARD, EMPTY, DARK, LIGHT = "#faf8ef", "#bbada0", "#cdc1b4", "#776e65", "#f9f6f2"
TILE = {1: "#eee4da", 2: "#ede0c8", 3: "#f2b179", 4: "#f59563", 5: "#f67c5f", 6: "#f65e3b", 7: "#edcf72",
        8: "#edcc61", 9: "#edc850", 10: "#edc53f", 11: "#edc22e"}
BOARD_PX, GAP = 640, 14
CELL = (BOARD_PX - 5 * GAP) / 4
BOARDS = {"fly": (W * 0.27 - BOARD_PX / 2, 330), "jev": (W * 0.73 - BOARD_PX / 2, 330)}
LABELS = {"fly": "Fly", "jev": "Jev"}
FONT = "C:/Windows/Fonts/segoeuib.ttf"
INTRO, OUTRO, D0, DECAY, DMIN = 0.9, 2.2, 0.16, 0.975, 0.012


def font(size):
    return ImageFont.truetype(FONT, int(size * S))


def ease(x):
    return 1 - (1 - x) ** 3


def durations(n):
    return np.maximum(DMIN, D0 * DECAY ** np.arange(n))


def cell_xy(board, r, c):
    x0, y0 = BOARDS[board]
    return x0 + GAP + c * (CELL + GAP), y0 + GAP + r * (CELL + GAP)


def draw_tile(d, board, x, y, exp, scale=1.0):
    size = CELL * scale
    cx, cy = x + CELL / 2, y + CELL / 2
    box = [(cx - size / 2) * S, (cy - size / 2) * S, (cx + size / 2) * S, (cy + size / 2) * S]
    d.rounded_rectangle(box, radius=6 * S * scale, fill=TILE.get(exp, "#3c3a32"))
    value = str(1 << exp)
    fsize = {1: 64, 2: 64, 3: 58, 4: 50}.get(len(value), 38) * scale
    f = font(fsize)
    color = DARK if exp <= 2 else LIGHT
    bbox = d.textbbox((0, 0), value, font=f)
    d.text((cx * S - (bbox[2] + bbox[0]) / 2, cy * S - (bbox[3] + bbox[1]) / 2), value, font=f, fill=color)


def draw_board_frame(d, board):
    x0, y0 = BOARDS[board]
    d.rounded_rectangle([x0 * S, y0 * S, (x0 + BOARD_PX) * S, (y0 + BOARD_PX) * S], radius=12 * S, fill=BOARD)
    for r in range(4):
        for c in range(4):
            x, y = cell_xy(board, r, c)
            d.rounded_rectangle([x * S, y * S, (x + CELL) * S, (y + CELL) * S], radius=6 * S, fill=EMPTY)


def draw_header(d, board, score):
    x0, y0 = BOARDS[board]
    f = font(76)
    d.text((x0 * S, (y0 - 118) * S), LABELS[board], font=f, fill=DARK)
    bw, bh = 170, 74
    bx, by = x0 + BOARD_PX - bw, y0 - 106
    d.rounded_rectangle([bx * S, by * S, (bx + bw) * S, (by + bh) * S], radius=6 * S, fill=BOARD)
    small, big = font(17), font(32)
    for text, fnt, yy, fill in (("SCORE", small, by + 18, "#eee4da"), (f"{score:,}", big, by + 48, "white")):
        bbox = d.textbbox((0, 0), text, font=fnt)
        d.text(((bx + bw / 2) * S - (bbox[2] + bbox[0]) / 2, yy * S - (bbox[3] + bbox[1]) / 2), text, font=fnt, fill=fill)


def draw_game_over(img, board, alpha):
    x0, y0 = BOARDS[board]
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([x0 * S, y0 * S, (x0 + BOARD_PX) * S, (y0 + BOARD_PX) * S], radius=12 * S,
                         fill=(238, 228, 218, int(186 * alpha)))
    f = font(64)
    text = "Game over"
    bbox = od.textbbox((0, 0), text, font=f)
    od.text(((x0 + BOARD_PX / 2) * S - (bbox[2] + bbox[0]) / 2, (y0 + BOARD_PX / 2) * S - (bbox[3] + bbox[1]) / 2),
            text, font=f, fill=(119, 110, 101, int(255 * alpha)))
    img.alpha_composite(overlay)


def board_at(d, board, start, steps, t, starts, durs):
    """Draw one board at time t; returns its score at t."""
    k = int(np.searchsorted(starts, t, side="right")) - 1
    draw_board_frame(d, board)
    if k < 0:
        for (r, c), tile in start.items():
            draw_tile(d, board, *cell_xy(board, r, c), tile.exp)
        return 0
    if k >= len(steps):
        for (r, c), tile in steps[-1]["after"].items():
            draw_tile(d, board, *cell_xy(board, r, c), tile.exp)
        return steps[-1]["score"]
    step = steps[k]
    p = (t - starts[k]) / durs[k]
    if p >= 1:                                   # after this game's last move: hold the final board
        for (r, c), tile in step["after"].items():
            draw_tile(d, board, *cell_xy(board, r, c), tile.exp)
        return step["score"]
    slide_end = 0.55
    if p < slide_end:
        q = ease(p / slide_end)
        for tile, (r0, c0), (r1, c1) in step["motions"]:
            x0, y0 = cell_xy(board, r0, c0)
            x1, y1 = cell_xy(board, r1, c1)
            draw_tile(d, board, x0 + (x1 - x0) * q, y0 + (y1 - y0) * q, tile.exp)
        return steps[k - 1]["score"] if k else 0
    q = (p - slide_end) / (1 - slide_end)
    merged = {m[0].id for m in step["merges"]}
    for (r, c), tile in step["after"].items():
        if tile is step["spawn"]:
            draw_tile(d, board, *cell_xy(board, r, c), tile.exp, scale=max(0.05, ease(q)))
        elif tile.id in merged:
            draw_tile(d, board, *cell_xy(board, r, c), tile.exp, scale=1 + 0.14 * np.sin(np.pi * min(1, q * 1.4)))
        else:
            draw_tile(d, board, *cell_xy(board, r, c), tile.exp)
    return step["score"]


def main():
    seed = int(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "renders" / f"fly-vs-jev-{seed}.mp4")
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    games = {}
    for board, folder in (("fly", "fly-none"), ("jev", "jev-rules")):
        rec = json.loads((ROOT / "runs" / folder / f"{seed}.json").read_text(encoding="utf8"))
        games[board] = replay(seed, rec["moves"])
    longest = max(len(g[1]) for g in games.values())
    durs = durations(longest)
    starts = INTRO + np.concatenate([[0], np.cumsum(durs)[:-1]])
    end_of = {b: starts[len(g[1]) - 1] + durs[len(g[1]) - 1] for b, g in games.items()}
    total = max(end_of.values()) + OUTRO
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n_frames = int(total * fps)
    for i in range(n_frames):
        t = i / fps
        img = Image.new("RGBA", (W * S, H * S), BG)
        d = ImageDraw.Draw(img)
        for board, (start, steps) in games.items():
            score = board_at(d, board, start, steps, t, starts, durs)
            draw_header(d, board, score)
        for board in games:
            if t > end_of[board] + 0.25:
                draw_game_over(img, board, min(1.0, (t - end_of[board] - 0.25) / 0.4))
        frame = img.convert("RGB").resize((W, H), Image.LANCZOS)
        ff.stdin.write(frame.tobytes())
        if i % 300 == 0:
            print(f"frame {i}/{n_frames}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out}: {total:.1f}s, fly {len(games['fly'][1])} moves, jev {len(games['jev'][1])} moves")


if __name__ == "__main__":
    main()
