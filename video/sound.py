"""Calm, synthesized sound for the Fly vs Jev video (2048 itself has no sound).

Events come from the render timeline: a soft tick per swipe, a sine pluck per merge whose pitch rises
with the tile (a pentatonic scale, so any run of merges stays consonant), a low two-note fall when a
player is out, and a warm chord when the biggest tile appears. The fly is panned left and Jev right.
Returns a stereo float array at 48 kHz; write it with write_wav().
"""
import wave

import numpy as np

SR = 48000
PENTATONIC = [0, 2, 4, 7, 9]                     # semitones within an octave


def pitch(exp):
    """Tile exponent -> frequency on a C major pentatonic scale starting at C4 for a 4."""
    step = max(0, exp - 2)
    semis = 12 * (step // 5) + PENTATONIC[step % 5]
    return 261.63 * 2 ** (semis / 12)


def pluck(freq, dur=0.5, decay=6.0):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, t / 0.004) * np.exp(-decay * t)
    return env * (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t) * np.exp(-3 * t))


def tick():
    rng = np.random.default_rng(7)
    n = int(SR * 0.03)
    noise = rng.standard_normal(n)
    kernel = np.ones(12) / 12                    # soften the click
    return np.convolve(noise, kernel, "same") * np.exp(-np.arange(n) / (SR * 0.006))


def chord(root, dur=2.4):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, t / 0.08) * np.exp(-1.6 * t)
    return env * sum(np.sin(2 * np.pi * root * r * t) * a for r, a in ((1, 1), (1.25, 0.7), (1.5, 0.6), (2, 0.35)))


def fall(freq):
    return np.concatenate([pluck(freq, 0.35, 5), pluck(freq * 0.84, 0.7, 3.5)])


def mix(events, total):
    """events: (time, kind, pan, exp). pan -1 = left, +1 = right."""
    out = np.zeros((int(SR * (total + 3)), 2))
    sounds = {"tick": (tick(), 0.035), "fall": (None, 0.16), "chord": (None, 0.16)}
    for time, kind, pan, exp in events:
        if kind == "tick":
            wave_, gain = sounds["tick"]
        elif kind == "merge":
            wave_, gain = pluck(pitch(exp)), 0.05 + 0.012 * min(exp, 11)
        elif kind == "fall":
            wave_, gain = fall(196.0), 0.16
        else:
            wave_, gain = chord(pitch(7) / 2), 0.13
        i = int(time * SR)
        seg = wave_[: max(0, len(out) - i)] * gain
        left, right = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
        out[i:i + len(seg), 0] += seg * left
        out[i:i + len(seg), 1] += seg * right
    # gentle limiter so dense merges never clip
    return np.tanh(out * 1.2) / np.tanh(1.2)


def write_wav(path, audio):
    data = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
