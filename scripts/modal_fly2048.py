"""Run the GPU-heavy steps on Modal (L4), reusing the Flytris connectome volume.

    modal volume put fly2048-runs runs/readout/positions.npz readout/positions.npz
    modal run scripts/modal_fly2048.py::build_readout

Outputs land on the fly2048-runs volume; fetch them with `modal volume get`.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
EYE_MAP = ROOT.parent / "Flytris" / "data" / "malecns" / "eye_map_2048.npz"

app = modal.App("fly2048")
data_volume = modal.Volume.from_name("flytris-data")
runs_volume = modal.Volume.from_name("fly2048-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.4.2", "pandas==2.2.3", "pyarrow==23.0.1", "torch==2.7.1", "numba==0.67.0")
    .env({"FLYTRIS_DATA": "/root/data/malecns", "FLY2048_EYE_MAP": "/root/eye_map_2048.npz",
          "FLY2048_RUNS": "/root/runs", "PYTHONUNBUFFERED": "1"})
    .add_local_file(EYE_MAP, "/root/eye_map_2048.npz")
    .add_local_dir(ROOT / "fly2048", "/root/fly2048", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", "/root/scripts", ignore=["__pycache__"])
)
gpu_job = {"image": image, "gpu": "L4", "cpu": 4, "memory": 16384, "timeout": 3600,
           "volumes": {"/root/data": data_volume, "/root/runs": runs_volume}}


def _run(*args):
    import subprocess
    import sys
    try:
        subprocess.run([sys.executable, *args], check=True, cwd="/root")
    finally:
        runs_volume.commit()


@app.function(**gpu_job)
def build_readout(positions: int = 4000, keep: int = 2048):
    _run("/root/scripts/build_readout.py", str(positions), str(keep))


@app.function(**{**gpu_job, "timeout": 3 * 3600})
def fly_games(first: int = 5000, last: int = 5016, control: str = "none"):
    _run("/root/scripts/fly_games.py", str(first), str(last), control)
