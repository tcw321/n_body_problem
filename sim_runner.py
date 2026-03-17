"""
Batch simulation runner — used for container smoke tests.

Runs N steps of the leapfrog simulation from a fixed seed and prints
the final positions and velocities as JSON to stdout, then exits.

Usage:
    python sim_runner.py [N] [steps] [seed]

Defaults: N=10, steps=10, seed=42
"""
import json
import sys

import numpy as np

from main import leapfrog_step


def run(N: int = 10, steps: int = 10, seed: int = 42) -> dict:
    np.random.seed(seed)
    pos  = np.random.randn(N, 3) * 5.0
    vel  = np.random.randn(N, 3) * 0.1
    mass = np.random.uniform(0.5, 2.0, N)

    for _ in range(steps):
        pos, vel = leapfrog_step(pos, vel, mass)

    return {"N": N, "steps": steps, "seed": seed,
            "pos": pos.tolist(), "vel": vel.tolist()}


if __name__ == "__main__":
    args = sys.argv[1:]
    N     = int(args[0]) if len(args) > 0 else 10
    steps = int(args[1]) if len(args) > 1 else 10
    seed  = int(args[2]) if len(args) > 2 else 42
    print(json.dumps(run(N, steps, seed)))
