"""
Phase 5 — Ray-based n-body simulation.

Replaces multiprocessing.Pool with Ray remote tasks.
The force computation (compute_forces_chunk) is unchanged — only the worker
dispatch mechanism changes.

Usage:
    import ray
    ray.init()
    pos, vel = nbody_ray_step(pos, vel, mass, n_workers=4)
    ray.shutdown()
"""
import numpy as np
import ray

from main import compute_forces_chunk


@ray.remote
def _ray_chunk(args):
    """Ray remote wrapper around compute_forces_chunk."""
    return compute_forces_chunk(args)


def nbody_ray_step(pos, vel, mass, n_workers=4, G=1.0, dt=0.01, softening=0.1):
    """
    Single simulation step using Ray remote tasks for parallel force computation.

    Ray.remote tasks replace Pool.map — the chunking logic and Euler integrator
    are identical to nbody_parallel_step_persistent in main.py.
    """
    N = len(mass)
    chunk_size = N // n_workers

    chunks = []
    for w in range(n_workers):
        i_start = w * chunk_size
        i_end = N if w == n_workers - 1 else i_start + chunk_size
        chunks.append((i_start, i_end, pos, mass, G, softening))

    futures = [_ray_chunk.remote(chunk) for chunk in chunks]
    results = ray.get(futures)

    acc = np.zeros((N, 3))
    for i_start, acc_chunk in results:
        acc[i_start:i_start + len(acc_chunk)] = acc_chunk

    vel = vel + acc * dt
    pos = pos + vel * dt
    return pos, vel
