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


def _parallel_acc(pos, mass, n_workers, G, softening):
    """Compute full acceleration array using Ray remote tasks."""
    N = len(mass)
    chunk_size = N // n_workers

    chunks = []
    for w in range(n_workers):
        i_start = w * chunk_size
        i_end = N if w == n_workers - 1 else i_start + chunk_size
        chunks.append((i_start, i_end, pos, mass, G, softening))

    results = ray.get([_ray_chunk.remote(chunk) for chunk in chunks])

    acc = np.zeros((N, 3))
    for i_start, acc_chunk in results:
        acc[i_start:i_start + len(acc_chunk)] = acc_chunk
    return acc


def nbody_ray_step(pos, vel, mass, n_workers=4, G=1.0, dt=0.01, softening=0.1):
    """
    Single leapfrog (KDK) step using Ray remote tasks for parallel force computation.

    Algorithm:
      1. acc      = forces(pos)              [parallel, round 1]
      2. vel_half = vel + acc * (dt/2)       [half kick]
      3. pos_new  = pos + vel_half * dt      [full drift]
      4. acc_new  = forces(pos_new)          [parallel, round 2]
      5. vel_new  = vel_half + acc_new*(dt/2)[half kick]
    """
    acc = _parallel_acc(pos, mass, n_workers, G, softening)
    vel_half = vel + acc * (dt / 2)
    pos_new = pos + vel_half * dt
    acc_new = _parallel_acc(pos_new, mass, n_workers, G, softening)
    vel_new = vel_half + acc_new * (dt / 2)
    return pos_new, vel_new
