"""
Phase 3 tests — Multi-core scaling and numerical consistency.

Run with:  pytest test_phase3.py -v
"""
import os
import numpy as np
import pytest
from multiprocessing import Pool

from main import nbody_step, nbody_parallel_step_persistent
from benchmark import time_steps, sweep_workers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_conditions(N=50, seed=7):
    np.random.seed(seed)
    pos  = np.random.randn(N, 3) * 5.0
    vel  = np.random.randn(N, 3) * 0.1
    mass = np.random.uniform(0.5, 2.0, N)
    return pos, vel, mass


# ---------------------------------------------------------------------------
# Test 1: 1-worker persistent pool matches single-process serial
#
# Both paths call compute_forces_chunk with identical arguments in the same
# order, so results should be bit-for-bit identical.
# ---------------------------------------------------------------------------

def test_1worker_parallel_matches_serial():
    N = 50
    pos, vel, mass = make_initial_conditions(N)
    G, dt, softening = 1.0, 0.01, 0.1
    steps = 5

    pos_s, vel_s = pos.copy(), vel.copy()
    for _ in range(steps):
        pos_s, vel_s = nbody_step(pos_s, vel_s, mass, G=G, dt=dt, softening=softening)

    pos_p, vel_p = pos.copy(), vel.copy()
    with Pool(1) as pool:
        for _ in range(steps):
            pos_p, vel_p = nbody_parallel_step_persistent(
                pos_p, vel_p, mass, pool, 1, G=G, dt=dt, softening=softening
            )

    np.testing.assert_array_equal(pos_s, pos_p,
        err_msg="1-worker parallel position differs from serial")
    np.testing.assert_array_equal(vel_s, vel_p,
        err_msg="1-worker parallel velocity differs from serial")


# ---------------------------------------------------------------------------
# Test 2: N-worker result matches 1-worker result (parallelism is correct)
#
# The chunked parallel computation must produce the same physics as the
# single-process version, regardless of how the bodies are divided.
# ---------------------------------------------------------------------------

def test_nworker_matches_1worker():
    n_workers = min(4, os.cpu_count())
    if n_workers < 2:
        pytest.skip("Need at least 2 CPUs to test multi-worker equivalence")

    N = 100
    pos, vel, mass = make_initial_conditions(N)
    G, dt, softening = 1.0, 0.01, 0.1
    steps = 5

    pos_1, vel_1 = pos.copy(), vel.copy()
    with Pool(1) as pool:
        for _ in range(steps):
            pos_1, vel_1 = nbody_parallel_step_persistent(
                pos_1, vel_1, mass, pool, 1, G=G, dt=dt, softening=softening
            )

    pos_n, vel_n = pos.copy(), vel.copy()
    with Pool(n_workers) as pool:
        for _ in range(steps):
            pos_n, vel_n = nbody_parallel_step_persistent(
                pos_n, vel_n, mass, pool, n_workers, G=G, dt=dt, softening=softening
            )

    np.testing.assert_allclose(pos_n, pos_1, rtol=1e-10,
        err_msg=f"{n_workers}-worker positions differ from 1-worker")
    np.testing.assert_allclose(vel_n, vel_1, rtol=1e-10,
        err_msg=f"{n_workers}-worker velocities differ from 1-worker")


# ---------------------------------------------------------------------------
# Test 3: timing harness returns an entry for every worker count in the sweep
# ---------------------------------------------------------------------------

def test_sweep_returns_all_worker_counts():
    max_workers = min(os.cpu_count(), 3)   # cap at 3 to keep test fast
    results = sweep_workers(N=100, steps=2, max_workers=max_workers)

    assert len(results) == max_workers, (
        f"Expected {max_workers} entries, got {len(results)}"
    )
    for w in range(1, max_workers + 1):
        assert w in results, f"Missing result for n_workers={w}"
        assert results[w] > 0, f"Non-positive time for n_workers={w}"


# ---------------------------------------------------------------------------
# Test 4: speedup is positive for large N with a persistent pool
#
# With enough bodies the parallel computation should outweigh the IPC
# overhead. Skipped on single-core machines.
# ---------------------------------------------------------------------------

def test_speedup_positive_large_N():
    if os.cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for a speedup test")

    N = 1000
    n_workers = min(4, os.cpu_count())
    pos, vel, mass = make_initial_conditions(N, seed=0)

    t1 = time_steps(pos, vel, mass, n_workers=1,        steps=3)
    tn = time_steps(pos, vel, mass, n_workers=n_workers, steps=3)

    assert tn < t1, (
        f"{n_workers}-worker time ({tn*1000:.1f} ms/step) not faster than "
        f"1-worker ({t1*1000:.1f} ms/step) at N={N}"
    )
