"""
Phase 5 tests — Ray distributed simulation.

Run with:  pytest test_phase5.py -v
"""
import numpy as np
import pytest
import ray

from main import nbody_step
from ray_sim import nbody_ray_step


# ---------------------------------------------------------------------------
# Session-scoped fixture: start Ray once, tear it down after all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def ray_session():
    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    ray.shutdown()


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
# Test 1: Ray worker output matches multiprocessing output (numerical equivalence)
# ---------------------------------------------------------------------------

def test_ray_matches_multiprocessing():
    N = 50
    pos, vel, mass = make_initial_conditions(N)
    G, dt, softening = 1.0, 0.01, 0.1
    steps = 5

    pos_mp, vel_mp = pos.copy(), vel.copy()
    for _ in range(steps):
        pos_mp, vel_mp = nbody_step(pos_mp, vel_mp, mass, G=G, dt=dt, softening=softening)

    pos_ray, vel_ray = pos.copy(), vel.copy()
    for _ in range(steps):
        pos_ray, vel_ray = nbody_ray_step(pos_ray, vel_ray, mass, n_workers=1,
                                          G=G, dt=dt, softening=softening)

    np.testing.assert_allclose(pos_ray, pos_mp, rtol=1e-10,
        err_msg="Ray positions differ from multiprocessing")
    np.testing.assert_allclose(vel_ray, vel_mp, rtol=1e-10,
        err_msg="Ray velocities differ from multiprocessing")


# ---------------------------------------------------------------------------
# Test 2: Simulation completes without error with 1 and 4 workers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_workers", [1, 4])
def test_ray_runs_with_n_workers(n_workers):
    pos, vel, mass = make_initial_conditions(N=50)
    for _ in range(3):
        pos, vel = nbody_ray_step(pos, vel, mass, n_workers=n_workers)
    assert np.all(np.isfinite(pos)), "Non-finite positions after Ray simulation"
    assert np.all(np.isfinite(vel)), "Non-finite velocities after Ray simulation"


# ---------------------------------------------------------------------------
# Test 3: Uneven chunks (N not divisible by n_workers)
# ---------------------------------------------------------------------------

def test_ray_handles_uneven_chunks():
    # N=53, n_workers=4 — last chunk gets the remainder
    N, n_workers = 53, 4
    assert N % n_workers != 0, "Test requires N not divisible by n_workers"

    pos, vel, mass = make_initial_conditions(N)
    for _ in range(3):
        pos, vel = nbody_ray_step(pos, vel, mass, n_workers=n_workers)

    assert pos.shape == (N, 3)
    assert np.all(np.isfinite(pos))


# ---------------------------------------------------------------------------
# Test 4: Ray shuts down cleanly (no hanging processes)
# ---------------------------------------------------------------------------

def test_ray_shutdown_clean():
    # Ray is initialised by the module fixture — verify it is running first
    assert ray.is_initialized(), "Ray should be initialized before shutdown test"

    ray.shutdown()
    assert not ray.is_initialized(), "Ray should not be initialized after shutdown"

    # Re-initialise so the module fixture teardown (ray.shutdown) is a safe no-op
    ray.init(num_cpus=4, ignore_reinit_error=True)
