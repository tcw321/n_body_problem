"""
Phase 2 tests — Visualization and simulation output.

Run with:  pytest test_phase2.py -v
"""
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display required

import numpy as np
import pytest
import matplotlib.pyplot as plt
from main import nbody_step, compute_forces_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_conditions(N=20, seed=42):
    np.random.seed(seed)
    pos  = np.random.randn(N, 3) * 5.0
    vel  = np.random.randn(N, 3) * 0.1
    mass = np.random.uniform(0.5, 2.0, N)
    return pos, vel, mass


# ---------------------------------------------------------------------------
# Test 1: simulation produces the correct number of snapshots
# ---------------------------------------------------------------------------

def test_snapshot_count():
    N, steps = 20, 30
    pos, vel, mass = make_initial_conditions(N)
    snapshots = []
    for _ in range(steps):
        pos, vel = nbody_step(pos, vel, mass)
        snapshots.append(pos.copy())
    assert len(snapshots) == steps


# ---------------------------------------------------------------------------
# Test 2: every snapshot has shape (N, 3) and contains only finite values
# ---------------------------------------------------------------------------

def test_snapshot_shape_and_finite():
    N, steps = 20, 50
    pos, vel, mass = make_initial_conditions(N)
    for _ in range(steps):
        pos, vel = nbody_step(pos, vel, mass)
        assert pos.shape == (N, 3), f"Expected shape ({N}, 3), got {pos.shape}"
        assert np.all(np.isfinite(pos)), "Position contains NaN or Inf"
        assert np.all(np.isfinite(vel)), "Velocity contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 3: bodies do not teleport
#
# After one step:  pos_new = pos + (vel + acc*dt) * dt
# Maximum possible displacement = (|vel|_max + |acc|_max * dt) * dt
# ---------------------------------------------------------------------------

def test_no_teleport():
    N = 20
    G, dt, softening = 1.0, 0.01, 0.3
    pos, vel, mass = make_initial_conditions(N)

    _, acc = compute_forces_chunk((0, N, pos, mass, G, softening))
    max_speed = np.max(np.linalg.norm(vel, axis=1))
    max_acc   = np.max(np.linalg.norm(acc, axis=1))
    max_allowed = (max_speed + max_acc * dt) * dt

    pos_before = pos.copy()
    pos_after, _ = nbody_step(pos, vel, mass, G=G, dt=dt, softening=softening)

    displacements = np.linalg.norm(pos_after - pos_before, axis=1)
    assert np.all(displacements <= max_allowed + 1e-10), (
        f"Max displacement {displacements.max():.6f} exceeds bound {max_allowed:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4: visualization smoke test
#
# create_animation must import cleanly and return a Figure and FuncAnimation
# without raising any exception. Uses the Agg backend (no display needed).
# ---------------------------------------------------------------------------

def test_visualization_constructs_without_error():
    from visualize import create_animation

    N = 10
    pos, vel, mass = make_initial_conditions(N)
    fig, ani = create_animation(pos, vel, mass)

    assert fig is not None
    assert ani is not None
    plt.close(fig)
