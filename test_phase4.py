"""
Phase 4 tests — Leapfrog integrator and energy conservation.

2-body circular orbit setup:
  G = 1, m1 = m2 = 1, separation r = 2
  Both bodies orbit the common centre of mass at radius R = 1.
  Required orbital speed: v = sqrt(G*m/(2*r)) = 0.5
  Period: T = 2*pi*R/v = 4*pi ≈ 12.57
"""
import numpy as np
import pytest
from main import euler_step, leapfrog_step, compute_energy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def circular_orbit(softening=0.1):
    """Two equal masses on a circular orbit around their common centre of mass."""
    pos  = np.array([[-1., 0., 0.],
                     [ 1., 0., 0.]])
    vel  = np.array([[ 0., -0.5, 0.],
                     [ 0.,  0.5, 0.]])
    mass = np.array([1., 1.])
    return pos, vel, mass


def run_steps(step_fn, pos, vel, mass, steps, **kw):
    for _ in range(steps):
        pos, vel = step_fn(pos, vel, mass, **kw)
    return pos, vel


# ---------------------------------------------------------------------------
# Test 1: Leapfrog conserves energy to 0.1% over 1000 steps
# ---------------------------------------------------------------------------

def test_leapfrog_energy_conservation():
    pos, vel, mass = circular_orbit()
    G, dt, softening = 1.0, 0.01, 0.1
    kw = dict(G=G, dt=dt, softening=softening)

    E0 = compute_energy(pos, vel, mass, G=G, softening=softening)
    pos, vel = run_steps(leapfrog_step, pos, vel, mass, steps=1000, **kw)
    E1 = compute_energy(pos, vel, mass, G=G, softening=softening)

    relative_drift = abs(E1 - E0) / abs(E0)
    assert relative_drift < 0.001, (
        f"Leapfrog energy drifted {relative_drift:.4%} — expected < 0.1%"
    )


# ---------------------------------------------------------------------------
# Test 2: Euler integrator fails the 0.1% threshold (energy drift is larger)
# ---------------------------------------------------------------------------

def test_euler_energy_drift_exceeds_threshold():
    pos, vel, mass = circular_orbit()
    G, dt, softening = 1.0, 0.01, 0.1
    kw = dict(G=G, dt=dt, softening=softening)

    E0 = compute_energy(pos, vel, mass, G=G, softening=softening)
    pos, vel = run_steps(euler_step, pos, vel, mass, steps=1000, **kw)
    E1 = compute_energy(pos, vel, mass, G=G, softening=softening)

    relative_drift = abs(E1 - E0) / abs(E0)
    assert relative_drift > 0.001, (
        f"Euler energy only drifted {relative_drift:.4%} — expected Euler to exceed 0.1%"
    )


# ---------------------------------------------------------------------------
# Test 3: Total momentum is conserved to machine precision (both integrators)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step_fn", [euler_step, leapfrog_step])
def test_momentum_conservation(step_fn):
    pos, vel, mass = circular_orbit()
    kw = dict(G=1.0, dt=0.01, softening=0.1)

    p0 = np.sum(mass[:, np.newaxis] * vel, axis=0)
    pos, vel = run_steps(step_fn, pos, vel, mass, steps=500, **kw)
    p1 = np.sum(mass[:, np.newaxis] * vel, axis=0)

    assert np.allclose(p0, p1, atol=1e-10), (
        f"{step_fn.__name__}: momentum changed by {np.max(np.abs(p1 - p0)):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: 2-body circular orbit stays circular under Leapfrog
# ---------------------------------------------------------------------------

def test_leapfrog_orbit_stays_circular():
    """
    The inter-body separation should remain close to its initial value
    throughout the simulation (orbit doesn't spiral in or out noticeably).
    """
    pos, vel, mass = circular_orbit()
    kw = dict(G=1.0, dt=0.01, softening=0.1)
    r0 = np.linalg.norm(pos[1] - pos[0])

    separations = []
    for _ in range(1000):
        pos, vel = leapfrog_step(pos, vel, mass, **kw)
        separations.append(np.linalg.norm(pos[1] - pos[0]))

    max_deviation = max(abs(r - r0) / r0 for r in separations)
    assert max_deviation < 0.05, (
        f"Orbit deviated {max_deviation:.2%} from circular — expected < 5%"
    )


# ---------------------------------------------------------------------------
# Test 5: leapfrog_step and euler_step agree at t=dt to first order in dt
# ---------------------------------------------------------------------------

def test_leapfrog_euler_agree_first_order():
    """
    Both integrators are identical to first order in dt: pos = pos0 + vel*dt + O(dt^2).
    The difference between them (O(dt^2)) must be much smaller than the
    displacement itself (O(dt)).
    """
    pos, vel, mass = circular_orbit()
    dt = 0.001
    kw = dict(G=1.0, dt=dt, softening=0.1)

    pos_e, _ = euler_step(pos.copy(), vel.copy(), mass, **kw)
    pos_lf, _ = leapfrog_step(pos.copy(), vel.copy(), mass, **kw)

    displacement = np.max(np.abs(pos_e - pos))   # O(dt)
    diff = np.max(np.abs(pos_e - pos_lf))         # O(dt^2)

    # diff / displacement ~ O(dt) = 0.001, so the ratio should be < dt*10
    assert diff < displacement * dt * 10, (
        f"Difference {diff:.2e} is not small enough relative to displacement {displacement:.2e}"
    )
