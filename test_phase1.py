"""
Phase 1 tests — Vectorized force computation.

Run with:  pytest test_phase1.py -v
"""
import numpy as np
import pytest
from main import compute_forces_chunk, compute_forces_chunk_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_args(pos, mass, i_start=0, i_end=None, G=1.0, softening=0.1):
    if i_end is None:
        i_end = len(mass)
    return (i_start, i_end, pos, mass, G, softening)


# ---------------------------------------------------------------------------
# Test 1: analytic 2-body result
#
# Two bodies on the x-axis separated by distance d.
# Body 0 at origin, body 1 at (d, 0, 0).
# Force on body 0 = G * m0 * m1 / (d^2 + s^2)^1.5 * (d, 0, 0)  (towards body 1)
# ---------------------------------------------------------------------------

def test_two_body_analytic():
    G, softening = 1.0, 0.0
    d = 3.0
    pos  = np.array([[0.0, 0.0, 0.0],
                     [d,   0.0, 0.0]])
    mass = np.array([1.0, 2.0])

    _, force = compute_forces_chunk(make_args(pos, mass, 0, 1, G=G, softening=softening))

    expected_x = G * mass[1] / d ** 2   # only x-component, softening=0
    assert force.shape == (1, 3)
    assert force[0, 0] == pytest.approx(expected_x, rel=1e-9)
    assert force[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert force[0, 2] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 2: vectorized output matches loop output (regression)
# ---------------------------------------------------------------------------

def test_vectorized_matches_loop():
    np.random.seed(0)
    N = 10
    pos  = np.random.randn(N, 3)
    mass = np.random.uniform(0.5, 2.0, N)
    args = make_args(pos, mass)

    _, force_vec  = compute_forces_chunk(args)
    _, force_loop = compute_forces_chunk_loop(args)

    np.testing.assert_allclose(force_vec, force_loop, rtol=1e-10,
                               err_msg="Vectorized and loop results differ")


# ---------------------------------------------------------------------------
# Test 3: Newton's third law — acceleration(i←j) == -acceleration(j←i)
#         for equal-mass bodies.
#
# Note: compute_forces_chunk returns gravitational acceleration
#   a_i = G * sum_j  m_j / r_ij^3 * (r_j - r_i)
# not force (m_i is absent). For a 2-body system with equal masses m,
# a_0 = G*m/r^3*(r_1-r_0)  and  a_1 = G*m/r^3*(r_0-r_1) = -a_0,
# so the accelerations are equal and opposite iff masses are equal.
# ---------------------------------------------------------------------------

def test_newtons_third_law():
    np.random.seed(1)
    pos  = np.array([[0.0, 0.0, 0.0],
                     [1.0, 2.0, 3.0]])
    mass = np.array([1.5, 1.5])          # equal masses required (see docstring)
    G, softening = 1.0, 0.05

    _, f0 = compute_forces_chunk(make_args(pos, mass, 0, 1, G=G, softening=softening))
    _, f1 = compute_forces_chunk(make_args(pos, mass, 1, 2, G=G, softening=softening))

    np.testing.assert_allclose(f0[0], -f1[0], rtol=1e-10,
                               err_msg="Accelerations not equal-and-opposite for equal masses")


# ---------------------------------------------------------------------------
# Test 4: symmetry — net force on the central body from a symmetric ring is zero
#
# Place 4 equal-mass bodies at the corners of a square around the origin.
# By symmetry the net force on any one of them from the other three is not
# necessarily zero, but the force on a 5th body placed exactly at the centre
# must be zero.
# ---------------------------------------------------------------------------

def test_symmetric_ring_zero_net_force():
    G, softening = 1.0, 1e-6
    r = 2.0
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)   # 5 ring bodies
    ring_pos = np.column_stack([r * np.cos(angles),
                                r * np.sin(angles),
                                np.zeros(5)])
    centre   = np.array([[0.0, 0.0, 0.0]])
    pos  = np.vstack([centre, ring_pos])                     # body 0 = centre
    mass = np.ones(len(pos))

    # Force on body 0 (centre) from all others
    _, force = compute_forces_chunk(make_args(pos, mass, 0, 1, G=G, softening=softening))

    np.testing.assert_allclose(force[0], 0.0, atol=1e-10,
                               err_msg="Net force on central body should be zero by symmetry")


# ---------------------------------------------------------------------------
# Test 5: softening prevents division by zero when two bodies coincide
# ---------------------------------------------------------------------------

def test_softening_prevents_division_by_zero():
    pos  = np.array([[1.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0]])   # identical positions
    mass = np.array([1.0, 1.0])

    _, force = compute_forces_chunk(make_args(pos, mass, softening=0.1))

    assert np.all(np.isfinite(force)), "Force contains NaN or Inf with coincident bodies"
