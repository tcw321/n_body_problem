"""
Phase 6 tests — WebSocket server and browser visualization.

Uses Starlette's TestClient, which runs the ASGI app (including its lifespan
and simulation loop) in a background thread. This avoids asyncio event-loop
scope conflicts on Windows and keeps tests straightforward synchronous code.

Run with:  pytest test_phase6.py -v
"""
import json
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from ws_server import create_app, numpy_array_to_payload, payload_to_numpy, N, DT


# ---------------------------------------------------------------------------
# Shared app fixture — one app instance for all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Start the app (including simulation loop lifespan) once for the module."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Test 1: Server accepts a WebSocket connection without error
# ---------------------------------------------------------------------------

def test_websocket_accepts_connection(client):
    with client.websocket_connect("/ws") as ws:
        assert ws is not None


# ---------------------------------------------------------------------------
# Test 2: Server sends one JSON frame per step with correct shape
# ---------------------------------------------------------------------------

def test_frame_shape_and_step_increments(client):
    with client.websocket_connect("/ws") as ws:
        prev_step = None
        for _ in range(5):
            raw   = ws.receive_text()
            frame = json.loads(raw)

            assert frame["N"] == N
            assert len(frame["bodies"]) == N
            assert all(k in frame["bodies"][0] for k in ("x", "y", "z", "mass"))

            if prev_step is not None:
                assert frame["step"] == prev_step + 1, "Steps must increment by 1"
            prev_step = frame["step"]


# ---------------------------------------------------------------------------
# Test 3: Client receives at least N frames within N * TICK_INTERVAL * 2 seconds
# ---------------------------------------------------------------------------

def test_throughput(client):
    EXPECTED = 30
    TICK     = 1 / 60
    deadline = EXPECTED * TICK * 2   # 2× ideal time as margin

    with client.websocket_connect("/ws") as ws:
        t_start = time.monotonic()
        for _ in range(EXPECTED):
            ws.receive_text()
        elapsed = time.monotonic() - t_start

    assert elapsed <= deadline, (
        f"Received {EXPECTED} frames in {elapsed:.2f}s, expected ≤ {deadline:.2f}s"
    )


# ---------------------------------------------------------------------------
# Test 4: Abruptly closed client does not crash the server
# ---------------------------------------------------------------------------

def test_bad_client_does_not_crash_server(client):
    # Open and immediately close without a proper WS close handshake
    with client.websocket_connect("/ws"):
        pass   # context manager closes on exit

    # Give the server a moment to process the disconnect
    time.sleep(0.2)

    # A new clean connection should still receive valid frames
    with client.websocket_connect("/ws") as ws:
        frame = json.loads(ws.receive_text())
        assert frame["N"] == N


# ---------------------------------------------------------------------------
# Test 5: Payload round-trips correctly (pure function — no server needed)
# ---------------------------------------------------------------------------

def test_payload_round_trip():
    rng  = np.random.default_rng(0)
    pos  = rng.standard_normal((N, 3))
    mass = rng.uniform(0.5, 2.0, N)
    step = 99

    payload             = numpy_array_to_payload(pos, mass, step)
    pos_rt, mass_rt, step_rt = payload_to_numpy(payload)

    assert step_rt == step
    np.testing.assert_allclose(pos_rt,  pos,  rtol=1e-10)
    np.testing.assert_allclose(mass_rt, mass, rtol=1e-10)
