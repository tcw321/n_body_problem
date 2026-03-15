"""
Phase 6 — FastAPI WebSocket server for real-time n-body visualization.

Each WebSocket connection runs its own simulation loop — clean, stateless per
connection, no shared state issues.

Run:
    uvicorn ws_server:app --reload

Then open http://localhost:8000 in a browser.
"""
import json
import time

import anyio
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from main import leapfrog_step
from visualize import circular_orbits_ic

# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------

N             = 20
G             = 1.0
DT            = 0.005
SOFTENING     = 0.1
TICK_INTERVAL = 1 / 60   # target ~60 fps broadcast rate


# ---------------------------------------------------------------------------
# Serialization helpers (pure functions — tested independently)
# ---------------------------------------------------------------------------

def numpy_array_to_payload(pos: np.ndarray, mass: np.ndarray, step: int) -> str:
    """Serialize simulation state to a JSON string."""
    bodies = [
        {"x": float(pos[i, 0]), "y": float(pos[i, 1]),
         "z": float(pos[i, 2]), "mass": float(mass[i])}
        for i in range(len(mass))
    ]
    return json.dumps({"step": step, "t": round(step * DT, 6),
                       "N": len(mass), "bodies": bodies})


def payload_to_numpy(payload: str):
    """Deserialize a JSON frame back to (pos, mass, step) numpy arrays."""
    data   = json.loads(payload)
    bodies = data["bodies"]
    pos    = np.array([[b["x"], b["y"], b["z"]] for b in bodies])
    mass   = np.array([b["mass"] for b in bodies])
    return pos, mass, data["step"]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Return a fresh FastAPI instance.  Using a factory keeps each test isolated.
    Each WebSocket connection runs its own independent simulation — no shared
    mutable state between connections.
    """
    _app = FastAPI()

    @_app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()

        pos, vel, mass = circular_orbits_ic(N=N, G=G)
        step = 0

        try:
            while True:
                tick_start = time.monotonic()

                pos, vel = leapfrog_step(pos, vel, mass,
                                         G=G, dt=DT, softening=SOFTENING)
                step += 1
                await ws.send_text(numpy_array_to_payload(pos, mass, step))

                elapsed    = time.monotonic() - tick_start
                sleep_for  = max(0.0, TICK_INTERVAL - elapsed)
                await anyio.sleep(sleep_for)

        except Exception:
            pass   # client disconnected — exit silently

    _app.mount("/", StaticFiles(directory="static", html=True), name="static")

    return _app


# Module-level app instance for uvicorn
app = create_app()
