# N-Body Problem — Development Plan

## Environment

Always use `.venv/Scripts/python.exe` for all Python and pytest commands.
Full path: `C:/Users/tcw32/PycharmProjects/n_body_problem1/.venv/Scripts/python.exe`

## Project Goal

Build a scalable n-body gravitational simulation with:
- Parallelism scaling from a single CPU to multiple machines / cloud servers
- A visual front end showing bodies moving in real time
- Incremental development as a learning exercise

## Starting Point

`main.py` — a working multiprocessing simulation that:
- Splits bodies into chunks, one chunk per worker process
- Computes gravitational forces independently per chunk (no shared state)
- Reassembles forces, then updates velocities and positions with a simple Euler integrator

---

## Development Phases

### Phase 1 — Vectorized Core
Replace the Python `for` loops in `compute_forces_chunk` with NumPy vectorized operations.

**Learn:** NumPy broadcasting, why vectorization is 10-100x faster than loops, memory vs. compute tradeoffs.

**Key change:** Compute all pairwise forces for a chunk in one NumPy expression instead of two nested loops.

**Tests (`test_phase1.py`):**
- Force on a single body from a known 2-body configuration matches analytic result
- Vectorized output matches loop-based output for N=10 random bodies (regression)
- Force on body i from body j is equal and opposite to force on body j from body i (Newton's 3rd law)
- Zero net force when all bodies are equidistant (symmetry)
- Softening parameter prevents division by zero when two bodies occupy the same position

---

### Phase 2 — Basic Visualization (single machine, live)
Add real-time 2D visualization using `matplotlib.animation` or `pygame`.

**Learn:** The simulation/render loop separation, how to decouple compute speed from display framerate.

**Key change:** Replace `print` with a live animated scatter plot. Reduce N to ~100 for interactive speed.

**Tests (`test_phase2.py`):**
- Simulation produces the correct number of position snapshots over N steps
- Each snapshot has shape `(N, 3)` with finite (non-NaN, non-inf) values
- Bodies do not teleport: max displacement per step is bounded by `|vel|*dt + epsilon`
- Visualization module imports and constructs without raising exceptions (smoke test)

---

### Phase 3 — Multi-core Scaling Benchmark
Benchmark worker count vs. wall time, understand Amdahl's Law in practice.

**Learn:** Process spawn overhead, when more workers stops helping, how to profile Python with `cProfile` and `time`.

**Key change:** Add a timing harness that sweeps `n_workers` from 1 to CPU count and plots throughput.

**Tests (`test_phase3.py`):**
- 1-worker result is numerically identical to the serial (single-process) result
- N-worker and 1-worker runs produce the same final positions to floating-point tolerance
- Timing harness returns a result for every worker count in the sweep without error
- Speedup is positive (multi-worker is not slower than single-worker for large N)

---

### Phase 4 — Structured Integrators
Replace the simple Euler integrator (`vel += acc*dt; pos += vel*dt`) with Leapfrog (symplectic), which conserves energy far better.

**Learn:** Numerical integration, why the choice of integrator matters for long simulations, energy drift.

**Key change:** Implement `leapfrog_step`. Add an energy monitor to compare Euler vs. Leapfrog drift over time.

**Tests (`test_phase4.py`):**
- Total energy (KE + PE) is conserved to within 0.1% over 1000 steps for Leapfrog (circular 2-body orbit)
- Euler integrator fails the same energy conservation threshold (confirms Leapfrog is better)
- Total momentum is conserved to machine precision for both integrators
- 2-body circular orbit remains circular (eccentricity stays near zero) under Leapfrog
- `leapfrog_step` and `euler_step` produce the same positions at t=0+dt to first order in dt

---

### Phase 5 — Distributed Computing with Ray
Replace `multiprocessing.Pool` with Ray, which works identically on one machine or a cluster with almost no code change.

**Learn:** The actor/task model, serialization, why Ray is a clean path from laptop to cloud.

**Key change:** Replace `Pool.map` with `ray.remote` tasks. Simulation logic stays the same — only worker dispatch changes.

**Tests (`test_phase5.py`):**
- Ray worker output matches multiprocessing worker output for the same input (numerical equivalence)
- Simulation completes without error when Ray is initialised with 1 and with 4 workers
- Ray tasks handle uneven chunk sizes (N not divisible by n_workers) correctly
- Teardown: Ray shuts down cleanly after simulation without hanging processes

---

### Phase 6 — Web Frontend (decoupled visualization)
Replace matplotlib with a browser-based visualization using WebSockets.

**Architecture:**
```
Simulation backend (Python/Ray)
        |
    WebSocket server (FastAPI)
        |
  Browser frontend (HTML5 Canvas or Three.js)
```

**Learn:** Client/server separation, streaming data to a browser, basic async Python.

**Tests (`test_phase6.py`):**
- WebSocket server starts and accepts a connection without error (async smoke test)
- Server sends one JSON frame per simulation step containing positions array of correct shape
- Client receives at least N frames in N*dt seconds (throughput check)
- Malformed or closed client connection does not crash the server
- Serialized position payload round-trips correctly (serialize → deserialize → compare)

---

### Phase 7 — Cloud Deployment
Run the simulation on multiple machines (cloud VMs or Ray cluster).

**Learn:** Ray cluster setup, Docker basics, how distributed state is managed.

**Options in order of simplicity:**
1. Ray on two local machines (LAN)
2. Ray on cloud VMs (AWS/GCP free tier)
3. Ray on Kubernetes (advanced)

**Tests (`test_phase7.py`):**
- Docker image builds without error (`docker build` exits 0)
- Container starts and runs a 10-step simulation, then exits cleanly
- Simulation output from containerised run matches local run for same seed and parameters
- Ray cluster health-check endpoint returns healthy status after startup
- End-to-end: submit job to cluster, retrieve results, verify numerical correctness

---

## Milestone Summary

| Phase | Milestone |
|-------|-----------|
| 1 | Vectorize force computation |
| 2 | Live matplotlib visualization |
| 3 | Benchmark multi-core scaling |
| 4 | Leapfrog integrator + energy monitor |
| 5 | Swap multiprocessing → Ray |
| 6 | Web frontend via WebSocket |
| 7 | Deploy to cloud cluster |
