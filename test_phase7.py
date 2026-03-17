"""
Phase 7 tests — Docker containerisation.

These tests require Docker to be installed and the daemon to be running.
They are skipped automatically if Docker is not available.

Run with:  pytest test_phase7.py -v
"""
import json
import subprocess
import numpy as np
import pytest

from sim_runner import run

IMAGE = "nbody:latest"


# ---------------------------------------------------------------------------
# Helper: check Docker is available
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available"
)


def docker_run(*args, timeout=30) -> subprocess.CompletedProcess:
    """Run a command inside a fresh container and return the result."""
    return subprocess.run(
        ["docker", "run", "--rm", IMAGE, *args],
        capture_output=True, text=True, timeout=timeout
    )


# ---------------------------------------------------------------------------
# Test 1: Docker image builds without error
# ---------------------------------------------------------------------------

@requires_docker
def test_image_builds():
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE, "."],
        capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"docker build failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Test 2: Container runs a 10-step simulation and exits cleanly
# ---------------------------------------------------------------------------

@requires_docker
def test_container_runs_simulation_and_exits():
    result = docker_run("python", "sim_runner.py", "10", "10", "42")
    assert result.returncode == 0, f"Container exited non-zero:\n{result.stderr}"

    data = json.loads(result.stdout)
    assert data["N"]     == 10
    assert data["steps"] == 10
    assert data["seed"]  == 42


# ---------------------------------------------------------------------------
# Test 3: Container output matches local output for the same seed
# ---------------------------------------------------------------------------

@requires_docker
def test_container_output_matches_local():
    N, steps, seed = 10, 10, 42

    # Local result
    local = run(N=N, steps=steps, seed=seed)

    # Container result
    result = docker_run("python", "sim_runner.py", str(N), str(steps), str(seed))
    assert result.returncode == 0, f"Container failed:\n{result.stderr}"
    container = json.loads(result.stdout)

    np.testing.assert_allclose(
        np.array(container["pos"]), np.array(local["pos"]), rtol=1e-10,
        err_msg="Container positions differ from local"
    )
    np.testing.assert_allclose(
        np.array(container["vel"]), np.array(local["vel"]), rtol=1e-10,
        err_msg="Container velocities differ from local"
    )
