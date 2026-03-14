"""
Phase 3 — Multi-core scaling benchmark.

Measures steps/second for varying worker counts and body counts,
then plots time-per-step and speedup curves so Amdahl's Law is
visible in the data.

Usage:
    python benchmark.py
"""
import os
import time
import multiprocessing
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from main import nbody_step, nbody_parallel_step_persistent


def make_initial_conditions(N, seed=42):
    np.random.seed(seed)
    pos  = np.random.randn(N, 3) * 10.0
    vel  = np.random.randn(N, 3) * 0.1
    mass = np.random.uniform(0.5, 2.0, N)
    return pos, vel, mass


def time_steps(pos, vel, mass, n_workers, steps, G=1.0, dt=0.01, softening=0.1):
    """
    Run `steps` simulation steps and return seconds-per-step.

    n_workers=1  → single-process nbody_step (no Pool overhead at all)
    n_workers>1  → persistent Pool created once, reused every step
    """
    pos, vel = pos.copy(), vel.copy()

    if n_workers == 1:
        start = time.perf_counter()
        for _ in range(steps):
            pos, vel = nbody_step(pos, vel, mass, G=G, dt=dt, softening=softening)
        elapsed = time.perf_counter() - start
    else:
        with Pool(n_workers) as pool:
            # Warm up: one throw-away step so process import/JIT costs
            # don't skew the first timed step.
            pos, vel = nbody_parallel_step_persistent(
                pos, vel, mass, pool, n_workers, G=G, dt=dt, softening=softening
            )
            start = time.perf_counter()
            for _ in range(steps):
                pos, vel = nbody_parallel_step_persistent(
                    pos, vel, mass, pool, n_workers, G=G, dt=dt, softening=softening
                )
            elapsed = time.perf_counter() - start

    return elapsed / steps


def sweep_workers(N, steps=5, max_workers=None):
    """
    Benchmark n_workers in [1 .. max_workers] for a fixed N.
    Returns dict {n_workers: seconds_per_step}.
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    pos, vel, mass = make_initial_conditions(N)
    results = {}

    print(f"\nN={N}, {steps} timed steps per worker count")
    print(f"{'workers':>8}  {'ms/step':>10}  {'speedup':>8}")
    print("-" * 34)

    baseline = None
    for n_workers in range(1, max_workers + 1):
        secs = time_steps(pos, vel, mass, n_workers, steps)
        results[n_workers] = secs
        if baseline is None:
            baseline = secs
        speedup = baseline / secs
        print(f"{n_workers:>8}  {secs * 1000:>10.1f}  {speedup:>8.2f}x")

    return results


def run_benchmark(N_values=(200, 500, 1000), steps=5, max_workers=None):
    """
    Run sweep_workers for each N value.
    Returns dict {N: {n_workers: seconds_per_step}}.
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    print(f"CPU count: {max_workers}")
    print("Pool is created once per worker-count and reused across steps.")

    all_results = {}
    for N in N_values:
        all_results[N] = sweep_workers(N, steps=steps, max_workers=max_workers)

    return all_results


def plot_results(results_by_N, save_path='benchmark_results.png'):
    """
    Two side-by-side charts:
      Left  — ms per step vs worker count
      Right — speedup vs worker count with ideal (linear) reference
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("N-Body Multiprocessing Benchmark  (persistent pool)", fontsize=13)

    all_workers = sorted(next(iter(results_by_N.values())).keys())

    for N, results in results_by_N.items():
        workers  = sorted(results.keys())
        times_ms = [results[w] * 1000 for w in workers]
        baseline = results[1]
        speedups = [baseline / results[w] for w in workers]

        ax1.plot(workers, times_ms, marker='o', label=f'N={N}')
        ax2.plot(workers, speedups, marker='o', label=f'N={N}')

    # Ideal linear speedup
    ax2.plot(all_workers, all_workers, 'k--', linewidth=1, label='ideal')

    ax1.set_xlabel('Workers')
    ax1.set_ylabel('ms per step')
    ax1.set_title('Time per step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Workers')
    ax2.set_ylabel('Speedup vs 1 worker')
    ax2.set_title("Speedup  (Amdahl's Law in practice)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"\nPlot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    results = run_benchmark(N_values=(200, 500, 1000), steps=5)
    plot_results(results)
