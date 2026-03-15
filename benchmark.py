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


def memory_required_gb(N, n_workers):
    """
    Estimate peak RAM for one simulation step.

    The dominant allocation is the diff array inside compute_forces_chunk:
        shape (chunk_size, N, 3)  float64
    where chunk_size = N // n_workers.

    Each worker holds one such array simultaneously, so total peak memory
    across all workers is n_workers × chunk_size × N × 3 × 8 bytes
    = N² × 3 × 8 bytes  (independent of n_workers, because total work is N²).

    However what matters for feasibility is whether any single worker's
    allocation fits in RAM, so we return the per-worker figure.
    """
    chunk_size = N // n_workers
    return chunk_size * N * 3 * 8 / 1e9   # GB per worker


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


def sweep_workers(N, steps=5, max_workers=None, memory_limit_gb=4.0):
    """
    Benchmark n_workers in [1 .. max_workers] for a fixed N.
    Returns dict {n_workers: seconds_per_step}.

    Skips the entire sweep and returns {} if the per-worker memory
    requirement exceeds memory_limit_gb (protects against OOM crashes).
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    # Check feasibility at max workers (best-case memory usage)
    mem_gb = memory_required_gb(N, max_workers)
    if mem_gb > memory_limit_gb:
        print(f"\nN={N}  SKIPPED")
        print(f"  Per-worker diff array requires {mem_gb:.1f} GB "
              f"(limit {memory_limit_gb} GB).")
        print(f"  N={N} needs an O(N log N) algorithm (e.g. Barnes-Hut) "
              f"to be tractable.")
        return {}

    pos, vel, mass = make_initial_conditions(N)
    results = {}

    print(f"\nN={N}, {steps} timed steps per worker count  "
          f"(peak RAM per worker: {memory_required_gb(N, 1)*1000:.0f} MB @ 1 worker)")
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


# Steps to use per N value — fewer steps for large N since each step is much slower
_STEPS_FOR_N = {200: 5, 500: 5, 1000: 5, 10_000: 2, 100_000: 2}


def run_benchmark(N_values=(200, 500, 1000, 10_000, 100_000),
                  max_workers=None, memory_limit_gb=4.0):
    """
    Run sweep_workers for each N value.
    Returns dict {N: {n_workers: seconds_per_step}}  (empty dict for skipped N).
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    print(f"CPU count: {max_workers}")
    print("Pool is created once per worker-count and reused across steps.")

    all_results = {}
    for N in N_values:
        steps = _STEPS_FOR_N.get(N, 3)
        all_results[N] = sweep_workers(N, steps=steps,
                                       max_workers=max_workers,
                                       memory_limit_gb=memory_limit_gb)

    return all_results


def plot_results(results_by_N, save_path='benchmark_results.png'):
    """
    Two side-by-side charts:
      Left  — ms per step vs worker count
      Right — speedup vs worker count with ideal (linear) reference

    Skipped N values (empty dicts) are omitted from the plot.
    """
    # Drop any N that was skipped
    plotable = {N: r for N, r in results_by_N.items() if r}
    if not plotable:
        print("No results to plot (all N values were skipped).")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("N-Body Multiprocessing Benchmark  (persistent pool)", fontsize=13)

    all_workers = sorted(next(iter(plotable.values())).keys())

    for N, results in plotable.items():
        workers  = sorted(results.keys())
        times_ms = [results[w] * 1000 for w in workers]
        baseline = results[1]
        speedups = [baseline / results[w] for w in workers]

        ax1.plot(workers, times_ms, marker='o', label=f'N={N:,}')
        ax2.plot(workers, speedups, marker='o', label=f'N={N:,}')

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
    results = run_benchmark(N_values=(200, 500, 1000))
    plot_results(results)
